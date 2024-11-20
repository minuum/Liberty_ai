from pinecone import Pinecone
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain import hub
from typing import Dict, List, Optional, Union, TypedDict
from langchain.schema import Document
import logging
import time
from dotenv import load_dotenv
import os
from data_processor import LegalDataProcessor
from search_engine import LegalSearchEngine
from langgraph.graph import StateGraph, END
from langchain_core.output_parsers import StrOutputParser
from datetime import datetime
from langchain.schema.runnable import RunnableConfig
import re

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    """에이전트 상태 정의"""
    question: str
    context: Union[List[Union[Document, str]], Dict[str, List[str]]]
    answer: str
    previous_answer: str
    rewrite_count: int
    rewrite_weight: float
    previous_weight: float
    original_weight: float
    combined_score: float

class LegalAgent:
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(LegalAgent, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, cache_mode: bool = False):
        """법률 에이전트 초기화"""
        if not self._initialized:
            try:
                # 기본 컴포넌트 초기화
                self._initialize_components(cache_mode)
                
                # 워크플로우 생성
                self.workflow = self._create_workflow()
                
                self._initialized = True
                
            except Exception as e:
                logger.error(f"에이전트 초기화 중 오류 발생: {str(e)}")
                raise

    def _initialize_components(self, cache_mode: bool):
        """기본 컴포넌트 초기화"""
        try:
            # Pinecone 초기화
            pc = Pinecone(api_key=PINECONE_API_KEY)
            self.pinecone_index = pc.Index(PINECONE_INDEX_NAME)
            logger.info("Pinecone 인덱스 초기화 완료")
            stats = self.pinecone_index.describe_index_stats()
            #logger.info(f"인덱스 통계: {stats}")
            
            # 데이터 프로세서 초기화
            self.data_processor = LegalDataProcessor(
                pinecone_api_key=PINECONE_API_KEY,
                index_name=PINECONE_INDEX_NAME,
                cache_dir="./liberty_agent/cached_vectors",
                cache_mode=cache_mode,
                encoder_path='./liberty_agent/KiwiBM25_sparse_encoder.pkl'
            )
            logger.info("데이터 프로세서 초기화 완료")
            
            # 리트리버 생성 (캐시 사용)
            retrievers, sparse_encoder = self.data_processor.create_retrievers(
                documents=None,
                use_faiss=True,
                use_kiwi=True,
                use_pinecone=True,
                cache_mode="load"
            )
            
            # 검색 엔진 초기화
            self.search_engine = LegalSearchEngine(
                retrievers=retrievers,
                sparse_encoder=sparse_encoder,
                pinecone_index=self.pinecone_index,
                namespace="liberty-db-namespace-legal-agent",
                cache_dir="./cached_vectors/search_engine"
            )
            logger.info("검색 엔진 초기화 완료")
            
            # 세션 종료 시 저장
            if cache_mode:
                import atexit
                atexit.register(
                    self.data_processor.save_retrievers,
                    retrievers=retrievers
                )
            
            # LLM 초기화
            self.llm = ChatOpenAI(
                model="gpt-4o-2024-08-06",
                temperature=0.1,
                api_key=OPENAI_API_KEY
            )
            logger.info("LLM 초기화 완료")
            
            # 프롬프트 로드
            self.answer_prompt = hub.pull("minuum/liberty-rag")
            self.rewrite_prompt = self._create_rewrite_prompt()
            
        except Exception as e:
            logger.error(f"에이전트 초기화 중 오류 발생: {str(e)}")
            raise

    def _create_workflow(self):
        """워크플로우 생성"""
        try:
            workflow = StateGraph(AgentState)
            
            # 1. 노드 추가
            workflow.add_node("retrieve", self._retrieve)
            workflow.add_node("quick_filter", self._quick_relevance_check)
            workflow.add_node("llm_answer", self._llm_answer)
            workflow.add_node("quality_check", self._detailed_quality_check)
            workflow.add_node("rewrite", self._rewrite)
            
            # 2. 시작점 설정
            workflow.set_entry_point("retrieve")
            
            # 3. 기본 플로우 연결
            workflow.add_edge("retrieve", "quick_filter")
            workflow.add_edge("llm_answer", "quality_check")
            
            # 4. 조건부 엣지 설정
            workflow.add_conditional_edges(
                "quick_filter",
                self._should_proceed,
                {
                    "proceed": "llm_answer",
                    "rewrite": "rewrite"
                }
            )
            
            workflow.add_conditional_edges(
                "quality_check",
                self._route_by_quality,
                {
                    "proceed": END,
                    "rewrite": "rewrite",
                    "re_retrieve": "retrieve"
                }
            )
            
            # 5. 재작성 노드 연결
            workflow.add_edge("rewrite", "retrieve")
            
            logger.info("""
            === 워크플로우 생성 완료 ===
            노드:
            - retrieve (시작)
            - quick_filter
            - llm_answer
            - quality_check
            - rewrite
            
            주요 경로:
            1. retrieve -> quick_filter -> llm_answer -> quality_check -> END
            2. retrieve -> quick_filter -> rewrite -> retrieve
            3. retrieve -> quick_filter -> llm_answer -> quality_check -> rewrite -> retrieve
            """)
            
            return workflow.compile()
            
        except Exception as e:
            logger.error(f"워크플로우 생성 중 오류: {str(e)}")
            raise

    def _route_by_relevance(self, state: AgentState) -> str:
        """라우팅 로직"""
        if state["rewrite_count"] >= 3:  # 재작성 횟수 제한
            return "grounded"  # 강제 종료
        
        if state["combined_score"] > 0.8:
            return "grounded"
        elif state["combined_score"] < 0.3:
            return "notGrounded"
        else:
            return "notSure"

    def _safe_retrieve(self, state: AgentState, max_retries: int = 3) -> AgentState:
        """검색 실패 시 복구 전략"""
        for attempt in range(max_retries):
            try:
                return self._retrieve(state)
            except Exception as e:
                logger.warning(f"검색 시도 {attempt + 1} 실패: {str(e)}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(1)  # 재시도 전 대기

    def _retrieve(self, state: AgentState) -> AgentState:
        """문서 검색"""
        try:
            logger.info(f"""
            === RETRIEVE 시작 ===
            질문: {state["question"]}
            재시도 횟수: {state.get("rewrite_count", 0)}
            """)
            
            # 재시도 횟수 너무 많으면 기본 응답 반환
            if state.get("rewrite_count", 0) >= 3:
                logger.warning("재시도 횟수 초과로 폴백 응답 반환")
                return self._create_fallback_response(state)
            
            # 검색 실행
            results = self.search_engine.search(state["question"])
            
            # 검색 결과 로깅
            logger.info(f"""
            === 검색 결과 ===
            결과 개수: {len(results)}
            첫 번째 결과 스코어: {results[0].metadata.get('adjusted_score') if results else 'N/A'}
            """)
            
            # 검색 결과가 없거나 빈 경우 폴백 메커니즘 실행
            if not results:
                logger.warning("검색 결과 없음 - 폴백 응답 생성")
                return self._create_fallback_response(state)
            
            # 상태 업데이트
            updated_state = state.copy()
            updated_state["context"] = results
            
            logger.info(f"""
            === RETRIEVE 완료 ===
            컨텍스트 길이: {len(results)}
            """)
            
            return AgentState(**updated_state)
            
        except Exception as e:
            logger.error(f"""
            === RETRIEVE 오류 ===
            오류 메시지: {str(e)}
            상태: {state}
            """)
            return self._create_fallback_response(state)

    def _create_fallback_response(self, state: AgentState) -> AgentState:
        """폴백 응답 생성"""
        # 질문 유형에 따른 기본 응답 선택
        basic_responses = {
            # 이혼/가족 관련 세부 응답
            "이혼 절차": """
            이혼 절차에 대한 기본 정보를 안내해드립니다:
            1. 협의이혼 절차
            2. 재판이혼 절차
            3. 필요 서류 안내
            4. 이혼 숙려기간
            자세한 상담은 법률전문가와 상담하시기를 권장드립니다.
            """,
            "위자료": """
            위자료에 대한 기본 정보를 안내해드립니다:
            1. 위자료 청구 요건
            2. 위자료 산정 기준
            3. 청구 절차
            4. 지급 방법
            자세한 상담은 법률전문가와 상담하시기를 권장드립니다.
            """,
            "양육권": """
            양육권에 대한 기본 정보를 안내해드립니다:
            1. 양육권자 결정 기준
            2. 양육비 산정
            3. 면접교섭권
            4. 양육권 변경
            자세한 상담은 법률전문가와 상담하시기를 권장드립니다.
            """,
            "재산분할": """
            재산분할에 대한 기본 정보를 안내해드립니다:
            1. 분할대상 재산 범위
            2. 분할 비율
            3. 청구 절차
            4. 시효
            자세한 상담은 법률전문가와 상담하시기를 권장드립니다.
            """,

            # 상속 관련 세부 응답
            "상속 순위": """
            상속 순위에 대한 기본 정보를 안내해드립니다:
            1. 법정상속인의 순위
            2. 상속분 산정
            3. 대습상속
            4. 상속인 결격사유
            자세한 상담은 법률전문가와 상담하시기를 권장드립니다.
            """,
            "유류분": """
            유류분에 대한 기본 정보를 안내해드립니다:
            1. 유류분 권리자
            2. 유류분 산정방법
            3. 청구 절차
            4. 시효
            자세한 상담은 법률전문가와 상담하시기를 권장드립니다.
            """,
            "상속포기": """
            상속포기에 대한 기본 정보를 안내해드립니다:
            1. 포기 절차
            2. 제출 서류
            3. 기간 제한
            4. 효력
            자세한 상담은 법률전문가와 상담하시기를 권장드립니다.
            """,
            "유언장": """
            유언장에 대한 기본 정보를 안내해드립니다:
            1. 유언의 방식
            2. 필수 요건
            3. 효력 발생
            4. 검인 절차
            자세한 상담은 법률전문가와 상담하시기를 권장드립니다.
            """,

            # 계약 관련 세부 응답
            "계약서 작성": """
            계약서 작성에 대한 기본 정보를 안내해드립니다:
            1. 필수 기재사항
            2. 계약조항 검토
            3. 특약사항 작성
            4. 서명 날인
            자세한 상담은 법률전문가와 상담하시기를 권장드립니다.
            """,
            "계약 해지": """
            계약 해지에 대한 기본 정보를 안내해드립니다:
            1. 해지 사유
            2. 해지 통보
            3. 위약금
            4. 손해배상
            자세한 상담은 법률전문가와 상담하시기를 권장드립니다.
            """,
            "손해배상": """
            손해배상에 대한 기본 정보를 안내해드립니다:
            1. 배상 범위
            2. 청구 절차
            3. 입증 방법
            4. 시효
            자세한 상담은 법률전문가와 상담하시기를 권장드립니다.
            """,
            "보증": """
            보증에 대한 기본 정보를 안내해드립니다:
            1. 보증의 종류
            2. 보증인의 책임
            3. 보증계약 체결
            4. 구상권
            자세한 상담은 법률전문가와 상담하시기를 권장드립니다.
            """,

            # 부동산 관련 세부 응답
            "매매": """
            부동산 매매에 대한 기본 정보를 안내해드립니다:
            1. 계약 절차
            2. 중도금 지급
            3. 소유권 이전
            4. 등기 절차
            자세한 상담은 법률전문가와 상담하시기를 권장드립니다.
            """,
            "임대차": """
            임대차에 대한 기본 정보를 안내해드립니다:
            1. 계약 체결
            2. 임차인 보호
            3. 보증금 반환
            4. 계약 갱신
            자세한 상담은 법률전문가와 상담하시기를 권장드립니다.
            """,
            "등기": """
            등기에 대한 기본 정보를 안내해드립니다:
            1. 등기 종류
            2. 신청 절차
            3. 구비서류
            4. 등기비용
            자세한 상담은 법률전문가와 상담하시기를 권장드립니다.
            """,
            "재개발": """
            재개발에 대한 기본 정보를 안내해드립니다:
            1. 사업 절차
            2. 조합 설립
            3. 권리산정
            4. 이주대책
            자세한 상담은 법률전문가와 상담하시기를 권장드립니다.
            """,

            # 형사 관련 세부 응답
            "고소/고발": """
            고소/고발에 대한 기본 정보를 안내해드립니다:
            1. 소/고발 방법
            2. 처리 절차
            3. 취하 방법
            4. 불기소 불복
            자세한 상담은 법률전문가와 상담하시기를 권장드립니다.
            """,
            "변호사 선임": """
            변호사 선임에 대한 기본 정보를 안내해드립니다:
            1. 국선변호인
            2. 사선변호인
            3. 선임 시기
            4. 비용
            자세한 상담은 법률전문가와 상담하시기를 권장드립니다.
            """,
            "형사절차": """
            형사절차에 대한 기본 정보를 안내해드립니다:
            1. 수사 절차
            2. 기소 여부
            3. 재판 진행
            4. 형 집행
            자세한 상담은 법률전문가와 상담하시기를 권장드립니다.
            """,
            "보석": """
            보석에 대한 기본 정보를 안내해드립니다:
            1. 신청 요건
            2. 절차
            3. 보증금
            4. 준수사항
            자세한 상담은 법률전문가와 상담하시기를 권장드립니다.
            """
        }
        
        # 질문 분석하여 적절한 응답 선택
        response = None
        for category, resp in basic_responses.items():
            if category in state["question"]:
                response = resp
                break
        
        if not response:
            response = "죄송합니다. 현재 해당 질문에 대한 정확한 답변을 제공하기 어렵습니다. 가까운 법률구조공단이나 변호사와 상담하시기를 권장드립니다."
        
        return AgentState(
            question=state["question"],
            context=[Document(page_content=response, metadata={"source": "fallback"})],
            answer=response,
            previous_answer=state.get("previous_answer", ""),
            rewrite_count=state.get("rewrite_count", 0),
            rewrite_weight=state.get("rewrite_weight", 0.0),
            previous_weight=state.get("previous_weight", 0.0),
            original_weight=state.get("original_weight", 1.0),
            combined_score=0.5  # 기본 응답의 신뢰도 점수
        )

    def _llm_answer(self, state: AgentState) -> AgentState:
        """LLM 사용한 답변 생성"""
        try:
            logger.info(f"""
                === LLM_ANSWER NODE 진입 ===
                질문: {state["question"]}
                컨텍스트 수: {len(state.get("context", []))}
            """)
            
            if not state.get("context"):
                logger.warning("컨텍스트 없음 - 폴백 응답 생성")
                return self._create_fallback_response(state)
            
            context = self._normalize_context(state["context"])
            context_text = "\n\n".join(self._safe_get_content(doc) for doc in context)
            
            chain = self.answer_prompt | self.llm | StrOutputParser()
            raw_answer = chain.invoke({
                "context": context_text,
                "question": state["question"],
                "original_weight": state.get("original_weight", 1.0),
                "rewrite_weight": state.get("rewrite_weight", 0.0)
            })
            
            formatted_answer = self._format_answer(raw_answer, state["context"])
            logger.info(f"formatted_answer: {formatted_answer}")
            updated_state = state.copy()
            updated_state["answer"] = formatted_answer
            
            logger.info(f"""
                === LLM_ANSWER NODE 완료 ===
                답변 길이: {len(formatted_answer)}
                컨텍스트 활용: {len(context)} documents
            """)
            
            return AgentState(**updated_state)
            
        except Exception as e:
            logger.error(f"답변 생성 중 오류: {str(e)}")
            return self._create_error_state(state)

    def _format_answer(self, answer: str, context: List[Document | str]) -> str:
        """답변 포맷팅"""
        references = []
        for doc in context:
            if isinstance(doc, Document):
                meta = doc.metadata
                ref = {
                    "판례번호": meta.get("caseNo", ""),
                    "법원": meta.get("courtName", ""),
                    "판결일자": meta.get("judgementDate", ""),
                    "사건명": meta.get("caseName", ""),
                    "사건종류": meta.get("caseType", "")
                }
                if any(ref.values()):
                    references.append(ref)

        formatted_answer = f"답변:\n{answer}"
        
        if references:
            formatted_answer += "\n\n참고 판례:"
            for i, ref in enumerate(references, 1):
                formatted_answer += f"""
{i}. {ref['법원']} {ref['판례번호']}
   - 판결일자: {ref['판결일자']}
   - 사건명: {ref['사건명']}
   - 사건종류: {ref['사건종류']}
"""
        
        return formatted_answer
    def _create_error_state(self, state: AgentState) -> AgentState:
        """에러 상태 생성"""
        return AgentState(
            question=state["question"],
            context=[Document(page_content="검색 시스템에 일시적인 문제가 발생했습니다.")],
            answer="죄송합니다. 일시적인 오류가 발생했습니다.",
            previous_answer="",
            rewrite_count=state.get("rewrite_count", 0),
            rewrite_weight=0.0,
            previous_weight=0.0,
            original_weight=1.0,
            combined_score=0.0
        )

    def _create_rewrite_prompt(self):
        """재작성 프롬프트 생성"""
        return ChatPromptTemplate.from_messages([
            (                
                    "system",
                    "You are a professional prompt rewriter. Your task is to generate questions to obtain additional information not shown in the given context. "
                    "Your generated questions will be used for web searches to find relevant information. "
                    "Consider the rewrite weight ({rewrite_weight:.2f}) to adjust the credibility of the previous answer. "
                    "The higher the weight, the more you should doubt the previous answer and focus on finding new information."
                    "The weight is calculated based on the number of times the question has been rewritten. "
                    "The higher the weight, the more you should doubt the previous answer and focus on finding new information."
                ),
                (
                    "human",
                    "Rewrite the question to obtain additional information for the answer. "
                    "\n\nInitial question:\n ------- \n{question}\n ------- \n"
                    "\n\nInitial context:\n ------- \n{context}\n ------- \n"
                    "\n\nInitial answer to the question:\n ------- \n{answer}\n ------- \n"
                    "\n\nRewrite weight: {rewrite_weight:.2f} (The higher this value, the more you should doubt the previous answer)"
                    "\n\nFormulate an improved question in Korean:"
                )
            ])

    def _rewrite(self, state: AgentState) -> AgentState:
        """질문 재작성"""
        try:
            logger.info(f"""
            === REWRITE NODE 진입 ===
            원래 질문: {state["question"]}
            현재 재작성 횟수: {state.get("rewrite_count", 0)}
            이전 가중치: {state.get("rewrite_weight", 0.0)}
            """)
            
            previous_weight = state.get("rewrite_weight", 0)
            rewrite_count = state.get("rewrite_count", 0) + 1
            rewrite_weight = min(rewrite_count * 0.1, 0.5)
            
            context = "\n\n".join([
                safe_get_content(doc) for doc in state["context"]
            ]) if state["context"] else ""
            
            revision_requirement = "major reframing" if rewrite_weight > 0.3 else "minor refinement"
            
            chain = self.rewrite_prompt | self.llm | StrOutputParser()
            new_question = chain.invoke({
                "question": state["question"],
                "context": context,
                "answer": state["answer"],
                "rewrite_count": rewrite_count,
                "previous_weight": previous_weight,
                "rewrite_weight": rewrite_weight,
                "revision_requirement": revision_requirement
            })
            
            logger.info(f"""
            === REWRITE NODE 종료 ===
            새로운 질문: {new_question}
            새로운 가중치: {rewrite_weight}
            반복 횟수: {rewrite_count}
            이전 가중치: {previous_weight:.2f}
            현재 가중치: {rewrite_weight:.2f}
            다음 노드: retrieve
            """)
            
            return AgentState(
                question=new_question,
                context=[],
                answer="",
                previous_answer=state["answer"],
                rewrite_count=rewrite_count,
                rewrite_weight=rewrite_weight,
                previous_weight=previous_weight,
                original_weight=state.get("original_weight", 1.0),
                combined_score=0.0
            )
            
        except Exception as e:
            logger.error(f"질문 재작성 중 오류: {str(e)}")
            return self._create_error_state(state)

    def _relevance_check(self, state: AgentState) -> AgentState:
            """답변 관련성 검사"""
            try:
                logger.info(f"""
                === RELEVANCE_CHECK NODE 진입 ===
                재작성 횟수: {state.get("rewrite_count", 0)}
                컨텍스트 수: {len(state.get("context", []))}
                답변 길이: {len(state.get("answer", ""))}
                """)
                
                # 컨텍스트 검증
                if not state.get("context"):
                    logger.warning("컨텍스트 없음 - notGrounded 반환")
                    #state["relevance"] = "notGrounded"
                    return self._update_state_score(state, 0.0, "notGrounded")
                
                # 신뢰도 계산
                combined_score = self.search_engine.validate_answer(
                    context=state.get("context", ""),
                    answer=state.get("answer", "")
                )
                
                # 결과 결정
                if combined_score >= 0.8:
                    return self._update_state_score(state, combined_score, "grounded")
                elif combined_score <= 0.3:
                    return self._update_state_score(state, combined_score, "notGrounded")
                else:
                    return self._update_state_score(state, combined_score, "notSure")
                    
            except Exception as e:
                logger.error(f"관련성 검사 중 오류: {str(e)}")
                return self._create_error_state(state)

    def _is_relevant(self, state: AgentState) -> str:
        """관련성 상태 반환"""
        return state["relevance"]

    # def _calculate_combined_score(
    #     self, 
    #     upstage_response: str, 
    #     kobert_score: float
    # ) -> float:
    #     """결합 점수 계산"""
    #     upstage_weight = 0.6
    #     kobert_weight = 0.4
        
    #     # upstage_response가 딕셔너리인 경우를 처리
    #     if isinstance(upstage_response, dict):
    #         # upstage_response에서 실제 응답 값을 추출
    #         upstage_result = upstage_response.get('result', 'notSure')
    #     else:
    #         upstage_result = upstage_response
        
    #     # 점수 매핑
    #     upstage_score = {
    #         "grounded": 1.0,
    #         "notGrounded": 0.0,
    #         "notSure": 0.33
    #     }.get(upstage_result, 0.0)
        
    #     return (upstage_weight * upstage_score) + (kobert_weight * kobert_score)

    def _get_relevance_status(self, score: float) -> str:
        """점수 기반 관련성 상태 결정"""
        if score >= 0.6:
            return "grounded"
        elif score <= 0.2:
            return "notGrounded"
        return "notSure"

    def process_query(self, query: str) -> Dict:
        """쿼리 처리"""
        try:
            # 워크플로우 초기 상태 설정
            initial_state = AgentState(
                question=query,
                context=[],
                answer="",
                previous_answer="",
                rewrite_count=0,
                rewrite_weight=0.0,
                previous_weight=0.0,
                original_weight=1.0,
                combined_score=0.0
            )
        
            # 워크플로우 실행
            app = self._create_workflow()
            outputs = []
            current_state = {}
            
            # 상태 스트리밍 및 처리
            for output in app.stream(initial_state):
                try:
                    # 현재 노드와 상태 추출
                    current_node = list(output.keys())[0]
                    current_state = output[current_node]
                    
                    # 주요 상태 변화 로깅
                    if current_state.get('answer'):
                        logger.info(f"""
                            Node: {current_node}
                            Answer Length: {len(current_state['answer'])}
                            Score: {current_state.get('combined_score', 0)}
                        """)
                    
                    outputs.append(current_state)
                except Exception as e:
                    logger.error(f"상태 처리 중 오류: {str(e)}")
                    continue
            
            # 최종 결과 처리
            final_state = outputs[-1] if outputs else current_state
            return {
                "answer": final_state.get("answer", "답변을 생성하지 못했습니다."),
                "confidence": final_state.get("combined_score", 0.0),
                "rewrite_count": final_state.get("rewrite_count", 0),
                "metadata": {
                    "quality_score": final_state.get("answer_quality", 0.0)
                }
            }
            
        except Exception as e:
            logger.error(f"쿼리 처리 중 오류: {str(e)}")
            return {
                "answer": "죄송합니다. 답변을 생성하는 중에 문제가 발생했습니다.",
                "confidence": 0.0
            }

    def _format_answer(self, answer: str, context: List[Document | str]) -> str:
        """답변 포맷팅 - 참고 자료 포함"""
        references = []
        for doc in context:
            if isinstance(doc, Document):
                meta = doc.metadata
                ref = {
                    "판례번호": meta.get("caseNo", ""),
                    "법원": meta.get("courtName", ""),
                    "판결일자": meta.get("judgementDate", ""),
                    "사건명": meta.get("caseName", ""),
                    "사건종류": meta.get("caseType", "")
                }
                if any(ref.values()):
                    references.append(ref)

        formatted_answer = f"""
답변:
{answer}
"""
        if references:
            formatted_answer += "\n참고 판례:"
            for i, ref in enumerate(references, 1):
                formatted_answer += f"""
{i}. {ref['법원']} {ref['판례번호']}
   - 판결일자: {ref['판결일자']}
   - 사건명: {ref['사건명']}
   - 사건종류: {ref['사건종류']}
"""
        
        return formatted_answer

    def _process_search_results(self, results: List[Document]) -> List[Document]:
        """검색 결과 처리"""
        try:
            processed_results = []
            for doc in results:
                if isinstance(doc, Document):
                    processed_results.append(doc)
                elif isinstance(doc, dict):
                    processed_results.append(Document(
                        page_content=doc.get('content', ''),
                        metadata=doc.get('metadata', {})
                    ))
            return processed_results
        except Exception as e:
            logger.error(f"검색 결과 처리 중 오류: {str(e)}")
            return []

    def _normalize_context(self, context: List[Document | str]) -> List[Document]:
        """컨텍스트 정규화"""
        try:
            normalized = []
            for item in context:
                if isinstance(item, Document):
                    normalized.append(item)
                elif isinstance(item, str):
                    normalized.append(Document(page_content=item))
            return normalized
        except Exception as e:
            logger.error(f"컨텍스트 정규화 중 오류: {str(e)}")
            return []

    def _should_continue_rewrite(self, state: AgentState) -> bool:
        """재작성 계속 여부 결정"""
        # 최대 재작성 횟수 제한
        if state.get("rewrite_count", 0) >= 3:
            return False
        
        # 이미 충분한 신뢰도를 얻은 경우
        if state.get("combined_score", 0) >= 0.6:
            return False
        
        # 컨텍스트가 충분한 경우
        if state.get("context") and len(state.get("context", [])) >= 2:
            return False
        
        return True

    def _safe_get_content(self, doc: Union[Document, str]) -> str:
        """문서 내용 안전하게 가져오기"""
        try:
            if isinstance(doc, Document):
                return doc.page_content
            return str(doc)
        except Exception as e:
            logger.error(f"문서 내용 가져오기 실패: {str(e)}")
            return ""

    def _analyze_query_intent(self, query: str) -> str:
        """쿼리 의도 분석"""
        try:
            prompt = f"""
            다음 법률 상담 질문의 의도를 분석하여 간단한 제목을 생성해주세요:
            질문: {query}

            규칙:
            1. 최대 20자 이내
            2. 핵심 법률 용어 포함
            3. 명사형으로 끝내기

            출력 형식:
            [제목만 출력]
            """
            
            response = self.llm.invoke(prompt).content
            return response.strip()
        except Exception as e:
            logger.error(f"쿼리 의도 해석 중 오류: {str(e)}")
            return f"법률상담_{datetime.now().strftime('%Y%m%d_%H%M')}"

    def _generate_answer(self, query: str, search_results: List[Document]) -> str:
        """검색 결과를 기반으로 답변 생성"""
        try:
            # 컨텍스트 준비
            contexts = []
            for doc in search_results[:3]:  # 상위 3개 문서만 사용
                if hasattr(doc, 'page_content'):
                    contexts.append(doc.page_content)
                elif isinstance(doc, str):
                    contexts.append(doc)
                
            context_text = "\n\n".join(contexts)
            
            # 프롬프트 템플릿 생성
            prompt = f"""다음 법률 질문에 대해, 제공된 컨텍스트를 기반으로 답변해주세요.

질문: {query}

컨텍스트:
{context_text}

답변 작성 규칙:
1. 정확한 법적 근거를 인용하세요
2. 이해하기 쉬운 용어를 사용하세요
3. 필요한 경우 단계별로 설명하세요
4. 주의사항이나 예외사항을 명시하세요
5. 추가 상담이 필요한 경우 이를 언급하세요

답변:"""

            # LLM을 사용하여 답변 생성
            response = self.llm.invoke(prompt).content
            
            # 답변 품질 검증
            if len(response.split()) < 10:  # 답변이 너무 짧은 경우
                return "죄송합니다. 충분한 정보를 찾지 못했습니다. 더 자세한 상담이 필요합니다."
            
            return response.strip()
        
        except Exception as e:
            logger.error(f"답변 생성 중 오류 발생: {str(e)}")
            return "죄송합니다. 답변을 생성하는 중에 문제가 발생했습니다."

    def _answer_check(self, state: AgentState) -> AgentState:
        """답변 품질 검증"""
        try:
            answer = state.get("answer", "")
            logger.info(f"""
                === ANSWER_CHECK NODE 진입 ===
                답변 길이: {len(answer)}
            """)
            
            # 답변 품질 검증
            quality_score = self._evaluate_answer_quality(answer, state["question"])
            
            # 상태 업데이트
            state["answer_quality"] = quality_score
            state["answer_status"] = "valid" if quality_score >= 0.7 else "invalid"
            
            logger.info(f"""
                === ANSWER_CHECK NODE 완료 ===
                품질 점수: {quality_score}
                상태: {state["answer_status"]}
            """)
            
            return state
            
        except Exception as e:
            logger.error(f"답변 검증 중 오류: {str(e)}")
            state["answer_status"] = "invalid"
            return state

    def _route_by_answer_quality(self, state: AgentState) -> str:
        """답변 품질에 따른 라우팅"""
        return state.get("answer_status", "invalid")

    def _evaluate_answer_quality(self, answer: str, question: str) -> float:
        """답변 품질 평가"""
        try:
            # 기본 품질 체크
            if len(answer.split()) < 20:
                return 0.3
            
            # 관련성 검사
            relevance_score = self.search_engine.validate_answer(
                context=answer,
                question=question
            )
            
            # 구조 검사
            structure_score = self._check_answer_structure(answer)
            
            # 최종 점수 계산
            return (relevance_score * 0.7 + structure_score * 0.3)
            
        except Exception as e:
            logger.error(f"답변 품질 평가 중 오류: {str(e)}")
            return 0.0

    def _check_answer_structure(self, answer: str) -> float:
        """답변 구조 검사"""
        try:
            score = 0.0
            
            # 법적 근거 포함 여부
            if "법" in answer or "조항" in answer:
                score += 0.3
            
            # 단계별 설명 포함 여부
            if "먼저" in answer or "다음" in answer:
                score += 0.3
            
            # 주의사항 포함 여부
            if "주의" in answer or "유의" in answer:
                score += 0.4
            
            return score
            
        except Exception as e:
            logger.error(f"답변 구조 검사 중 오류: {str(e)}")
            return 0.0

    def _detailed_quality_check(self, state: AgentState) -> AgentState:
        """답변 품질 상세 평가"""
        try:
            answer = state["answer"]
            context = state["context"]
            
            # Upstage + KoBERT 검증
            combined_score = self.search_engine.validate_answer(
                context=context,
                answer=answer,
                upstage_weight=0.3,  # Upstage 가중치
                kobert_weight=0.7    # KoBERT 가중치
            )
            
            # 상태 업데이트
            state["combined_score"] = combined_score
            
            logger.info(f"""
            === 상세 품질 검사 완료 ===
            결합 점수: {combined_score:.3f}
            기준:
            - 높음 (≥ 0.8): 진행
            - 중간 (0.3-0.8): 재작성
            - 낮음 (< 0.3): 재검색
            """)
            
            return state
            
        except Exception as e:
            logger.error(f"품질 검사 중 오류: {str(e)}")
            state["combined_score"] = 0.0
            return state
    def create_prompt_template(self):
        return hub.pull("minuum/liberty-rag")
        
    def _quick_relevance_check(self, state: AgentState) -> AgentState:
        """빠른 관련성 검사"""
        try:
            logger.info("""
            === QUICK_FILTER NODE 진입 ===
            컨텍스트 수: %d
            """, len(state["context"]))
            
            # 검색엔진의 evaluate_context_quality 활용
            context_quality = self.search_engine.evaluate_context_quality(
                state["context"], 
                state["question"]
            )
            
            # 빠른 필터링 점수 계산
            state["quick_filter_score"] = context_quality
            
            logger.info("""
            === QUICK_FILTER NODE 완료 ===
            필터링 점수: %.3f
            """, context_quality)
            
            return state
            
        except Exception as e:
            logger.error(f"빠른 필터링 중 오류: {str(e)}")
            return self._create_error_state(state)
        
    def _should_proceed(self, state: AgentState) -> str:
        quick_score = state.get("quick_filter_score", 0.0)
        
        if quick_score >= 0.3:  # 임계값 하향 조정
            return "proceed"
        elif state["rewrite_count"] >= 3:
            return "proceed"
        else:
            return "rewrite"
        
    def _route_by_quality(self, state: AgentState) -> str:
        """품질 점수에 따른 라우팅 결정"""
        try:
            score = state.get("combined_score", 0.0)
            rewrite_count = state.get("rewrite_count", 0)
            
            logger.info(f"""
            === 품질 기반 라우팅 ===
            품질 점수: {score:.3f}
            재작성 횟수: {rewrite_count}
            """)
            
            if score >= 0.8:
                return "proceed"
            elif score < 0.3:
                return "re_retrieve"
            else:
                return "rewrite"
                
        except Exception as e:
            logger.error(f"라우팅 결정 중 오류: {str(e)}")
            return "proceed"  # 오류 시 안전하게 진행
    def format_answer(self, answer: str) -> str:
        """답변 포맷팅"""
        # 복사 버튼 태그 제거
        answer = answer.replace('class="copy-button">', '')
        # 기타 HTML 태그 제거
        answer = re.sub(r'<[^>]+>', '', answer)
        return answer.strip()

    def selected_category_prompt(self, category: str, subcategory: str) -> str:
        """카테고리별 최적화된 프롬프트 반환"""
        
        category_prompts = {
            "이혼/가족": {
                "이혼 절차": """
                    이혼 절차에 대해 다음 내용을 포함하여 상세히 설명해주세요:
                    1. 협의이혼과 재판이혼의 차이점과 각각의 진행 절차
                    2. 필요한 서류와 준비사항 목록
                    3. 소요 기간과 비용 상세 내역
                    4. 법원 조정 절차와 진행 방법
                    5. 이혼 후 조치사항과 주의점
                    
                    답변 형식:
                    - 단계별 구분하여 설명
                    - 필요 서류 체크리스트 포함
                    - 구체적인 비용과 기간 명시
                    - 민법 제834조~제843조 관련 내용 인용
                    """,
                
                "위자료": """
                    위자료 청구와 관련하여 다음 내용을 포함하여 설명해주세요:
                    1. 위자료 청구권의 법적 근거와 성립 요건
                    2. 위자료 산정 기준과 일반적인 금액 범위
                    3. 위자료 청구 절차와 필요 서류
                    4. 위자료 증액/감액 사유와 판례
                    5. 위자료 지급 방식과 강제집행 방법
                    
                    답변 형식:
                    - 청구권자별 설명
                    - 계산 사례 포함
                    - 판례 기준 설명
                    - 민법 제843조, 제806조 관련 내용 인용
                    """,
                
                "양육권": """
                    양육권 관련하여 다음 내용을 포함하여 설명해주세요:
                    1. 양육권과 친권의 차이점
                    2. 양육권 결정 기준과 고려사항
                    3. 양육권 변경 사유와 절차
                    4. 면접교섭권의 내용과 행사방법
                    5. 양육비 산정과 청구 방법
                    
                    답변 형식:
                    - 양육권/친권 개념 명확히 구분
                    - 구체적인 사례 제시
                    - 법원 판단 기준 설명
                    - 민법 제837조, 제837조의2 관련 내용 인용
                    """,
                
                "재산분할": """
                    재산분할과 관련하여 다음 내용을 포함하여 설명해주세요:
                    1. 재산분할의 대상과 범위
                    2. 분할 비율 결정 기준과 고려사항
                    3. 재산분할 청구 절차와 시효
                    4. 숨긴 재산 발견시 대응방법
                    5. 재산분할 협의와 조정 방법
                    
                    답변 형식:
                    - 재산 유형별 분류 설명
                    - 구체적 산정 방식 제시
                    - 실제 판례 사례 포함
                    - 민법 제839조의2 관련 내용 인용
                    """
            },
            
            "상속": {
                "상속 순위": """
                    상속 순위에 대해 다음 내용을 포함하여 설명해주세요:
                    1. 법정상속인의 순위와 상속분
                    2. 대습상속의 요건과 효과
                    3. 상속인 결격사유와 효과
                    4. 상속인 확인 방법과 절차
                    5. 상속순위 관련 분쟁 해결방법
                    
                    답변 형식:
                    - 상속순위 도표 형식 설명
                    - 구체적 사례로 설명
                    - 상속분 계산 예시 포함
                    - 민법 제1000조~제1003조 관련 내용 인용
                    """,
                
                "유류분": """
                    유류분에 대해 다음 내용을 포함하여 설명해주세요:
                    1. 유류분의 법적 의미와 청구권자
                    2. 유류분 산정 방법과 범위
                    3. 유류분 청구 절차와 시효
                    4. 유류분 반환 방법과 범위
                    5. 유류분 포기와 합의 방법
                    
                    답변 형식:
                    - 청구권자별 설명
                    - 계산 사례 포함
                    - 판례 기준 설명
                    - 민법 제1112조~제1118조 관련 내용 인용
                    """,
                
                "상속포기": """
                    상속포기에 대해 다음 내용을 포함하여 설명해주세요:
                    1. 상속포기의 요건과 효과
                    2. 포기 신고 절차와 필요서류
                    3. 포기 기간과 철회 가능성
                    4. 한정승인과의 차이점
                    5. 포기 후 법적 효과
                    
                    답변 형식:
                    - 절차별 상세 설명
                    - 필요 서류 목록 제시
                    - 주의사항 강조
                    - 민법 제1041조~제1044조 관련 내용 인용
                    """,
                
                "유언장": """
                    유언장 작성과 관련하여 다음 내용을 포함하여 설명해주세요:
                    1. 유언장의 종류와 효력
                    2. 유언장 작성 방법과 요건
                    3. 유언 집행 절차
                    4. 유언장 보관과 공개 방법
                    5. 유언 무효 사유와 대응
                    
                    답변 형식:
                    - 유언 종류별 설명
                    - 작성 요령 상세 안내
                    - 구체적 사례 포함
                    - 민법 제1060조~제1072조 관련 내용 인용
                    """
            },
            
            "계약": {
                "계약서 작성": """
                    계약서 작성에 대해 다음 내용을 포함하여 설명해주세요:
                    1. 계약서 필수 기재사항
                    2. 계약 조항별 주의사항
                    3. 특약 조항 작성 방법
                    4. 계약서 검토 포인트
                    5. 공증 절차와 효과
                    
                    답변 형식:
                    - 항목별 체크리스트
                    - 조항별 예시문구
                    - 일반적 실수사례
                    - 민법 제527조~제535조 관련 내용 인용
                    """,
                
                "계약 해지": """
                    계약 해지에 대해 다음 내용을 포함하여 설명해주세요:
                    1. 계약 해지와 해제의 차이
                    2. 적법한 해지 사유와 요건
                    3. 해지 통보 방법과 절차
                    4. 위약금과 손해배상
                    5. 원상회복 의무 범위
                    
                    답변 형식:
                    - 해지/해제 구분 설명
                    - 절차별 설명
                    - 구체적 사례 포함
                    - 민법 제543조~제553조 관련 내용 인용
                    """,
                
                "손해배상": """
                    손해배상에 대해 다음 내용을 포함하여 설명해주세요:
                    1. 손해배상의 요건과 범위
                    2. 배상액 산정 방법
                    3. 청구 절차와 시효
                    4. 입증책임과 증거자료
                    5. 합의와 조정 방법
                    
                    답변 형식:
                    - 배상 유형별 설명
                    - 계산 사례 포함
                    - 판례 기준 설명
                    - 민법 제390조~제399조 관련 내용 인용
                    """,
                
                "보증": """
                    보증계약에 대해 다음 내용을 포함하여 설명해주세요:
                    1. 보증의 종류와 효력
                    2. 보증인의 권리와 의무
                    3. 보증 책임의 범위
                    4. 보증인 보호 제도
                    5. 구상권 행사 방법
                    
                    답변 형식:
                    - 보증 유형별 설명
                    - 책임 범위 명확화
                    - 구체적 사례 포함
                    - 민법 제428조~제448조 관련 내용 인용
                    """
            },
            
            "부동산": {
                "매매": """
                    부동산 매매와 관련하여 다음 내용을 포함하여 설명해주세요:
                    1. 매매계약 절차와 필수사항
                    2. 계약금/중도금/잔금 지급
                    3. 소유권 이전 등기 절차
                    4. 하자담보책임과 분쟁해결
                    5. 취득세 등 세금 문제
                    
                    답변 형식:
                    - 단계별 절차 설명
                    - 체크리스트 제시
                    - 구체적 사례 포함
                    - 민법 제568조~제584조 관련 내용 인용
                    """,
                
                "임대차": """
                    부동산 임대차에 대해 다음 내용을 포함하여 설명해주세요:
                    1. 주택임대차보호법의 주요 내용
                    2. 계약 체결시 주의사항
                    3. 임차인의 권리보호 방법
                    4. 보증금 반환 절차
                    5. 분쟁 해결 방법
                    
                    답변 형식:
                    - 임차인 보호제도 설명
                    - 구체적 사례 제시
                    - 계약시 체크포인트
                    - 주택임대차보호법 관련 조항 인용
                    """,
                
                "등기": """
                    부동산 등기에 대해 다음 내용을 포함하여 설명해주세요:
                    1. 등기의 종류와 효력
                    2. 등기 신청 절차와 서류
                    3. 등기 비용과 기간
                    4. 등기부 확인사항
                    5. 등기 관련 분쟁해결
                    
                    답변 형식:
                    - 등기 종류별 설명
                    - 절차 체크리스트
                    - 비용 상세 안내
                    - 부동산등기법 관련 조항 인용
                    """,
                
                "재개발": """
                    재개발/재건축에 대해 다음 내용을 포함하여 설명해주세요:
                    1. 재개발과 재건축의 차이
                    2. 사업 진행 절차와 기간
                    3. 조합원의 권리와 의무
                    4. 분담금 산정 방식
                    5. 이주와 보상 문제
                    
                    답변 형식:
                    - 단계별 진행과정 설명
                    - 권리변동 설명
                    - 구체적 사례 포함
                    - 도시정비법 관련 조항 인용
                    """
            },
            
            "형사": {
                "고소/고발": """
                    고소/고발에 대한 기본 정보를 안내해드립니다:
                    1. 고소와 고발의 차이점
                    2. 고소장/고발장 작성 방법
                    3. 처리 절차
                    4. 불기소 불복
                    자세한 상담은 법률전문가와 상담하시기를 권장드립니다.
                    """,
                
                "변호사 선임": """
                    변호사 선임에 대한 기본 정보를 안내해드립니다:
                    1. 국선변호인
                    2. 사선변호인
                    3. 선임 시기
                    4. 비용
                    자세한 상담은 법률전문가와 상담하시기를 권장드립니다.
                    """,
                
                "형사절차": """
                    형사절차에 대한 기본 정보를 안내해드립니다:
                    1. 수사 절차
                    2. 기소 여부
                    3. 재판 진행
                    4. 형 집행
                    자세한 상담은 법률전문가와 상담하시기를 권장드립니다.
                    """,
                
                "보석": """
                    보석제도에 대해 다음 내용을 포함하여 설명해주세요:
                    1. 보석의 요건과 제한사유
                    2. 보석 신청 절차와 서류
                    3. 보석보증금 산정기준
                    4. 보석 후 준수사항
                    5. 보석 취소 사유
                    
                    답변 형식:
                    - 신청 절차 설명
                    - 보증금 기준 안내
                    - 구체적 사례 포함
                    - 형사소송법 제94조~제100조 관련 내용 인용
                    """
            }
        }
        
        # 기본 프롬프트
        default_prompt = f"""
            {subcategory}에 대해 다음과 같이 설명해주세요:
            1. 법적 정의와 의미
            2. 관련 절차와 방법
            3. 주의사항과 팁
            4. 관련 법률과 판례
            
            답변 형식:
            - 객관적 사실 중심
            - 단계별 설명
            - 구체적 예시 포함
            - 관련 법률 조항 인용
        """
        
        try:
            return category_prompts[category][subcategory].strip()
        except KeyError:
            return default_prompt.strip()       
        
def safe_get_content(doc: Union[Document, str]) -> str:
    try:
        return doc.page_content if hasattr(doc, 'page_content') else str(doc)
    except Exception as e:
        logger.warning(f"Content extraction failed: {e}")
        return str(doc)

def normalize_context(context: Union[Dict, List]) -> List[Document]:
    if isinstance(context, dict):
        return [DocumentWrapper(doc, category) 
                for category, docs in context.items() 
                for doc in docs]
    return [DocumentWrapper(doc) for doc in context]


class DocumentWrapper:
    def __init__(self, content: Union[str, Document], category: str = None):
        self.content = content
        self.category = category
        
    @property
    def page_content(self) -> str:
        if isinstance(self.content, Document):
            return self.content.page_content
        return str(self.content)



