from pinecone import Pinecone
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain import hub
from typing import Dict, List, Union, TypedDict
from langchain.schema import Document
import logging
import time
from dotenv import load_dotenv
import os
from data_processor import LegalDataProcessor
from search_engine import LegalSearchEngine
from langgraph.graph import StateGraph, END
from langchain_core.output_parsers import StrOutputParser

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
    def __init__(self, cache_mode: bool = False):
        """법률 에이전트 초기화"""
        try:
            # Pinecone 초기화
            pc = Pinecone(api_key=PINECONE_API_KEY)
            self.pinecone_index = pc.Index(PINECONE_INDEX_NAME)
            logger.info("Pinecone 인덱스 초기화 완료")
            stats = self.pinecone_index.describe_index_stats()
            logger.info(f"인덱스 통계: {stats}")
            
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
            
            # 워크플로우 초기화
            self.workflow = self._create_workflow()
            logger.info("워크플로우 초기화 완료")
            
            # 프롬프트 로드
            self.answer_prompt = hub.pull("minuum/liberty-rag")
            self.rewrite_prompt = self._create_rewrite_prompt()
            
        except Exception as e:
            logger.error(f"에이전트 초기화 중 오류 발생: {str(e)}")
            raise

    def _create_workflow(self) -> StateGraph:
        """워크플로우 그래프 생성"""
        workflow = StateGraph(AgentState)
        
        # 노드 추가
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("llm_answer", self._llm_answer)
        workflow.add_node("rewrite", self._rewrite)
        workflow.add_node("relevance_check", self._relevance_check)
        
        # 엣지 설정
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "llm_answer")
        workflow.add_edge("llm_answer", "relevance_check")
        workflow.add_conditional_edges(
            "relevance_check",
            self._is_relevant,
            {
                "grounded": END,
                "notGrounded": "rewrite",
                "notSure": "rewrite"
            }
        )
        workflow.add_edge("rewrite", "retrieve")
        
        return workflow.compile()

    
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
            === RETRIEVE NODE 디버깅 ===
            검색 쿼리: {state["question"]}
            하이브리드 검색기 상태: {hasattr(self.search_engine, 'hybrid_retriever')}
            네임스페이스: {self.search_engine.namespace}
            """)
            
            # 검색 실행
            results = self.search_engine.hybrid_search(state["question"])
            
            # 결과를 Document 객체로 변환
            processed_results = []
            
            # 딕셔너리 형태로 반환된 경우 처리
            if isinstance(results, dict):
                for category, docs in results.items():
                    for doc in docs:
                        if isinstance(doc, str):
                            processed_results.append(Document(
                                page_content=doc,
                                metadata={"category": category}
                            ))
                        elif isinstance(doc, Document):
                            processed_results.append(doc)
                        else:
                            logger.warning(f"Unexpected document type: {type(doc)}")
                            
            # 리스트 형태로 반환된 경우 처리
            elif isinstance(results, list):
                for doc in results:
                    if isinstance(doc, str):
                        processed_results.append(Document(
                            page_content=doc,
                            metadata={"category": "general"}
                        ))
                    elif isinstance(doc, Document):
                        processed_results.append(doc)
                    else:
                        logger.warning(f"Unexpected document type: {type(doc)}")
            
            # 검색 결과가 없는 경우 폴백 메커니즘 실행
            if not processed_results:
                logger.warning("검색 결과 없음 - 폴백 메커니즘 실행")
                fallback_results = self.search_engine._get_fallback_results(
                    state["question"],
                    self.search_engine._analyze_query_intent(state["question"])
                )
                processed_results = [
                    Document(
                        page_content=doc if isinstance(doc, str) else str(doc),
                        metadata={"category": "fallback"}
                    ) for doc in fallback_results
                ]
            
            # 상태 업데이트
            updated_state = state.copy()
            updated_state["context"] = processed_results
            
            logger.info(f"""
            === RETRIEVE NODE 종료 ===
            검색된 문서 수: {len(processed_results)}
            다음 노드: llm_answer
            """)
            
            return AgentState(**updated_state)
            
        except Exception as e:
            logger.error(f"문서 검색 중 오류: {str(e)}")
            # 에러 발생 시에도 기본 컨텍스트 제공
            fallback_doc = Document(
                page_content="검색 시스템에 일시적인 문제가 발생했습니다. 일반적인 법률 정보를 제공합니다.",
                metadata={"source": "fallback", "reliability": "low"}
            )
            return AgentState(
                question=state["question"],
                context=[fallback_doc],
                answer="",
                previous_answer="",
                rewrite_count=state.get("rewrite_count", 0),
                rewrite_weight=state.get("rewrite_weight", 0.0),
                previous_weight=state.get("previous_weight", 0.0),
                original_weight=state.get("original_weight", 1.0),
                combined_score=0.0
            )
    def _retrieve(self, state: AgentState) -> AgentState:
        """문서 검색"""
        try:
            results = self.search_engine.hybrid_search(state["question"])
            processed_results = self._process_search_results(results)
            
            updated_state = state.copy()
            updated_state["context"] = processed_results
            
            return AgentState(**updated_state)
            
        except Exception as e:
            logger.error(f"문서 검색 중 오류: {str(e)}")
            return self._create_error_state(state)

    def _llm_answer(self, state: AgentState) -> AgentState:
        """LLM 사용한 답변 생성"""
        try:
            context = self._normalize_context(state["context"])
            context_text = "\n\n".join(self._safe_get_content(doc) for doc in context)
            
            chain = self.answer_prompt | self.llm | StrOutputParser()
            raw_answer = chain.invoke({
                "context": context_text,
                "question": state["question"]
            })
            
            formatted_answer = self._format_answer(raw_answer, state["context"])
            
            updated_state = state.copy()
            updated_state["answer"] = formatted_answer
            
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
                답변 길이: {len(state["answer"])}
                """)
                
                # context를 문자열로 변환
                context_str = "\n\n".join([
                    f"문서 {i+1}:\n{doc.page_content}" 
                    for i, doc in enumerate(state["context"])
                ]) if state["context"] else ""
                
                # Upstage 검사
                upstage_response = self.search_engine.upstage_checker.run({
                    "context": context_str,
                    "answer": state["answer"]
                })
                
                # KoBERT 검사
                kobert_score = self.search_engine.validate_answer(
                    context=context_str,
                    answer=state["answer"]
                )
                
                # 결합 점수 계산
                combined_score = self._calculate_combined_score(
                    upstage_response, 
                    kobert_score
                )
                
                # state 복사 후 업데이트
                updated_state = state.copy()
                updated_state["relevance"] = self._get_relevance_status(combined_score)
                updated_state["combined_score"] = combined_score
                
                logger.info(f"""
                === RELEVANCE_CHECK NODE 종료 ===
                결합 점수: {combined_score:.2f}
                관련성 상태: {updated_state["relevance"]}
                다음 노드: {updated_state["relevance"]}
                """)
                
                return AgentState(**updated_state)
                
            except Exception as e:
                logger.error(f"관련성 검사 중 오류: {str(e)}")
                raise

    def _is_relevant(self, state: AgentState) -> str:
        """관련성 상태 반환"""
        return state["relevance"]

    def _calculate_combined_score(
        self, 
        upstage_response: str, 
        kobert_score: float
    ) -> float:
        """결합 점수 계산"""
        upstage_weight = 0.6
        kobert_weight = 0.4
        
        # upstage_response가 딕셔너리인 경우를 처리
        if isinstance(upstage_response, dict):
            # upstage_response에서 실제 응답 값을 추출
            upstage_result = upstage_response.get('result', 'notSure')
        else:
            upstage_result = upstage_response
        
        # 점수 매핑
        upstage_score = {
            "grounded": 1.0,
            "notGrounded": 0.0,
            "notSure": 0.33
        }.get(upstage_result, 0.0)
        
        return (upstage_weight * upstage_score) + (kobert_weight * kobert_score)

    def _get_relevance_status(self, score: float) -> str:
        """점수 기반 관련성 상태 결정"""
        if score >= 0.6:
            return "grounded"
        elif score <= 0.2:
            return "notGrounded"
        return "notSure"

    def process_query(self, query: str) -> Dict:
        """쿼리 처리 메인 함수"""
        try:
            # 초기 상태 설정
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
            
            # 그래프 실행
            final_state = self.workflow.invoke(initial_state)
            
            return {
                "answer": final_state["answer"],
                "confidence": final_state.get("combined_score", 0.0),
                "rewrites": final_state.get("rewrite_count", 0)
            }
            
        except Exception as e:
            logger.error(f"Error in process_query: {str(e)}")
            logger.error(f"State at error: {initial_state}")
            return {
                "error": "죄송합니다. 답변을 생성하는 중에 문제가 발생했습니다. 잠시 후 다시 시도해주세요.",
                "confidence": 0.0,
                "rewrites": 0
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