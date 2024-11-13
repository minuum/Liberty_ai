from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Dict, List
import operator
from dotenv import load_dotenv
import os
import logging
from data_processor import LegalDataProcessor
from search_engine import LegalSearchEngine
from pinecone import Pinecone
import streamlit as st
from langchain import hub 

# 로깅 설정
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
 
# 환경 변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# 시스템 프롬프트 정의
SYSTEM_PROMPT = """당신은 법률 전문 AI 어시스턴트입니다. 
주어진 법률 문서와 판례를 기반으로 정확하고 객관적인 답변을 제공해야 합니다.

다음 지침을 따라주세요:
1. 법률 용어를 정확하게 사용하세요
2. 관련 판례와 법령을 인용할 때는 출처를 명시하세요
3. 불확실한 내용에 대해서는 명확히 그 불확실성을 표현하세요
4. 개인의 구체적인 법률 자문이 필요한 경우, 전문 법률가와의 상담을 권장하세요

컨텍스트: {context}
질문: {question}

답변 형식:
1. 관련 법령/판례 요약
2. 구체적 답변
3. 주의사항/제한사항
"""

class AgentState(TypedDict):
    """에이전트 상태 정의"""
    question: str
    context: List[str]
    answer: str
    previous_answer: str
    rewrite_count: int
    rewrite_weight: float
    previous_weight: float
    original_weight: float
    combined_score: float

class LegalAgent:
    def __init__(self):
        """법률 에이전트 초기화"""
        try:
            # Pinecone 초기화
            pc = Pinecone(api_key=PINECONE_API_KEY)
            self.pinecone_index = pc.Index(PINECONE_INDEX_NAME)
            logger.info("Pinecone 인덱스 초기화 완료")
            
            # 데이터 프로세서 초기화
            self.data_processor = LegalDataProcessor(
                pinecone_api_key=PINECONE_API_KEY,
                index_name=PINECONE_INDEX_NAME,
                load_encoder=True,
                encoder_path="sparse_encoder.pkl"
            )
            logger.info("데이터 프로세서 초기화 완료")
            
            # sparse encoder 가져오기
            sparse_encoder = self.data_processor.get_encoder()
            if sparse_encoder is None:
                raise ValueError("Sparse encoder를 로드하지 못했습니다.")
            logger.info("Sparse encoder 로드 완료")
            
            # 검색 엔진 초기화
            self.search_engine = LegalSearchEngine(
                pinecone_index=self.pinecone_index,
                namespace="liberty-rag-json-namespace-02",
                use_combined_check=True
            )
            logger.info("검색 엔진 초기화 완료")
            
            # 하이브리드 검색기 설정
            self.search_engine.setup_hybrid_retriever(sparse_encoder)
            logger.info("하이브리드 검색기 설정 완료")
            
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
            self.rewrite_prompt = ChatPromptTemplate.from_messages([
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
            logger.info("프롬프트 로드 완료")
            
        except Exception as e:
            logger.error(f"에이전트 초기화 중 오류 발생: {str(e)}")
            raise
        
    def _create_workflow(self) -> StateGraph:
        """워크플로우 그래프 생성"""
        workflow = StateGraph(AgentState)
        
        # 노드 추가
        workflow.add_node("retrieve", self._retrieve_document)
        workflow.add_node("llm_answer", self._llm_answer)
        workflow.add_node("rewrite", self._rewrite)
        workflow.add_node("relevance_check", self._relevance_check)
        
        # 엣지 추가
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "llm_answer")
        workflow.add_edge("llm_answer", "relevance_check")
        
        # 조건부 엣지 추가
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
        
    def _retrieve_document(self, state: AgentState) -> AgentState:
        """문서 검색"""
        try:
            logger.info(f"""
            === RETRIEVE NODE 디버깅 ===
            검색 쿼리: {state["question"]}
            하이브리드 검색기 상태: {self.search_engine.hybrid_retriever is not None}
            네임스페이스: {self.search_engine.namespace}
            """)
            
            context = self.search_engine.hybrid_search(state["question"])
            
            logger.info(f"""
            검색 결과:
            - 문서 수: {len(context)}
            - 첫 번째 문서 길이: {len(context[0].page_content) if context else 0}
            - 메타데이터: {context[0].metadata if context else None}
            """)
            
            return AgentState(
                question=state["question"],
                context=context,
                rewrite_count=state.get("rewrite_count", 0)
            )
        except Exception as e:
            logger.error(f"문서 검색 중 오류: {str(e)}")
            raise

    def _llm_answer(self, state: AgentState) -> AgentState:
        """LLM 답변 생성"""
        try:
            logger.info(f"""
            === LLM_ANSWER NODE 진입 ===
            질문: {state["question"]}
            재작성 횟수: {state.get("rewrite_count", 0)}
            이전 답변 존재: {"있음" if state.get("previous_answer") else "없음"}
            재작성 가중치: {state.get("rewrite_weight", 0.0)}
            """)
            
            # 중요한 에러 로그만 남김
            if state.get("rewrite_weight") is None:
                logger.error(f"rewrite_weight is None in state")
            
            # ERROR 레벨로 변경하여 반드시 보이도록 함
            logger.error(f"State received in _llm_answer: {state}")
            logger.error(f"rewrite_weight value: {state.get('rewrite_weight')}")
            context = "\n\n".join([
                f"문서 {i+1}:\n{doc.page_content}" 
                for i, doc in enumerate(state["context"])
            ]) if state["context"] else ""
            
            # 디버깅을 위한 상태 로깅 추가
            logger.error(f"Current state values:")
            logger.error(f"rewrite_weight: {state.get('rewrite_weight')}")
            logger.error(f"previous_weight: {state.get('previous_weight')}")
            logger.error(f"rewrite_count: {state.get('rewrite_count')}")
            
            # None 체크 추가
            rewrite_weight = state.get("rewrite_weight")
            if rewrite_weight is None:
                logger.error(f"rewrite_weight is None in state: {state}")
                rewrite_weight = 0.0  # 기본값 설정
            
            try:
                exploration_type = "More exploratory and critical" if rewrite_weight > 0.3 else "Slightly refined"
            except TypeError as e:
                logger.error(f"TypeError in exploration_type comparison: {e}")
                logger.error(f"rewrite_weight type: {type(rewrite_weight)}")
                raise
            
            # 새로운 프롬프트 템플릿
            answer_prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    """You are analyzing this question for the {rewrite_count}th time.
                    
                    Previous weight: {previous_weight:.2f}
                    Current weight: {rewrite_weight:.2f}
                    
                    This weight progression indicates:
                    - Initial response (weight 0.0): Direct answer from context
                    - Current response (weight {rewrite_weight:.2f}): {exploration_type}
                    
                    Your task is to {revision_type} the previous interpretation.
                    
                    Context:
                    {context}
                    
                    Previous answer:
                    {previous_answer}
                    
                    Remember to maintain legal accuracy and provide Korean responses."""
                ),
                ("human", "{question}")
            ])
            
            # 체인 실행
            chain = answer_prompt | self.llm | StrOutputParser()
            
            # 기본값 설정
            rewrite_weight = state.get("rewrite_weight", 0.0)
            exploration_type = "More exploratory and critical" if rewrite_weight > 0.3 else "Slightly refined"
            revision_type = "significantly revise" if rewrite_weight > 0.3 else "slightly adjust"
            
            response = chain.invoke({
                "question": state["question"],
                "context": context,
                "previous_answer": state.get("previous_answer", ""),
                "rewrite_count": state.get("rewrite_count", 0),
                "previous_weight": state.get("previous_weight", 0.0),  # 기본값 추가
                "rewrite_weight": rewrite_weight,  # 이미 기본값이 설정된 변수 사용
                "exploration_type": exploration_type,
                "revision_type": revision_type
            })
            
            # 수정된 부분: state 업데이트 방식 변경
            updated_state = state.copy()
            updated_state["previous_answer"] = state.get("answer", "")
            updated_state["answer"] = response
            
            logger.info(f"""
            === LLM_ANSWER NODE 종료 ===
            답변 길이: {len(response)}
            다음 노드: relevance_check
            """)
            
            return AgentState(**updated_state)
            
        except Exception as e:
            logger.error(f"답변 생성 중 오류: {str(e)}")
            raise

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
                f"문서 {i+1}:\n{doc.page_content}" 
                for i, doc in enumerate(state["context"])
            ]) if state["context"] else ""
            
            # 새로운 재작성 프롬프트
            rewrite_prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    """Rewrite iteration: {rewrite_count}
                    Weight change: {previous_weight:.2f} → {rewrite_weight:.2f}
                    
                    As the weight increases:
                    1. Question complexity should increase
                    2. Scope should broaden
                    3. Alternative viewpoints should be explored
                    
                    Current stage requires: {revision_requirement}
                    
                    Generate an improved question in Korean."""
                ),
                (
                    "human",
                    "Initial question:\n------- \n{question}\n------- \n"
                    "Context:\n------- \n{context}\n------- \n"
                    "Previous answer:\n------- \n{answer}\n------- \n"
                )
            ])
            
            chain = rewrite_prompt | self.llm | StrOutputParser()
            
            revision_requirement = "major reframing" if rewrite_weight > 0.3 else "minor refinement"
            
            new_question = chain.invoke({
                "question": state["question"],
                "context": context,
                "answer": state["answer"],
                "rewrite_count": rewrite_count,
                "previous_weight": previous_weight,
                "rewrite_weight": rewrite_weight,
                "revision_requirement": revision_requirement
            })
            
            # 상태 로깅 추가
            logger.info(f"""
            Weight Progress:
            - Iteration: {rewrite_count}
            - Previous Weight: {previous_weight:.2f}
            - Current Weight: {rewrite_weight:.2f}
            - Expected Changes: {'Significant' if rewrite_weight > 0.3 else 'Minor'}
            """)
            
            logger.info(f"""
            === REWRITE NODE 종료 ===
            새로운 질문: {new_question}
            새로운 가중치: {rewrite_weight}
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
            raise

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
        if score >= 0.7:
            return "grounded"
        elif score <= 0.3:
            return "notGrounded"
        return "notSure"

    def process_query(self, question: str) -> Dict:
        """질문 처리 및 답변 생성"""
        try:
            initial_state = AgentState(
                question=question,
                context=[],
                answer="",
                previous_answer="",
                rewrite_count=0,
                rewrite_weight=0.0,
                previous_weight=0.0,
                original_weight=1.0,
                combined_score=0.0
            )
            
            # 초기 상태 로깅
            logger.info(f"Initial state created with values:")
            logger.info(f"rewrite_weight: {initial_state.get('rewrite_weight')}")
            logger.info(f"previous_weight: {initial_state.get('previous_weight')}")
            
            config = RunnableConfig(
                recursion_limit=5,
                configurable={"thread_id": "LEGAL-AGENT"}
            )
            
            result = self.workflow.invoke(initial_state, config=config)
            return {
                "answer": result["answer"],
                "confidence": result["combined_score"],
                "rewrites": result["rewrite_count"]
            }
        except Exception as e:
            logger.error(f"Error in process_query: {str(e)}")
            logger.error(f"State at error: {initial_state}")
            raise
            
# Streamlit UI용 함수
def create_ui():
    
    
    st.title("법률 AI 어시스턴트")
    
    # 초기화 상태 표시
    init_status = st.empty()
    
    # 세션 상태 초기화
    if "initialized" not in st.session_state:
        init_status.info("시스템을 초기화하는 중입니다...")
        try:
            # Pinecone 초기화 확인
            pc = Pinecone(api_key=PINECONE_API_KEY)
            index = pc.Index(PINECONE_INDEX_NAME)
            logger.info("Pinecone 연결 성공")
            
            # 에이전트 초기화
            st.session_state.agent = LegalAgent()
            st.session_state.messages = []
            st.session_state.initialized = True
            
            init_status.success("시스템 초기화가 완료되었습니다!")
            logger.info("에이전트 초기화 성공")
            
        except Exception as e:
            error_msg = f"시스템 초기화 실패: {str(e)}"
            init_status.error(error_msg)
            logger.error(error_msg)
            st.stop()
    
    # 초기화 완료 후 UI 표시
    if st.session_state.initialized:
        # 채팅 히스토리 표시
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
        # 사용자 입력
        if prompt := st.chat_input("질문을 입력하세요"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
                
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown("답변을 생성하는 중입니다...")
                
                try:
                    response = st.session_state.agent.process_query(prompt)
                    
                    if "error" in response:
                        full_response = f"오류가 발생했습니다: {response['error']}"
                    else:
                        full_response = f"""
                        {response['answer']}
                        
                        신뢰도: {response['confidence']:.2f}
                        질문 재작성 횟수: {response['rewrites']}    
                        """
                        
                    message_placeholder.markdown(full_response)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": full_response}
                    )
                    
                except Exception as e:
                    error_message = f"답변 생성 중 오류 발생: {str(e)}"
                    logger.error(error_message)
                    message_placeholder.markdown(error_message)

if __name__ == "__main__":
    try:
        create_ui()
    except Exception as e:
        logger.error(f"애플리케이션 실행 중 오류 발생: {str(e)}")
        st.error(f"애플리케이션 오류: {str(e)}")
