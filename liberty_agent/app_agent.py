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

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
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
    rewrite_count: int
    combined_score: float

class LegalAgent:
    def __init__(self):
        """법률 에이전트 초기화"""
        # 검색 엔진 초기화
        self.search_engine = LegalSearchEngine(
            pinecone_index=PINECONE_INDEX_NAME,
            namespace="legal-agent"
        )
        
        # LLM 초기화
        self.llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0.1,
            api_key=OPENAI_API_KEY
        )
        
        # 워크플로우 그래프 초기화
        self.workflow = self._create_workflow()
        
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
            context = self.search_engine.hybrid_search(state["question"])
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
            rewrite_count = state.get("rewrite_count", 0)
            rewrite_weight = min(rewrite_count * 0.1, 0.5)
            original_weight = 1 - rewrite_weight
            
            response = self.llm.invoke({
                "question": state["question"],
                "context": state["context"],
                "rewrite_weight": rewrite_weight,
                "original_weight": original_weight
            })
            
            return AgentState(
                question=state["question"],
                context=state["context"],
                answer=response,
                rewrite_count=rewrite_count,
                rewrite_weight=rewrite_weight,
                original_weight=original_weight
            )
        except Exception as e:
            logger.error(f"답변 생성 중 오류: {str(e)}")
            raise

    def _rewrite(self, state: AgentState) -> AgentState:
        """질문 재작성"""
        try:
            rewrite_count = state.get("rewrite_count", 0) + 1
            rewrite_weight = min(rewrite_count * 0.1, 0.5)
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", "질문을 재작성하여 추가 정보를 얻으세요..."),
                ("human", "{question}\n\n컨텍스트:\n{context}\n\n이전 답변:\n{answer}")
            ])
            
            chain = prompt | self.llm | StrOutputParser()
            new_question = chain.invoke({
                "question": state["question"],
                "context": state["context"],
                "answer": state["answer"]
            })
            
            return AgentState(
                question=new_question,
                rewrite_count=rewrite_count,
                rewrite_weight=rewrite_weight,
                original_weight=1-rewrite_weight
            )
        except Exception as e:
            logger.error(f"질문 재작성 중 오류: {str(e)}")
            raise

    def _relevance_check(self, state: AgentState) -> AgentState:
        """답변 관련성 검사"""
        try:
            # Upstage 검사
            upstage_response = self.search_engine.upstage_checker.run({
                "context": state["context"],
                "answer": state["answer"]
            })
            
            # KoBERT 검사
            kobert_score = self.search_engine.validate_answer(
                context=state["context"],
                answer=state["answer"]
            )
            
            # 결합 점수 계산
            combined_score = self._calculate_combined_score(
                upstage_response, 
                kobert_score
            )
            
            return AgentState(
                **state,
                relevance=self._get_relevance_status(combined_score),
                combined_score=combined_score
            )
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
        
        upstage_score = {
            "grounded": 1.0,
            "notGrounded": 0.0,
            "notSure": 0.33
        }.get(upstage_response, 0.0)
        
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
                rewrite_count=0,
                combined_score=0.0
            )
            
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
            logger.error(f"질문 처리 중 오류 발생: {str(e)}")
            return {
                "error": str(e),
                "answer": "죄송합니다. 답변을 생성하는 중에 오류가 발생했습니다."
            }

# Streamlit UI용 함수
def create_ui():
    import streamlit as st
    
    st.title("법률 AI 어시스턴트")
    
    # 세션 상태 초기화
    if "agent" not in st.session_state:
        st.session_state.agent = LegalAgent()
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
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

if __name__ == "__main__":
    create_ui()
