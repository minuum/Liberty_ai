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
        
        # 프롬프트 템플릿 초기화
        self.prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)
        
        # 출력 파서 초기화
        self.output_parser = StrOutputParser()
        
        # 워크플로우 그래프 초기화
        self.workflow = self._create_workflow()
        
    def _create_workflow(self) -> StateGraph:
        """워크플로우 그래프 생성"""
        workflow = StateGraph(AgentState)
        
        # 노드 정의
        def retrieve(state: AgentState) -> AgentState:
            """관련 문서 검색"""
            results = self.search_engine.hybrid_search(state["question"])
            state["context"] = [doc.page_content for doc in results]
            return state
            
        def generate_answer(state: AgentState) -> AgentState:
            """답변 생성"""
            chain = self.prompt | self.llm | self.output_parser
            state["answer"] = chain.invoke({
                "context": "\n\n".join(state["context"]),
                "question": state["question"]
            })
            return state
            
        def validate_answer(state: AgentState) -> AgentState:
            """답변 검증"""
            context = "\n\n".join(state["context"])
            validation = self.search_engine.validate_answer(
                context=context,
                answer=state["answer"]
            )
            state["combined_score"] = validation["combined_score"]
            return state
            
        def rewrite_question(state: AgentState) -> AgentState:
            """질문 재작성"""
            rewrite_prompt = f"""
            원래 질문: {state['question']}
            이 질문을 더 구체적이고 명확하게 재작성해주세요.
            """
            state["question"] = self.llm.invoke(rewrite_prompt)
            state["rewrite_count"] = state.get("rewrite_count", 0) + 1
            return state
            
        # 노드 추가
        workflow.add_node("retrieve", retrieve)
        workflow.add_node("generate", generate_answer)
        workflow.add_node("validate", validate_answer)
        workflow.add_node("rewrite", rewrite_question)
        
        # 조건부 라우팅
        def should_rewrite(state: AgentState) -> str:
            if state["combined_score"] >= 0.7:
                return "end"
            elif state["rewrite_count"] >= 3:
                return "end"
            else:
                return "rewrite"
                
        # 엣지 설정
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", "validate")
        workflow.add_conditional_edges(
            "validate",
            should_rewrite,
            {
                "end": END,
                "rewrite": "retrieve"
            }
        )
        
        return workflow.compile()
        
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
