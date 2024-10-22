import streamlit as st
import os
from dotenv import load_dotenv
from langchain_teddynote import logging
from rag.pdf import PDFRetrievalChain
from typing import TypedDict
from transformers import AutoModel, AutoTokenizer
import torch
from langchain_upstage import UpstageGroundednessCheck
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from rag.utils import format_docs, format_searched_docs
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolInvocation
import pprint
from langgraph.errors import GraphRecursionError
from langchain_core.runnables import RunnableConfig
import streamlit as st
from menu import menu



# 환경 변수 로드
load_dotenv()

# LangSmith 로깅 설정
logging.langsmith("liberty_langgraph")

# PDF 체인 설정
@st.cache_resource
def setup_pdf_chain():
    pdf_chain = PDFRetrievalChain(["data/Minbub Selected Provisions.pdf"])
    pdf_chain.create_embedding()
    pdf = pdf_chain.create_chain()
    return pdf.retriever, pdf.chain

pdf_retriever, pdf_chain = setup_pdf_chain()

# GraphState 정의
class GraphState(TypedDict):
    question: str
    context: str
    answer: str
    rewrite_weight: float
    original_weight: float
    relevance: str
    rewrite_count: int
    combined_score: float

# KoBERT 모델 및 토크나이저 로드
@st.cache_resource
def load_kobert_model():
    model = AutoModel.from_pretrained("monologg/kobert")
    tokenizer = AutoTokenizer.from_pretrained("monologg/kobert", trust_remote_code=True, clean_up_tokenization_spaces=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, tokenizer, device

model, tokenizer, device = load_kobert_model()

# Upstage Groundedness Check 설정
upstage_ground_checker = UpstageGroundednessCheck()

# 여기에 이전에 정의한 함수들을 포함시킵니다 (retrieve_document, llm_answer, rewrite, kobert_relevance_check, relevance_check, is_relevant, final_output)
from langchain_upstage import UpstageGroundednessCheck
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from rag.utils import format_docs, format_searched_docs

# 업스테이지 문서 관련성 체크 기능을 설정합니다. https://upstage.ai
upstage_ground_checker = UpstageGroundednessCheck()


# 문서에서 검색하여 관련성 있는 문서를 찾습니다.
def retrieve_document(state: GraphState) -> GraphState:
    # 문서에서 검색하여 관련성 있는 문서를 찾습니다.
    retrieved_docs = pdf_retriever.invoke(state["question"])

    # 검색된 문서를 형식화합니다.
    retrieved_docs = format_docs(retrieved_docs)

    # 검색된 문서를 context 키에 저장합니다.
    return GraphState(context=retrieved_docs)


# # LLM을 사용하여 답변을 생성합니다.
# def llm_answer(state: GraphState) -> GraphState:
#     question = state["question"]
#     context = state["context"]

#     # 체인을 호출하여 답변을 생성합니다.
#     response = pdf_chain.invoke({"question": question, "context": context})

    return GraphState(answer=response)

def llm_answer(state: GraphState) -> GraphState:
    question = state["question"]
    context = state["context"]
    #재작성 횟수
    rewrite_count = state.get("rewrite_count", 0)
    # 재작성 횟수에 따른 가중치 조정
    rewrite_weight = state["rewrite_weight"]  # 최대 0.5까지 증가
    original_weight =state["original_weight"]

    response = pdf_chain.invoke({"question": question, "context": context, "rewrite_weight": rewrite_weight, "original_weight": original_weight})

    return GraphState(answer=response, rewrite_count=rewrite_count)

def rewrite(state):
    
    state["rewrite_count"] = (state.get("rewrite_count", 0) + 1)  # 카운터 증가
    print("rewrite_count : ", state["rewrite_count"])

    state["rewrite_weight"] = min(state["rewrite_count"] * 0.1, 0.5)  # 최대 0.5까지 증가
    state["original_weight"] = 1 - state["rewrite_weight"]
    print("state['rewrite_weight'] : ", state["rewrite_weight"])
    print("state['original_weight'] : ", state["original_weight"])

    question = state["question"]
    answer = state["answer"]
    context = state["context"]
    rewrite_weight = state["rewrite_weight"]
    
    prompt = ChatPromptTemplate.from_messages(
        [
            (
            "system",
            "You are a professional prompt rewriter. Your task is to generate questions to obtain additional information not shown in the given context. "
            "Your generated questions will be used for web searches to find relevant information. "
            "Consider the rewrite weight ({{rewrite_weight:.2f}}) to adjust the credibility of the previous answer. "
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
            "\n\nRewrite weight: {{rewrite_weight:.2f}} (The higher this value, the more you should doubt the previous answer)"
            "\n\nFormulate an improved question in Korean:"
        ),
    ]
)

    # Question rewriting model
    model = ChatOpenAI(temperature=0, model="gpt-4o-2024-08-06")

    chain = prompt | model | StrOutputParser()
    response = chain.invoke(
        {"question": question, "answer": answer, "context": context,"rewrite_weight":rewrite_weight}
    )
    print(f"질문 재작성 횟수: {state['rewrite_count']}")  # 현재 카운트 출력
    return GraphState(question=response,rewrite_count=state["rewrite_count"])


def kobert_relevance_check(context, answer):
    # 각각의 입력을 토크나이즈
    context_inputs = tokenizer(context, return_tensors="pt", truncation=True, padding=True)
    answer_inputs = tokenizer(answer, return_tensors="pt", truncation=True, padding=True)
    
    # 디바이스로 이동
    context_inputs = {k: v.to(device) for k, v in context_inputs.items()}
    answer_inputs = {k: v.to(device) for k, v in answer_inputs.items()}
    
    with torch.no_grad():
        context_outputs = model(**context_inputs)
        answer_outputs = model(**answer_inputs)
    
    # 마지막 히든 스테이트에서 평균 풀링
    context_embedding = context_outputs.last_hidden_state.mean(dim=1)  # (batch_size, hidden_size)
    answer_embedding = answer_outputs.last_hidden_state.mean(dim=1)
    
    # 코사인 유사도 계산
    similarity = torch.nn.functional.cosine_similarity(context_embedding, answer_embedding)
    
    # Convert the similarity value to a range between 0 and 1
    relevance_score = (similarity.item() + 1) / 2
    
    return relevance_score

def relevance_check(state: GraphState) -> GraphState:
    #print("relevance_check", state)
    print("relevance_check")
    # 기존 upstage_ground_checker 실행
    upstage_response = upstage_ground_checker.run(
        {"context": state["context"], "answer": state["answer"],"rewrite_weight":state["rewrite_weight"],"original_weight":state["original_weight"]}
    )
    print("upstage_response : ", upstage_response)
    # 설정에 따라 분기
    use_combined_check =True # True: upstage와 kobert 함께 사용, False: upstage만 사용
    
    if use_combined_check:
        # koBERT 모델을 이용한 관련성 체크
        kobert_score = kobert_relevance_check(state["context"], state["answer"])
        # 두 결과 합산 (가중 평균)
        upstage_weight = 0.6
        kobert_weight = 0.4
                
        if upstage_response == "grounded":
            upstage_score = 1.0
        elif upstage_response == "notGrounded":
            upstage_score = 0.0
        else:  # 'notSure'
            upstage_score = 0.33 
        
        combined_score = upstage_weight * upstage_score + kobert_weight * kobert_score
        
        if combined_score >= 0.7:
            final_relevance = "grounded"
        elif combined_score <= 0.3:
            final_relevance = "notGrounded"
        else:
            final_relevance = "notSure"
        print("upstage_score : ", upstage_score)
        print("kobert_score : ", kobert_score)
        print("combined_score : ", combined_score)
        print("final_relevance : ", final_relevance)
        
    else:
        # upstage 결과만 사용
        final_relevance = upstage_response
        print("final_relevance (upstage only) : ", final_relevance)
    
    return GraphState(
        relevance=final_relevance, question=state["question"], answer=state["answer"], combined_score=combined_score
    )


def is_relevant(state: GraphState) -> GraphState:
    return state["relevance"]
# 최종 출력을 위한 함수 수정
def final_output(state: GraphState) -> str:
    final_answer = state["answer"][-1].content
    rewrite_count = state.get("rewrite_count", 0)
    return f"최종 답변: {final_answer}\n\n총 질문 재작성 횟수: {rewrite_count}"

# Streamlit 앱 시작
st.title("Liberty LangGraph QA 챗봇")
# 세션 상태 초기화
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    st.session_state.logged_in = True
    st.rerun()

def logout():
    st.session_state.logged_in = False
    st.rerun()

# 로그인 버튼
if not st.session_state.logged_in:
    st.title("로그인")
    if st.button("로그인"):
        login()
else:
    st.title("환영합니다!")
    if st.button("로그아웃"):
        logout()

# 메뉴 호출
menu()

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 이전 메시지 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 처리
if prompt := st.chat_input("질문을 입력하세요:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 챗봇 응답 생성
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # GraphState 초기화
        inputs = GraphState(
            question=prompt,
            rewrite_count=0,
            rewrite_weight=0,
            original_weight=1,
        )

        # 그래프 설정
        workflow = StateGraph(GraphState)

        # 노드 추가
        workflow.add_node("retrieve", retrieve_document)
        workflow.add_node("llm_answer", llm_answer)
        workflow.add_node("rewrite", rewrite)
        workflow.add_node("relevance_check", relevance_check)

        # 엣지 추가
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "llm_answer")
        workflow.add_edge("llm_answer", "relevance_check")
        workflow.add_conditional_edges(
            "relevance_check",
            is_relevant,
            {
                "grounded": END,
                "notGrounded": "rewrite",
                "notSure": "rewrite",
            },
        )
        workflow.add_edge("rewrite", "retrieve")

        # 그래프 컴파일
        app = workflow.compile()

        # 설정
        config = RunnableConfig(
            recursion_limit=15, configurable={"thread_id": "CORRECTIVE-SEARCH-RAG"}
        )

        try:
            outputs = []
            for i, output in enumerate(app.stream(inputs, config=config)):
                outputs.append(output)
                # 진행 상황 업데이트
                progress = f"처리 중... ({i + 1}번째 단계)"
                full_response = progress + "\n\n" + str(output)
                message_placeholder.markdown(full_response + "▌")

            # 최종 결과 표시
            final_output = outputs[-1]
            final_answer = final_output.get('relevance_check', {}).get('answer', "답변을 찾을 수 없습니다.")
            rewrite_count = final_output.get('relevance_check', {}).get('rewrite_count', 0)
            combined_score = final_output.get('relevance_check', {}).get('combined_score', 0)  # combined_score 확인
            
            full_response = f"{final_answer}\n\n총 질문 재작성 횟수: {rewrite_count}\n결합 점수: {combined_score}"
            message_placeholder.markdown(full_response)

        except GraphRecursionError as e:
            full_response = f"재귀 한도에 도달했습니다: {e}\n최종 답변: {outputs[-1] if outputs else '답변을 생성하지 못했습니다.'}"
            message_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})

st.write("---")
st.write("© 2024 Liberty LangGraph QA 챗봇")