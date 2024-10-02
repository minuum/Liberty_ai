import os
import time
import torch
import streamlit as st
from transformers import AutoTokenizer, BertForSequenceClassification
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langchain_upstage import UpstageEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from typing import TypedDict, Annotated, Sequence
import operator
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# BERT 모델 및 토크나이저 로드
bert_tokenizer = AutoTokenizer.from_pretrained("monologg/kobert", trust_remote_code=True)
bert_model = BertForSequenceClassification.from_pretrained("monologg/kobert", trust_remote_code=True)

# ChatOpenAI 모델 초기화
gpt_llm = ChatOpenAI(
    model_name="gpt-4o-mini-2024-07-18",  # 또는 다른 OpenAI 모델
    temperature=0.2,
    openai_api_key=os.getenv("OPENAI_API_KEY")  # .env 파일에서 API 키 로드
)

# 임베딩 및 벡터스토어 설정
passage_embeddings = UpstageEmbeddings(model="solar-embedding-1-large-passage")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
documents = [
    "계약은 양 당사자 간의 합의에 의해 성립됩니다.",
    "계약의 위반 시 손해배상이 청구될 수 있습니다.",
    "계약서에 명시된 조항은 법적 구속력을 가집니다."
]

fs = LocalFileStore("./cache/")
cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
    passage_embeddings, fs, namespace=passage_embeddings.model
)

texts = []
for doc in documents:
    texts.extend(text_splitter.split_text(doc))

FAISS_INDEX_PATH = "faiss_index"

def create_vector_store():
    if os.path.exists(FAISS_INDEX_PATH):
        print("기존 FAISS 인덱스를 로드합니다.")
        vector_store = FAISS.load_local(FAISS_INDEX_PATH, cached_embeddings, allow_dangerous_deserialization=True)
    else:
        print("FAISS 인덱스가 없습니다. 새로 생성합니다.")
        vector_store = create_new_vector_store()
    
    return vector_store

def create_new_vector_store():
    print("새로운 벡터 저장소를 생성합니다.")
    vector_store = FAISS.from_texts(texts, cached_embeddings)
    
    # 벡터 저장소를 로컬에 저장
    vector_store.save_local(FAISS_INDEX_PATH)
    
    return vector_store

vectorstore = create_vector_store()
print("벡터스토어 설정 완료.")

# 상태 정의
class AgentState(TypedDict):
    messages: Annotated[Sequence[HumanMessage], operator.add]

# 노드 함수들
def evaluate_question(state):
    start_time = time.time()
    messages = state['messages']
    question = messages[-1].content
    inputs = bert_tokenizer(question, return_tensors='pt')
    outputs = bert_model(**inputs)
    classification = torch.argmax(outputs.logits, dim=1).item()
    classification_map = {0: "일반 법률 질문", 1: "복잡한 법률 질문"}
    evaluation = classification_map.get(classification, "알 수 없음")
    end_time = time.time()
    print(f"평가 단계 실행 시간: {end_time - start_time:.2f}초")
    return {"messages": [HumanMessage(content=f"평가 결과: {evaluation}")]}

def retrieve_info(state):
    start_time = time.time()
    messages = state['messages']
    query = messages[-1].content + " " + messages[0].content
    docs = vectorstore.similarity_search(query, k=3)
    retrieved_info = "\n".join([doc.page_content for doc in docs])
    end_time = time.time()
    print(f"검색 단계 실행 시간: {end_time - start_time:.2f}초")
    return {"messages": [HumanMessage(content=f"검색된 정보: {retrieved_info}")]}

def generate_response(state):
    start_time = time.time()
    messages = state['messages']
    evaluation = messages[1].content
    retrieved_info = messages[2].content
    user_input = messages[0].content
    
    prompt_template = PromptTemplate(
        input_variables=["evaluation", "retrieved_info", "user_input"],
        template="""
        당신은 전문 법률 AI 어시스턴트 Liberty입니다.
        
        {evaluation}
        {retrieved_info}
        
        사용자 질문: {user_input}
        
        위의 평가 결과와 검색된 정보를 바탕으로 사용자에게 도움이 되는 법률 답변을 제공하세요.
        """
    )
    
    prompt = prompt_template.format(
        evaluation=evaluation,
        retrieved_info=retrieved_info,
        user_input=user_input
    )
    response = gpt_llm.invoke(prompt)
    end_time = time.time()
    print(f"생성 단계 실행 시간: {end_time - start_time:.2f}초")
    return {"messages": [HumanMessage(content=f"AI 응답: {response.content}")]}

# 그래프 생성
def create_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("평가", evaluate_question)
    workflow.add_node("검색", retrieve_info)
    workflow.add_node("생성", generate_response)
    
    workflow.set_entry_point("평가")
    workflow.add_edge("평가", "검색")
    workflow.add_edge("검색", "생성")
    workflow.add_edge("생성", END)
    
    return workflow.compile()

# Streamlit UI 설정
def setup_page():
    st.set_page_config(page_title="Liberty_ai", layout="wide")
    st.title("Liberty")

def create_sidebar():
    with st.sidebar:
        st.header("Settings")
        model = st.selectbox("모델 선택", ["gpt-4o-mini-2024-07-18","gpt-4o"])
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
    return model, temperature

def display_chat_history(chat_history):
    for i, message in enumerate(chat_history):
        if message["role"] == "user":
            st.chat_message("user", message["content"])
        else:
            st.chat_message("assistant", message["content"])

def user_input_section():
    user_input = st.text_area("원하는 법률 답변을 입력하세요!", height=100)
    send_button = st.button("➡")
    return user_input, send_button

def display_sources(sources):
    with st.expander("Sources"):
        for i, source in enumerate(sources, 1):
            st.write(f"{i}. {source}")

def main():
    setup_page()
    model, temperature = create_sidebar()
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    display_chat_history(st.session_state.chat_history)
    
    user_input, send_button = user_input_section()
    
    if send_button and user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # 그래프 실행
        graph = create_graph()
        inputs = {"messages": [HumanMessage(content=user_input)]}
        for output in graph.stream(inputs):
            for key, value in output.items():
                if key == "생성":
                    st.session_state.chat_history.append({"role": "assistant", "content": value['messages'][-1].content})
        
        # 소스 표시 (실제 구현 시 AI 응답과 함께 생성되어야 함)
        sources = ["Source 1", "Source 2", "Source 3"]
        display_sources(sources)
        
        st.rerun()

if __name__ == "__main__":
    main()