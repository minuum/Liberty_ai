import os
import time
import torch
import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader 
from transformers import AutoTokenizer, BertForSequenceClassification
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.vectorstores import FAISS, Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langchain_upstage import UpstageEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from typing import TypedDict, Annotated, Sequence
import operator
from dotenv import load_dotenv

# 한글 불용어 사전 불러오기 (불용어 사전 출처: https://www.ranks.nl/stopwords/korean)

# pinecone 사용


from pinecone import Pinecone, ServerlessSpec
from langchain_teddynote.community.pinecone import create_index
from langchain_teddynote.community.pinecone import (create_sparse_encoder,fit_sparse_encoder, load_sparse_encoder)
from langchain_teddynote.korean import stopwords
import glob
from langchain_teddynote.community.pinecone import preprocess_documents
from langchain_teddynote.community.pinecone import upsert_documents_parallel
# 환경 변수 로드
load_dotenv()

# 벡터 저장소 타입 설정 (기본값: pinecone)
VECTOR_STORE_TYPE = os.getenv("VECTOR_STORE_TYPE", "pinecone")

# Pinecone 설정
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "liberty-index")
# FAISS 설정
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss_index")

# BERT 모델 및 토크나이저 로드
bert_tokenizer = AutoTokenizer.from_pretrained("monologg/kobert", trust_remote_code=True)
bert_model = BertForSequenceClassification.from_pretrained("monologg/kobert", trust_remote_code=True)

# ChatOpenAI 모델 초기화
gpt_llm = ChatOpenAI(
    model_name="gpt-4o-mini-2024-07-18",  # 또는 다른 OpenAI 모델
    temperature=0.2,
    openai_api_key=os.getenv("OPENAI_API_KEY")  # .env 파일에서 API 키 로드
)

# 임베딩 및 텍스트 스플리터 설정
passage_embeddings = UpstageEmbeddings(model="solar-embedding-1-large-query")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
stopword = stopwords()
# 캐시된 임베딩 설정
fs = LocalFileStore("./cache/")
cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
    passage_embeddings, fs, namespace=passage_embeddings.model
)

def load_documents(data_dir="./data"):
    documents = []
    files = sorted(glob.glob("data/*.pdf"))
    for filename in files:
        loader = PDFPlumberLoader(filename)
        documents.extend(loader.load_and_split(text_splitter))
    return documents


def preprocess_contents():
    split_docs = load_documents()
    print(split_docs)
    contents, metadatas = preprocess_documents(
        split_docs=split_docs,
        metadata_keys=["source", "page", "author"],
        min_length=5,
        use_basename=True,
    )
    return contents, metadatas
def Sparse_encoder(contents):
    sparse_encoder_path = "./sparse_encoder.pkl"
    
    if os.path.exists(sparse_encoder_path):
        # 이미 학습된 sparse encoder가 존재하는 경우 불러옵니다.
        sparse_encoder = load_sparse_encoder(sparse_encoder_path)
    else:
        # 학습된 sparse encoder가 없는 경우 새로 생성하고 학습합니다.
        sparse_encoder = create_sparse_encoder(stopwords(), mode="kiwi")
        saved_path = fit_sparse_encoder(
            sparse_encoder=sparse_encoder, 
            contents=contents, 
            save_path=sparse_encoder_path
        )
    
    return sparse_encoder

def Pinecone_upsert():
    contents, metadatas = preprocess_contents()
    upsert_documents_parallel(
    index=PINECONE_INDEX_NAME,  # Pinecone 인덱스
    namespace="liberty_namespace-01",  # Pinecone namespace
    contents=contents,  # 이전에 전처리한 문서 내용
    metadatas=metadatas,  # 이전에 전처리한 문서 메타데이터
    sparse_encoder=Sparse_encoder(contents),  # Sparse encoder
    embedder=passage_embeddings,
    batch_size=64,
        max_workers=30,
    )
from langchain_teddynote.community.pinecone import init_pinecone_index
def Pinecone_init():
    pinecone_params=init_pinecone_index(
        index_name=PINECONE_INDEX_NAME,  # Pinecone 인덱스 이름
    namespace="liberty_namespace-01",  # Pinecone Namespace
    api_key=os.environ["PINECONE_API_KEY"],  # Pinecone API Key
    sparse_encoder_path="./sparse_encoder.pkl",  # Sparse Encoder 저장경로(save_path)
    stopwords=stopwords(),  # 불용어 사전
    tokenizer="kiwi",
    embeddings=passage_embeddings,
   # Dense Embedder
    top_k=5,  # Top-K 문서 반환 개수
    alpha=0.5, 

    )
    return pinecone_params
from langchain_teddynote.community.pinecone import PineconeKiwiHybridRetriever

# 검색기 생성
pinecone_retriever = PineconeKiwiHybridRetriever(**(Pinecone_init()))

def create_vector_store(documents, store_type=VECTOR_STORE_TYPE):
    if store_type == "pinecone":
        if not PINECONE_API_KEY:
            print("Pinecone API 키가 설정되지 않았습니다. FAISS를 사용합니다.")
            return create_vector_store(documents, "faiss")
        
        try:
            pc_index = create_index(        
                api_key=os.environ["PINECONE_API_KEY"],
                index_name="teddynote-db-index",  # 인덱스 이름을 지정합니다.
                dimension=4096,  # Embedding 차원과 맞춥니다. (OpenAIEmbeddings: 1536, UpstageEmbeddings: 4096)
                metric="dotproduct",  # 유사도 측정 방법을 지정합니다. (dotproduct, euclidean, cosine)
            )

        except Exception as e:
            print(f"Pinecone 초기화 중 오류 발생: {e}")
            print("FAISS를 대체 옵션으로 사용합니다.")
            return create_vector_store(documents, "faiss")
    
    elif store_type == "faiss":
        if os.path.exists(FAISS_INDEX_PATH):
            print("기존 FAISS 인덱스를 로드합니다.")
            return FAISS.load_local(FAISS_INDEX_PATH, cached_embeddings, allow_dangerous_deserialization=True)
        else:
            print("새로운 FAISS 벡터 저장소를 생성합니다.")
            vector_store = FAISS.from_texts(documents, cached_embeddings)
            vector_store.save_local(FAISS_INDEX_PATH)
            return vector_store
    
    raise ValueError("지원되지 않는 벡터 저장소 타입입니다.")

def setup_vector_store():
    start_time = time.time()
    
    print("문서 로딩 중...")
    documents = load_documents()
    print(f"{len(documents)}개의 텍스트 청크가 생성되었습니다.")
    
    print(f"{VECTOR_STORE_TYPE} 벡터 저장소 생성 중...")
    vector_store = create_vector_store(documents)
    
    end_time = time.time()
    print(f"벡터 저장소 설정 완료. 총 소요 시간: {end_time - start_time:.2f}초")
    
    return vector_store

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
        st.session_state.chat_history = [{"role": "assistant", "content": "안녕하세요. 법률 관련 질문에 답변해 드리겠습니다. 어떤 도움이 필요하신가요?"}]
    
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("법률 관련 질문을 입력해주세요:"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # 그래프 실행
            graph = create_graph()
            inputs = {"messages": [HumanMessage(content=prompt)]}
            for output in graph.stream(inputs):
                for key, value in output.items():
                    if key == "생성":
                        assistant_response = value['messages'][-1].content
                        for chunk in assistant_response.split():
                            full_response += chunk + " "
                            time.sleep(0.05)
                            message_placeholder.markdown(full_response + "▌")
                        message_placeholder.markdown(full_response)
            
            st.session_state.chat_history.append({"role": "assistant", "content": full_response})
        
        # 소스 표시 (실제 구현 시 AI 응답과 함께 생성되어야 함)
        sources = ["Source 1", "Source 2", "Source 3"]
        display_sources(sources)

    # 피드백 수집 (사이드바로 이동)
    with st.sidebar:
        st.subheader("피드백")
        feedback = st.text_area("답변에 대한 피드백을 남겨주세요:")
        if st.button("피드백 제출"):
            if feedback:
                st.success("피드백이 제출되었습니다. 감사합니다!")
                # 여기에 피드백 처리 로직을 추가할 수 있습니다.
            else:
                st.warning("피드백을 입력해주세요.")

if __name__ == "__main__":
    vectorstore = setup_vector_store()
    main()