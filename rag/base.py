from langchain import hub
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_upstage import UpstageEmbeddings
from langchain_openai import ChatOpenAI
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_core.output_parsers import StrOutputParser
from abc import ABC, abstractmethod
from operator import itemgetter
import os

class RetrievalChain(ABC):
    def __init__(self):
        self.source_uri = None
        self.k = 5
        self.FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss_index")
        self.cached_embeddings = None
    @abstractmethod
    def load_documents(self, source_uris):
        """loader를 사용하여 문서를 로드합니다."""
        pass

    @abstractmethod
    def create_text_splitter(self):
        """text splitter를 생성합니다."""
        pass

    def split_documents(self, docs, text_splitter):
        """text splitter를 사용하여 문서를 분할합니다."""
        return text_splitter.split_documents(docs)

    def create_embedding(self):
        fs = LocalFileStore("./cache/")
        passage_embeddings = UpstageEmbeddings(model="solar-embedding-1-large-query")
        self.cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
    passage_embeddings, fs, namespace=passage_embeddings.model
        )
        return self.cached_embeddings
    #추후 FAISS -> Finecone으로 벡터스토어 변경 필요

    
    
    def create_vectorstore(self, split_docs):

        if os.path.exists(self.FAISS_INDEX_PATH):
            print("기존 FAISS 인덱스를 로드합니다.")
            return FAISS.load_local(self.FAISS_INDEX_PATH, self.cached_embeddings, allow_dangerous_deserialization=True)
        else:
            print("새로운 FAISS 벡터 저장소를 생성합니다.")
            vector_store = FAISS.from_documents(split_docs, self.cached_embeddings)
            vector_store.save_local(self.FAISS_INDEX_PATH)
            return vector_store
 

    def create_retriever(self, vectorstore):
        # MMR을 사용하여 검색을 수행하는 retriever를 생성합니다.
        dense_retriever = vectorstore.as_retriever(
            search_type="mmr", search_kwargs={"k": self.k}
        )
        return dense_retriever

    def create_model(self):
        return ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    def create_prompt(self):
        return hub.pull("minuum/liberty-rag")

    @staticmethod
    def format_docs(docs):
        return "\n".join(docs)

    def create_chain(self):
        docs = self.load_documents(self.source_uri)
        text_splitter = self.create_text_splitter()
        split_docs = self.split_documents(docs, text_splitter)
        self.vectorstore = self.create_vectorstore(split_docs)
        self.retriever = self.create_retriever(self.vectorstore)
        model = self.create_model()
        prompt = self.create_prompt()
        self.chain = (
                {
                    "question": itemgetter("question"),
                    "context": itemgetter("context"),
                    "rewrite_weight": itemgetter("rewrite_weight"),
                    "original_weight": itemgetter("original_weight")
                }
                | prompt
                | model
                | StrOutputParser()
            )
        return self
    
