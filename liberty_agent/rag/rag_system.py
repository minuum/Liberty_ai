import logging
from typing import List, Tuple
import numpy as np
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_upstage import UpstageEmbeddings

logger = logging.getLogger(__name__)

class SimpleRAGSystem:
    def __init__(self, 
                 faiss_cache_dir: str, 
                 embedding_model: UpstageEmbeddings,
                 llm_model_name: str = "gpt-4o-2024-08-06",
                 llm_temperature: float = 0.1,
                 llm_api_key: str = None):
        self.llm = ChatOpenAI(
            model_name=llm_model_name, 
            temperature=llm_temperature,
            api_key=llm_api_key
        )
        self.dense_embedder = embedding_model
        
        try:
            logger.info(f"FAISS 벡터 저장소 로드 중... ({faiss_cache_dir})")
            self.vectorstore = FAISS.load_local(
                faiss_cache_dir,
                self.dense_embedder,
                allow_dangerous_deserialization=True
            )
            logger.info("FAISS 벡터 저장소 로드 완료.")
        except Exception as e:
            logger.error(f"FAISS 벡터 저장소 로드 중 오류 발생: {e}")
            self.vectorstore = None
            raise

    def retrieve(self, query: str, top_k: int = 3) -> Tuple[List[Document], List[float]]:
        """FAISS 벡터 저장소를 사용하여 유사한 문서를 검색하고 L2 거리를 반환합니다."""
        if self.vectorstore is None:
            logger.warning("FAISS 벡터 저장소가 로드되지 않았습니다. 빈 결과를 반환합니다.")
            return [], []
            
        try:
            results_with_scores = self.vectorstore.similarity_search_with_score(
                query,
                k=top_k
            )
            retrieved_docs = [doc for doc, score in results_with_scores]
            l2_distances = [score for doc, score in results_with_scores]
            
            logger.info(f"'{query}'에 대해 {len(retrieved_docs)}개의 문서 검색됨.")
            return retrieved_docs, l2_distances
        except Exception as e:
            logger.error(f"문서 검색 중 오류 발생: {e}")
            return [], []

    def generate_answer_with_context(self, query: str, retrieved_docs: List[Document]) -> str:
        """검색된 문맥을 바탕으로 LLM을 사용하여 답변을 생성합니다."""
        if not retrieved_docs:
            logger.warning("제공된 컨텍스트가 없습니다. 컨텍스트 없이 답변 생성 시도.")
            return "관련 정보를 찾을 수 없어 답변을 생성할 수 없습니다."

        context_str = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "당신은 법률 전문가입니다. 주어진 컨텍스트를 바탕으로 사용자의 질문에 정확하고 간결하게 답변해주세요. 컨텍스트에 없는 내용은 언급하지 마세요."),
            ("human", "컨텍스트:\n{context}\n\n질문: {query}\n\n답변:")
        ])
        
        chain = prompt_template | self.llm
        response = chain.invoke({"context": context_str, "query": query})
        
        logger.info(f"'{query}'에 대한 답변 생성 완료.")
        return response.content 