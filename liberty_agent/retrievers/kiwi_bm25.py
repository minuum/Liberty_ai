from typing import Any, Callable, Dict, Iterable, List, Optional, Union
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.pydantic_v1 import Field
from operator import itemgetter
import numpy as np
import pickle
import os
from tqdm import tqdm
from kiwipiepy import Kiwi
import logging

logger = logging.getLogger(__name__)

class CustomKiwiBM25Retriever(BaseRetriever):
    """Kiwi 토크나이저를 사용하는 BM25 검색기"""

    vectorizer: Any = Field(description="BM25 벡터라이저")
    docs: List[Document] = Field(repr=False, description="문서 리스트")
    k: int = Field(default=20, description="반환할 문서 수")
    
    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_documents(
        cls,
        documents: Iterable[Document],
        k: int = 4,
        **kwargs: Any
    ) -> "CustomKiwiBM25Retriever":
        """문서로부터 검색기 생성"""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("rank_bm25를 설치해주세요: pip install rank_bm25")

        texts, metadatas = zip(*((d.page_content, d.metadata) for d in documents))
        
        # Kiwi 토크나이저로 텍스트 전처리
        kiwi = Kiwi()
        texts_processed = []
        for text in tqdm(texts, desc="텍스트 토큰화 중"):
            tokens = kiwi.tokenize(text)
            texts_processed.append([token.form for token in tokens])

        vectorizer = BM25Okapi(texts_processed)
        docs = [Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)]
        
        return cls(vectorizer=vectorizer, docs=docs, k=k, **kwargs)

    def save_local(self, folder_path: str, index_name: str) -> None:
        """검색기를 로컬에 저장"""
        try:
            os.makedirs(folder_path, exist_ok=True)
            
            # 벡터라이저와 문서 저장
            with open(os.path.join(folder_path, f"{index_name}_vectorizer.pkl"), 'wb') as f:
                pickle.dump(self.vectorizer, f)
            with open(os.path.join(folder_path, f"{index_name}_docs.pkl"), 'wb') as f:
                pickle.dump(self.docs, f)
                
            logger.info(f"KiwiBM25 검색기 저장 완료: {folder_path}")
            
        except Exception as e:
            logger.error(f"KiwiBM25 검색기 저장 실패: {str(e)}")
            raise

    @classmethod
    def load_local(cls, folder_path: str, index_name: str, **kwargs) -> "CustomKiwiBM25Retriever":
        """저장된 검색기 로드"""
        try:
            # 벡터라이저와 문서 로드
            with open(os.path.join(folder_path, f"{index_name}_vectorizer.pkl"), 'rb') as f:
                vectorizer = pickle.load(f)
            with open(os.path.join(folder_path, f"{index_name}_docs.pkl"), 'rb') as f:
                docs = pickle.load(f)
                
            logger.info(f"KiwiBM25 검색기 로드 완료: {folder_path}")
            return cls(vectorizer=vectorizer, docs=docs, **kwargs)
            
        except Exception as e:
            logger.error(f"KiwiBM25 검색기 로드 실패: {str(e)}")
            raise

    def _get_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """쿼리와 관련된 문서 검색"""
        try:
            # 쿼리 토큰화
            kiwi = Kiwi()
            tokens = kiwi.tokenize(query)
            processed_query = [token.form for token in tokens]
            
            # 문서 검색
            return self.vectorizer.get_top_n(processed_query, self.docs, n=self.k)
            
        except Exception as e:
            logger.error(f"문서 검색 중 오류 발생: {str(e)}")
            return []

    def search_with_score(self, query: str, top_k: Optional[int] = None) -> List[Document]:
        """점수와 함께 문서 검색"""
        try:
            kiwi = Kiwi()
            tokens = kiwi.tokenize(query)
            processed_query = [token.form for token in tokens]
            
            scores = self.vectorizer.get_scores(processed_query)
            normalized_scores = self._softmax(scores)
            
            k = top_k if top_k is not None else self.k
            indices = np.argsort(normalized_scores)[-k:][::-1]
            
            results = []
            for idx in indices:
                doc = self.docs[idx]
                doc.metadata["score"] = float(normalized_scores[idx])
                results.append(doc)
                
            return results
            
        except Exception as e:
            logger.error(f"점수 계산 중 오류 발생: {str(e)}")
            return []

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """점수의 소프트맥스 계산"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum() 