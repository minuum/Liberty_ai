from typing import Any, List
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import Field
import logging
from typing import Optional
from uuid import uuid4

logger = logging.getLogger(__name__)

class HybridRetriever(BaseRetriever):
    """Dense와 Sparse 검색기를 결합한 하이브리드 검색기"""
    
    dense_retriever: Any = Field(description="Dense 검색기")
    sparse_retriever: Any = Field(description="Sparse 검색기")
    k: int = Field(default=20, description="반환할 문서 수")
    dense_weight: float = Field(default=0.7, description="Dense 검색 가중치")
    sparse_weight: float = Field(default=0.3, description="Sparse 검색 가중치")
    logger.info(f"HybridRetriver 생성 완료")
    class Config:
        arbitrary_types_allowed = True

    async def _aget_relevant_documents(
        self,
        query: str,
        *, 
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> List[Document]:
        dense_docs = await self.dense_retriever.invoke(query)
        sparse_docs = await self.sparse_retriever.invoke(query) 
        
        return self._merge_documents(dense_docs, sparse_docs)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
        **kwargs: Any,
    ) -> List[Document]:
        # CallbackManager 초기화
        if run_manager is None:
            run_manager = CallbackManagerForRetrieverRun(
                run_id=uuid4(),
                handlers=[],
                inheritable_handlers=[],
                tags=["hybrid_retriever"],
                metadata={},
            )

        dense_docs = self.dense_retriever.invoke(query)
        sparse_docs = self.sparse_retriever.invoke(query)
        
        return self._merge_documents(dense_docs, sparse_docs)

    def _merge_documents(
        self,
        dense_docs: List[Document],
        sparse_docs: List[Document],
    ) -> List[Document]:
        """양쪽 검색기에서 문서를 가져와 가중치를 적용하여 결합"""
        try:
            # 문서별 점수 계산
            doc_scores = {}
            
            # Dense 검색 결과 처리
            for i, doc in enumerate(dense_docs):
                score = (len(dense_docs) - i) / len(dense_docs) * self.dense_weight
                doc_scores[doc.page_content] = (doc, score)
            
            # Sparse 검색 결과 처리
            for i, doc in enumerate(sparse_docs):
                score = (len(sparse_docs) - i) / len(sparse_docs) * self.sparse_weight
                if doc.page_content in doc_scores:
                    doc_scores[doc.page_content] = (
                        doc, 
                        doc_scores[doc.page_content][1] + score
                    )
                else:
                    doc_scores[doc.page_content] = (doc, score)
            
            # 점수순 정렬 및 상위 문서 반환 (스코어 포함)
            sorted_docs = sorted(
                doc_scores.values(), 
                key=lambda x: x[1], 
                reverse=True
            )
            return [
                Document(
                    page_content=doc.page_content,
                    metadata={
                        **doc.metadata,
                        "score": score  # 스코어 정보 추가
                    }
                ) 
                for doc, score in sorted_docs[:self.k]
            ]
            
        except Exception as e:
            logger.error(f"하이브리드 검색 중 오류 발생: {str(e)}")
            return [] 