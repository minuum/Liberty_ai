from typing import List, Dict
from dataclasses import dataclass
import time
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.evaluation import load_evaluator
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class RetrieverTestResult:
    name: str
    recall_at_k: float
    precision_at_k: float
    ndcg_at_k: float
    latency: float
    memory_usage: float

class RetrieverTester:
    def __init__(self, test_queries: List[str], relevant_docs: List[List[str]]):
        """
        Args:
            test_queries: 테스트할 쿼리 목록
            relevant_docs: 각 쿼리에 대한 관련 문서 목록
        """
        self.test_queries = test_queries
        self.relevant_docs = relevant_docs
        self.results = []
        
    def _calculate_recall(self, retrieved: List[str], relevant: List[str]) -> float:
        """
        재현율(Recall) 계산
        """
        if not relevant:  # 관련 문서가 없는 경우
            return 0.0
        relevant_retrieved = set(retrieved) & set(relevant)
        return len(relevant_retrieved) / len(relevant)

    def _calculate_precision(self, retrieved: List[str], relevant: List[str]) -> float:
        """
        정밀도(Precision) 계산
        """
        if not retrieved:  # 검색된 문서가 없는 경우
            return 0.0
        relevant_retrieved = set(retrieved) & set(relevant)
        return len(relevant_retrieved) / len(retrieved)

    def _calculate_ndcg(self, retrieved: List[str], relevant: List[str], k: int) -> float:
        """
        NDCG(Normalized Discounted Cumulative Gain) 계산
        """
        if not relevant or not retrieved:
            return 0.0

        # DCG 계산
        dcg = 0
        for i, doc in enumerate(retrieved[:k]):
            rel = 1 if doc in relevant else 0
            dcg += rel / np.log2(i + 2)  # i+2 because i starts at 0

        # Ideal DCG 계산
        idcg = 0
        for i in range(min(len(relevant), k)):
            idcg += 1 / np.log2(i + 2)

        if idcg == 0:
            return 0.0

        return dcg / idcg

    def _get_memory_usage(self) -> float:
        """
        현재 프로세스의 메모리 사용량 반환 (MB 단위)
        """
        import psutil
        import os
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # Convert to MB

    def test_hybrid_retriever(self, search_engine, k: int = 5) -> RetrieverTestResult:
        """하이브리드 검색기 테스트"""
        start_time = time.time()
        metrics = {
            "recall": [],
            "precision": [],
            "ndcg": []
        }
        
        for query, relevant in zip(self.test_queries, self.relevant_docs):
            results = search_engine.hybrid_search(query, top_k=k)
            retrieved_docs = [doc.page_content for doc in results]
            
            # 메트릭 계산
            metrics["recall"].append(self._calculate_recall(retrieved_docs, relevant))
            metrics["precision"].append(self._calculate_precision(retrieved_docs, relevant))
            metrics["ndcg"].append(self._calculate_ndcg(retrieved_docs, relevant, k))
            
        return RetrieverTestResult(
            name="Hybrid",
            recall_at_k=np.mean(metrics["recall"]),
            precision_at_k=np.mean(metrics["precision"]),
            ndcg_at_k=np.mean(metrics["ndcg"]),
            latency=(time.time() - start_time) / len(self.test_queries),
            memory_usage=self._get_memory_usage()
        )
        
    def test_faiss_retriever(self, embeddings, documents: List[str], k: int = 5) -> RetrieverTestResult:
        """FAISS 검색기 테스트"""
        start_time = time.time()
        
        # FAISS 초기화
        vectorstore = FAISS.from_texts(documents, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        
        metrics = {
            "recall": [],
            "precision": [],
            "ndcg": []
        }
        
        for query, relevant in zip(self.test_queries, self.relevant_docs):
            results = retriever.get_relevant_documents(query)
            retrieved_docs = [doc.page_content for doc in results]
            
            metrics["recall"].append(self._calculate_recall(retrieved_docs, relevant))
            metrics["precision"].append(self._calculate_precision(retrieved_docs, relevant))
            metrics["ndcg"].append(self._calculate_ndcg(retrieved_docs, relevant, k))
            
        return RetrieverTestResult(
            name="FAISS",
            recall_at_k=np.mean(metrics["recall"]),
            precision_at_k=np.mean(metrics["precision"]),
            ndcg_at_k=np.mean(metrics["ndcg"]),
            latency=(time.time() - start_time) / len(self.test_queries),
            memory_usage=self._get_memory_usage()
        )
        
    def test_bm25_retriever(self, documents: List[str], k: int = 5) -> RetrieverTestResult:
        """BM25 검색기 테스트"""
        start_time = time.time()
        
        # BM25 초기화
        retriever = BM25Retriever.from_texts(documents)
        retriever.k = k
        
        metrics = {
            "recall": [],
            "precision": [],
            "ndcg": []
        }
        
        for query, relevant in zip(self.test_queries, self.relevant_docs):
            results = retriever.get_relevant_documents(query)
            retrieved_docs = [doc.page_content for doc in results]
            
            metrics["recall"].append(self._calculate_recall(retrieved_docs, relevant))
            metrics["precision"].append(self._calculate_precision(retrieved_docs, relevant))
            metrics["ndcg"].append(self._calculate_ndcg(retrieved_docs, relevant, k))
            
        return RetrieverTestResult(
            name="BM25",
            recall_at_k=np.mean(metrics["recall"]),
            precision_at_k=np.mean(metrics["precision"]),
            ndcg_at_k=np.mean(metrics["ndcg"]),
            latency=(time.time() - start_time) / len(self.test_queries),
            memory_usage=self._get_memory_usage()
        )
    
    def visualize_results(self):
        """테스트 결과 시각화"""
        results_df = pd.DataFrame([vars(result) for result in self.results])
        
        # 메트릭별 비교 그래프
        plt.figure(figsize=(15, 10))
        
        # Recall, Precision, NDCG 비교
        plt.subplot(2, 2, 1)
        metrics = ['recall_at_k', 'precision_at_k', 'ndcg_at_k']
        df_melted = pd.melt(results_df, id_vars=['name'], value_vars=metrics)
        sns.barplot(data=df_melted, x='name', y='value', hue='variable')
        plt.title('Retrieval Metrics Comparison')
        
        # Latency 비교
        plt.subplot(2, 2, 2)
        sns.barplot(data=results_df, x='name', y='latency')
        plt.title('Latency Comparison')
        
        # Memory Usage 비교
        plt.subplot(2, 2, 3)
        sns.barplot(data=results_df, x='name', y='memory_usage')
        plt.title('Memory Usage Comparison')
        
        plt.tight_layout()
        return plt 