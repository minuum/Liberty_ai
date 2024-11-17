import logging
from typing import List, Dict, Optional

from langchain.schema import Document
from langchain.vectorstores import FAISS
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset,SingleTurnSample
from ragas.metrics import (
    ContextRecall,
    ContextPrecision,
    ContextEntityRecall,
    Faithfulness,
    AnswerRelevancy,
    AnswerCorrectness,
)

logger = logging.getLogger(__name__)

class RAGEvaluator:
    def __init__(
        self, 
        retrievers: Dict, 
        llm: Optional[object] = None,
        test_cases: List[Dict] = None
    ):
        """
        Args:
            retrievers: 평가할 리트리버들의 딕셔너리 (faiss, kiwi, pinecone, hybrid)
            llm: 생성 평가를 위한 LLM 모델
            test_cases: 테스트 케이스 리스트
        """
        self.retrievers = retrievers
        self.llm = llm
        self.test_cases = test_cases
        
        # Retrieval 평가 메트릭
        self.retrieval_metrics = [
            ContextRecall(),
            ContextPrecision(),
            ContextEntityRecall()
        ]
        
        # Generation 평가 메트릭
        self.generation_metrics = [
            Faithfulness(),
            AnswerRelevancy(),
            AnswerCorrectness()
        ]

    def get_retriever_results(self, retriever_name: str, retriever, query: str, k: int = 10):
        """리트리버 타입별 검색 실행"""
        try:
            if retriever is None:
                logger.error(f"{retriever_name} 리트리버가 None입니다.")
                return []
            
            if retriever_name == 'faiss':
                return retriever.similarity_search(query, k=k)
            elif retriever_name == 'hybrid':
                return retriever._get_relevant_documents(query)
            elif retriever_name == 'kiwi':
                return retriever.search_with_score(query, top_k=k)
            elif retriever_name == 'pinecone':
                return retriever.invoke(query)
        except Exception as e:
            logger.error(f"{retriever_name} 검색 중 오류 발생: {str(e)}")
            return []

    def prepare_retrieval_data(self) -> Dict[str, EvaluationDataset]:
        """RAGAS Retrieval 평가를 위한 데이터셋 준비"""
        datasets = {}
        
        for retriever_name, retriever in self.retrievers.items():
            samples = []
            for test_case in self.test_cases:
                query = test_case["query"]
                ground_truth = test_case.get("ground_truths", [""])[0]
                try:
                    results = self.get_retriever_results(retriever_name, retriever, query)
                    if results:
                        contexts = [doc.page_content for doc in results]
                        sample = SingleTurnSample(
                            user_input=query,
                            retrieved_contexts=contexts,
                            reference=ground_truth
                        )
                        samples.append(sample)
                except Exception as e:
                    logger.error(f"{retriever_name} 데이터 준비 중 오류 발생: {str(e)}")
            if samples:
                datasets[retriever_name] = EvaluationDataset(samples=samples)
        return datasets

    def prepare_generation_data(self, retriever_name: str) -> EvaluationDataset:
        """RAGAS Generation 평가를 위한 데이터셋 준비"""
        if not self.llm:
            raise ValueError("Generation 평가를 위해서는 LLM이 필요합니다.")
            
        samples = []
        
        for test_case in self.test_cases:
            query = test_case["query"]
            ground_truth = test_case.get("ground_truths", [""])[0]
            
            try:
                results = self.get_retriever_results(
                    retriever_name, 
                    self.retrievers[retriever_name], 
                    query
                )
                
                if results:
                    contexts = [doc.page_content for doc in results]
                    
                    # LLM을 사용하여 답변 생성
                    prompt = f"질문: {query}\n\n컨텍스트: {' '.join(contexts)}\n\n답변:"
                    answer = self.llm.invoke(prompt)
                    
                    sample = SingleTurnSample(
                        user_input=query,
                        retrieved_contexts=contexts,
                        response=answer.content if hasattr(answer, 'content') else str(answer),
                        reference=ground_truth
                    )
                    
                    samples.append(sample)
                
            except Exception as e:
                logger.error(f"Generation 데이터 준비 중 오류 발생: {str(e)}")
        
        return EvaluationDataset(samples=samples)

    def evaluate_retrieval(self):
        """리트리버 성능 평가 실행"""
        try:
            eval_datasets = self.prepare_retrieval_data()
            
            results = {}
            for retriever_name, eval_dataset in eval_datasets.items():
                if len(eval_dataset) > 0:
                    evaluation_result = evaluate(
                        eval_dataset,
                        metrics=self.retrieval_metrics
                    )
                    results[retriever_name] = evaluation_result
                        
            return results
                
        except Exception as e:
            logger.error(f"Retrieval 평가 중 오류 발생: {str(e)}")
            raise

    def evaluate_generation(self, retriever_name: str):
        """생성 모델 성능 평가 실행"""
        try:
            eval_dataset = self.prepare_generation_data(retriever_name)
            
            if len(eval_dataset) > 0:
                results = evaluate(
                    eval_dataset,
                    metrics=self.generation_metrics
                )
                return results
            
        except Exception as e:
            logger.error(f"Generation 평가 중 오류 발생: {str(e)}")
            raise

    def visualize_results(self, results: Dict, eval_type: str = "retrieval"):
        """평가 결과 시각화"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd

        viz_data = []

        if eval_type == "retrieval":
            for retriever_name, result in results.items():
                for metric_name, score in result._repr_dict.items():
                    viz_data.append({
                        "Retriever": retriever_name,
                        "Metric": metric_name,
                        "Score": score
                    })
        else:  # generation
            for metric_name, score in results._repr_dict.items():
                viz_data.append({
                    "Retriever": "Generation",
                    "Metric": metric_name,
                    "Score": score
                })

        df = pd.DataFrame(viz_data)

        # 각 지표별로 서브플롯 생성
        metrics = df['Metric'].unique()
        num_metrics = len(metrics)
        fig, axes = plt.subplots(1, num_metrics, figsize=(5 * num_metrics, 6))

        if num_metrics == 1:
            axes = [axes]

        for i, metric in enumerate(metrics):
            ax = axes[i]
            metric_data = df[df['Metric'] == metric]
            sns.barplot(
                data=metric_data,
                x="Retriever",
                y="Score",
                ax=ax
            )
            ax.set_title(metric)
            ax.set_ylim(0, 1)
            ax.set_xlabel('Retriever')
            ax.set_ylabel('Score')

        plt.tight_layout()
        return plt  # plt 객체를 반환합니다.
    
    def visualize_results_table(self, results: Dict, eval_type: str = "retrieval"):
        """평가 결과를 표 형태로 출력"""
        import pandas as pd

        data = []

        if eval_type == "retrieval":
            for retriever_name, result in results.items():
                for metric_name, score in result._repr_dict.items():
                    data.append({
                        "Retriever": retriever_name,
                        "Metric": metric_name,
                        "Score": score
                    })
        else:  # generation
            for metric_name, score in results._repr_dict.items():
                data.append({
                    "Retriever": "Generation",
                    "Metric": metric_name,
                    "Score": score
                })

        df = pd.DataFrame(data)
        pivot_df = df.pivot(index='Retriever', columns='Metric', values='Score').reset_index()
        print(pivot_df.to_string(index=False))