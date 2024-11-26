import logging
from typing import List, Dict, Optional
from tqdm import tqdm
from langchain.schema import Document
from langchain.vectorstores import FAISS
from ragas import evaluate
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from ragas.dataset_schema import EvaluationDataset,SingleTurnSample
from ragas.metrics import (
    ContextRecall,
    ContextPrecision,
    ContextEntityRecall,
    Faithfulness,
    AnswerRelevancy,
    AnswerCorrectness,
)
import time
from uuid import uuid4
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
        self.k=20
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
        
        # 검색기별 최적 파라미터 설정
        self.retriever_configs = {
            'pinecone': {
                'default': {
                    'alpha': 0.75,
                    'k': 20,
                    'rerank': False
                },
                'high_precision': {
                    'alpha': 0.8,
                    'k': 30,
                    'rerank': True,
                    'rerank_model': "bge-reranker-v2-m3"
                },
                'high_recall': {
                    'alpha': 0.6,
                    'k': 40,
                    'rerank': False
                }
            }
        }

    def get_retriever_results(self, retriever_name: str, retriever, query: str, mode: str = 'default'):
        """검색기별 최적화된 검색 실행"""
        try:
            if retriever_name == 'faiss':
                docs = retriever.similarity_search_with_score(query, search_kwargs={"k": self.k})
                return self._normalize_docs(docs)
            
            elif retriever_name == 'hybrid':
                docs = retriever._get_relevant_documents(query)
                return self._normalize_docs(docs)
            
            elif retriever_name == 'kiwi':
                docs = retriever.search_with_score(query)
                return self._normalize_docs(docs)
            
            elif retriever_name == 'pinecone':
                from langchain.callbacks.manager import CallbackManagerForRetrieverRun
                
                config = self.retriever_configs['pinecone'][mode]
                search_kwargs = {
                    "k": config['k'],
                    "alpha": config['alpha'],
                    "score": True,
                    "include_metadata": True
                }
                logger.info(f"Pinecone 검색기 호출 완료: {search_kwargs}")
                
                if config.get('rerank'):
                    search_kwargs.update({
                        "rerank": True,
                        "rerank_model": config.get('rerank_model', "bge-reranker-v2-m3"),
                        "top_n": config['k'],
                        "return_rerank_score": True
                    })
                
                run_manager = CallbackManagerForRetrieverRun(
                    run_id=uuid4(),
                    handlers=[],
                    inheritable_handlers=[],
                    tags=["pinecone_retriever"],
                    metadata={}
                )
                
                docs = retriever._get_relevant_documents(
                    query, 
                    search_kwargs=search_kwargs, 
                    run_manager=run_manager
                )
                
                normalized_docs = []
                for doc in docs:
                    score = None
                    if hasattr(doc, 'score'):
                        score = doc.score
                    elif isinstance(doc.metadata, dict) and 'score' in doc.metadata:
                        score = doc.metadata['score']
                    elif isinstance(doc.metadata, dict) and '_score' in doc.metadata:
                        score = doc.metadata['_score']
                        
                    if score is not None:
                        normalized_score = float(score)
                        if config.get('rerank'):
                            normalized_score = score
                        else:
                            normalized_score = (score + 1) / 2
                        
                        normalized_docs.append(Document(
                            page_content=doc.page_content,
                            metadata={
                                **doc.metadata,
                                'score': normalized_score
                            }
                        ))
                    else:
                        normalized_docs.append(doc)
                
                logger.info(f"Pinecone 검색기 결과 (스코어 포함): {normalized_docs}")
                return normalized_docs
                
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

    def prepare_generation_data(self, retriever_name: str, batch_size: int = 5) -> EvaluationDataset:
        """RAGAS Generation 평가를 위한 데이터셋 준비"""
        if not self.llm:
            raise ValueError("Generation 평가를 위해서는 LLM이 필요합니다.")
        
        samples = []
        
        # 배치 단위로 처리
        for i in range(0, len(self.test_cases), batch_size):
            batch = self.test_cases[i:i + batch_size]
            
            for test_case in batch:
                try:
                    # 필수 필드 추출
                    query = test_case["query"]
                    ground_truth = test_case["ground_truth"]
                    
                    # 리트리버로 컨텍스트 가져오기
                    results = self.get_retriever_results(
                        retriever_name, 
                        self.retrievers[retriever_name], 
                        query
                    )
                    
                    if results:
                        # 검색된 컨텍스트 추출
                        retrieved_contexts = [doc.page_content for doc in results]
                        
                        # LLM으로 답변 생성
                        prompt = f"질문: {query}\n\n컨텍스트: {' '.join(retrieved_contexts)}\n\n답변:"
                        answer = self.llm.invoke(prompt)
                        response = answer.content if hasattr(answer, 'content') else str(answer)
                        
                        # API 속도 제한 방지를 위한 대기
                        time.sleep(3)  # 3초 대기
                        
                        # SingleTurnSample 생성
                        sample = SingleTurnSample(
                            user_input=query,  # question 대신 user_input 사용
                            retrieved_contexts=retrieved_contexts,
                            response=response,
                            reference=ground_truth
                        )
                        samples.append(sample)
                    
                except Exception as e:
                    logger.error(f"Generation 데이터 준비 중 오류 발생: {str(e)}")
        
        if not samples:
            logger.warning("생성된 평가 샘플이 없습니다.")
            return EvaluationDataset(samples=[])
        
        return EvaluationDataset(samples=samples)

    def _normalize_docs(self, docs: List[Document]) -> List[Document]:
        """문서 형식 정규화"""
        normalized = []
        for doc in docs:
            if isinstance(doc, Document):
                normalized.append(doc)
            elif isinstance(doc, dict):
                normalized.append(Document(
                    page_content=doc.get('content', ''),
                    metadata=doc.get('metadata', {})
                ))
            else:
                normalized.append(Document(
                    page_content=str(doc),
                    metadata={}
                ))
        return normalized

    def prepare_evaluation_sample(self, test_case: Dict) -> SingleTurnSample:
        """RAGAS 평가용 샘플 준비"""
        return SingleTurnSample(
            question=test_case["query"],
            ground_truths=[test_case["ground_truth"]],
            contexts=[c["content"] for c in test_case["contexts"]],
            context_metadata=[c["metadata"] for c in test_case["contexts"]]
        )

    def evaluate_retrieval(self) -> Dict:
        """검색 성능 평가"""
        results = {}
        
        for retriever_name, retriever in self.retrievers.items():
            samples = []
            
            for test_case in tqdm(self.test_cases, desc=f"{retriever_name} 평가 중"):
                try:
                    retrieved_docs = self.get_retriever_results(
                        retriever_name, retriever, test_case["query"]
                    )
                    if retrieved_docs:
                        sample = SingleTurnSample(
                            question=test_case["query"],
                            ground_truths=[test_case["ground_truth"]],
                            contexts=[c["content"] for c in test_case["contexts"]],
                            retrieved_contexts=[doc.page_content for doc in retrieved_docs],
                            reference=test_case["ground_truth"],
                            user_input=test_case["query"]
                        )
                        samples.append(sample)
                except Exception as e:
                    logger.error(f"{retriever_name} 평가 중 오류: {str(e)}")
            
            if samples:
                try:
                    # RAGAS 데이터셋으로 변환하여 평가
                    dataset = EvaluationDataset(samples=samples)
                    results[retriever_name] = evaluate(
                        dataset,
                        metrics=self.retrieval_metrics,
                        llm=self.llm
                    )
                except Exception as e:
                    logger.error(f"{retriever_name} 평가 중 오류: {str(e)}")
        
        return results

    def evaluate_generation(self) -> Dict:
        """모든 리트리버의 생성 모델 성능 평가 실행"""
        results = {}
        
        for retriever_name in tqdm(self.retrievers.keys(), desc="Generation 평가 중"):
            try:
                logger.info(f"{retriever_name} 리트리버의 생성 모델 평가 시작")
                eval_dataset = self.prepare_generation_data(retriever_name)
                
                if len(eval_dataset) > 0:
                    logger.info(f"{retriever_name}: {len(eval_dataset)}개의 평가 데이터 준비 완료")
                    result = evaluate(
                        eval_dataset,
                        metrics=self.generation_metrics
                    )
                    results[retriever_name] = result
                    logger.info(f"{retriever_name} 평가 완료: {result._repr_dict}")
                else:
                    logger.warning(f"{retriever_name}: 평가할 데이터가 없습니다")
                    
            except Exception as e:
                logger.error(f"{retriever_name} Generation 평가 중 오류 발생: {str(e)}")
        
        if results:
            logger.info("모든 리트리버의 생성 ��델 평가가 완료되었습니다")
        else:
            logger.warning("평가 결과가 없습니다")
        
        return results

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
            for retriever_name, result in results.items():
                for metric_name, score in result._repr_dict.items():
                    viz_data.append({
                        "Retriever": retriever_name,
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
        import pandas as pd
        if not results:  # 결과가 비어있는 경우 처리
            print("평가 결과가 없습니다.")
            return
            
        data = []
        if eval_type == "retrieval":
            for retriever_name, result in results.items():
                if hasattr(result, '_repr_dict'):
                    for metric_name, score in result._repr_dict.items():
                        data.append({
                            "Retriever": retriever_name,
                            "Metric": metric_name,
                            "Score": score
                        })
        
        if data:  # 데이터가 있는 경우만 처리
            df = pd.DataFrame(data)
            if not df.empty:
                pivot_df = df.pivot(index='Retriever', columns='Metric', values='Score')
                print(pivot_df.to_string())
        else:
            print("시각화할 데이터가 없습니다.")