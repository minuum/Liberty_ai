import logging
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from langchain.schema import Document

from liberty_agent.search_engine import LegalSearchEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SearchWeightEvaluator:
    def __init__(self, search_engine: LegalSearchEngine):
        self.search_engine = search_engine
        self.test_queries = self._get_test_queries()
        self.results_dir = Path("./evaluation_results")
        self.results_dir.mkdir(exist_ok=True)
        
    def _get_test_queries(self) -> Dict[str, List[str]]:
        """테스트용 쿼리 세트 반환"""
        return {
            "민사": [
                "임대차 계약 해지 요건은 무엇인가요?",
                "주택임대차보호법상 계약갱신요구권의 행사방법은?",
                "부동산 매매계약 위약금 청구 요건은?"
            ],
            "형사": [
                "특수폭행죄의 구성요건은 무엇인가요?",
                "사기죄의 기수시기는 언제인가요?",
                "정당방위의 성립요건은?"
            ],
            "행정": [
                "행정처분 취소소송의 제소기간은?",
                "영업정지처분 효력정지신청 요건은?",
                "건축허가 거부처분 취소소송의 요건은?"
            ]
        }
    def _safe_get_content(self, doc: Document) -> str:
        """문서 내용을 안전하게 가져오기"""
        if isinstance(doc, Document):
            return doc.page_content
        elif isinstance(doc, str):
            return doc
        elif isinstance(doc, list):
            return ' '.join([self._safe_get_content(d) for d in doc])
        else:
            return str(doc)
    def evaluate_retrieval_quality(
        self,
        query: str,
        results: List[Document],
        evaluation_type: str
    ) -> float:
        """검색 결과 품질 평가"""
        try:
            if not results:
                return 0.0
            
            contents= [self._safe_get_content(doc) for doc in results]
            if evaluation_type == "keyword":
                # _calculate_keyword_similarity -> _calculate_keyword_match로 변경
                return self.search_engine._calculate_keyword_match(contents, query)
            elif evaluation_type == "semantic":
                # _calculate_semantic_similarity -> evaluate_context_quality로 변경
                return self.search_engine.evaluate_context_quality(results, query)
            elif evaluation_type == "metadata":
                return self.search_engine._calculate_metadata_reliability(contents)
            elif evaluation_type == "hybrid":
                keyword_score = self.search_engine._calculate_keyword_match(contents, query)
                semantic_score = self.search_engine.evaluate_context_quality(contents, query)
                metadata_score = self.search_engine._calculate_metadata_reliability(results)
                
                # 하이브리드 점수 계산 (현재 가중치 설정 반영)
                weights = self.search_engine.evaluate_context_quality.weights
                return (
                    keyword_score * weights['keyword'] +
                    semantic_score * weights['semantic'] +
                    metadata_score * weights['metadata']
                )
            else:
                raise ValueError(f"Unknown evaluation type: {evaluation_type}")
                
        except Exception as e:
            logger.error(f"검색 결과 품질 평가 중 오류: {str(e)}")
            return 0.0

    def run_comprehensive_evaluation(
        self,
        weight_configs: List[Dict]
    ) -> pd.DataFrame:
        """종합적인 평가 실행"""
        all_results = []
        evaluation_types = ["keyword", "semantic", "metadata", "hybrid"]
        
        for config in tqdm(weight_configs, desc="가중치 설정 평가 중"):
            # 검색 엔진 가중치 설정
            self.search_engine.weights = {
                'semantic': config["dense"],
                'keyword': config["keyword"],
                'metadata': config["metadata"]
            }
            
            for area, queries in self.test_queries.items():
                for query in queries:
                    # 검색 실행
                    search_results = self.search_engine.hybrid_search(query)
                    
                    # 각 평가 방식별 점수 계산
                    for eval_type in evaluation_types:
                        quality_score = self.evaluate_retrieval_quality(
                            query,
                            search_results,
                            eval_type
                        )
                        
                        result = {
                            "weights": str(config),
                            "legal_area": area,
                            "query": query,
                            "evaluation_type": eval_type,
                            "quality_score": quality_score
                        }
                        
                        all_results.append(result)
        
        results_df = pd.DataFrame(all_results)
        self._save_results(results_df)
        self._visualize_results(results_df)
        
        return results_df

    def _visualize_results(self, df: pd.DataFrame):
        """결과 시각화"""
        # 1. 평가 방식별 성능 비교
        plt.figure(figsize=(12, 6))
        sns.boxplot(
            data=df,
            x='evaluation_type',
            y='quality_score'
        )
        plt.title('평가 방식별 검색 품질 분포')
        plt.savefig(self.results_dir / 'evaluation_type_performance.png')
        plt.close()

        # 2. 가중치 설정별 하이브리드 성능
        hybrid_df = df[df['evaluation_type'] == 'hybrid']
        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=hybrid_df,
            x='weights',
            y='quality_score'
        )
        plt.title('가중치 설정별 하이브리드 검색 성능')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.results_dir / 'weight_performance.png')
        plt.close()

    def _save_results(self, df: pd.DataFrame):
        """평가 결과 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"retrieval_evaluation_{timestamp}.csv"
        df.to_csv(self.results_dir / filename, index=False)
        logger.info(f"평가 결과 저장 완료: {filename}")