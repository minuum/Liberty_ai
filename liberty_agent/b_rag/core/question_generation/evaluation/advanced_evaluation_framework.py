"""
고도화된 질문 생성 평가 프레임워크
Advanced Question Generation Evaluation Framework
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime

class EvaluationLevel(Enum):
    """평가 수준 정의"""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

class QuestionType(Enum):
    """질문 유형 정의"""
    DIRECT = "direct"
    INDIRECT = "indirect"
    COMPARATIVE = "comparative"
    CONDITIONAL = "conditional"
    HYPOTHETICAL = "hypothetical"

@dataclass
class QuestionMetrics:
    """개별 질문 평가 지표"""
    # 기본 품질 지표 (0.0-1.0)
    legal_accuracy: float
    relevance: float
    difficulty_appropriateness: float
    clarity: float
    educational_value: float
    
    # 고급 평가 지표
    semantic_consistency: float
    factual_grounding: float
    practical_utility: float
    cognitive_load: float
    
    # 메타 정보
    question_type: QuestionType
    target_level: EvaluationLevel
    confidence: float

@dataclass
class DistributionMetrics:
    """분포 관련 평가 지표"""
    yes_count: int
    no_count: int
    ratio: str
    balance_score: float  # 0.0-1.0 (자연스러운 분포일수록 높음)
    extremity_penalty: float  # 극단적 편향에 대한 페널티
    naturalness_score: float  # 분포의 자연스러움

@dataclass
class DiversityMetrics:
    """다양성 평가 지표"""
    perspective_diversity: float  # 관점 다양성
    linguistic_diversity: float   # 언어적 다양성
    cognitive_diversity: float    # 인지적 다양성
    type_diversity: float        # 질문 유형 다양성
    level_progression: float     # 레벨별 점진성

class AdvancedEvaluationFramework:
    """고도화된 평가 프레임워크"""
    
    def __init__(self):
        self.evaluation_history = []
        self.benchmark_scores = self._load_benchmark_scores()
    
    def _load_benchmark_scores(self) -> Dict[str, float]:
        """벤치마크 점수 로드"""
        return {
            "legal_accuracy": 0.85,
            "relevance": 0.80,
            "clarity": 0.90,
            "educational_value": 0.85,
            "semantic_consistency": 0.80,
            "overall_quality": 0.82
        }
    
    def evaluate_comprehensive(self, 
                             questions: List[Dict], 
                             gt_question: str,
                             judgment_text: str) -> Dict[str, Any]:
        """종합적 평가 수행"""
        
        # 1. 개별 질문 평가
        individual_metrics = []
        for q in questions:
            metrics = self._evaluate_individual_question(q, gt_question, judgment_text)
            individual_metrics.append(metrics)
        
        # 2. 분포 평가
        distribution_metrics = self._evaluate_distribution(questions)
        
        # 3. 다양성 평가
        diversity_metrics = self._evaluate_diversity(questions)
        
        # 4. 일관성 평가
        consistency_metrics = self._evaluate_consistency(questions, gt_question)
        
        # 5. 종합 점수 계산
        overall_score = self._calculate_overall_score(
            individual_metrics, distribution_metrics, 
            diversity_metrics, consistency_metrics
        )
        
        # 6. 상세 분석 리포트 생성
        detailed_report = self._generate_detailed_report(
            individual_metrics, distribution_metrics,
            diversity_metrics, consistency_metrics, overall_score
        )
        
        return {
            "timestamp": datetime.now().isoformat(),
            "individual_metrics": individual_metrics,
            "distribution_metrics": distribution_metrics.__dict__,
            "diversity_metrics": diversity_metrics.__dict__,
            "consistency_metrics": consistency_metrics,
            "overall_score": overall_score,
            "detailed_report": detailed_report,
            "benchmark_comparison": self._compare_with_benchmark(overall_score),
            "improvement_suggestions": self._generate_improvement_suggestions(
                individual_metrics, distribution_metrics, diversity_metrics
            )
        }
    
    def _evaluate_individual_question(self, 
                                    question: Dict, 
                                    gt_question: str,
                                    judgment_text: str) -> QuestionMetrics:
        """개별 질문 평가"""
        
        # 기본 품질 지표 계산
        legal_accuracy = self._calculate_legal_accuracy(question, judgment_text)
        relevance = self._calculate_relevance(question, gt_question)
        difficulty_appropriateness = self._calculate_difficulty_appropriateness(question)
        clarity = self._calculate_clarity(question)
        educational_value = self._calculate_educational_value(question)
        
        # 고급 평가 지표 계산
        semantic_consistency = self._calculate_semantic_consistency(question, gt_question)
        factual_grounding = self._calculate_factual_grounding(question, judgment_text)
        practical_utility = self._calculate_practical_utility(question)
        cognitive_load = self._calculate_cognitive_load(question)
        
        # 메타 정보 추출
        question_type = self._identify_question_type(question)
        target_level = self._identify_target_level(question)
        confidence = question.get('confidence', 0.8)
        
        return QuestionMetrics(
            legal_accuracy=legal_accuracy,
            relevance=relevance,
            difficulty_appropriateness=difficulty_appropriateness,
            clarity=clarity,
            educational_value=educational_value,
            semantic_consistency=semantic_consistency,
            factual_grounding=factual_grounding,
            practical_utility=practical_utility,
            cognitive_load=cognitive_load,
            question_type=question_type,
            target_level=target_level,
            confidence=confidence
        )
    
    def _evaluate_distribution(self, questions: List[Dict]) -> DistributionMetrics:
        """분포 평가"""
        yes_count = sum(1 for q in questions if q.get('expected_answer') == 'Yes')
        no_count = len(questions) - yes_count
        
        ratio = f"{yes_count}:{no_count}"
        
        # 자연스러운 분포 점수 계산
        balance_score = self._calculate_balance_score(yes_count, no_count)
        
        # 극단적 편향 페널티
        extremity_penalty = self._calculate_extremity_penalty(yes_count, no_count)
        
        # 분포의 자연스러움
        naturalness_score = self._calculate_naturalness_score(yes_count, no_count)
        
        return DistributionMetrics(
            yes_count=yes_count,
            no_count=no_count,
            ratio=ratio,
            balance_score=balance_score,
            extremity_penalty=extremity_penalty,
            naturalness_score=naturalness_score
        )
    
    def _evaluate_diversity(self, questions: List[Dict]) -> DiversityMetrics:
        """다양성 평가"""
        
        # 관점 다양성
        perspective_diversity = self._calculate_perspective_diversity(questions)
        
        # 언어적 다양성
        linguistic_diversity = self._calculate_linguistic_diversity(questions)
        
        # 인지적 다양성
        cognitive_diversity = self._calculate_cognitive_diversity(questions)
        
        # 질문 유형 다양성
        type_diversity = self._calculate_type_diversity(questions)
        
        # 레벨별 점진성
        level_progression = self._calculate_level_progression(questions)
        
        return DiversityMetrics(
            perspective_diversity=perspective_diversity,
            linguistic_diversity=linguistic_diversity,
            cognitive_diversity=cognitive_diversity,
            type_diversity=type_diversity,
            level_progression=level_progression
        )
    
    def _evaluate_consistency(self, questions: List[Dict], gt_question: str) -> Dict[str, float]:
        """일관성 평가"""
        
        # 의미적 일관성
        semantic_consistency = np.mean([
            self._calculate_semantic_consistency(q, gt_question) 
            for q in questions
        ])
        
        # 논리적 일관성
        logical_consistency = self._calculate_logical_consistency(questions)
        
        # 법리적 일관성
        legal_consistency = self._calculate_legal_consistency(questions)
        
        return {
            "semantic_consistency": semantic_consistency,
            "logical_consistency": logical_consistency,
            "legal_consistency": legal_consistency,
            "overall_consistency": np.mean([
                semantic_consistency, logical_consistency, legal_consistency
            ])
        }
    
    def _calculate_overall_score(self, 
                               individual_metrics: List[QuestionMetrics],
                               distribution_metrics: DistributionMetrics,
                               diversity_metrics: DiversityMetrics,
                               consistency_metrics: Dict[str, float]) -> Dict[str, float]:
        """종합 점수 계산"""
        
        # 개별 질문 평균 점수
        avg_individual = {
            "legal_accuracy": np.mean([m.legal_accuracy for m in individual_metrics]),
            "relevance": np.mean([m.relevance for m in individual_metrics]),
            "clarity": np.mean([m.clarity for m in individual_metrics]),
            "educational_value": np.mean([m.educational_value for m in individual_metrics]),
            "semantic_consistency": np.mean([m.semantic_consistency for m in individual_metrics])
        }
        
        # 가중 평균 계산 (각 영역별 가중치)
        weights = {
            "individual_quality": 0.4,
            "distribution_quality": 0.2,
            "diversity_quality": 0.2,
            "consistency_quality": 0.2
        }
        
        individual_quality = np.mean(list(avg_individual.values()))
        distribution_quality = distribution_metrics.naturalness_score
        diversity_quality = np.mean([
            diversity_metrics.perspective_diversity,
            diversity_metrics.type_diversity,
            diversity_metrics.level_progression
        ])
        consistency_quality = consistency_metrics["overall_consistency"]
        
        overall_score = (
            weights["individual_quality"] * individual_quality +
            weights["distribution_quality"] * distribution_quality +
            weights["diversity_quality"] * diversity_quality +
            weights["consistency_quality"] * consistency_quality
        )
        
        return {
            "individual_quality": individual_quality,
            "distribution_quality": distribution_quality,
            "diversity_quality": diversity_quality,
            "consistency_quality": consistency_quality,
            "overall_score": overall_score,
            "grade": self._assign_grade(overall_score)
        }
    
    def _generate_detailed_report(self, 
                                individual_metrics: List[QuestionMetrics],
                                distribution_metrics: DistributionMetrics,
                                diversity_metrics: DiversityMetrics,
                                consistency_metrics: Dict[str, float],
                                overall_score: Dict[str, float]) -> Dict[str, Any]:
        """상세 분석 리포트 생성"""
        
        return {
            "summary": {
                "total_questions": len(individual_metrics),
                "overall_grade": overall_score["grade"],
                "overall_score": overall_score["overall_score"],
                "distribution": distribution_metrics.ratio
            },
            "strengths": self._identify_strengths(
                individual_metrics, distribution_metrics, diversity_metrics
            ),
            "weaknesses": self._identify_weaknesses(
                individual_metrics, distribution_metrics, diversity_metrics
            ),
            "quality_breakdown": {
                "excellent": len([m for m in individual_metrics if self._get_question_score(m) >= 0.9]),
                "good": len([m for m in individual_metrics if 0.8 <= self._get_question_score(m) < 0.9]),
                "fair": len([m for m in individual_metrics if 0.7 <= self._get_question_score(m) < 0.8]),
                "poor": len([m for m in individual_metrics if self._get_question_score(m) < 0.7])
            },
            "level_analysis": self._analyze_by_level(individual_metrics),
            "type_analysis": self._analyze_by_type(individual_metrics)
        }
    
    def _compare_with_benchmark(self, overall_score: Dict[str, float]) -> Dict[str, Any]:
        """벤치마크와 비교"""
        
        comparisons = {}
        for metric, score in overall_score.items():
            if metric in self.benchmark_scores:
                benchmark = self.benchmark_scores[metric]
                difference = score - benchmark
                comparisons[metric] = {
                    "current": score,
                    "benchmark": benchmark,
                    "difference": difference,
                    "status": "above" if difference > 0 else "below" if difference < 0 else "equal"
                }
        
        return {
            "comparisons": comparisons,
            "overall_performance": "above_benchmark" if overall_score["overall_score"] > self.benchmark_scores["overall_quality"] else "below_benchmark"
        }
    
    def _generate_improvement_suggestions(self,
                                        individual_metrics: List[QuestionMetrics],
                                        distribution_metrics: DistributionMetrics,
                                        diversity_metrics: DiversityMetrics) -> List[str]:
        """개선 제안 생성"""
        
        suggestions = []
        
        # 개별 질문 품질 개선
        avg_legal_accuracy = np.mean([m.legal_accuracy for m in individual_metrics])
        if avg_legal_accuracy < 0.8:
            suggestions.append("법적 정확성 향상을 위해 판결문 근거를 더 명확히 반영하세요.")
        
        avg_clarity = np.mean([m.clarity for m in individual_metrics])
        if avg_clarity < 0.8:
            suggestions.append("질문의 명확성을 높이기 위해 더 구체적인 표현을 사용하세요.")
        
        # 분포 개선
        if distribution_metrics.naturalness_score < 0.7:
            suggestions.append("더 자연스러운 Yes/No 분포를 위해 GT 질문의 의미를 더 깊이 분석하세요.")
        
        # 다양성 개선
        if diversity_metrics.perspective_diversity < 0.7:
            suggestions.append("다양한 법적 관점(원고/피고/법원/학계)을 더 균형있게 포함하세요.")
        
        if diversity_metrics.type_diversity < 0.7:
            suggestions.append("직접/간접/비교/조건부 등 다양한 질문 유형을 활용하세요.")
        
        return suggestions
    
    # 헬퍼 메서드들 (실제 계산 로직)
    def _calculate_legal_accuracy(self, question: Dict, judgment_text: str) -> float:
        """법적 정확성 계산"""
        # 실제 구현에서는 NLP 모델이나 법률 온톨로지 활용
        return question.get('legal_accuracy', 0.8)
    
    def _calculate_relevance(self, question: Dict, gt_question: str) -> float:
        """관련성 계산"""
        return question.get('relevance', 0.8)
    
    def _calculate_difficulty_appropriateness(self, question: Dict) -> float:
        """난이도 적절성 계산"""
        return question.get('difficulty_appropriateness', 0.8)
    
    def _calculate_clarity(self, question: Dict) -> float:
        """명확성 계산"""
        return question.get('clarity', 0.8)
    
    def _calculate_educational_value(self, question: Dict) -> float:
        """교육적 가치 계산"""
        return question.get('educational_value', 0.8)
    
    def _calculate_semantic_consistency(self, question: Dict, gt_question: str) -> float:
        """의미적 일관성 계산"""
        return question.get('semantic_consistency', 0.8)
    
    def _calculate_factual_grounding(self, question: Dict, judgment_text: str) -> float:
        """사실적 근거 계산"""
        return question.get('factual_grounding', 0.8)
    
    def _calculate_practical_utility(self, question: Dict) -> float:
        """실무적 유용성 계산"""
        return question.get('practical_utility', 0.8)
    
    def _calculate_cognitive_load(self, question: Dict) -> float:
        """인지적 부하 계산"""
        return question.get('cognitive_load', 0.5)
    
    def _identify_question_type(self, question: Dict) -> QuestionType:
        """질문 유형 식별"""
        return QuestionType(question.get('question_type', 'direct'))
    
    def _identify_target_level(self, question: Dict) -> EvaluationLevel:
        """대상 수준 식별"""
        level = question.get('level', 5)
        if level <= 2:
            return EvaluationLevel.EXPERT
        elif level <= 4:
            return EvaluationLevel.ADVANCED
        elif level <= 6:
            return EvaluationLevel.INTERMEDIATE
        else:
            return EvaluationLevel.BASIC
    
    def _calculate_balance_score(self, yes_count: int, no_count: int) -> float:
        """균형 점수 계산"""
        total = yes_count + no_count
        if total == 0:
            return 0.0
        
        ratio = min(yes_count, no_count) / max(yes_count, no_count)
        return ratio
    
    def _calculate_extremity_penalty(self, yes_count: int, no_count: int) -> float:
        """극단적 편향 페널티 계산"""
        total = yes_count + no_count
        if total == 0:
            return 0.0
        
        # 10:0 또는 0:10인 경우 최대 페널티
        if yes_count == 0 or no_count == 0:
            return 1.0
        
        # 9:1 또는 1:9인 경우 중간 페널티
        if min(yes_count, no_count) == 1:
            return 0.5
        
        return 0.0
    
    def _calculate_naturalness_score(self, yes_count: int, no_count: int) -> float:
        """자연스러움 점수 계산"""
        # 7:3, 8:2, 6:4 등이 자연스러운 분포로 간주
        total = yes_count + no_count
        if total == 0:
            return 0.0
        
        ratio = min(yes_count, no_count) / total
        
        # 0.2-0.4 범위가 자연스러운 분포
        if 0.2 <= ratio <= 0.4:
            return 1.0
        elif 0.1 <= ratio < 0.2 or 0.4 < ratio <= 0.5:
            return 0.8
        elif 0.05 <= ratio < 0.1:
            return 0.5
        else:
            return 0.2
    
    def _calculate_perspective_diversity(self, questions: List[Dict]) -> float:
        """관점 다양성 계산"""
        perspectives = set()
        for q in questions:
            perspectives.add(q.get('legal_perspective', 'general'))
        
        # 최대 5개 관점 (요건/효과/비교/실무/예외)
        return min(len(perspectives) / 5.0, 1.0)
    
    def _calculate_linguistic_diversity(self, questions: List[Dict]) -> float:
        """언어적 다양성 계산"""
        # 실제 구현에서는 어휘 다양성, 문장 구조 다양성 등을 분석
        return 0.8
    
    def _calculate_cognitive_diversity(self, questions: List[Dict]) -> float:
        """인지적 다양성 계산"""
        # 실제 구현에서는 인지적 복잡성 분석
        return 0.8
    
    def _calculate_type_diversity(self, questions: List[Dict]) -> float:
        """질문 유형 다양성 계산"""
        types = set()
        for q in questions:
            types.add(q.get('question_type', 'direct'))
        
        # 최대 6개 유형
        return min(len(types) / 6.0, 1.0)
    
    def _calculate_level_progression(self, questions: List[Dict]) -> float:
        """레벨별 점진성 계산"""
        # 레벨별 난이도가 점진적으로 변하는지 확인
        levels = [q.get('level', 5) for q in questions]
        if len(set(levels)) == len(levels):  # 모든 레벨이 다름
            return 1.0
        else:
            return 0.8
    
    def _calculate_logical_consistency(self, questions: List[Dict]) -> float:
        """논리적 일관성 계산"""
        # 실제 구현에서는 논리적 모순 검사
        return 0.85
    
    def _calculate_legal_consistency(self, questions: List[Dict]) -> float:
        """법리적 일관성 계산"""
        # 실제 구현에서는 법리적 모순 검사
        return 0.85
    
    def _get_question_score(self, metrics: QuestionMetrics) -> float:
        """개별 질문 종합 점수"""
        return np.mean([
            metrics.legal_accuracy,
            metrics.relevance,
            metrics.clarity,
            metrics.educational_value,
            metrics.semantic_consistency
        ])
    
    def _assign_grade(self, score: float) -> str:
        """점수에 따른 등급 부여"""
        if score >= 0.9:
            return "A+"
        elif score >= 0.85:
            return "A"
        elif score >= 0.8:
            return "B+"
        elif score >= 0.75:
            return "B"
        elif score >= 0.7:
            return "C+"
        elif score >= 0.65:
            return "C"
        else:
            return "D"
    
    def _identify_strengths(self, individual_metrics, distribution_metrics, diversity_metrics) -> List[str]:
        """강점 식별"""
        strengths = []
        
        avg_legal_accuracy = np.mean([m.legal_accuracy for m in individual_metrics])
        if avg_legal_accuracy >= 0.9:
            strengths.append("법적 정확성이 매우 우수함")
        
        if distribution_metrics.naturalness_score >= 0.8:
            strengths.append("자연스러운 Yes/No 분포 달성")
        
        if diversity_metrics.perspective_diversity >= 0.8:
            strengths.append("다양한 법적 관점 포함")
        
        return strengths
    
    def _identify_weaknesses(self, individual_metrics, distribution_metrics, diversity_metrics) -> List[str]:
        """약점 식별"""
        weaknesses = []
        
        avg_clarity = np.mean([m.clarity for m in individual_metrics])
        if avg_clarity < 0.7:
            weaknesses.append("질문의 명확성 부족")
        
        if distribution_metrics.extremity_penalty > 0.5:
            weaknesses.append("극단적인 Yes/No 편향")
        
        if diversity_metrics.type_diversity < 0.6:
            weaknesses.append("질문 유형의 다양성 부족")
        
        return weaknesses
    
    def _analyze_by_level(self, individual_metrics: List[QuestionMetrics]) -> Dict[str, Any]:
        """레벨별 분석"""
        level_analysis = {}
        
        for level in EvaluationLevel:
            level_metrics = [m for m in individual_metrics if m.target_level == level]
            if level_metrics:
                avg_score = np.mean([self._get_question_score(m) for m in level_metrics])
                level_analysis[level.value] = {
                    "count": len(level_metrics),
                    "average_score": avg_score,
                    "grade": self._assign_grade(avg_score)
                }
        
        return level_analysis
    
    def _analyze_by_type(self, individual_metrics: List[QuestionMetrics]) -> Dict[str, Any]:
        """유형별 분석"""
        type_analysis = {}
        
        for qtype in QuestionType:
            type_metrics = [m for m in individual_metrics if m.question_type == qtype]
            if type_metrics:
                avg_score = np.mean([self._get_question_score(m) for m in type_metrics])
                type_analysis[qtype.value] = {
                    "count": len(type_metrics),
                    "average_score": avg_score,
                    "grade": self._assign_grade(avg_score)
                }
        
        return type_analysis

# 사용 예시
def example_usage():
    """사용 예시"""
    
    # 평가 프레임워크 초기화
    evaluator = AdvancedEvaluationFramework()
    
    # 샘플 질문 데이터
    sample_questions = [
        {
            "level": 1,
            "question": "민법 제470조에 따른 동업자의 채권 준점유자 지위가 인정되는가?",
            "expected_answer": "Yes",
            "legal_accuracy": 0.95,
            "relevance": 0.90,
            "clarity": 0.85,
            "educational_value": 0.90,
            "question_type": "direct",
            "legal_perspective": "요건중심"
        },
        # ... 더 많은 질문들
    ]
    
    gt_question = "동업자가 채권의 준점유자에 해당하지 아니한다고 할 수 있는가?"
    judgment_text = "판결문 내용..."
    
    # 종합 평가 수행
    evaluation_result = evaluator.evaluate_comprehensive(
        sample_questions, gt_question, judgment_text
    )
    
    # 결과 출력
    print(f"Overall Score: {evaluation_result['overall_score']['overall_score']:.3f}")
    print(f"Grade: {evaluation_result['overall_score']['grade']}")
    print(f"Distribution: {evaluation_result['distribution_metrics']['ratio']}")
    
    return evaluation_result

if __name__ == "__main__":
    example_usage() 