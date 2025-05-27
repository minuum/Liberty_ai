"""
B-RAG 프로젝트 실험 설정
질문 생성론 검증을 위한 실험 파라미터 관리
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path

@dataclass
class QuestionGenerationConfig:
    """질문 생성 설정"""
    # 모델 설정
    model_name: str = "gpt-4o-2024-08-06"
    temperature: float = 0.1
    
    # 질문 생성 파라미터
    questions_per_level: int = 1
    total_levels: int = 10
    
    # 검증 설정
    min_consistency_rate: float = 0.8
    max_retries: int = 3

@dataclass
class RAGExperimentConfig:
    """RAG 실험 설정"""
    # 모델 설정
    embedding_model: str = "solar-embedding-1-large"
    llm_model: str = "gpt-4o-2024-08-06"
    llm_temperature: float = 0.1
    
    # RAG 파라미터
    top_k: int = 3
    similarity_threshold: float = 0.7
    
    # Boost RAG 설정
    max_boost_iterations: int = 3
    confidence_threshold: float = 0.9

@dataclass
class ExperimentPipeline:
    """전체 실험 파이프라인 설정"""
    # 기본 설정
    experiment_name: str = "b_rag_experiment"
    output_dir: Path = Path("results")
    
    # 실험 단계 설정
    run_question_generation: bool = True
    run_standard_rag: bool = True
    run_boost_rag: bool = True
    run_comparison_analysis: bool = True
    
    # 데이터 설정
    sample_documents: List[str] = field(default_factory=lambda: [
        """
        대법원 1982. 11. 9. 선고 80다3135 판결
        
        【판시사항】
        동업자가 채권의 준점유자에 해당하지 아니한다고 할 수 있다.
        
        【판결요지】
        민법 제470조에 따르면 채권의 준점유자에게 변제한 경우에도 
        변제자가 선의이고 과실이 없으면 유효한 변제가 된다.
        그러나 단순한 동업관계만으로는 채권의 준점유자로 볼 수 없다.
        """,
        """
        대법원 2020. 5. 14. 선고 2018다12345 판결
        
        계약 해지로 인한 손해배상청구권은 일반 채권으로서 
        민법 제162조 제1항에 따라 10년의 소멸시효에 걸린다.
        
        다만, 불법행위로 인한 손해배상청구권과는 구별되며,
        계약 해지 시점부터 시효가 진행된다.
        """
    ])
    
    # GT 질문들
    gt_questions: List[str] = field(default_factory=lambda: [
        "동업자가 채권의 준점유자에 해당하지 아니 한다고 할 수 있는가?",
        "계약 해지 시 손해배상청구권이 소멸시효에 걸리는가?"
    ])
    
    # 성능 목표
    target_metrics: Dict[str, float] = field(default_factory=lambda: {
        "question_consistency_rate": 0.8,
        "boost_rag_improvement": 0.1,  # 10% 개선 목표
        "yes_answer_increase": 2  # 평균 2개 이상 Yes 답변 증가
    })

@dataclass
class BRAGConfig:
    """B-RAG 프로젝트 통합 설정"""
    question_generation: QuestionGenerationConfig = field(default_factory=QuestionGenerationConfig)
    rag_experiment: RAGExperimentConfig = field(default_factory=RAGExperimentConfig)
    pipeline: ExperimentPipeline = field(default_factory=ExperimentPipeline)
    
    def __post_init__(self):
        """설정 검증 및 초기화"""
        # 출력 디렉토리 생성
        self.pipeline.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 설정 검증
        self._validate_config()
    
    def _validate_config(self):
        """설정 유효성 검증"""
        assert self.question_generation.total_levels > 0, "총 레벨 수는 0보다 커야 합니다"
        assert 0 <= self.question_generation.temperature <= 1, "Temperature는 0-1 사이여야 합니다"
        assert self.rag_experiment.top_k > 0, "Top-K는 0보다 커야 합니다"
        assert len(self.pipeline.gt_questions) > 0, "GT 질문이 최소 1개는 있어야 합니다"
    
    def to_dict(self) -> Dict:
        """설정을 딕셔너리로 변환"""
        return {
            "question_generation": {
                "model_name": self.question_generation.model_name,
                "temperature": self.question_generation.temperature,
                "questions_per_level": self.question_generation.questions_per_level,
                "total_levels": self.question_generation.total_levels,
                "min_consistency_rate": self.question_generation.min_consistency_rate,
                "max_retries": self.question_generation.max_retries
            },
            "rag_experiment": {
                "embedding_model": self.rag_experiment.embedding_model,
                "llm_model": self.rag_experiment.llm_model,
                "llm_temperature": self.rag_experiment.llm_temperature,
                "top_k": self.rag_experiment.top_k,
                "similarity_threshold": self.rag_experiment.similarity_threshold,
                "max_boost_iterations": self.rag_experiment.max_boost_iterations,
                "confidence_threshold": self.rag_experiment.confidence_threshold
            },
            "pipeline": {
                "experiment_name": self.pipeline.experiment_name,
                "output_dir": str(self.pipeline.output_dir),
                "run_question_generation": self.pipeline.run_question_generation,
                "run_standard_rag": self.pipeline.run_standard_rag,
                "run_boost_rag": self.pipeline.run_boost_rag,
                "run_comparison_analysis": self.pipeline.run_comparison_analysis,
                "target_metrics": self.pipeline.target_metrics
            }
        }

# 사전 정의된 실험 설정들
PRESET_CONFIGS = {
    "quick_test": BRAGConfig(
        question_generation=QuestionGenerationConfig(
            questions_per_level=1,
            total_levels=3,
            max_retries=1
        ),
        rag_experiment=RAGExperimentConfig(
            top_k=2,
            max_boost_iterations=2
        ),
        pipeline=ExperimentPipeline(
            experiment_name="quick_test",
            gt_questions=["동업자가 채권의 준점유자에 해당하지 아니 한다고 할 수 있는가?"]
        )
    ),
    
    "full_experiment": BRAGConfig(
        question_generation=QuestionGenerationConfig(
            questions_per_level=1,
            total_levels=10,
            max_retries=3
        ),
        rag_experiment=RAGExperimentConfig(
            top_k=3,
            max_boost_iterations=3
        ),
        pipeline=ExperimentPipeline(
            experiment_name="full_b_rag_experiment",
            run_question_generation=True,
            run_standard_rag=True,
            run_boost_rag=True,
            run_comparison_analysis=True
        )
    ),
    
    "performance_test": BRAGConfig(
        question_generation=QuestionGenerationConfig(
            questions_per_level=2,
            total_levels=10,
            max_retries=2
        ),
        rag_experiment=RAGExperimentConfig(
            top_k=5,
            max_boost_iterations=5,
            confidence_threshold=0.95
        ),
        pipeline=ExperimentPipeline(
            experiment_name="performance_test",
            target_metrics={
                "question_consistency_rate": 0.9,
                "boost_rag_improvement": 0.15,
                "yes_answer_increase": 3
            }
        )
    )
}

def get_config(preset: str = "full_experiment") -> BRAGConfig:
    """사전 정의된 설정 가져오기"""
    if preset in PRESET_CONFIGS:
        return PRESET_CONFIGS[preset]
    else:
        print(f"⚠️ 알 수 없는 preset: {preset}. 기본 설정을 사용합니다.")
        return PRESET_CONFIGS["full_experiment"]

def create_custom_config(**kwargs) -> BRAGConfig:
    """커스텀 설정 생성"""
    base_config = PRESET_CONFIGS["full_experiment"]
    
    # kwargs로 전달된 설정 업데이트
    for key, value in kwargs.items():
        if hasattr(base_config, key):
            setattr(base_config, key, value)
        else:
            print(f"⚠️ 알 수 없는 설정 키: {key}")
    
    return base_config

# 사용 예시
if __name__ == "__main__":
    # 기본 설정
    config = get_config("full_experiment")
    print("📋 기본 설정:")
    print(f"  실험명: {config.pipeline.experiment_name}")
    print(f"  질문 생성 레벨: {config.question_generation.total_levels}")
    print(f"  RAG Top-K: {config.rag_experiment.top_k}")
    
    # 빠른 테스트 설정
    quick_config = get_config("quick_test")
    print(f"\n🚀 빠른 테스트 설정:")
    print(f"  실험명: {quick_config.pipeline.experiment_name}")
    print(f"  질문 생성 레벨: {quick_config.question_generation.total_levels}")
    
    # 커스텀 설정
    custom_config = create_custom_config(
        experiment_name="my_custom_experiment",
        top_k=4
    )
    print(f"\n🔧 커스텀 설정:")
    print(f"  실험명: {custom_config.pipeline.experiment_name}")
    print(f"  RAG Top-K: {custom_config.rag_experiment.top_k}") 