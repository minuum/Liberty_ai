"""
Yes/No 질문 생성을 위한 Pydantic 스키마
Langchain structured output과 호환
"""

from pydantic import BaseModel, Field
from typing import List, Literal
from enum import Enum

class TargetAudience(str, Enum):
    """대상 독자 유형"""
    EXPERT = "전문가 및 전공자"
    INTERMEDIATE = "대학생 및 직장인 중급자"
    BEGINNER = "일반 성인 및 초심자"
    NON_EXPERT = "학생 및 비전문가"
    LAW_STUDENT = "법학과 학부생 및 법무 실무진"
    HIGH_SCHOOL = "고등학생 및 일반 직장인"
    MIDDLE_SCHOOL = "중학생 및 법률 초심자"
    ELEMENTARY = "초등학생 및 완전 초심자"
    KINDERGARTEN = "유치원생 및 법률 지식이 전혀 없는 사람"
    SPECIAL_EDUCATION = "언어 발달 단계의 어린이 및 특수 교육"

class YesNoAnswer(str, Enum):
    """Yes/No 답변 타입"""
    YES = "Yes"
    NO = "No"

class LevelQuestion(BaseModel):
    """특정 레벨의 Yes/No 질문"""
    level: int = Field(description="질문의 난이도 레벨 (1-10)", ge=1, le=10)
    question: str = Field(description="Yes 또는 No로 답변 가능한 질문")
    target_audience: str = Field(description="이 레벨의 대상 독자")
    reasoning: str = Field(description="이 레벨에서 이 질문을 생성한 이유")
    expected_answer: YesNoAnswer = Field(description="예상되는 답변 (Yes 또는 No)")
    confidence: float = Field(description="답변에 대한 확신도", ge=0.0, le=1.0)

class GenerationMetadata(BaseModel):
    """질문 생성 메타데이터"""
    total_questions: int = Field(description="생성된 총 질문 수", default=10)
    difficulty_range: str = Field(description="난이도 범위", default="1-10")
    question_type: str = Field(description="질문 유형", default="Yes/No")
    semantic_equivalence: bool = Field(description="의미론적 동등성 여부", default=True)

class TenLevelYesNoQuestions(BaseModel):
    """GT_Q와 의미론적으로 동일한 10개 레벨의 Yes/No 질문들"""
    gt_question: str = Field(description="원본 GT 질문")
    document_summary: str = Field(description="판결문의 핵심 내용 요약")
    questions: List[LevelQuestion] = Field(
        description="레벨 1-10의 Yes/No 질문들", 
        min_items=10, 
        max_items=10
    )
    semantic_consistency: str = Field(
        description="모든 질문이 GT_Q와 의미적으로 동일함을 확인하는 설명"
    )
    generation_metadata: GenerationMetadata = Field(
        description="질문 생성 관련 메타데이터"
    )

    def validate_levels(self) -> bool:
        """레벨 1-10이 모두 포함되어 있는지 확인"""
        levels = [q.level for q in self.questions]
        return set(levels) == set(range(1, 11))

    def get_question_by_level(self, level: int) -> LevelQuestion:
        """특정 레벨의 질문 반환"""
        for question in self.questions:
            if question.level == level:
                return question
        raise ValueError(f"Level {level} question not found")

    def get_positive_questions(self) -> List[LevelQuestion]:
        """Yes 답변이 예상되는 질문들 반환"""
        return [q for q in self.questions if q.expected_answer == YesNoAnswer.YES]

    def get_negative_questions(self) -> List[LevelQuestion]:
        """No 답변이 예상되는 질문들 반환"""
        return [q for q in self.questions if q.expected_answer == YesNoAnswer.NO]

    def get_consistency_rate(self) -> float:
        """답변 일관성 비율 계산 (모든 질문이 같은 답변을 가져야 함)"""
        if not self.questions:
            return 0.0
        
        first_answer = self.questions[0].expected_answer
        consistent_count = sum(1 for q in self.questions if q.expected_answer == first_answer)
        return consistent_count / len(self.questions)

class RAGAnswer(BaseModel):
    """RAG 시스템의 Yes/No 답변"""
    answer: YesNoAnswer = Field(description="Yes 또는 No 답변")
    confidence: float = Field(description="답변에 대한 확신도 (0.0-1.0)", ge=0.0, le=1.0)
    reasoning: str = Field(description="답변의 근거가 되는 판결문 내용")
    retrieved_context: str = Field(description="검색된 관련 문서 내용")

class ExperimentResult(BaseModel):
    """실험 결과"""
    gt_question: str = Field(description="원본 GT 질문")
    generated_questions: TenLevelYesNoQuestions = Field(description="생성된 10개 레벨 질문들")
    rag_results: List[RAGAnswer] = Field(description="각 질문에 대한 RAG 답변들")
    positive_count: int = Field(description="Yes 답변 개수")
    positive_rate: float = Field(description="Yes 답변 비율", ge=0.0, le=1.0)
    level_distribution: dict = Field(description="레벨별 Yes 답변 분포")
    experiment_type: Literal["standard", "boost"] = Field(description="실험 유형")

class BoostExperimentResult(BaseModel):
    """Boost RAG 실험 결과"""
    gt_question: str = Field(description="원본 GT 질문")
    best_result: ExperimentResult = Field(description="가장 좋은 결과")
    all_iterations: List[ExperimentResult] = Field(description="모든 반복 실험 결과")
    improvement: int = Field(description="개선된 Yes 답변 개수")
    best_iteration: int = Field(description="최고 성능을 보인 반복 횟수")
    experiment_type: Literal["boost"] = Field(description="실험 유형", default="boost")

# 사용 예시를 위한 샘플 데이터
SAMPLE_TARGET_AUDIENCES = {
    1: TargetAudience.EXPERT,
    2: TargetAudience.INTERMEDIATE,
    3: TargetAudience.BEGINNER,
    4: TargetAudience.NON_EXPERT,
    5: TargetAudience.LAW_STUDENT,
    6: TargetAudience.HIGH_SCHOOL,
    7: TargetAudience.MIDDLE_SCHOOL,
    8: TargetAudience.ELEMENTARY,
    9: TargetAudience.KINDERGARTEN,
    10: TargetAudience.SPECIAL_EDUCATION
}

def create_sample_question(level: int, question_text: str, expected_answer: YesNoAnswer) -> LevelQuestion:
    """샘플 질문 생성 헬퍼 함수"""
    return LevelQuestion(
        level=level,
        question=question_text,
        target_audience=SAMPLE_TARGET_AUDIENCES[level].value,
        reasoning=f"Level {level}에 적합한 난이도와 언어 수준으로 구성",
        expected_answer=expected_answer,
        confidence=0.9
    ) 