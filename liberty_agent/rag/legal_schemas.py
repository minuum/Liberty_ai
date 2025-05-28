from typing import List, Optional, Literal
from pydantic import BaseModel, Field

# 난이도 타입 정의
DifficultyLevel = Literal["입문", "기초", "중급", "고급", "전문가"]

class DocumentAnalysis(BaseModel):
    """법률 문서 분석 결과 스키마"""
    key_points: List[str] = Field(..., description="문서의 핵심 내용 목록")
    legal_issues: List[str] = Field(..., description="주요 법적 쟁점 목록")
    keywords: List[str] = Field(..., description="주요 키워드 목록")
    document_type: str = Field(..., description="문서 유형(판례/법령/계약서 등)")
    complexity_level: DifficultyLevel = Field(..., description="""문서 난이도:
        입문: 법률 지식이 전혀 없는 일반인도 이해 가능
        기초: 기본적인 법률 용어와 개념 이해 필요
        중급: 관련 법령과 판례에 대한 기본 지식 필요
        고급: 심화된 법률 지식과 관련 판례 이해 필요
        전문가: 해당 분야의 전문적인 법률 지식 필요""")

class LegalQuestion(BaseModel):
    """법률 질문 스키마"""
    question: str = Field(..., description="생성된 법률 질문")
    reasoning: str = Field(..., description="출제 의도 및 학습 포인트")
    strategy: str = Field(...,
                        description="질문 전략",
                        enum=["사실관계 이해", "법리 해석", "판례 적용", "실무 적용", "종합 분석"])
    difficulty: DifficultyLevel = Field(...,
                          description="""난이도:
                          입문: 법률 지식 없이도 풀 수 있는 기본 개념 문제
                          기초: 기본적인 법률 용어와 개념 이해가 필요한 문제
                          중급: 관련 법령과 판례 지식이 필요한 문제
                          고급: 심화된 법률 지식과 판례 분석이 필요한 문제
                          전문가: 전문적 법률 지식과 실무 경험이 필요한 문제""")
    keywords: List[str] = Field(..., description="문제와 관련된 주요 키워드")

class QuestionSet(BaseModel):
    """질문 세트 스키마"""
    questions: List[LegalQuestion] = Field(..., description="생성된 질문 목록")

class QAExample(BaseModel):
    """QA 데이터셋 예시 스키마"""
    question: str = Field(..., description="생성된 법률 질문 또는 원본 질문")
    reference_answer: str = Field(..., description="질문에 대한 기준 답변")
    difficulty: DifficultyLevel # Literal 타입 직접 사용
    strategy: str # Literal 타입으로 변경 가능: Literal["사실관계 이해", "법리 해석", "판례 적용", "실무 적용", "종합 분석"]
    keywords: List[str] = Field(..., description="질문과 관련된 키워드 목록")

class RAGInput(BaseModel):
    """RAG 시스템 입력 스키마"""
    query: str
    policy_level: Optional[int] = None
    original_difficulty: Optional[DifficultyLevel] = None

class RAGOutput(BaseModel):
    """RAG 시스템 출력 및 평가 결과 스키마"""
    input_query: str
    retrieved_contexts: List[str]
    generated_answer: str
    reference_answer: str
    cosine_similarity_score: Optional[float] = None
    l2_distances: Optional[List[float]] = None
    policy_level: Optional[int] = None
    original_difficulty: Optional[DifficultyLevel] = None 