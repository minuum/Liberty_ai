"""
진정한 자연스러운 분포 Yes/No 질문 생성기
하드코딩 없이 LLM이 GT 질문의 의미를 파악하여 자율적으로 생성
"""

import json
import time
from typing import Optional, List
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

try:
    from liberty_agent.b_rag.core.schemas.yesno_question_schemas import (
        TenLevelYesNoQuestions, LevelQuestion, YesNoAnswer, GenerationMetadata
    )
except ImportError:
    from core.schemas.yesno_question_schemas import (
        TenLevelYesNoQuestions, LevelQuestion, YesNoAnswer, GenerationMetadata
    )


class GTAnalysis(BaseModel):
    """GT 질문 분석 결과"""
    core_meaning: str = Field(description="GT 질문의 핵심 의미")
    expected_answer: str = Field(description="Yes 또는 No")
    legal_reasoning: str = Field(description="법적 근거와 논리")


class QuestionData(BaseModel):
    """개별 질문 데이터"""
    level: int = Field(description="질문 레벨 (1-10)")
    question: str = Field(description="생성된 질문")
    target_audience: str = Field(description="대상 독자")
    approach: str = Field(description="접근 방식")
    expected_answer: str = Field(description="Yes 또는 No")
    confidence: float = Field(description="확신도 (0.0-1.0)")
    reasoning: str = Field(description="생성 이유")


class GenerationSummary(BaseModel):
    """생성 요약"""
    total_questions: int = Field(description="총 질문 수")
    yes_count: int = Field(description="Yes 답변 개수")
    no_count: int = Field(description="No 답변 개수")
    natural_distribution: str = Field(description="자연스러운 분포 설명")
    semantic_consistency: str = Field(description="의미적 일관성 확인")


class TrulyNaturalResponse(BaseModel):
    """진정한 자연스러운 응답 구조"""
    gt_analysis: GTAnalysis
    questions: List[QuestionData]
    generation_summary: GenerationSummary


class TrulyNaturalQuestionGenerator:
    """진정한 자연스러운 분포 질문 생성기"""
    
    def __init__(
        self, 
        model_name: str = "gpt-4o-2024-08-06", 
        temperature: float = 0.1,
        openai_api_key: Optional[str] = None
    ):
        self.model_name = model_name
        self.temperature = temperature
        
        # LLM 초기화
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=openai_api_key
        )
        
        # 구조화된 출력을 위한 LLM
        self.structured_llm = self.llm.with_structured_output(TrulyNaturalResponse)
        
        # 프롬프트 로드
        self.prompt_template = self._load_prompt()
        
        print(f"✅ TrulyNaturalQuestionGenerator 초기화 완료")
        print(f"   모델: {model_name}")
        print(f"   온도: {temperature}")
    
    def _load_prompt(self) -> ChatPromptTemplate:
        """진정한 자연스러운 분포 프롬프트 로드"""
        try:
            prompt_path = Path(__file__).parent / "prompts" / "minu" / "truly_natural_yesno_generator.txt"
            
            if not prompt_path.exists():
                print(f"⚠️ 프롬프트 파일을 찾을 수 없습니다: {prompt_path}")
                return self._create_default_prompt()
            
            with open(prompt_path, 'r', encoding='utf-8') as f:
                prompt_content = f.read()
            
            system_prompt = prompt_content
            human_prompt = """
GT 질문: {gt_question}

판결문 내용:
{document_content}

고려할 키워드: {keywords_to_consider}

위 정보를 바탕으로 진정한 자연스러운 분포의 10개 레벨 Yes/No 질문을 생성해주세요.
분포 조작 없이, GT 질문의 의미에 따라 자연스럽게 생성하세요.
"""
            
            return ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", human_prompt)
            ])
            
        except Exception as e:
            print(f"❌ 프롬프트 로드 실패: {e}")
            return self._create_default_prompt()
    
    def _create_default_prompt(self) -> ChatPromptTemplate:
        """기본 프롬프트 생성"""
        system_prompt = """
당신은 법률 교육 전문가입니다. 주어진 GT 질문과 판결문을 바탕으로 의미적으로 동등한 10개 레벨의 Yes/No 질문을 생성해주세요.

핵심 원칙:
1. 의미적 동등성: 모든 질문은 GT 질문과 동일한 법적 결론을 도출해야 함
2. 자연스러운 생성: 분포를 인위적으로 조작하지 말고, 각 레벨별로 독립적으로 생성
3. 레벨별 특성화: 1-2(Expert), 3-4(Advanced), 5-6(Intermediate), 7-8(Basic), 9-10(Foundation)

중요: 결과 분포가 10:0, 9:1, 8:2 등 어떤 형태든 자연스러운 결과로 수용하세요.
"""
        
        human_prompt = """
GT 질문: {gt_question}
판결문: {document_content}
키워드: {keywords_to_consider}

진정한 자연스러운 분포의 질문들을 생성해주세요.
"""
        
        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])
    
    def generate_truly_natural_questions(
        self, 
        gt_question: str, 
        document_content: str = "", 
        keywords_to_consider: Optional[str] = None
    ) -> TenLevelYesNoQuestions:
        """진정한 자연스러운 분포 질문 생성"""
        print(f"\n🌱 진정한 자연스러운 분포 질문 생성 시작...")
        print(f"GT 질문: {gt_question}")
        
        start_time = time.time()
        
        try:
            # LLM 체인 구성
            chain = self.prompt_template | self.structured_llm
            
            # 질문 생성
            response = chain.invoke({
                "gt_question": gt_question,
                "document_content": document_content or "판결문 내용이 제공되지 않았습니다.",
                "keywords_to_consider": keywords_to_consider or "법률 용어, 판례, 법리"
            })
            
            # 응답을 TenLevelYesNoQuestions 형태로 변환
            questions = []
            for q_data in response.questions:
                questions.append(LevelQuestion(
                    level=q_data.level,
                    question=q_data.question,
                    target_audience=q_data.target_audience,
                    reasoning=q_data.reasoning,
                    expected_answer=YesNoAnswer.YES if q_data.expected_answer.upper() == "YES" else YesNoAnswer.NO,
                    confidence=q_data.confidence
                ))
            
            result = TenLevelYesNoQuestions(
                gt_question=gt_question,
                document_summary=f"진정한 자연스러운 분포 생성 - {response.gt_analysis.core_meaning}",
                questions=questions,
                semantic_consistency=response.generation_summary.semantic_consistency,
                generation_metadata=GenerationMetadata(
                    total_questions=response.generation_summary.total_questions,
                    difficulty_range="1-10",
                    question_type="Yes/No",
                    semantic_equivalence=True
                )
            )
            
            generation_time = time.time() - start_time
            
            # 결과 분석
            yes_count = sum(1 for q in questions if q.expected_answer == YesNoAnswer.YES)
            no_count = len(questions) - yes_count
            consistency_rate = result.get_consistency_rate()
            
            print(f"\n✅ 진정한 자연스러운 분포 생성 완료!")
            print(f"   생성 시간: {generation_time:.2f}초")
            print(f"   분포: Yes {yes_count}개, No {no_count}개 ({yes_count}:{no_count})")
            print(f"   일관성: {consistency_rate:.1%}")
            print(f"   GT 분석: {response.gt_analysis.core_meaning}")
            print(f"   예상 답변: {response.gt_analysis.expected_answer}")
            
            # 분포 자연스러움 평가
            if yes_count == 10 or no_count == 10:
                print(f"   🎯 완전 일관성 (10:0) - GT 질문의 명확한 의미 반영")
            elif min(yes_count, no_count) == 1:
                print(f"   🌟 매우 자연스러운 분포 (9:1) - 높은 일관성과 최소 다양성")
            elif min(yes_count, no_count) == 2:
                print(f"   ✅ 자연스러운 분포 (8:2) - 적절한 일관성과 다양성")
            else:
                print(f"   📊 균형적 분포 - 복합적 의미 해석")
            
            return result
            
        except Exception as e:
            print(f"❌ 진정한 자연스러운 분포 생성 실패: {e}")
            raise e
    
    def validate_natural_result(self, result: TenLevelYesNoQuestions) -> dict:
        """자연스러운 결과 검증"""
        questions = result.questions
        
        # 기본 검증
        if len(questions) != 10:
            return {"valid": False, "reason": "질문 개수가 10개가 아닙니다."}
        
        levels = [q.level for q in questions]
        if set(levels) != set(range(1, 11)):
            return {"valid": False, "reason": "레벨 1-10이 모두 포함되지 않았습니다."}
        
        # 분포 분석
        yes_count = sum(1 for q in questions if q.expected_answer == YesNoAnswer.YES)
        no_count = len(questions) - yes_count
        consistency_rate = result.get_consistency_rate()
        
        # 자연스러움 평가
        if consistency_rate >= 0.9:
            naturalness = "매우 자연스러운"
        elif consistency_rate >= 0.8:
            naturalness = "자연스러운"
        elif consistency_rate >= 0.7:
            naturalness = "적절한"
        else:
            naturalness = "복합적 의미"
        
        return {
            "valid": True,
            "distribution": f"{yes_count}:{no_count}",
            "consistency_rate": consistency_rate,
            "naturalness": naturalness,
            "analysis": f"{naturalness} 분포 - GT 질문의 의미에 따른 자연스러운 결과"
        }
    
    def save_natural_questions(
        self, 
        questions: TenLevelYesNoQuestions, 
        output_path: str,
        include_analysis: bool = True
    ) -> None:
        """자연스러운 질문들을 파일로 저장"""
        try:
            # 검증 결과 포함
            validation = self.validate_natural_result(questions)
            
            output_data = {
                "generation_type": "truly_natural_distribution",
                "gt_question": questions.gt_question,
                "document_summary": questions.document_summary,
                "questions": [
                    {
                        "level": q.level,
                        "question": q.question,
                        "target_audience": q.target_audience,
                        "reasoning": q.reasoning,
                        "expected_answer": q.expected_answer.value,
                        "confidence": q.confidence
                    }
                    for q in questions.questions
                ],
                "semantic_consistency": questions.semantic_consistency,
                "generation_metadata": questions.generation_metadata.dict()
            }
            
            if include_analysis:
                output_data["natural_analysis"] = validation
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            print(f"💾 진정한 자연스러운 분포 질문들이 저장되었습니다: {output_path}")
            
        except Exception as e:
            print(f"❌ 파일 저장 중 오류: {e}")


# 사용 예시
if __name__ == "__main__":
    # 테스트용 샘플 데이터
    sample_gt_question = "동업자가 채권의 준점유자에 해당하지 아니 한다고 할 수 있는가?"
    sample_document = """
    대법원 1982. 11. 9. 선고 80다3135 판결
    
    동업자가 채권의 준점유자에 해당하지 아니한다고 할 수 있다.
    민법 제470조에 따르면 채권의 준점유자에게 변제한 경우에도 
    변제자가 선의이고 과실이 없으면 유효한 변제가 된다.
    그러나 단순한 동업관계만으로는 채권의 준점유자로 볼 수 없다.
    """
    
    # 진정한 자연스러운 분포 생성기 초기화
    generator = TrulyNaturalQuestionGenerator(
        model_name="gpt-4o-2024-08-06",
        temperature=0.1
    )
    
    # 질문 생성
    result = generator.generate_truly_natural_questions(
        gt_question=sample_gt_question,
        document_content=sample_document,
        keywords_to_consider="동업자, 채권의 준점유자, 민법 제470조, 변제"
    )
    
    # 결과 출력
    print(f"\n📋 생성된 질문들:")
    for q in result.questions:
        print(f"Level {q.level}: {q.question}")
        print(f"  대상: {q.target_audience}")
        print(f"  예상답변: {q.expected_answer.value}")
        print()
    
    # 검증 및 저장
    validation = generator.validate_natural_result(result)
    print(f"\n🔍 자연스러움 검증: {validation}")
    
    generator.save_natural_questions(result, "truly_natural_questions.json") 