"""
통합된 Yes/No 질문 생성기
10개 레벨을 하나의 프롬프트로 처리하고 Langchain structured output 사용
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_upstage import ChatUpstage
from dotenv import load_dotenv

from .schemas.yesno_question_schemas import (
    TenLevelYesNoQuestions, 
    LevelQuestion, 
    YesNoAnswer,
    SAMPLE_TARGET_AUDIENCES
)

# 환경 변수 로드
load_dotenv()

class UnifiedYesNoQuestionGenerator:
    """통합된 Yes/No 질문 생성기"""
    
    def __init__(self, model_name: str = "solar-1-mini-chat"):
        """
        초기화
        
        Args:
            model_name: 사용할 Upstage 모델명
        """
        self.llm = ChatUpstage(model=model_name)
        self.structured_llm = self.llm.with_structured_output(TenLevelYesNoQuestions)
        self.prompt_template = self._load_unified_prompt()
        
        print(f"✅ UnifiedYesNoQuestionGenerator 초기화 완료 (모델: {model_name})")
    
    def _load_unified_prompt(self) -> ChatPromptTemplate:
        """통합 프롬프트 로드"""
        try:
            prompt_path = Path(__file__).parent / "prompts" / "minu" / "unified_yesno_question_generator.txt"
            
            with open(prompt_path, 'r', encoding='utf-8') as f:
                system_prompt = f.read()
            
            # Few-shot 예시 추가
            few_shot_examples = self._get_few_shot_examples()
            
            full_system_prompt = f"{system_prompt}\n\n{few_shot_examples}"
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", full_system_prompt),
                ("human", """
                GT 질문: {gt_question}
                
                판결문 내용:
                {document_content}
                
                참고 키워드: {keywords_to_consider}
                
                위 정보를 바탕으로 GT 질문과 의미론적으로 동일한 10개 레벨의 Yes/No 질문을 생성해주세요.
                """)
            ])
            
            return prompt
            
        except FileNotFoundError:
            print("⚠️ 통합 프롬프트 파일을 찾을 수 없습니다. 기본 프롬프트를 사용합니다.")
            return self._create_default_prompt()
    
    def _get_few_shot_examples(self) -> str:
        """Few-shot 예시 생성"""
        return """
**예시:**

GT 질문: "동업자가 채권의 준점유자에 해당하지 아니 한다고 할 수 있는가?"

예상 출력:
```json
{
  "gt_question": "동업자가 채권의 준점유자에 해당하지 아니 한다고 할 수 있는가?",
  "document_summary": "동업자와 채권의 준점유자 개념에 관한 민법 제470조 관련 판례",
  "questions": [
    {
      "level": 1,
      "question": "민법 제470조에 따라 동업자가 채권의 준점유자로 인정되지 않는가?",
      "target_audience": "전문가 및 전공자",
      "reasoning": "법조문을 직접 인용하여 전문적 법리 해석을 요구",
      "expected_answer": "Yes",
      "confidence": 0.95
    },
    {
      "level": 5,
      "question": "동업 관계에 있는 사람이 채권의 준점유자(실제 권리자처럼 보이는 사람)가 될 수 있나요?",
      "target_audience": "법학과 학부생 및 법무 실무진",
      "reasoning": "법률 용어에 쉬운 설명을 병기하여 이해를 도움",
      "expected_answer": "No",
      "confidence": 0.90
    },
    {
      "level": 10,
      "question": "같이 일하는 사람이 돈 받을 수 있어요?",
      "target_audience": "언어 발달 단계의 어린이 및 특수 교육",
      "reasoning": "가장 기본적인 단어로 핵심 개념만 전달",
      "expected_answer": "No",
      "confidence": 0.80
    }
  ],
  "semantic_consistency": "모든 질문은 동업자가 채권의 준점유자가 될 수 없다는 동일한 법리를 다루고 있으며, 각 레벨에 맞는 언어 수준으로 표현되었습니다.",
  "generation_metadata": {
    "total_questions": 10,
    "difficulty_range": "1-10",
    "question_type": "Yes/No",
    "semantic_equivalence": true
  }
}
```
"""
    
    def _create_default_prompt(self) -> ChatPromptTemplate:
        """기본 프롬프트 생성"""
        system_prompt = """
        당신은 법률 질문 생성 전문가입니다. 
        주어진 GT 질문과 판결문을 바탕으로 의미론적으로 동일하지만 
        난이도가 다른 10개의 Yes/No 질문을 생성해야 합니다.
        
        모든 질문은 Yes 또는 No로만 답변 가능해야 하며,
        GT 질문과 의미론적으로 완전히 동일해야 합니다.
        """
        
        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "GT 질문: {gt_question}\n판결문: {document_content}\n키워드: {keywords_to_consider}")
        ])
    
    def generate_ten_level_questions(
        self, 
        gt_question: str, 
        document_content: str, 
        keywords_to_consider: Optional[str] = None
    ) -> TenLevelYesNoQuestions:
        """
        10개 레벨의 Yes/No 질문 생성
        
        Args:
            gt_question: 원본 GT 질문
            document_content: 판결문 전체 내용
            keywords_to_consider: 참고할 키워드들
            
        Returns:
            TenLevelYesNoQuestions: 생성된 10개 레벨 질문들
        """
        try:
            print(f"🔄 10개 레벨 Yes/No 질문 생성 시작...")
            print(f"📝 GT 질문: {gt_question[:50]}...")
            
            # 키워드가 없으면 기본값 설정
            if not keywords_to_consider:
                keywords_to_consider = "법률 용어, 판례, 법리"
            
            # 프롬프트 체인 구성
            chain = self.prompt_template | self.structured_llm
            
            # 질문 생성 실행
            result = chain.invoke({
                "gt_question": gt_question,
                "document_content": document_content,
                "keywords_to_consider": keywords_to_consider
            })
            
            # 결과 검증
            if not self._validate_result(result):
                print("⚠️ 생성된 결과가 유효하지 않습니다. 재시도합니다.")
                return self._retry_generation(gt_question, document_content, keywords_to_consider)
            
            print(f"✅ 10개 레벨 질문 생성 완료")
            print(f"📊 일관성 비율: {result.get_consistency_rate():.2%}")
            
            return result
            
        except Exception as e:
            print(f"❌ 질문 생성 중 오류 발생: {e}")
            return self._create_fallback_questions(gt_question, document_content)
    
    def _validate_result(self, result: TenLevelYesNoQuestions) -> bool:
        """생성된 결과 검증"""
        try:
            # 기본 검증
            if not result.validate_levels():
                print("❌ 레벨 1-10이 모두 포함되지 않았습니다.")
                return False
            
            # Yes/No 질문 형식 검증
            for question in result.questions:
                if not self._is_yesno_question(question.question):
                    print(f"❌ Level {question.level} 질문이 Yes/No 형식이 아닙니다: {question.question}")
                    return False
            
            # 의미론적 일관성 검증 (간단한 키워드 기반)
            consistency_rate = result.get_consistency_rate()
            if consistency_rate < 0.8:  # 80% 이상 일관성 요구
                print(f"❌ 답변 일관성이 낮습니다: {consistency_rate:.2%}")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ 결과 검증 중 오류: {e}")
            return False
    
    def _is_yesno_question(self, question: str) -> bool:
        """Yes/No 질문인지 확인"""
        # 간단한 휴리스틱 검증
        yesno_indicators = [
            "인가", "인지", "할 수 있", "가능한", "맞는", "옳은", "틀린",
            "해당하", "포함되", "적용되", "인정되", "성립하", "유효한"
        ]
        
        question_lower = question.lower()
        return any(indicator in question_lower for indicator in yesno_indicators) or question.endswith("?")
    
    def _retry_generation(
        self, 
        gt_question: str, 
        document_content: str, 
        keywords_to_consider: str,
        max_retries: int = 2
    ) -> TenLevelYesNoQuestions:
        """재시도 로직"""
        for attempt in range(max_retries):
            try:
                print(f"🔄 재시도 {attempt + 1}/{max_retries}")
                
                # 프롬프트에 추가 지침 포함
                enhanced_prompt = self.prompt_template.partial(
                    additional_instruction="이전 시도에서 실패했습니다. 더욱 명확한 Yes/No 질문을 생성해주세요."
                )
                
                chain = enhanced_prompt | self.structured_llm
                result = chain.invoke({
                    "gt_question": gt_question,
                    "document_content": document_content,
                    "keywords_to_consider": keywords_to_consider
                })
                
                if self._validate_result(result):
                    print(f"✅ 재시도 {attempt + 1}에서 성공")
                    return result
                    
            except Exception as e:
                print(f"❌ 재시도 {attempt + 1} 실패: {e}")
        
        print("❌ 모든 재시도 실패. Fallback 질문을 생성합니다.")
        return self._create_fallback_questions(gt_question, document_content)
    
    def _create_fallback_questions(
        self, 
        gt_question: str, 
        document_content: str
    ) -> TenLevelYesNoQuestions:
        """Fallback 질문 생성"""
        print("🔧 Fallback 질문 생성 중...")
        
        # 기본 Yes/No 질문 템플릿
        base_question = f"{gt_question}에 대한 답변이 긍정적인가요?"
        
        fallback_questions = []
        for level in range(1, 11):
            audience = SAMPLE_TARGET_AUDIENCES[level].value
            
            # 레벨에 따라 질문 복잡도 조정
            if level <= 3:
                question = f"이 법률 문제에 대한 답변이 '예'인가요?"
            elif level <= 6:
                question = f"이 상황에서 법적으로 긍정적인 결과가 나오나요?"
            else:
                question = f"이것이 맞는 일인가요?"
            
            fallback_questions.append(LevelQuestion(
                level=level,
                question=question,
                target_audience=audience,
                reasoning=f"Fallback 질문 - Level {level}",
                expected_answer=YesNoAnswer.YES,
                confidence=0.5
            ))
        
        return TenLevelYesNoQuestions(
            gt_question=gt_question,
            document_summary="Fallback 모드로 생성된 요약",
            questions=fallback_questions,
            semantic_consistency="Fallback 모드에서 생성된 기본 질문들",
            generation_metadata={
                "total_questions": 10,
                "difficulty_range": "1-10",
                "question_type": "Yes/No",
                "semantic_equivalence": False  # Fallback이므로 False
            }
        )
    
    def save_questions_to_file(
        self, 
        questions: TenLevelYesNoQuestions, 
        output_path: str
    ) -> None:
        """생성된 질문들을 파일로 저장"""
        try:
            output_data = {
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
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            print(f"💾 질문들이 저장되었습니다: {output_path}")
            
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
    
    # 질문 생성기 초기화
    generator = UnifiedYesNoQuestionGenerator()
    
    # 질문 생성
    result = generator.generate_ten_level_questions(
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
    
    # 파일 저장
    generator.save_questions_to_file(result, "sample_yesno_questions.json") 