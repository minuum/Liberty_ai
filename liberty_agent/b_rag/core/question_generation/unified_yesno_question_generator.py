"""
통합된 Yes/No 질문 생성기
10개 레벨을 하나의 프롬프트로 처리하고 Langchain structured output 사용
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from core.schemas.yesno_question_schemas import (
    TenLevelYesNoQuestions, 
    LevelQuestion, 
    YesNoAnswer,
    SAMPLE_TARGET_AUDIENCES
)

# 환경 변수 로드
load_dotenv()

class UnifiedYesNoQuestionGenerator:
    """통합된 Yes/No 질문 생성기"""
    
    def __init__(self, 
                 model_name: str = "gpt-4o-2024-08-06", 
                 temperature: float = 0.1,
                 openai_api_key: Optional[str] = None):
        """
        초기화
        
        Args:
            model_name: 사용할 OpenAI 모델명
            temperature: 모델 온도 설정
            openai_api_key: OpenAI API 키 (환경변수에서 자동 로드)
        """
        self.llm = ChatOpenAI(
            model=model_name, 
            temperature=temperature, 
            api_key=openai_api_key or os.getenv("OPENAI_API_KEY")
        )
        # function calling 방식으로 structured output 사용
        try:
            self.structured_llm = self.llm.with_structured_output(
                TenLevelYesNoQuestions, 
                method="function_calling", 
                include_raw=False
            )
        except Exception as e:
            print(f"⚠️ Structured output 설정 실패: {e}")
            print("기본 LLM을 사용하고 JSON 파싱을 시도합니다.")
            self.structured_llm = self.llm
        self.prompt_template = self._load_unified_prompt()
        
        print(f"✅ UnifiedYesNoQuestionGenerator 초기화 완료 (모델: {model_name}, 온도: {temperature})")
    
    def _load_unified_prompt(self) -> ChatPromptTemplate:
        """통합 프롬프트 로드"""
        try:
            # 개선된 프롬프트 파일 우선 시도
            prompt_path = Path(__file__).parent / "prompts" / "minu" / "unified_yesno_question_generator_v2.txt"
            
            if not prompt_path.exists():
                # 기존 프롬프트 파일 사용
                prompt_path = Path(__file__).parent / "prompts" / "minu" / "unified_yesno_question_generator.txt"
            
            with open(prompt_path, 'r', encoding='utf-8') as f:
                system_prompt = f.read()
                
            print(f"✅ 프롬프트 로드 완료: {prompt_path.name}")
            
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

예상 출력 형식:
- Level 1 (전문가): "민법 제470조에 따라 동업자가 채권의 준점유자로 인정되지 않는가?"
- Level 5 (중급): "동업 관계에 있는 사람이 채권의 준점유자가 될 수 있나요?"  
- Level 10 (초급): "같이 일하는 사람이 돈 받을 수 있어요?"

모든 질문의 예상 답변은 동일해야 하며, 각 레벨에 맞는 언어 수준으로 표현되어야 합니다.
"""
    
    def _create_default_prompt(self) -> ChatPromptTemplate:
        """기본 프롬프트 생성"""
        system_prompt = """
당신은 법률 질문 생성 전문가입니다. 주어진 GT 질문과 판결문을 바탕으로 의미론적으로 동일하지만 난이도가 다른 10개 레벨의 Yes/No 질문을 생성해야 합니다.

핵심 원칙:
1. 모든 질문은 GT 질문과 정확히 같은 의미를 가져야 합니다
2. 모든 질문은 Yes 또는 No로만 답변 가능해야 합니다
3. 레벨 1(전문가)부터 레벨 10(특수교육)까지 언어 수준을 조정합니다
4. 모든 질문의 예상 답변은 동일해야 합니다

레벨별 가이드:
- Level 1-3: 법조문 인용, 전문 용어 사용
- Level 4-6: 법률 용어에 간단한 설명 병기
- Level 7-10: 일상 언어로 핵심 개념만 전달

반드시 JSON 형식으로 응답하세요.
        """
        
        human_prompt = """
GT 질문: {gt_question}

판결문 내용:
{document_content}

참고 키워드: {keywords_to_consider}

위 정보를 바탕으로 GT 질문과 의미론적으로 동일한 10개 레벨의 Yes/No 질문을 JSON 형식으로 생성해주세요.
        """
        
        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
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
            response = chain.invoke({
                "gt_question": gt_question,
                "document_content": document_content,
                "keywords_to_consider": keywords_to_consider
            })
            
            # Structured output이 실패한 경우 JSON 파싱 시도
            if isinstance(response, str):
                result = self._parse_json_response(response, gt_question, document_content)
            else:
                result = response
            
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
    
    def _parse_json_response(self, response: str, gt_question: str, document_content: str) -> TenLevelYesNoQuestions:
        """JSON 응답 파싱"""
        try:
            import json
            import re
            
            # JSON 블록 추출
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # JSON 블록이 없으면 전체 응답에서 JSON 찾기
                json_str = response
            
            # JSON 파싱
            data = json.loads(json_str)
            
            # TenLevelYesNoQuestions 객체 생성
            from core.schemas.yesno_question_schemas import GenerationMetadata
            
            questions = []
            for q_data in data.get("questions", []):
                questions.append(LevelQuestion(
                    level=q_data["level"],
                    question=q_data["question"],
                    target_audience=q_data["target_audience"],
                    reasoning=q_data["reasoning"],
                    expected_answer=YesNoAnswer(q_data["expected_answer"]),
                    confidence=q_data["confidence"]
                ))
            
            metadata = data.get("generation_metadata", {})
            return TenLevelYesNoQuestions(
                gt_question=data.get("gt_question", gt_question),
                document_summary=data.get("document_summary", "AI 생성 요약"),
                questions=questions,
                semantic_consistency=data.get("semantic_consistency", "AI 생성 일관성 설명"),
                generation_metadata=GenerationMetadata(
                    total_questions=metadata.get("total_questions", len(questions)),
                    difficulty_range=metadata.get("difficulty_range", "1-10"),
                    question_type=metadata.get("question_type", "Yes/No"),
                    semantic_equivalence=metadata.get("semantic_equivalence", True)
                )
            )
            
        except Exception as e:
            print(f"❌ JSON 파싱 실패: {e}")
            print(f"응답 내용: {response[:200]}...")
            return self._create_fallback_questions(gt_question, document_content)
    
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
        
        from core.schemas.yesno_question_schemas import GenerationMetadata
        
        return TenLevelYesNoQuestions(
            gt_question=gt_question,
            document_summary="Fallback 모드로 생성된 요약",
            questions=fallback_questions,
            semantic_consistency="Fallback 모드에서 생성된 기본 질문들",
            generation_metadata=GenerationMetadata(
                total_questions=10,
                difficulty_range="1-10",
                question_type="Yes/No",
                semantic_equivalence=False  # Fallback이므로 False
            )
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
    
    # 질문 생성기 초기화 (OpenAI 모델 사용)
    generator = UnifiedYesNoQuestionGenerator(
        model_name="gpt-4o-2024-08-06",
        temperature=0.1
    )
    
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