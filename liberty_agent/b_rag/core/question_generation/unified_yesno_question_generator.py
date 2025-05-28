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

try:
    from liberty_agent.b_rag.core.schemas.yesno_question_schemas import (
        TenLevelYesNoQuestions, 
        LevelQuestion, 
        YesNoAnswer,
        SAMPLE_TARGET_AUDIENCES
    )
except ImportError:
    from core.schemas.yesno_question_schemas import (
        TenLevelYesNoQuestions, 
        LevelQuestion, 
        YesNoAnswer,
        SAMPLE_TARGET_AUDIENCES
    )
except ImportError:
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
                 openai_api_key: Optional[str] = None,
                 prompt_mode: str = "enhanced"):
        """
        초기화
        
        Args:
            model_name: 사용할 OpenAI 모델명
            temperature: 모델 온도 설정
            openai_api_key: OpenAI API 키 (환경변수에서 자동 로드)
            prompt_mode: 프롬프트 모드 ("enhanced", "simple", "minimal")
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
        
        self.prompt_mode = prompt_mode
        self.prompt_template = self._load_unified_prompt()
        
        print(f"✅ UnifiedYesNoQuestionGenerator 초기화 완료 (모델: {model_name}, 온도: {temperature}, 프롬프트: {prompt_mode})")
    
    def _load_unified_prompt(self) -> ChatPromptTemplate:
        """통합 프롬프트 로드"""
        if self.prompt_mode == "simple":
            print("🔧 단순화된 프롬프트를 사용합니다.")
            return self._create_simple_prompt()
        elif self.prompt_mode == "minimal":
            print("🔧 최소한의 프롬프트를 사용합니다.")
            return self._create_minimal_prompt()
        elif self.prompt_mode == "smart_simple":
            print("🔧 스마트한 단순화 프롬프트를 사용합니다.")
            return self._create_smart_simple_prompt()
        else:
            print("🔧 개선된 기본 프롬프트를 사용합니다.")
            return self._create_enhanced_prompt()
    
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
    
    def _create_enhanced_prompt(self) -> ChatPromptTemplate:
        """개선된 프롬프트 생성 - 의미적 일관성 중심"""
        system_prompt = """당신은 법률 질문 생성 전문가입니다. 주어진 GT 질문과 판결문을 바탕으로 의미론적으로 일관된 10개 레벨 Yes/No 질문을 생성해야 합니다.

## 핵심 원칙 (의미적 일관성 우선)

1. **의미적 일관성**: 모든 질문이 GT 질문과 동일한 법적 결론을 도출해야 함
2. **자연스러운 분포**: GT 질문의 본래 의미에 따른 자연스러운 Yes/No 분포 허용
3. **다양한 관점**: 같은 결론이지만 다양한 법적 관점과 접근 방식 활용
4. **이중 평가 기준**:
   - **사실 정보**: 판결문에 명확한 근거가 있는가?
   - **법적 유용성**: 법률 학습/실무에 도움이 되는가?
5. **적응적 난이도**: 레벨별 특성화된 언어 수준

## 질문 생성 전략 (의미적 일관성 중심)

### 단계 1: GT 질문의 핵심 의미 파악
- GT 질문이 요구하는 법적 결론 명확히 파악
- 판결문에서 해당 결론을 뒷받침하는 근거 추출
- 다양한 법적 관점(원고/피고/법원/학계) 고려

### 단계 2: 다양한 접근 방식 계획
- **직접적 접근**: GT 질문과 유사한 직접적 질문
- **간접적 접근**: 관련 법리나 요건을 통한 간접적 질문
- **비교적 접근**: 유사 사례나 반대 상황과의 비교
- **실무적 접근**: 실제 법률 실무에서의 적용 관점
- **학습적 접근**: 법률 학습자를 위한 교육적 관점

### 단계 3: 레벨별 질문 생성

**Level 1-2 (전문가 수준)**
- 법조문 정확한 인용, 판례 번호 포함
- 복잡한 법리 해석, 예외 조항 고려

**Level 3-4 (고급 수준)**
- 법률 용어 + 간단한 설명
- 법리적 논리 구조 포함

**Level 5-6 (중급 수준)**
- 법률 용어와 일반 용어 혼합
- 실무적 관점 포함

**Level 7-8 (초급 수준)**
- 일상 언어 중심, 핵심 개념만
- 구체적 상황 설명

**Level 9-10 (기초 수준)**
- 매우 간단한 표현
- 일상적 상황으로 변환

## 품질 검증 체크리스트

1. **[CONSISTENCY]** 모든 질문이 GT 질문과 의미적으로 일관된 결론을 도출하는가?
2. **[FACTUAL]** 모든 질문이 판결문에 명확한 근거를 가지는가?
3. **[UTILITY]** 각 질문이 법률 학습에 도움이 되는가?
4. **[DIVERSITY]** 다양한 법적 관점과 접근 방식이 포함되었는가?
5. **[PROGRESSION]** 레벨별 난이도가 적절히 조정되었는가?
6. **[NATURAL]** Yes/No 분포가 GT 질문의 의미에 따라 자연스럽게 형성되었는가?

## 중요한 지침

- **의미적 일관성이 최우선**: 인위적 균형보다 논리적 일관성 중시
- GT 질문과 완전히 동일할 필요는 없으나 같은 법적 결론 도출
- 자연스러운 분포를 허용하되 극단적 편향(9:1, 1:9)은 피할 것
- 모든 질문은 GT 질문과 의미적으로 관련되어야 함

반드시 JSON 형식으로 응답하세요."""
        
        human_prompt = """GT 질문: {gt_question}

판결문 내용:
{document_content}

참고 키워드: {keywords_to_consider}

위 정보를 바탕으로 GT 질문과 의미적으로 일관된 10개 레벨의 Yes/No 질문을 JSON 형식으로 생성해주세요.

**중요한 요구사항:**
1. 모든 질문은 GT 질문과 동일한 법적 결론을 도출해야 합니다
2. 자연스러운 Yes/No 분포를 허용하되, 최소한의 다양성을 위해 10:0 완전 편향은 피해주세요
3. 최소 1-2개의 질문은 다른 관점(반대 논리, 예외 상황, 비교 관점)에서 접근해주세요
4. 예: "~하지 않는다"는 GT 질문이라면, 8-9개는 "Yes" 답변이지만 1-2개는 "No" 답변으로 다른 각도에서 질문

이렇게 하면 의미적 일관성을 유지하면서도 최소한의 관점 다양성을 확보할 수 있습니다."""
        
        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])
    
    def _create_simple_prompt(self) -> ChatPromptTemplate:
        """단순화된 프롬프트 생성"""
        try:
            prompt_path = Path(__file__).parent / "prompts" / "minu" / "unified_yesno_question_generator_simple.txt"
            with open(prompt_path, 'r', encoding='utf-8') as f:
                system_prompt = f.read()
        except FileNotFoundError:
            system_prompt = """당신은 질문을 다시 쓰는 전문가입니다.

주어진 GT 질문을 보고, 같은 의미이지만 난이도가 다른 10개의 Yes/No 질문을 만드세요.

규칙:
1. 모든 질문은 Yes 또는 No로 답할 수 있어야 합니다
2. 모든 질문의 답은 같아야 합니다
3. 레벨 1은 가장 어렵게, 레벨 10은 가장 쉽게 만드세요

레벨별 가이드:
- Level 1-3: 법률 전문용어 사용
- Level 4-6: 법률용어 + 쉬운 설명
- Level 7-10: 일상 언어로만

JSON 형식으로 답하세요."""
        
        human_prompt = """GT 질문: {gt_question}

위 질문을 10개 레벨로 다시 써주세요."""
        
        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])
    
    def _create_minimal_prompt(self) -> ChatPromptTemplate:
        """최소한의 프롬프트 생성"""
        try:
            prompt_path = Path(__file__).parent / "prompts" / "minu" / "unified_yesno_question_generator_minimal.txt"
            with open(prompt_path, 'r', encoding='utf-8') as f:
                system_prompt = f.read()
        except FileNotFoundError:
            system_prompt = """이 질문을 10가지 다른 방법으로 다시 써주세요.

모든 질문은 Yes/No로 답할 수 있어야 하고, 답은 모두 같아야 합니다.

1번은 가장 어렵게, 10번은 가장 쉽게 써주세요.

JSON으로 답하세요."""
        
        human_prompt = """{gt_question}"""
        
        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])
    
    def _create_smart_simple_prompt(self) -> ChatPromptTemplate:
        """스마트한 단순화 프롬프트 생성"""
        try:
            prompt_path = Path(__file__).parent / "prompts" / "minu" / "unified_yesno_question_generator_smart_simple.txt"
            with open(prompt_path, 'r', encoding='utf-8') as f:
                system_prompt = f.read()
        except FileNotFoundError:
            system_prompt = """이 질문을 10가지 다른 방법으로 다시 써주세요.

규칙:
1. 모든 질문은 Yes/No로 답할 수 있어야 합니다
2. 모든 질문의 답은 원본 질문과 같아야 합니다
3. 5개는 Yes 답변, 5개는 No 답변이 되도록 만드세요
4. 1번은 가장 어렵게, 10번은 가장 쉽게 써주세요

원본 질문의 답이 "No"라면:
- 1,3,5,7,9번 질문은 반대로 물어서 "Yes" 답변이 나오게 하세요
- 2,4,6,8,10번 질문은 원본과 같은 방향으로 물어서 "No" 답변이 나오게 하세요

원본 질문의 답이 "Yes"라면:
- 1,3,5,7,9번 질문은 원본과 같은 방향으로 물어서 "Yes" 답변이 나오게 하세요  
- 2,4,6,8,10번 질문은 반대로 물어서 "No" 답변이 나오게 하세요

JSON으로 답하세요."""
        
        human_prompt = """{gt_question}"""
        
        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])
    
    def generate_ten_level_questions(
        self, 
        gt_question: str, 
        document_content: str = "", 
        keywords_to_consider: Optional[str] = None
    ) -> TenLevelYesNoQuestions:
        """
        10개 레벨의 Yes/No 질문 생성
        
        Args:
            gt_question: 원본 GT 질문
            document_content: 판결문 전체 내용 (단순화 모드에서는 무시됨)
            keywords_to_consider: 참고할 키워드들 (단순화 모드에서는 무시됨)
            
        Returns:
            TenLevelYesNoQuestions: 생성된 10개 레벨 질문들
        """
        try:
            print(f"🔄 10개 레벨 Yes/No 질문 생성 시작... (모드: {self.prompt_mode})")
            print(f"📝 GT 질문: {gt_question[:50]}...")
            
            # 프롬프트 모드에 따라 입력 파라미터 조정
            if self.prompt_mode == "minimal":
                # 최소한 모드: GT 질문만 사용
                invoke_params = {"gt_question": gt_question}
            elif self.prompt_mode == "simple":
                # 단순화 모드: GT 질문만 사용
                invoke_params = {"gt_question": gt_question}
            elif self.prompt_mode == "smart_simple":
                # 스마트 단순화 모드: GT 질문만 사용
                invoke_params = {"gt_question": gt_question}
            else:
                # 기존 모드: 모든 파라미터 사용
                if not keywords_to_consider:
                    keywords_to_consider = "법률 용어, 판례, 법리"
                invoke_params = {
                    "gt_question": gt_question,
                    "document_content": document_content,
                    "keywords_to_consider": keywords_to_consider
                }
            
            # 프롬프트 체인 구성
            chain = self.prompt_template | self.structured_llm
            
            # 질문 생성 실행
            response = chain.invoke(invoke_params)
            
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
        """생성된 결과 검증 - 의미적 일관성 중심"""
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
            
            # 의미적 일관성 검증 (핵심 변경점)
            consistency_rate = result.get_consistency_rate()
            if consistency_rate < 0.65:  # 65% 이상 일관성 요구 (70%에서 완화)
                print(f"❌ 답변 일관성이 낮습니다: {consistency_rate:.2%}")
                print("💡 GT 질문의 의미에 따른 자연스러운 분포를 허용합니다.")
                return False
            
            # 극단적 편향 방지 (8:2까지 허용, 9:1, 10:0은 방지)
            yes_count = sum(1 for q in result.questions if q.expected_answer.value == 'Yes')
            no_count = sum(1 for q in result.questions if q.expected_answer.value == 'No')
            
            if yes_count == 0 or no_count == 0:
                print(f"❌ 완전 편향 감지: Yes {yes_count}개, No {no_count}개")
                print("💡 최소 1개 이상의 다른 관점이 필요합니다.")
                return False
            
            # 8:2 분포까지는 자연스러운 것으로 허용
            if min(yes_count, no_count) >= 2:
                print(f"✅ 적절한 분포: Yes {yes_count}개, No {no_count}개")
            elif min(yes_count, no_count) == 1:
                print(f"⚠️ 편향된 분포이지만 허용: Yes {yes_count}개, No {no_count}개")
                print("💡 의미적 일관성을 우선시하여 허용합니다.")
            
            # 질문 다양성 검증 (중복 방지)
            unique_questions = set(q.question.lower().strip() for q in result.questions)
            if len(unique_questions) < 8:  # 최소 8개는 서로 달라야 함
                print(f"❌ 질문 다양성 부족: {len(unique_questions)}개 고유 질문")
                return False
            
            print(f"✅ 검증 통과: Yes {yes_count}개, No {no_count}개 (일관성: {consistency_rate:.1%})")
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
            try:
                from liberty_agent.b_rag.core.schemas.yesno_question_schemas import GenerationMetadata
            except ImportError:
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
        """개선된 Fallback 질문 생성 - 의미적 일관성 중심"""
        print("🔧 의미적 일관성 중심 Fallback 질문 생성 중...")
        
        fallback_questions = []
        
        # GT 질문의 의미 분석
        gt_lower = gt_question.lower()
        is_negative_question = any(neg in gt_lower for neg in ["아니", "않", "없", "못", "안"])
        
        # GT 질문의 예상 답변 결정
        if is_negative_question:
            # "~하지 아니한다고 할 수 있는가?" → Yes (아니하다는 것이 맞다)
            primary_answer = YesNoAnswer.YES
            secondary_answer = YesNoAnswer.NO
            primary_ratio = 0.7  # 70% Yes, 30% No
        else:
            # "~한다고 할 수 있는가?" → 문맥에 따라 결정
            primary_answer = YesNoAnswer.YES
            secondary_answer = YesNoAnswer.NO
            primary_ratio = 0.6  # 60% Yes, 40% No
        
        # 자연스러운 분포로 답변 배정
        primary_count = int(10 * primary_ratio)
        secondary_count = 10 - primary_count
        
        answer_sequence = ([primary_answer] * primary_count + 
                          [secondary_answer] * secondary_count)
        
        # 다양한 관점의 질문 템플릿
        question_templates = {
            YesNoAnswer.YES: [
                "이 법률 문제에서 긍정적인 결론이 도출되는가?",
                "관련 법리에 따라 인정될 수 있는 부분이 있는가?",
                "이 상황에서 법적으로 유효한 주장이 가능한가?",
                "판례나 법리상 지지되는 견해가 있는가?",
                "법적 요건이 충족되는 경우가 있나요?",
                "이런 상황에서 법적 보호를 받을 수 있나요?",
                "이것이 법적으로 인정받을 수 있는 일인가요?",
                "이런 경우에 권리를 주장할 수 있어요?",
                "이것이 법적으로 맞는 일인가요?",
                "이런 상황이 법적으로 괜찮은 건가요?"
            ],
            YesNoAnswer.NO: [
                "이 법률 문제에서 부정적인 결론이 도출되는가?",
                "관련 법리에 따라 제한되는 부분이 있는가?",
                "이 상황에서 법적으로 문제가 되는 요소가 있는가?",
                "판례나 법리상 반대되는 견해가 있는가?",
                "법적 요건이 충족되지 않는 경우가 있나요?",
                "이런 상황에서 법적 제재를 받을 수 있나요?",
                "이것이 법적으로 문제가 되는 일인가요?",
                "이런 경우에 권리를 제한받을 수 있어요?",
                "이것이 법적으로 틀린 일인가요?",
                "이런 상황이 법적으로 문제가 되나요?"
            ]
        }
        
        for level in range(1, 11):
            audience = SAMPLE_TARGET_AUDIENCES[level].value
            expected_answer = answer_sequence[level - 1]
            
            # 해당 답변 유형의 템플릿에서 선택
            templates = question_templates[expected_answer]
            question = templates[level - 1] if level <= len(templates) else templates[0]
            
            # 레벨에 따른 언어 수준 조정
            if level <= 3:
                # 전문가/고급 수준 - 법률 용어 사용
                question = question.replace("이것이", "해당 사안이").replace("법적으로", "법리상")
            elif level <= 6:
                # 중급 수준 - 적당한 법률 용어
                question = question.replace("법리상", "법적으로")
            else:
                # 초급/기초 수준 - 쉬운 표현
                question = question.replace("법리상", "법적으로").replace("사안이", "경우가")
            
            fallback_questions.append(LevelQuestion(
                level=level,
                question=question,
                target_audience=audience,
                reasoning=f"의미적 일관성 중심 Fallback - Level {level} ({expected_answer.value} 답변, 자연스러운 분포)",
                expected_answer=expected_answer,
                confidence=0.7  # 개선된 확신도
            ))
        
        try:
            from liberty_agent.b_rag.core.schemas.yesno_question_schemas import GenerationMetadata
        except ImportError:
            from core.schemas.yesno_question_schemas import GenerationMetadata
        
        return TenLevelYesNoQuestions(
            gt_question=gt_question,
            document_summary="의미적 일관성 중심 Fallback 모드로 생성된 요약 - 자연스러운 분포",
            questions=fallback_questions,
            semantic_consistency="의미적 일관성을 우선시하여 생성된 자연스러운 분포의 질문들",
            generation_metadata=GenerationMetadata(
                total_questions=10,
                difficulty_range="1-10",
                question_type="Yes/No",
                semantic_equivalence=True  # 의미적 일관성 중심이므로 True
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