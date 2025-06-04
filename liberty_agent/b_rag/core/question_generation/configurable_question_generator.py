"""
확장 가능한 Yes/No 질문 생성기
- 질문 개수 설정 가능 (5개, 10개, 20개 등)
- 여러 케이스 배치 처리
- 확신도 기준 설정 가능
- 난이도 모드 확장 지원
"""

import os
import time
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# 기존 스키마 재사용
from liberty_agent.b_rag.core.schemas.yesno_question_schemas import (
    LevelQuestion, 
    YesNoAnswer, 
    GenerationMetadata
)

@dataclass
class GenerationConfig:
    """질문 생성 설정"""
    num_questions: int = 10  # 생성할 질문 개수
    min_level: int = 1       # 최소 레벨
    max_level: int = 10      # 최대 레벨
    difficulty_mode: str = "enhanced"  # 난이도 모드
    confidence_threshold: float = 0.95  # 확신도 임계값
    temperature: float = 0.2  # LLM 온도
    model_name: str = "gpt-4o-2024-08-06"
    
    def get_level_range(self) -> List[int]:
        """레벨 범위 생성"""
        if self.num_questions <= (self.max_level - self.min_level + 1):
            # 질문 수가 레벨 범위보다 작거나 같으면 균등 분배
            step = (self.max_level - self.min_level) / (self.num_questions - 1)
            return [int(self.min_level + i * step) for i in range(self.num_questions)]
        else:
            # 질문 수가 더 많으면 레벨 반복
            base_levels = list(range(self.min_level, self.max_level + 1))
            levels = []
            while len(levels) < self.num_questions:
                levels.extend(base_levels)
            return levels[:self.num_questions]

@dataclass 
class TestCase:
    """테스트 케이스 정의"""
    name: str
    gt_question: str
    document_content: str
    keywords: Optional[str] = None
    expected_answer: str = "No"  # 기본값
    metadata: Optional[Dict[str, Any]] = None

class FlexibleYesNoQuestions(BaseModel):
    """확장 가능한 Yes/No 질문 컬렉션"""
    gt_question: str = Field(description="원본 GT 질문")
    document_summary: str = Field(description="판결문의 핵심 내용 요약")
    questions: List[LevelQuestion] = Field(description="생성된 질문들")
    semantic_consistency: str = Field(description="의미적 일관성 설명")
    generation_metadata: GenerationMetadata = Field(description="생성 메타데이터")
    
    def get_consistency_rate(self) -> float:
        """답변 일관성 비율 계산"""
        if not self.questions:
            return 0.0
        
        first_answer = self.questions[0].expected_answer
        consistent_count = sum(1 for q in self.questions if q.expected_answer == first_answer)
        return consistent_count / len(self.questions)
    
    def get_question_by_level(self, level: int) -> Optional[LevelQuestion]:
        """특정 레벨의 질문 반환"""
        for question in self.questions:
            if question.level == level:
                return question
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """통계 정보 반환"""
        yes_count = sum(1 for q in self.questions if q.expected_answer == YesNoAnswer.YES)
        no_count = len(self.questions) - yes_count
        
        avg_confidence = sum(q.confidence for q in self.questions) / len(self.questions)
        level_distribution = {}
        for q in self.questions:
            level_distribution[q.level] = level_distribution.get(q.level, 0) + 1
            
        return {
            "total_questions": len(self.questions),
            "yes_count": yes_count,
            "no_count": no_count,
            "consistency_rate": self.get_consistency_rate(),
            "avg_confidence": avg_confidence,
            "level_distribution": level_distribution,
            "level_range": f"{min(q.level for q in self.questions)}-{max(q.level for q in self.questions)}"
        }

class ConfigurableQuestionGenerator:
    """확장 가능한 질문 생성기"""
    
    def __init__(self, config: GenerationConfig, openai_api_key: Optional[str] = None):
        self.config = config
        self.llm = ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=openai_api_key or os.getenv("OPENAI_API_KEY")
        )
        
        # Structured output 설정
        try:
            self.structured_llm = self.llm.with_structured_output(
                FlexibleYesNoQuestions,
                method="function_calling", 
                include_raw=False
            )
        except Exception as e:
            print(f"⚠️ Structured output 설정 실패: {e}")
            self.structured_llm = self.llm
            
        self.prompt_template = self._create_prompt()
        
        print(f"✅ ConfigurableQuestionGenerator 초기화 완료")
        print(f"   - 질문 개수: {config.num_questions}개")
        print(f"   - 레벨 범위: {config.min_level}-{config.max_level}")
        print(f"   - 난이도 모드: {config.difficulty_mode}")
        print(f"   - 확신도 임계값: {config.confidence_threshold}")
    
    def _create_prompt(self) -> ChatPromptTemplate:
        """동적 프롬프트 생성"""
        levels = self.config.get_level_range()
        level_descriptions = self._generate_level_descriptions(levels)
        
        system_prompt = f"""당신은 법률 질문 생성 전문가입니다. 주어진 GT 질문과 판결문을 바탕으로 **의미론적으로 완전히 동일하지만 구체성 수준이 다른** {self.config.num_questions}개의 Yes/No 질문을 생성해야 합니다.

## 🎯 핵심 원칙: 의미론적 동일성 (Semantic Equivalence)

### 생성할 질문 구성
- 총 질문 개수: **{self.config.num_questions}개**
- 레벨 범위: **{self.config.min_level}-{self.config.max_level}**
- 난이도 모드: **{self.config.difficulty_mode}**

### 레벨별 특성
{level_descriptions}

### 핵심 요구사항
1. **의미론적 동일성**: 모든 질문이 GT 질문과 정확히 같은 법적 상황을 다뤄야 함
2. **답변 일관성**: 모든 질문이 동일한 Yes/No 답변을 가져야 함  
3. **핵심 키워드 보존**: GT 질문의 주요 키워드를 각 레벨에 맞게 반드시 포함
4. **새로운 정보 금지**: 원래 없던 조건, 시간, 장소, 당사자 추가 금지
5. **확신도 기준**: 모든 답변의 확신도가 {self.config.confidence_threshold} 이상이어야 함

반드시 JSON 형식으로 {self.config.num_questions}개의 질문을 정확히 생성하세요."""

        human_prompt = """GT 질문: {gt_question}

판결문 내용:
{document_content}

참고 키워드: {keywords_to_consider}

위 GT 질문과 **정확히 같은 법적 상황**을 다루되, **구체성 수준만 다른** {num_questions}개의 Yes/No 질문을 JSON 형식으로 생성해주세요.

🎯 **핵심 요구사항 (절대 준수):**
1. **핵심 키워드 보존**: GT 질문의 주요 키워드를 각 레벨에 맞게 반드시 포함
2. **의미 동일성**: 모든 질문이 GT와 정확히 같은 법적 상황을 다뤄야 함
3. **답변 일관성**: 모든 질문이 동일한 Yes/No 답변을 가져야 함
4. **확신도 기준**: 모든 답변의 확신도가 {confidence_threshold} 이상
5. **새로운 정보 금지**: 원래 없던 조건, 시간, 장소, 당사자 추가 금지""".format(
            num_questions=self.config.num_questions,
            confidence_threshold=self.config.confidence_threshold
        )
        
        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])
    
    def _generate_level_descriptions(self, levels: List[int]) -> str:
        """레벨 설명 생성"""
        descriptions = []
        for level in sorted(set(levels)):
            if level <= 2:
                desc = f"**Level {level}**: 최고 구체성 - 모든 법률 용어, 조문 번호, 전문적 표현 유지"
            elif level <= 4:
                desc = f"**Level {level}**: 높은 구체성 - 핵심 법률 용어 유지, 부가 설명 추가"
            elif level <= 6:
                desc = f"**Level {level}**: 중간 구체성 - 법률 용어와 일반 용어 혼합"
            elif level <= 8:
                desc = f"**Level {level}**: 낮은 구체성 - 대부분 일반 용어, 단순한 문장"
            else:
                desc = f"**Level {level}**: 최저 구체성 - 완전한 일상 언어, 최대한 간단"
            descriptions.append(desc)
        return "\n".join(descriptions)
    
    def generate_questions(
        self,
        gt_question: str,
        document_content: str = "",
        keywords_to_consider: Optional[str] = None
    ) -> Tuple[FlexibleYesNoQuestions, Dict[str, Any]]:
        """단일 케이스 질문 생성"""
        print(f"🎯 질문 생성 시작: {self.config.num_questions}개 질문")
        print(f"📝 GT 질문: {gt_question[:50]}...")
        
        start_time = time.time()
        
        invoke_params = {
            "gt_question": gt_question,
            "document_content": document_content,
            "keywords_to_consider": keywords_to_consider or "자동 추출"
        }
        
        try:
            # API 호출
            chain = self.prompt_template | self.structured_llm
            result = chain.invoke(invoke_params)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # 결과 검증
            success = self._validate_result(result, gt_question)
            
            # 통계 정보
            stats = result.get_statistics() if hasattr(result, 'get_statistics') else {}
            
            metadata = {
                "execution_time": execution_time,
                "api_called": execution_time >= 3.0,
                "success": success,
                "config": {
                    "num_questions": self.config.num_questions,
                    "level_range": f"{self.config.min_level}-{self.config.max_level}",
                    "difficulty_mode": self.config.difficulty_mode,
                    "confidence_threshold": self.config.confidence_threshold
                },
                "statistics": stats
            }
            
            print(f"✅ 질문 생성 완료!")
            print(f"   실행 시간: {execution_time:.2f}초")
            print(f"   생성된 질문: {len(result.questions) if hasattr(result, 'questions') else 0}개")
            print(f"   API 호출: {'Yes' if execution_time >= 3.0 else 'No'}")
            
            return result, metadata
            
        except Exception as e:
            print(f"❌ 질문 생성 실패: {e}")
            # Fallback 생성
            fallback_result = self._create_fallback_questions(gt_question, document_content)
            metadata = {
                "execution_time": 0,
                "api_called": False,
                "success": False,
                "error": str(e)
            }
            return fallback_result, metadata
    
    def batch_generate_questions(
        self,
        test_cases: List[TestCase],
        save_individual: bool = True,
        output_dir: str = "results"
    ) -> Dict[str, Any]:
        """배치 처리: 여러 케이스 질문 생성"""
        print(f"🚀 배치 질문 생성 시작: {len(test_cases)}개 케이스")
        print(f"📁 출력 디렉토리: {output_dir}")
        
        # 출력 디렉토리 생성
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        batch_results = {
            "config": {
                "num_questions": self.config.num_questions,
                "level_range": f"{self.config.min_level}-{self.config.max_level}",
                "difficulty_mode": self.config.difficulty_mode,
                "confidence_threshold": self.config.confidence_threshold,
                "total_cases": len(test_cases)
            },
            "results": [],
            "summary": {
                "successful_cases": 0,
                "failed_cases": 0,
                "total_questions_generated": 0,
                "avg_execution_time": 0,
                "api_calls_made": 0
            }
        }
        
        total_execution_time = 0
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n{'='*60}")
            print(f"📋 케이스 {i}/{len(test_cases)}: {test_case.name}")
            print(f"{'='*60}")
            
            # 질문 생성
            result, metadata = self.generate_questions(
                test_case.gt_question,
                test_case.document_content,
                test_case.keywords
            )
            
            # 결과 저장
            case_result = {
                "case_id": i,
                "case_name": test_case.name,
                "gt_question": test_case.gt_question,
                "result": result.dict() if hasattr(result, 'dict') else str(result),
                "metadata": metadata,
                "test_case_metadata": test_case.metadata or {}
            }
            
            batch_results["results"].append(case_result)
            
            # 개별 파일 저장
            if save_individual:
                individual_file = f"{output_dir}/case_{i:02d}_{test_case.name.replace(' ', '_')}.json"
                self._save_json(case_result, individual_file)
                print(f"💾 개별 결과 저장: {individual_file}")
            
            # 통계 업데이트
            if metadata.get("success", False):
                batch_results["summary"]["successful_cases"] += 1
                batch_results["summary"]["total_questions_generated"] += len(result.questions) if hasattr(result, 'questions') else 0
            else:
                batch_results["summary"]["failed_cases"] += 1
            
            total_execution_time += metadata.get("execution_time", 0)
            if metadata.get("api_called", False):
                batch_results["summary"]["api_calls_made"] += 1
            
            # 진행률 출력
            progress = (i / len(test_cases)) * 100
            print(f"📊 진행률: {progress:.1f}% ({i}/{len(test_cases)})")
        
        # 최종 통계
        if batch_results["summary"]["successful_cases"] > 0:
            batch_results["summary"]["avg_execution_time"] = total_execution_time / len(test_cases)
        
        # 배치 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_file = f"{output_dir}/batch_results_{timestamp}.json"
        self._save_json(batch_results, batch_file)
        
        print(f"\n{'='*60}")
        print(f"🎉 배치 처리 완료!")
        print(f"   성공한 케이스: {batch_results['summary']['successful_cases']}/{len(test_cases)}")
        print(f"   총 생성 질문: {batch_results['summary']['total_questions_generated']}개")
        print(f"   평균 실행 시간: {batch_results['summary']['avg_execution_time']:.2f}초")
        print(f"   API 호출: {batch_results['summary']['api_calls_made']}회")
        print(f"💾 배치 결과: {batch_file}")
        print(f"{'='*60}")
        
        return batch_results
    
    def _validate_result(self, result: FlexibleYesNoQuestions, gt_question: str) -> bool:
        """결과 검증"""
        try:
            if not hasattr(result, 'questions') or not result.questions:
                return False
            
            if len(result.questions) != self.config.num_questions:
                print(f"⚠️ 질문 개수 불일치: {len(result.questions)}개 (예상: {self.config.num_questions}개)")
                return False
            
            # 답변 일관성 검증
            consistency_rate = result.get_consistency_rate()
            if consistency_rate < 1.0:
                print(f"⚠️ 답변 일관성 부족: {consistency_rate:.1%}")
                return False
            
            # 확신도 검증
            low_confidence_questions = [
                q for q in result.questions 
                if q.confidence < self.config.confidence_threshold
            ]
            if low_confidence_questions:
                print(f"⚠️ 낮은 확신도 질문 {len(low_confidence_questions)}개")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ 검증 오류: {e}")
            return False
    
    def _create_fallback_questions(
        self, 
        gt_question: str, 
        document_content: str
    ) -> FlexibleYesNoQuestions:
        """Fallback 질문 생성"""
        levels = self.config.get_level_range()
        questions = []
        
        for i, level in enumerate(levels):
            fallback_question = LevelQuestion(
                level=level,
                question=f"[Fallback {i+1}] {gt_question}",
                target_audience=f"Level {level} 대상",
                reasoning=f"API 실패로 인한 Fallback 질문 (Level {level})",
                expected_answer=YesNoAnswer.NO,
                confidence=0.5
            )
            questions.append(fallback_question)
        
        return FlexibleYesNoQuestions(
            gt_question=gt_question,
            document_summary="API 실패로 인한 Fallback 응답",
            questions=questions,
            semantic_consistency="Fallback 모드에서는 의미적 일관성을 보장할 수 없습니다.",
            generation_metadata=GenerationMetadata(
                total_questions=len(questions),
                difficulty_range=f"{self.config.min_level}-{self.config.max_level}",
                question_type="Yes/No (Fallback)",
                semantic_equivalence=False
            )
        )
    
    def _save_json(self, data: Dict[str, Any], filepath: str) -> None:
        """JSON 파일 저장"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            print(f"⚠️ 파일 저장 실패 {filepath}: {e}")

# 편의 함수들
def create_standard_config(num_questions: int = 10) -> GenerationConfig:
    """표준 설정 생성"""
    return GenerationConfig(
        num_questions=num_questions,
        min_level=1,
        max_level=10,
        difficulty_mode="enhanced",
        confidence_threshold=0.85,
        temperature=0.2
    )

def create_quick_config(num_questions: int = 5) -> GenerationConfig:
    """빠른 테스트용 설정"""
    return GenerationConfig(
        num_questions=num_questions,
        min_level=2,
        max_level=8,
        difficulty_mode="semantic",
        confidence_threshold=0.8,
        temperature=0.1
    )

def create_comprehensive_config(num_questions: int = 20) -> GenerationConfig:
    """포괄적 테스트용 설정"""
    return GenerationConfig(
        num_questions=num_questions,
        min_level=1,
        max_level=10,
        difficulty_mode="enhanced",
        confidence_threshold=0.9,
        temperature=0.3
    ) 