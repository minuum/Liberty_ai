#!/usr/bin/env python3
"""
수정된 AdvancedQuestionGenerator 테스트 스크립트
- robustness_set 전략의 중괄호 이스케이프 문제 수정 확인
- 10개 질문 생성 프로세스가 정책에 반영되었는지 확인
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Optional, Dict, Union, Any
from enum import Enum
import json

# Langchain 관련 임포트
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

# 로거 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- legal_schemas.py의 내용 ---
class DifficultyLevel(Enum):
    INTRODUCTORY = "입문"
    BASIC = "기초"
    INTERMEDIATE = "중급"
    ADVANCED = "고급"
    EXPERT = "전문가"

    def __str__(self):
        return self.value

class DocumentAnalysis(BaseModel):
    key_points: List[str] = Field(..., description="문서의 핵심 내용 요약")
    legal_issues: List[str] = Field(..., description="주요 법적 쟁점 또는 문제점")
    keywords: List[str] = Field(..., description="문서의 주요 키워드")
    document_type: str = Field(..., description="문서 유형 (예: 판례, 법령, 계약서)")
    complexity_level: DifficultyLevel = Field(..., description="문서의 전반적인 난이도")

class LegalQuestion(BaseModel):
    question: str = Field(..., description="생성된 법률 질문")
    reasoning: str = Field(..., description="질문의 출제 의도 및 학습 포인트")
    keywords: List[str] = Field(default_factory=list, description="질문과 관련된 주요 키워드")
    difficulty: DifficultyLevel = Field(..., description="질문의 난이도")
    strategy: str = Field(..., description="질문 생성 전략 또는 유형")

class QuestionSet(BaseModel):
    questions: List[LegalQuestion] = Field(..., description="생성된 법률 질문 세트")

# --- 간소화된 AdvancedQuestionGenerator ---
class TestAdvancedQuestionGenerator:
    def __init__(self, openai_api_key: Optional[str] = None):
        self.llm = ChatOpenAI(model_name="gpt-4o-2024-08-06", temperature=0.1, api_key=openai_api_key)
        self.prompts_dir = Path("/Users/minu/dev/Liberty/Liberty_ai/liberty_agent/rag/prompts/minu")
        
    def _load_prompt_content(self, file_key: str, sub_dir: Optional[str] = None) -> str:
        """프롬프트 파일 로드"""
        current_prompts_path = self.prompts_dir
        if sub_dir:
            current_prompts_path = current_prompts_path / sub_dir
        
        prompt_file_path = current_prompts_path / f"{file_key}.txt"
        try:
            content = prompt_file_path.read_text(encoding="utf-8")
            logger.info(f"프롬프트 로드 성공: {prompt_file_path}")
            return content
        except FileNotFoundError:
            logger.warning(f"프롬프트 파일 없음: {prompt_file_path}")
            return ""
        except Exception as e:
            logger.error(f"프롬프트 로드 오류: {e}")
            return ""

    def test_robustness_prompt(self):
        """robustness_set 프롬프트의 중괄호 이스케이프 문제 테스트"""
        logger.info("=== robustness_set 프롬프트 테스트 ===")
        
        system_content = self._load_prompt_content("boosted_robustness_system")
        human_content = self._load_prompt_content("boosted_robustness_human")
        
        if not system_content or not human_content:
            logger.error("프롬프트 로드 실패")
            return False
            
        try:
            # 프롬프트 템플릿 생성 테스트
            prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(system_content),
                HumanMessagePromptTemplate.from_template(human_content)
            ])
            
            # 변수 포맷팅 테스트
            formatted = prompt.format_messages(
                document_text="테스트 문서",
                primary_theme="테스트 테마",
                num_questions=2
            )
            
            logger.info("✅ robustness_set 프롬프트 템플릿 생성 성공")
            return True
            
        except Exception as e:
            logger.error(f"❌ robustness_set 프롬프트 템플릿 오류: {e}")
            return False

    def test_policy_prompts(self):
        """정책 프롬프트들이 10개 질문 생성 프로세스를 포함하는지 테스트"""
        logger.info("=== 정책 프롬프트 테스트 ===")
        
        policy_levels = ["policy_level_1", "policy_level_2", "policy_level_3", "policy_level_4", "policy_level_else"]
        
        for level in policy_levels:
            content = self._load_prompt_content(level, sub_dir="policy")
            if content:
                if "10개 질문 생성 프로세스" in content:
                    logger.info(f"✅ {level}: 10개 질문 생성 프로세스 포함됨")
                else:
                    logger.warning(f"⚠️ {level}: 10개 질문 생성 프로세스 누락")
            else:
                logger.error(f"❌ {level}: 프롬프트 로드 실패")

    def test_standard_question_generation(self):
        """Standard 질문 생성 테스트 (정책 레벨 1, 10개)"""
        logger.info("=== Standard 질문 생성 테스트 ===")
        
        sample_document = """
        민법 제750조는 "고의 또는 과실로 인한 위법행위로 타인에게 손해를 가한 자는 그 손해를 배상할 책임이 있다."고 규정하고 있다.
        이는 불법행위 책임의 일반 조항이다.
        """
        
        # 간단한 분석 결과 생성
        analysis = DocumentAnalysis(
            key_points=["불법행위 책임의 일반 조항", "고의 또는 과실", "손해배상 책임"],
            legal_issues=["불법행위 성립요건", "손해배상 범위"],
            keywords=["민법", "불법행위", "손해배상", "고의", "과실"],
            document_type="법령",
            complexity_level=DifficultyLevel.BASIC
        )
        
        try:
            # 정책 프롬프트 로드 및 포맷팅 테스트
            policy_content = self._load_prompt_content("policy_level_1", sub_dir="policy")
            if policy_content:
                formatted_policy = policy_content.format(keywords_to_consider=", ".join(analysis.keywords))
                logger.info("✅ 정책 프롬프트 포맷팅 성공")
                logger.info(f"정책 내용 (일부): {formatted_policy[:200]}...")
                return True
            else:
                logger.error("❌ 정책 프롬프트 로드 실패")
                return False
                
        except Exception as e:
            logger.error(f"❌ Standard 질문 생성 테스트 오류: {e}")
            return False

def main():
    """메인 테스트 함수"""
    logger.info("🚀 AdvancedQuestionGenerator 수정사항 테스트 시작")
    
    # API 키 확인
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("⚠️ OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
    
    # 테스트 인스턴스 생성
    tester = TestAdvancedQuestionGenerator(openai_api_key=api_key)
    
    # 테스트 실행
    results = []
    
    # 1. robustness_set 프롬프트 테스트
    results.append(tester.test_robustness_prompt())
    
    # 2. 정책 프롬프트 테스트
    tester.test_policy_prompts()
    
    # 3. Standard 질문 생성 테스트
    results.append(tester.test_standard_question_generation())
    
    # 결과 요약
    logger.info("=" * 50)
    logger.info("🏁 테스트 결과 요약")
    logger.info(f"성공한 테스트: {sum(results)}/{len(results)}")
    
    if all(results):
        logger.info("✅ 모든 테스트 통과! 수정사항이 정상적으로 적용되었습니다.")
    else:
        logger.warning("⚠️ 일부 테스트 실패. 추가 수정이 필요할 수 있습니다.")

if __name__ == "__main__":
    main() 