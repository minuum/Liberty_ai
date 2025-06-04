#!/usr/bin/env python3
"""
개선된 키워드 추출 및 GT 질문 생성 시스템
- AI 기반 지능형 키워드 추출 (하드코딩 제거)
- 문맥 기반 GT 질문 생성 (규칙 기반 → AI 기반)
"""

import os
import re
from typing import List, Dict, Tuple, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

class KeywordExtractionResult(BaseModel):
    """키워드 추출 결과"""
    primary_keywords: List[str] = Field(description="핵심 키워드 3-5개")
    secondary_keywords: List[str] = Field(description="보조 키워드 2-3개")
    legal_domain: str = Field(description="법률 분야 (민사, 형사, 행정 등)")
    main_topic: str = Field(description="문서의 주요 법적 쟁점")

class GTQuestionResult(BaseModel):
    """GT 질문 생성 결과"""
    gt_question: str = Field(description="생성된 GT 질문")
    reasoning: str = Field(description="질문 생성 근거")
    difficulty_level: str = Field(description="난이도 (초급/중급/고급)")
    legal_principle: str = Field(description="관련 법리")

class ImprovedKeywordExtractor:
    """개선된 AI 기반 키워드 추출기"""
    
    def __init__(self, model_name: str = "gpt-4o-2024-08-06"):
        self.llm = ChatOpenAI(
            model=model_name, 
            temperature=0.1,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        self.keyword_llm = self.llm.with_structured_output(
            KeywordExtractionResult,
            method="function_calling",
            include_raw=False
        )
        
        self.gt_question_llm = self.llm.with_structured_output(
            GTQuestionResult,
            method="function_calling", 
            include_raw=False
        )
        
        self.keyword_prompt = self._create_keyword_prompt()
        self.gt_question_prompt = self._create_gt_question_prompt()
        
        print(f"✅ ImprovedKeywordExtractor 초기화 완료 (모델: {model_name})")
    
    def _create_keyword_prompt(self) -> ChatPromptTemplate:
        """키워드 추출 프롬프트 생성"""
        system_prompt = """당신은 법률 문서 분석 전문가입니다. 주어진 법률 문서에서 핵심 키워드를 지능적으로 추출해야 합니다.

## 추출 원칙

1. **맥락 중심**: 문서의 실제 법적 쟁점과 관련된 키워드만 추출
2. **중요도 순**: 핵심 키워드(3-5개) > 보조 키워드(2-3개)
3. **법적 정확성**: 법률 용어의 정확한 의미와 사용법 고려
4. **불필요한 키워드 제외**: 단순히 언급되었다고 모든 키워드를 포함하지 않음

## 키워드 분류

- **핵심 키워드**: 문서의 주요 법적 쟁점과 직접 관련
- **보조 키워드**: 배경 정보나 부차적 쟁점 관련
- **법률 분야**: 민사, 형사, 행정, 상사, 가사, 헌법 등
- **주요 쟁점**: 한 문장으로 요약한 핵심 법적 문제

반드시 JSON 형식으로 응답하세요."""

        human_prompt = """다음 법률 문서를 분석하여 지능적으로 키워드를 추출해주세요:

문서 내용:
{document_content}

위 문서에서 실제 법적 쟁점과 관련된 핵심 키워드만을 추출하고, 법률 분야와 주요 쟁점을 파악해주세요."""

        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])
    
    def _create_gt_question_prompt(self) -> ChatPromptTemplate:
        """GT 질문 생성 프롬프트 생성"""
        system_prompt = """당신은 법률 교육 전문가입니다. 주어진 법률 문서와 키워드 분석 결과를 바탕으로 적절한 GT(Ground Truth) 질문을 생성해야 합니다.

## GT 질문 생성 원칙

1. **문서 중심**: 실제 문서 내용에서 다루는 법적 쟁점을 질문으로 변환
2. **Yes/No 형식**: 명확하게 예/아니오로 답할 수 있는 질문
3. **법적 정확성**: 법률 전문용어의 정확한 사용
4. **적절한 난이도**: 너무 쉽지도 어렵지도 않은 적정 수준

## 질문 패턴

- **권리/의무**: "~권리가 인정되는가?", "~의무가 발생하는가?"
- **성립/인정**: "~가 성립하는가?", "~가 인정되는가?"
- **효력/유효성**: "~의 효력이 있는가?", "~가 유효한가?"
- **위법/적법**: "~가 위법한가?", "~가 적법한가?"

반드시 JSON 형식으로 응답하세요."""

        human_prompt = """다음 정보를 바탕으로 적절한 GT 질문을 생성해주세요:

문서 내용:
{document_content}

키워드 분석 결과:
- 핵심 키워드: {primary_keywords}
- 법률 분야: {legal_domain}
- 주요 쟁점: {main_topic}

위 정보를 종합하여 문서의 핵심 법적 쟁점을 다루는 Yes/No 형식의 GT 질문을 생성해주세요."""

        return ChatPromptTemplate.from_messages([
            ("system", system_prompt), 
            ("human", human_prompt)
        ])
    
    def extract_keywords_intelligently(self, document_content: str) -> KeywordExtractionResult:
        """AI 기반 지능형 키워드 추출"""
        try:
            print("🔍 AI 기반 키워드 추출 중...")
            
            # 문서 내용 전처리
            cleaned_content = self._preprocess_content(document_content)
            
            # AI 키워드 추출
            chain = self.keyword_prompt | self.keyword_llm
            result = chain.invoke({"document_content": cleaned_content})
            
            print(f"✅ 키워드 추출 완료:")
            print(f"  핵심: {', '.join(result.primary_keywords)}")
            print(f"  보조: {', '.join(result.secondary_keywords)}")
            print(f"  분야: {result.legal_domain}")
            print(f"  쟁점: {result.main_topic}")
            
            return result
            
        except Exception as e:
            print(f"❌ AI 키워드 추출 실패: {e}")
            return self._fallback_keyword_extraction(document_content)
    
    def generate_gt_question_intelligently(
        self, 
        document_content: str, 
        keyword_result: KeywordExtractionResult
    ) -> GTQuestionResult:
        """AI 기반 지능형 GT 질문 생성"""
        try:
            print("📝 AI 기반 GT 질문 생성 중...")
            
            # 문서 내용 전처리
            cleaned_content = self._preprocess_content(document_content)
            
            # AI GT 질문 생성
            chain = self.gt_question_prompt | self.gt_question_llm
            result = chain.invoke({
                "document_content": cleaned_content,
                "primary_keywords": ", ".join(keyword_result.primary_keywords),
                "legal_domain": keyword_result.legal_domain,
                "main_topic": keyword_result.main_topic
            })
            
            print(f"✅ GT 질문 생성 완료:")
            print(f"  질문: {result.gt_question}")
            print(f"  근거: {result.reasoning}")
            print(f"  난이도: {result.difficulty_level}")
            
            return result
            
        except Exception as e:
            print(f"❌ AI GT 질문 생성 실패: {e}")
            return self._fallback_gt_question_generation(document_content, keyword_result)
    
    def _preprocess_content(self, content: str) -> str:
        """문서 내용 전처리"""
        # 과도한 공백 제거
        content = re.sub(r'\s+', ' ', content)
        
        # 특수 문자 정리
        content = re.sub(r'[^\w\s가-힣.,?!():\-]', ' ', content)
        
        # 길이 제한 (토큰 제한 고려)
        if len(content) > 2000:
            content = content[:2000] + "..."
        
        return content.strip()
    
    def _fallback_keyword_extraction(self, document_content: str) -> KeywordExtractionResult:
        """Fallback 키워드 추출"""
        print("🔧 Fallback 키워드 추출 모드")
        
        # 간단한 휴리스틱 기반 키워드 추출
        content_lower = document_content.lower()
        
        primary_keywords = []
        secondary_keywords = []
        
        # 법률 분야 판별
        if any(term in content_lower for term in ["계약", "소유권", "손해배상"]):
            legal_domain = "민사"
            primary_keywords = ["계약", "권리", "의무"]
        elif any(term in content_lower for term in ["절도", "폭행", "사기"]):
            legal_domain = "형사"
            primary_keywords = ["범죄", "구성요건", "책임"]
        elif any(term in content_lower for term in ["행정처분", "행정소송"]):
            legal_domain = "행정"
            primary_keywords = ["행정", "처분", "소송"]
        else:
            legal_domain = "일반"
            primary_keywords = ["법률", "판단", "적용"]
        
        secondary_keywords = ["판례", "법리", "해석"]
        main_topic = "법적 판단이 필요한 사안"
        
        return KeywordExtractionResult(
            primary_keywords=primary_keywords,
            secondary_keywords=secondary_keywords,
            legal_domain=legal_domain,
            main_topic=main_topic
        )
    
    def _fallback_gt_question_generation(
        self, 
        document_content: str, 
        keyword_result: KeywordExtractionResult
    ) -> GTQuestionResult:
        """Fallback GT 질문 생성"""
        print("🔧 Fallback GT 질문 생성 모드")
        
        # 법률 분야별 기본 질문 패턴
        domain_patterns = {
            "민사": "이 사건에서 당사자의 권리가 인정되는가?",
            "형사": "해당 행위가 범죄를 구성한다고 볼 수 있는가?",
            "행정": "행정처분이 적법하다고 할 수 있는가?",
            "일반": "이 사건에서 법적 판단이 타당한가?"
        }
        
        gt_question = domain_patterns.get(
            keyword_result.legal_domain, 
            "이 사건에서 법적 판단이 필요한가?"
        )
        
        return GTQuestionResult(
            gt_question=gt_question,
            reasoning="Fallback 모드로 생성된 기본 패턴 질문",
            difficulty_level="중급",
            legal_principle="일반적인 법적 판단 원리"
        )
    
    def process_document_completely(self, document_content: str) -> Tuple[KeywordExtractionResult, GTQuestionResult]:
        """문서 완전 처리: 키워드 추출 + GT 질문 생성"""
        print(f"🚀 문서 완전 처리 시작...")
        
        # 1단계: AI 기반 키워드 추출
        keyword_result = self.extract_keywords_intelligently(document_content)
        
        # 2단계: AI 기반 GT 질문 생성
        gt_question_result = self.generate_gt_question_intelligently(
            document_content, 
            keyword_result
        )
        
        print(f"✅ 문서 처리 완료!")
        return keyword_result, gt_question_result

# 노트북에서 사용할 편의 함수들
def extract_keywords_with_ai(document_content: str) -> List[str]:
    """노트북용 AI 키워드 추출 함수"""
    extractor = ImprovedKeywordExtractor()
    result = extractor.extract_keywords_intelligently(document_content)
    
    # 기존 노트북 형식에 맞게 리스트로 반환
    all_keywords = result.primary_keywords + result.secondary_keywords
    return all_keywords[:5]  # 최대 5개만 반환하여 과도한 키워드 방지

def generate_gt_question_with_ai(document_content: str) -> str:
    """노트북용 AI GT 질문 생성 함수"""
    extractor = ImprovedKeywordExtractor()
    keyword_result, gt_question_result = extractor.process_document_completely(document_content)
    
    return gt_question_result.gt_question

# 사용 예시
if __name__ == "__main__":
    # 테스트용 샘플 문서
    sample_document = """
    대법원 1982. 11. 9. 선고 80다3135 판결
    
    동업자가 채권의 준점유자에 해당하지 아니한다고 할 수 있다.
    민법 제470조에 따르면 채권의 준점유자에게 변제한 경우에도 
    변제자가 선의이고 과실이 없으면 유효한 변제가 된다.
    그러나 단순한 동업관계만으로는 채권의 준점유자로 볼 수 없다.
    """
    
    # AI 기반 처리
    extractor = ImprovedKeywordExtractor()
    keyword_result, gt_question_result = extractor.process_document_completely(sample_document)
    
    print(f"\n📋 최종 결과:")
    print(f"키워드: {keyword_result.primary_keywords}")
    print(f"GT 질문: {gt_question_result.gt_question}")
    print(f"법률 분야: {keyword_result.legal_domain}") 