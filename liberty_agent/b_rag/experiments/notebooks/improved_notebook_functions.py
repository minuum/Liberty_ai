#!/usr/bin/env python3
"""
노트북용 개선된 함수들
- 기존 하드코딩된 함수들을 AI 기반으로 대체
- 노트북에서 간편하게 import해서 사용 가능
"""

import os
import sys
from pathlib import Path

# 경로 설정 (노트북에서 import할 때 필요)
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

try:
    from improved_keyword_extraction import extract_keywords_with_ai, generate_gt_question_with_ai
    AI_AVAILABLE = True
    print("✅ AI 기반 키워드 추출 및 GT 질문 생성 사용 가능")
except ImportError as e:
    AI_AVAILABLE = False
    print(f"⚠️ AI 기반 함수 import 실패: {e}")
    print("기존 하드코딩 방식을 fallback으로 사용합니다.")

def extract_keywords_from_content_improved(content: str) -> list:
    """
    개선된 키워드 추출 함수
    - AI 사용 가능하면 AI 기반 추출
    - 불가능하면 기존 하드코딩 방식 사용
    """
    if AI_AVAILABLE:
        try:
            return extract_keywords_with_ai(content)
        except Exception as e:
            print(f"⚠️ AI 키워드 추출 실패, fallback 사용: {e}")
    
    # Fallback: 기존 하드코딩 방식 (하지만 개선됨)
    legal_keywords = [
        "채권", "준점유자", "계약", "해제", "취소", "소유권", "손해배상", "시효", 
        "효력", "책임", "민법", "판례", "무효", "불법행위", "과실", "고의",
        "청구권", "소멸", "완성", "성립", "인정", "변제", "이행", "의무",
        "권리", "법률행위", "의사표시", "합의", "당사자", "법원", "판결"
    ]
    
    keywords = []
    for keyword in legal_keywords:
        if keyword in content:
            keywords.append(keyword)
    
    # 최대 5개로 제한하여 과도한 키워드 방지
    return keywords[:5]

def generate_gt_question_from_content_improved(content: str) -> str:
    """
    개선된 GT 질문 생성 함수
    - AI 사용 가능하면 AI 기반 생성
    - 불가능하면 기존 규칙 기반 방식 사용
    """
    if AI_AVAILABLE:
        try:
            return generate_gt_question_with_ai(content)
        except Exception as e:
            print(f"⚠️ AI GT 질문 생성 실패, fallback 사용: {e}")
    
    # Fallback: 기존 규칙 기반 방식
    content_lower = content.lower()
    
    # 키워드 기반 질문 생성
    if "채권" in content and "준점유" in content:
        return "동업자가 채권의 준점유자에 해당하지 아니 한다고 할 수 있는가?"
    elif "계약" in content and ("해제" in content or "취소" in content):
        return "계약의 해제권이 발생하는가?"
    elif "소유권" in content and "이전" in content:
        return "소유권 이전의 효력이 인정되는가?"
    elif "손해배상" in content:
        return "손해배상의 범위가 제한되는가?"
    elif "시효" in content and ("완성" in content or "소멸" in content):
        return "시효가 완성되었다고 볼 수 있는가?"
    elif "무효" in content:
        return "해당 법률행위가 무효라고 할 수 있는가?"
    elif "취소" in content:
        return "취소권을 행사할 수 있는가?"
    elif "불법행위" in content:
        return "불법행위 책임이 성립하는가?"
    elif "과실" in content:
        return "과실이 인정된다고 볼 수 있는가?"
    elif "효력" in content:
        return "해당 법률 행위의 효력이 인정되는가?"
    elif "책임" in content:
        return "법적 책임이 발생한다고 볼 수 있는가?"
    elif "청구" in content:
        return "이 사건에서 청구가 인정될 수 있는가?"
    elif "판결" in content or "판례" in content:
        return "이 판례의 법리가 적용될 수 있는가?"
    else:
        return "이 사건에서 당사자의 주장이 인정될 수 있는가?"

# 노트북에서 간편하게 사용할 수 있는 alias
extract_keywords_from_content = extract_keywords_from_content_improved
generate_gt_question_from_content = generate_gt_question_from_content_improved

# 사용 예시
if __name__ == "__main__":
    # 테스트
    sample_content = """
    대법원 1982. 11. 9. 선고 80다3135 판결
    
    동업자가 채권의 준점유자에 해당하지 아니한다고 할 수 있다.
    민법 제470조에 따르면 채권의 준점유자에게 변제한 경우에도 
    변제자가 선의이고 과실이 없으면 유효한 변제가 된다.
    그러나 단순한 동업관계만으로는 채권의 준점유자로 볼 수 없다.
    """
    
    print("🔍 키워드 추출 테스트:")
    keywords = extract_keywords_from_content(sample_content)
    print(f"추출된 키워드: {keywords}")
    
    print("\n📝 GT 질문 생성 테스트:")
    gt_question = generate_gt_question_from_content(sample_content)
    print(f"생성된 GT 질문: {gt_question}") 