#!/usr/bin/env python3
"""
의미론적 동일성 문제 직접 테스트
- 모듈 reload 확인
- Fallback 질문 검증
"""

import sys
import os
import importlib
from pathlib import Path

# 경로 설정
current_dir = Path(__file__).parent
sys.path.append(str(current_dir / "../../../"))

# 환경변수 설정
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

def test_semantic_equivalence():
    """의미론적 동일성 테스트"""
    print("🎯 의미론적 동일성 테스트 시작")
    
    try:
        # 모듈 import (강제 reload)
        print("📦 질문 생성기 모듈 로드 중...")
        
        # 기존 모듈이 있으면 reload
        module_name = 'liberty_agent.b_rag.core.question_generation.unified_yesno_question_generator'
        if module_name in sys.modules:
            print("🔄 기존 모듈 reload 중...")
            module = sys.modules[module_name]
            importlib.reload(module)
        else:
            module = importlib.import_module(module_name)
        
        UnifiedYesNoQuestionGenerator = module.UnifiedYesNoQuestionGenerator
        print("✅ 질문 생성기 로드 완료")
        
        # 생성기 초기화
        generator = UnifiedYesNoQuestionGenerator(
            model_name="gpt-4o-2024-08-06",
            temperature=0.0
        )
        
        # 테스트 케이스
        gt_question = "동업자가 채권의 준점유자에 해당하지 아니 한다고 할 수 있는가?"
        document_content = """
        민법 제470조는 채권의 준점유자에 대한 변제는 변제자가 선의이며 과실없는 때에 한하여 효력이 있다고 규정하고 있다.
        그러나 단순한 동업관계만으로는 채권의 준점유자로 볼 수 없다.
        """
        
        print(f"📝 GT 질문: {gt_question}")
        print(f"📄 문서: {document_content.strip()[:50]}...")
        
        # 질문 생성 실행
        print("\n🔄 질문 생성 시작...")
        result = generator.generate_ten_level_questions(
            gt_question=gt_question,
            document_content=document_content,
            keywords_to_consider="동업자, 채권의 준점유자, 민법 제470조"
        )
        
        # 결과 분석
        if result and result.questions:
            print(f"\n✅ 질문 생성 완료!")
            print(f"   생성된 질문: {len(result.questions)}개")
            print(f"   일관성: {result.get_consistency_rate():.1%}")
            
            # 전체 질문 출력 및 의미론적 동일성 검증
            print(f"\n📋 생성된 질문 목록:")
            
            semantic_score = 0
            for i, q in enumerate(result.questions, 1):
                question_text = q.question
                
                # 간단한 의미론적 동일성 체크
                gt_keywords = ["동업자", "채권", "준점유자", "해당"]
                preserved_keywords = sum(1 for kw in gt_keywords 
                                       if kw in question_text or 
                                          any(alt in question_text for alt in get_alternatives(kw)))
                
                semantic_valid = preserved_keywords >= 2  # 최소 2개 키워드 보존
                if semantic_valid:
                    semantic_score += 1
                
                status = "✅" if semantic_valid else "❌"
                print(f"  {i:2d}. {status} Level {q.level} ({q.expected_answer.value}): {question_text}")
                if not semantic_valid:
                    print(f"      ⚠️ 의미 변질: 핵심 키워드 {preserved_keywords}/{len(gt_keywords)}개만 보존")
            
            semantic_rate = semantic_score / len(result.questions)
            print(f"\n📊 의미론적 동일성 점수: {semantic_rate:.1%} ({semantic_score}/{len(result.questions)})")
            
            if semantic_rate < 0.7:
                print("❌ 의미론적 동일성 실패: 대부분 질문이 GT와 다른 의미")
                print("🔧 Fallback 시스템 검증 필요")
            else:
                print("✅ 의미론적 동일성 성공: 대부분 질문이 GT와 동일한 의미")
                
        else:
            print("❌ 질문 생성 완전 실패")
            
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()

def get_alternatives(keyword):
    """키워드의 대안 표현들"""
    alternatives = {
        "동업자": ["같이 사업하는", "함께 일하는", "파트너", "동업"],
        "채권": ["돈을 받을 권리", "권리", "빚"],
        "준점유자": ["권리가 있는 것처럼", "점유자", "권리자"],
        "해당": ["맞", "적용", "인정"]
    }
    return alternatives.get(keyword, [])

if __name__ == "__main__":
    test_semantic_equivalence() 