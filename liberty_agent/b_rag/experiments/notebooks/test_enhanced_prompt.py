#!/usr/bin/env python3
"""
개선된 B-RAG 프롬프트 테스트 스크립트
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

import json
from liberty_agent.b_rag.core.question_generation.unified_yesno_question_generator import UnifiedYesNoQuestionGenerator

def test_enhanced_prompt():
    """개선된 프롬프트 테스트"""
    print("=== 개선된 B-RAG 프롬프트 테스트 ===\n")
    
    # 질문 생성기 초기화
    try:
        generator = UnifiedYesNoQuestionGenerator()
        print("✅ 질문 생성기 초기화 완료\n")
    except Exception as e:
        print(f"❌ 질문 생성기 초기화 실패: {e}")
        return
    
    # 테스트 데이터
    test_cases = [
        {
            "gt_question": "동업자가 채권의 준점유자에 해당하지 아니 한다고 할 수 있는가?",
            "document": """
            동업관계에서 채권의 준점유자 인정에 관한 판례입니다. 
            법원은 동업자의 지위와 채권 준점유자의 요건을 검토하였습니다.
            민법 제470조에 따르면 채권의 준점유자는 특정 요건을 충족해야 합니다.
            본 사안에서 동업자는 이러한 요건을 충족하지 못한다고 판단되었습니다.
            """,
            "keywords": "동업자, 채권, 준점유자, 민법 제470조"
        },
        {
            "gt_question": "계약 해지 시 손해배상 청구가 가능한가?",
            "document": """
            계약 해지와 손해배상에 관한 판례입니다.
            계약 당사자 일방이 계약을 해지한 경우의 손해배상 책임을 다룹니다.
            민법상 계약 해지권 행사와 손해배상 청구권은 별개의 권리입니다.
            법원은 해지권 행사가 손해배상 청구를 배제하지 않는다고 판시하였습니다.
            """,
            "keywords": "계약 해지, 손해배상, 청구권"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"🧪 테스트 케이스 {i}")
        print(f"GT 질문: {test_case['gt_question']}")
        print(f"키워드: {test_case['keywords']}")
        print("-" * 50)
        
        try:
            # 질문 생성
            result = generator.generate_ten_level_questions(
                gt_question=test_case['gt_question'],
                document_content=test_case['document'],
                keywords_to_consider=test_case['keywords']
            )
            
            # 결과 분석
            questions = result.questions if hasattr(result, 'questions') else []
            
            if not questions:
                print("❌ 질문이 생성되지 않았습니다.")
                continue
                
            print(f"✅ {len(questions)}개 질문 생성 완료")
            
            # Yes/No 분배 확인
            yes_count = sum(1 for q in questions if q.expected_answer == 'Yes')
            no_count = sum(1 for q in questions if q.expected_answer == 'No')
            
            print(f"📊 답변 분배: Yes {yes_count}개, No {no_count}개")
            
            # 균형도 평가
            balance_ratio = min(yes_count, no_count) / max(yes_count, no_count) if max(yes_count, no_count) > 0 else 0
            print(f"⚖️ 균형도: {balance_ratio:.2f} (1.0이 완벽한 균형)")
            
            # 샘플 질문 출력
            print("\n📝 샘플 질문들:")
            for j, q in enumerate(questions[:5]):  # 처음 5개만
                print(f"  Level {q.level}: {q.question}")
                print(f"    → 답변: {q.expected_answer}, 확신도: {q.confidence:.2f}")
                if hasattr(q, 'legal_perspective'):
                    print(f"    → 관점: {q.legal_perspective}")
                print()
            
            # 일관성 체크
            if hasattr(result, 'get_consistency_rate'):
                consistency = result.get_consistency_rate()
                print(f"🎯 일관성 비율: {consistency:.2%}")
            
            print("=" * 60)
            print()
            
        except Exception as e:
            print(f"❌ 테스트 실패: {e}")
            import traceback
            traceback.print_exc()
            print("=" * 60)
            print()

def analyze_prompt_improvements():
    """프롬프트 개선 효과 분석"""
    print("=== 프롬프트 개선 효과 분석 ===\n")
    
    improvements = [
        "✅ 의미적 동등성 → 의미적 관련성으로 완화",
        "✅ 균형잡힌 Yes/No 분배 (4-6개씩)",
        "✅ FIT-RAG 기반 이중 평가 (사실성 + 유용성)",
        "✅ SummRAG 논리적 단계별 처리",
        "✅ Know When to Fuse 적응적 접근",
        "✅ 다양한 법적 관점 포함",
        "✅ 품질 검증 체크리스트 추가"
    ]
    
    for improvement in improvements:
        print(improvement)
    
    print("\n🎯 기대 효과:")
    print("- Yes/No 답변 편향 해결")
    print("- 질문 다양성 증가")
    print("- 실무 활용성 향상")
    print("- 시스템 강인성 개선")

if __name__ == "__main__":
    analyze_prompt_improvements()
    print()
    test_enhanced_prompt() 