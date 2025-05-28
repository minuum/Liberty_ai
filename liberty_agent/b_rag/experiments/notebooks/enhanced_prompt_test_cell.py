# ===== 개선된 B-RAG 프롬프트 테스트 셀 =====
# 노트북에서 이 코드를 복사하여 새 셀에 붙여넣고 실행하세요

import sys
import os
from pathlib import Path

# 경로 설정
if '/Users/minu/dev/Liberty/Liberty_ai' not in sys.path:
    sys.path.append('/Users/minu/dev/Liberty/Liberty_ai')

# 환경변수 설정 (필요시)
os.environ['PYTHONPATH'] = '/Users/minu/dev/Liberty/Liberty_ai'

print("=== 개선된 B-RAG 프롬프트 테스트 ===\n")

try:
    from liberty_agent.b_rag.core.question_generation.unified_yesno_question_generator import UnifiedYesNoQuestionGenerator
    print("✅ 모듈 import 성공")
except Exception as e:
    print(f"❌ Import 실패: {e}")
    print("경로를 확인하고 다시 시도하세요.")

# 질문 생성기 초기화
try:
    generator = UnifiedYesNoQuestionGenerator()
    print("✅ 질문 생성기 초기화 완료\n")
except Exception as e:
    print(f"❌ 초기화 실패: {e}")

# 테스트 케이스 1: 동업자 채권 준점유자
print("🧪 테스트 케이스 1: 동업자 채권 준점유자")
print("-" * 50)

test_gt = "동업자가 채권의 준점유자에 해당하지 아니 한다고 할 수 있는가?"
test_doc = """
동업관계에서 채권의 준점유자 인정에 관한 판례입니다. 
법원은 동업자의 지위와 채권 준점유자의 요건을 검토하였습니다.
민법 제470조에 따르면 채권의 준점유자는 특정 요건을 충족해야 합니다.
본 사안에서 동업자는 이러한 요건을 충족하지 못한다고 판단되었습니다.
따라서 동업자가 채권의 준점유자에 해당하지 않는다고 보는 것이 타당합니다.
"""
test_keywords = "동업자, 채권, 준점유자, 민법 제470조"

print(f"GT 질문: {test_gt}")
print(f"키워드: {test_keywords}")
print()

try:
    # 질문 생성 실행
    print("🔄 질문 생성 중...")
    result = generator.generate_ten_level_questions(
        gt_question=test_gt,
        document_content=test_doc,
        keywords_to_consider=test_keywords
    )
    
    # 결과 분석
    questions = result.questions if hasattr(result, 'questions') else []
    
    if questions:
        print(f"✅ {len(questions)}개 질문 생성 완료")
        
        # Yes/No 분배 확인
        yes_count = sum(1 for q in questions if q.expected_answer == 'Yes')
        no_count = sum(1 for q in questions if q.expected_answer == 'No')
        
        print(f"📊 답변 분배: Yes {yes_count}개, No {no_count}개")
        
        # 균형도 계산
        if max(yes_count, no_count) > 0:
            balance_ratio = min(yes_count, no_count) / max(yes_count, no_count)
            print(f"⚖️ 균형도: {balance_ratio:.2f} (1.0이 완벽한 균형)")
            
            if balance_ratio >= 0.6:
                print("✅ 균형잡힌 분배 달성!")
            else:
                print("⚠️ 분배 불균형 - 프롬프트 추가 조정 필요")
        
        # 샘플 질문 출력
        print("\n📝 생성된 질문 샘플:")
        for i, q in enumerate(questions[:6]):  # 처음 6개
            print(f"  Level {q.level}: {q.question}")
            print(f"    → 답변: {q.expected_answer}, 확신도: {q.confidence:.2f}")
            
            # 새로운 필드들 확인
            if hasattr(q, 'legal_perspective'):
                print(f"    → 관점: {q.legal_perspective}")
            if hasattr(q, 'question_type'):
                print(f"    → 유형: {q.question_type}")
            if hasattr(q, 'factual_score'):
                print(f"    → 사실성: {q.factual_score:.2f}, 유용성: {q.utility_score:.2f}")
            print()
        
        # 전체 질문 리스트
        print("📋 전체 질문 목록:")
        for q in questions:
            print(f"Level {q.level:2d} ({q.expected_answer:3s}): {q.question}")
        
        # 개선 효과 분석
        print("\n🎯 개선 효과 분석:")
        
        # 1. 답변 다양성
        unique_answers = set(q.expected_answer for q in questions)
        print(f"- 답변 다양성: {len(unique_answers)}가지 답변 유형")
        
        # 2. 확신도 분포
        confidences = [q.confidence for q in questions]
        avg_confidence = sum(confidences) / len(confidences)
        print(f"- 평균 확신도: {avg_confidence:.3f}")
        
        # 3. 레벨별 분포
        level_distribution = {}
        for q in questions:
            level_distribution[q.level] = level_distribution.get(q.level, 0) + 1
        print(f"- 레벨 분포: {dict(sorted(level_distribution.items()))}")
        
        # 4. 일관성 체크
        if hasattr(result, 'get_consistency_rate'):
            consistency = result.get_consistency_rate()
            print(f"- 일관성 비율: {consistency:.2%}")
        
    else:
        print("❌ 질문이 생성되지 않았습니다.")
        
except Exception as e:
    print(f"❌ 테스트 실패: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)

# 개선 사항 요약
print("\n💡 프롬프트 개선 사항 요약:")
improvements = [
    "✅ 의미적 동등성 → 의미적 관련성으로 완화",
    "✅ 균형잡힌 Yes/No 분배 목표 (4-6개씩)",
    "✅ FIT-RAG 기반 이중 평가 (사실성 + 유용성)",
    "✅ SummRAG 논리적 단계별 처리 도입",
    "✅ Know When to Fuse 적응적 접근 적용",
    "✅ 다양한 법적 관점 포함 (원고/피고/법원/일반)",
    "✅ 품질 검증 체크리스트 추가",
    "✅ 레벨별 특성화 강화"
]

for improvement in improvements:
    print(improvement)

print("\n🚀 다음 단계:")
print("1. 더 많은 테스트 케이스로 검증")
print("2. 실제 B-RAG 실험에 적용")
print("3. 성능 지표 개선 확인")
print("4. 하이브리드 검색 시스템 통합")

print("\n✨ 테스트 완료!") 