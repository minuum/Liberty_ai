# ===== 단순화된 프롬프트 테스트 =====
# 노트북에서 이 코드를 복사하여 새 셀에 붙여넣고 실행하세요

import sys
import os
from pathlib import Path

# 경로 설정
if '/Users/minu/dev/Liberty/Liberty_ai' not in sys.path:
    sys.path.append('/Users/minu/dev/Liberty/Liberty_ai')

# 환경변수 설정 (필요시)
os.environ['PYTHONPATH'] = '/Users/minu/dev/Liberty/Liberty_ai'

print("=== 단순화된 프롬프트 테스트 ===\n")

try:
    from liberty_agent.b_rag.core.question_generation.unified_yesno_question_generator import UnifiedYesNoQuestionGenerator
    print("✅ 모듈 import 성공")
except Exception as e:
    print(f"❌ Import 실패: {e}")
    print("경로를 확인하고 다시 시도하세요.")

# 테스트할 GT 질문
test_gt = "동업자가 채권의 준점유자에 해당하지 아니 한다고 할 수 있는가?"

print(f"🧪 테스트 GT 질문: {test_gt}")
print("-" * 60)

# 방법 1: 단순화된 프롬프트 테스트
print("\n🔬 방법 1: 단순화된 프롬프트 (simple)")
print("-" * 40)

try:
    generator_simple = UnifiedYesNoQuestionGenerator(prompt_mode="simple")
    print("✅ 단순화된 질문 생성기 초기화 완료")
    
    # 질문 생성 (문서 내용 없이 GT 질문만 사용)
    result_simple = generator_simple.generate_ten_level_questions(
        gt_question=test_gt,
        document_content="",  # 빈 문서
        keywords_to_consider=""  # 빈 키워드
    )
    
    if result_simple and result_simple.questions:
        print(f"✅ {len(result_simple.questions)}개 질문 생성 완료")
        
        # 답변 분배 확인
        yes_count = sum(1 for q in result_simple.questions if q.expected_answer.value == 'Yes')
        no_count = sum(1 for q in result_simple.questions if q.expected_answer.value == 'No')
        print(f"📊 답변 분배: Yes {yes_count}개, No {no_count}개")
        
        # 샘플 질문 출력
        print("\n📝 생성된 질문 (처음 5개):")
        for i, q in enumerate(result_simple.questions[:5]):
            print(f"  Level {q.level}: {q.question}")
            print(f"    → 답변: {q.expected_answer.value}")
        
    else:
        print("❌ 질문이 생성되지 않았습니다.")
        
except Exception as e:
    print(f"❌ 단순화된 프롬프트 테스트 실패: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)

# 방법 2: 최소한의 프롬프트 테스트
print("\n🔬 방법 2: 최소한의 프롬프트 (minimal)")
print("-" * 40)

try:
    generator_minimal = UnifiedYesNoQuestionGenerator(prompt_mode="minimal")
    print("✅ 최소한의 질문 생성기 초기화 완료")
    
    # 질문 생성 (GT 질문만 사용)
    result_minimal = generator_minimal.generate_ten_level_questions(
        gt_question=test_gt,
        document_content="",  # 빈 문서
        keywords_to_consider=""  # 빈 키워드
    )
    
    if result_minimal and result_minimal.questions:
        print(f"✅ {len(result_minimal.questions)}개 질문 생성 완료")
        
        # 답변 분배 확인
        yes_count = sum(1 for q in result_minimal.questions if q.expected_answer.value == 'Yes')
        no_count = sum(1 for q in result_minimal.questions if q.expected_answer.value == 'No')
        print(f"📊 답변 분배: Yes {yes_count}개, No {no_count}개")
        
        # 샘플 질문 출력
        print("\n📝 생성된 질문 (처음 5개):")
        for i, q in enumerate(result_minimal.questions[:5]):
            print(f"  Level {q.level}: {q.question}")
            print(f"    → 답변: {q.expected_answer.value}")
        
    else:
        print("❌ 질문이 생성되지 않았습니다.")
        
except Exception as e:
    print(f"❌ 최소한의 프롬프트 테스트 실패: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)

# 결과 비교 및 추천
print("\n💡 프롬프트 비교 결과:")
print("1. 단순화된 프롬프트 (simple): GT 질문 + 기본 지침")
print("2. 최소한의 프롬프트 (minimal): GT 질문만 + 최소 지침")

print("\n🎯 추천:")
print("- 빠른 테스트: minimal 모드 사용")
print("- 균형잡힌 품질: simple 모드 사용")

print("\n✨ 테스트 완료!")

if __name__ == "__main__":
    # 스크립트로 실행할 때만 실행
    pass 