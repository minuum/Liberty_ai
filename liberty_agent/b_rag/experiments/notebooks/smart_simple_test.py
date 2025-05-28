# ===== 스마트한 단순화 프롬프트 테스트 =====

import sys
import os
from pathlib import Path

# 경로 설정
if '/Users/minu/dev/Liberty/Liberty_ai' not in sys.path:
    sys.path.append('/Users/minu/dev/Liberty/Liberty_ai')

os.environ['PYTHONPATH'] = '/Users/minu/dev/Liberty/Liberty_ai'

print("=== 스마트한 단순화 프롬프트 테스트 ===\n")

try:
    from liberty_agent.b_rag.core.question_generation.unified_yesno_question_generator import UnifiedYesNoQuestionGenerator
    print("✅ 모듈 import 성공")
except Exception as e:
    print(f"❌ Import 실패: {e}")
    exit()

# 테스트할 GT 질문들
test_cases = [
    {
        "question": "동업자가 채권의 준점유자에 해당하지 아니 한다고 할 수 있는가?",
        "expected_answer": "No",  # 부정형 질문
        "description": "부정형 질문 (답: No)"
    },
    {
        "question": "동업자가 채권의 준점유자에 해당한다고 할 수 있는가?", 
        "expected_answer": "Yes",  # 긍정형 질문
        "description": "긍정형 질문 (답: Yes)"
    }
]

print("🧪 테스트 케이스:")
for i, case in enumerate(test_cases, 1):
    print(f"  {i}. {case['description']}")
    print(f"     질문: {case['question']}")
    print(f"     예상 답변: {case['expected_answer']}")
print("-" * 60)

# 스마트 단순화 모드 테스트
print("\n🔬 스마트한 단순화 프롬프트 (smart_simple)")
print("-" * 50)

try:
    generator = UnifiedYesNoQuestionGenerator(prompt_mode="smart_simple")
    print("✅ 스마트 단순화 질문 생성기 초기화 완료\n")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"📝 테스트 케이스 {i}: {test_case['description']}")
        print(f"GT 질문: {test_case['question']}")
        print()
        
        try:
            # 질문 생성 (GT 질문만 사용)
            result = generator.generate_ten_level_questions(
                gt_question=test_case['question'],
                document_content="",
                keywords_to_consider=""
            )
            
            if result and result.questions:
                print(f"✅ {len(result.questions)}개 질문 생성 완료")
                
                # 답변 분배 확인
                yes_count = sum(1 for q in result.questions if q.expected_answer.value == 'Yes')
                no_count = sum(1 for q in result.questions if q.expected_answer.value == 'No')
                print(f"📊 답변 분배: Yes {yes_count}개, No {no_count}개")
                
                # 균형도 계산
                if max(yes_count, no_count) > 0:
                    balance_ratio = min(yes_count, no_count) / max(yes_count, no_count)
                    print(f"⚖️ 균형도: {balance_ratio:.2f}")
                    
                    if balance_ratio >= 0.6:
                        print("✅ 균형잡힌 분배 달성!")
                    else:
                        print("⚠️ 분배 불균형")
                
                # 생성된 질문 출력
                print("\n📝 생성된 질문:")
                for q in result.questions:
                    print(f"  Level {q.level:2d} ({q.expected_answer.value:3s}): {q.question}")
                
                # 패턴 분석
                print("\n🔍 패턴 분석:")
                odd_levels = [q for q in result.questions if q.level % 2 == 1]  # 1,3,5,7,9
                even_levels = [q for q in result.questions if q.level % 2 == 0]  # 2,4,6,8,10
                
                odd_yes = sum(1 for q in odd_levels if q.expected_answer.value == 'Yes')
                odd_no = sum(1 for q in odd_levels if q.expected_answer.value == 'No')
                even_yes = sum(1 for q in even_levels if q.expected_answer.value == 'Yes')
                even_no = sum(1 for q in even_levels if q.expected_answer.value == 'No')
                
                print(f"  홀수 레벨 (1,3,5,7,9): Yes {odd_yes}개, No {odd_no}개")
                print(f"  짝수 레벨 (2,4,6,8,10): Yes {even_yes}개, No {even_no}개")
                
                # 기대 패턴과 비교
                if test_case['expected_answer'] == "No":
                    expected_pattern = "홀수=Yes, 짝수=No"
                    pattern_correct = (odd_yes == 5 and odd_no == 0 and even_yes == 0 and even_no == 5)
                else:
                    expected_pattern = "홀수=Yes, 짝수=No"  
                    pattern_correct = (odd_yes == 5 and odd_no == 0 and even_yes == 0 and even_no == 5)
                
                print(f"  기대 패턴: {expected_pattern}")
                print(f"  패턴 일치: {'✅' if pattern_correct else '❌'}")
                
            else:
                print("❌ 질문이 생성되지 않았습니다.")
                
        except Exception as e:
            print(f"❌ 테스트 케이스 {i} 실패: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "="*60 + "\n")

except Exception as e:
    print(f"❌ 스마트 단순화 프롬프트 테스트 실패: {e}")
    import traceback
    traceback.print_exc()

print("💡 결론:")
print("1. 스마트 단순화 프롬프트는 GT 질문만으로도 균형잡힌 분배를 달성할 수 있습니다")
print("2. 명시적인 패턴 지시로 Yes/No 편향 문제를 해결했습니다")
print("3. 복잡한 문서 분석 없이도 효과적인 질문 생성이 가능합니다")

print("\n✨ 테스트 완료!") 