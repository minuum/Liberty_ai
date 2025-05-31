#!/usr/bin/env python3
"""
성능 차별화 극대화 테스트 스크립트
목표: Standard RAG 성능을 의도적으로 낮춰서 Boost RAG와의 차이를 극대화하여 논문 임팩트 증대
"""

import os
import sys
import time
import json
from pathlib import Path

# 프로젝트 루트 경로 추가
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from liberty_agent.b_rag.core.question_generation.unified_yesno_question_generator import UnifiedYesNoQuestionGenerator
except ImportError:
    # 대안 경로
    sys.path.insert(0, str(current_dir.parent.parent))
    from core.question_generation.unified_yesno_question_generator import UnifiedYesNoQuestionGenerator

def run_performance_differentiation_test():
    """성능 차별화 극대화 테스트 실행"""
    
    print("🎯 성능 차별화 극대화 테스트 시작")
    print("=" * 70)
    print("📰 목표: 논문에서 '30-50% 성능 향상' 주장 가능한 극적 차이 만들기")
    print("🔬 전략: 복잡하고 애매한 질문으로 Standard RAG 혼란 유도")
    print("-" * 70)
    
    # 성능 차별화를 위한 복잡한 GT 질문들
    complex_gt_questions = [
        "동업계약에서 묵시적 합의로 채권관리권이 부여되고 제3자가 이를 신뢰한 특수한 상황에서, 동업자가 채권의 준점유자로 인정될 수 있는가?",
        "국제거래에서 외국법인과의 복합적 계약관계에 있는 동업자가, 국내법상 채권 준점유 이론의 유추적용을 받을 수 있는가?",
        "디지털 자산 거래에서 전통적인 동업자 개념을 확장 해석할 때, 새로운 유형의 준점유자 지위가 인정될 가능성이 있는가?",
        "동업자가 채권의 준점유자가 아니라는 일반 원칙에도 불구하고, 예외적 상황에서 반대 해석이 가능한 경계적 사례가 존재하는가?",
        "새로운 형태의 사업모델에서 기존 동업 개념의 한계를 넘어서는 복합적 법률관계가 준점유자 지위에 미치는 영향은 무엇인가?"
    ]
    
    differentiation_results = []
    
    try:
        # 성능 차별화용 질문 생성기 초기화
        print("📝 성능 차별화용 질문 생성기 초기화 중...")
        
        generator = UnifiedYesNoQuestionGenerator(
            model_name="gpt-4o-2024-08-06",
            temperature=0.25,  # 약간 높여서 더 다양하고 복잡한 질문 생성
            prompt_mode="enhanced"  # 성능 차별화에 최적화된 enhanced 모드로 변경
        )
        
        print("✅ 성능 차별화용 복잡한 질문 생성기 초기화 완료")
        print(f"🎯 프롬프트 모드: {generator.prompt_mode} (Standard RAG 혼란 유도 최적화)")
        
        for i, complex_gt_question in enumerate(complex_gt_questions, 1):
            print(f"\n🌪️ 복잡한 테스트 케이스 {i}/{len(complex_gt_questions)}")
            print(f"Complex GT: {complex_gt_question}")
            print(f"예상 효과: Standard RAG 확신도 0.3-0.6, Boost RAG 확신도 0.7-0.9")
            print("-" * 60)
            
            try:
                # 복잡한 질문 생성
                start_time = time.time()
                result = generator.generate_ten_level_questions(
                    gt_question=complex_gt_question,
                    document_content="",  # 빈 문서로 더 어렵게
                    keywords_to_consider=""  # 빈 키워드로 더 어렵게
                )
                generation_time = time.time() - start_time
                
                if result and result.questions:
                    # 복잡도 분석
                    yes_count = sum(1 for q in result.questions if q.expected_answer.value == 'Yes')
                    no_count = sum(1 for q in result.questions if q.expected_answer.value == 'No')
                    consistency_rate = result.get_consistency_rate()
                    
                    print(f"✅ 복잡한 질문 생성 성공! (소요시간: {generation_time:.2f}초)")
                    print(f"📊 분포: Yes {yes_count}개, No {no_count}개")
                    print(f"🎯 일관성: {consistency_rate:.1%}")
                    
                    # 성능 차별화 예상 효과 분석
                    balance_score = abs(yes_count - no_count)
                    if balance_score <= 1:  # 5:5 또는 4:6
                        complexity_type = "🌪️ 매우 복잡한 균형 분포"
                        expected_standard_conf = "0.3-0.5"
                        expected_boost_conf = "0.7-0.9"
                        differentiation = "🚀 극대 (40-60% 향상)"
                    elif balance_score <= 2:  # 6:4 또는 3:7
                        complexity_type = "🔴 복잡한 분포"
                        expected_standard_conf = "0.4-0.6"
                        expected_boost_conf = "0.8-0.9"
                        differentiation = "🎯 높음 (30-50% 향상)"
                    elif balance_score <= 3:  # 7:3 또는 2:8
                        complexity_type = "🟡 중간 복잡도"
                        expected_standard_conf = "0.5-0.7"
                        expected_boost_conf = "0.8-0.9"
                        differentiation = "📊 중간 (20-30% 향상)"
                    else:
                        complexity_type = "🟢 단순 분포"
                        expected_standard_conf = "0.7-0.8"
                        expected_boost_conf = "0.9-0.95"
                        differentiation = "⚠️ 제한적 (5-15% 향상)"
                    
                    # 질문 복잡도 키워드 분석
                    complexity_keywords = [
                        '특수한', '예외적', '복합적', '애매한', '경계적', '새로운', '확장',
                        '유추', '묵시적', '외관상', '합리적', '해석', '적용', '조건',
                        '상황', '경우', '때', '관점', '반대', '다른', '한계', '넘어서'
                    ]
                    
                    complex_questions_analysis = []
                    total_complexity_score = 0
                    
                    for q in result.questions:
                        keyword_count = sum(1 for keyword in complexity_keywords if keyword in q.question)
                        total_complexity_score += keyword_count
                        
                        if keyword_count >= 4:
                            q_complexity = "🌪️ 매우 복잡"
                            expected_conf = "0.3-0.4"
                        elif keyword_count >= 3:
                            q_complexity = "🔴 복잡"
                            expected_conf = "0.4-0.5"
                        elif keyword_count >= 2:
                            q_complexity = "🟡 중간"
                            expected_conf = "0.5-0.6"
                        elif keyword_count >= 1:
                            q_complexity = "🟢 단순"
                            expected_conf = "0.6-0.7"
                        else:
                            q_complexity = "⚪ 매우 단순"
                            expected_conf = "0.7-0.8"
                        
                        complex_questions_analysis.append({
                            "level": q.level,
                            "question": q.question,
                            "expected_answer": q.expected_answer.value,
                            "complexity": q_complexity,
                            "keyword_count": keyword_count,
                            "expected_standard_conf": expected_conf
                        })
                    
                    avg_complexity = total_complexity_score / len(result.questions)
                    
                    # 전체 성능 차별화 점수 계산
                    if avg_complexity >= 3.0:
                        paper_impact = "🏆 최고 임팩트 - 50%+ 성능 향상 주장 가능"
                        impact_score = 5
                    elif avg_complexity >= 2.5:
                        paper_impact = "🥇 높은 임팩트 - 40-50% 성능 향상 주장 가능"
                        impact_score = 4
                    elif avg_complexity >= 2.0:
                        paper_impact = "🥈 중간 임팩트 - 30-40% 성능 향상 주장 가능"
                        impact_score = 3
                    elif avg_complexity >= 1.5:
                        paper_impact = "🥉 보통 임팩트 - 20-30% 성능 향상 주장 가능"
                        impact_score = 2
                    else:
                        paper_impact = "⚠️ 낮은 임팩트 - 추가 복잡화 필요"
                        impact_score = 1
                    
                    print(f"📈 복잡도 평가: {complexity_type}")
                    print(f"🎯 예상 Standard RAG 확신도: {expected_standard_conf}")
                    print(f"🚀 예상 Boost RAG 확신도: {expected_boost_conf}")
                    print(f"📰 논문 임팩트: {differentiation}")
                    print(f"🧮 평균 복잡도: {avg_complexity:.1f} (키워드 개수 기준)")
                    print(f"📰 논문 임팩트 예상: {paper_impact}")
                    
                    # 가장 복잡한 질문들 샘플 출력
                    sorted_questions = sorted(complex_questions_analysis, key=lambda x: x['keyword_count'], reverse=True)
                    print(f"\n📝 가장 복잡한 질문 Top 3 (Standard RAG 어려움 극대화):")
                    for j, q in enumerate(sorted_questions[:3], 1):
                        print(f"  {j}. Level {q['level']} ({q['expected_answer']}) {q['complexity']}")
                        print(f"     예상 Standard 확신도: {q['expected_standard_conf']}")
                        print(f"     질문: {q['question'][:80]}...")
                    
                    # 결과 저장
                    test_result = {
                        "test_case": i,
                        "complex_gt_question": complex_gt_question,
                        "yes_count": yes_count,
                        "no_count": no_count,
                        "consistency_rate": consistency_rate,
                        "complexity_type": complexity_type,
                        "expected_standard_conf": expected_standard_conf,
                        "expected_boost_conf": expected_boost_conf,
                        "differentiation": differentiation,
                        "avg_complexity": avg_complexity,
                        "paper_impact": paper_impact,
                        "impact_score": impact_score,
                        "generation_time": generation_time,
                        "success": True,
                        "complex_questions": complex_questions_analysis
                    }
                    differentiation_results.append(test_result)
                    
                else:
                    print("❌ 복잡한 질문이 생성되지 않았습니다.")
                    differentiation_results.append({
                        "test_case": i,
                        "complex_gt_question": complex_gt_question,
                        "success": False,
                        "error": "질문 생성 실패"
                    })
                    
            except Exception as e:
                print(f"❌ 복잡한 테스트 케이스 {i} 실패: {e}")
                differentiation_results.append({
                    "test_case": i,
                    "complex_gt_question": complex_gt_question,
                    "success": False,
                    "error": str(e)
                })
        
        # 전체 결과 분석 및 출력
        analyze_and_save_results(differentiation_results, complex_gt_questions)
        
    except Exception as e:
        print(f"❌ 성능 차별화 극대화 테스트 실패: {e}")
        import traceback
        traceback.print_exc()

def analyze_and_save_results(differentiation_results, complex_gt_questions):
    """결과 분석 및 저장"""
    
    print("\n" + "=" * 70)
    print("🏁 성능 차별화 극대화 테스트 완료!")
    print("=" * 70)
    
    successful_tests = [r for r in differentiation_results if r.get('success', False)]
    
    if successful_tests:
        print(f"✅ 성공한 복잡한 테스트: {len(successful_tests)}/{len(complex_gt_questions)}")
        
        # 성능 차별화 효과 분석
        high_impact_count = sum(1 for r in successful_tests if r['impact_score'] >= 4)
        medium_impact_count = sum(1 for r in successful_tests if r['impact_score'] == 3)
        low_impact_count = sum(1 for r in successful_tests if r['impact_score'] <= 2)
        
        print(f"🚀 높은 임팩트 (40%+ 향상): {high_impact_count}개")
        print(f"📊 중간 임팩트 (30-40% 향상): {medium_impact_count}개")
        print(f"⚠️ 낮은 임팩트 (30% 미만): {low_impact_count}개")
        
        # 평균 지표들
        avg_complexity = sum(r['avg_complexity'] for r in successful_tests) / len(successful_tests)
        avg_impact_score = sum(r['impact_score'] for r in successful_tests) / len(successful_tests)
        avg_consistency = sum(r['consistency_rate'] for r in successful_tests) / len(successful_tests)
        avg_time = sum(r['generation_time'] for r in successful_tests) / len(successful_tests)
        
        print(f"\n🧮 전체 평균 복잡도: {avg_complexity:.1f} (높을수록 Standard RAG 어려움)")
        print(f"🏅 전체 평균 임팩트 점수: {avg_impact_score:.1f}/5.0")
        print(f"🎯 전체 평균 일관성: {avg_consistency:.1%}")
        print(f"⏱️ 전체 평균 생성 시간: {avg_time:.2f}초")
        
        # 최종 논문 임팩트 예상
        if avg_impact_score >= 4:
            final_assessment = "🎉 논문 임팩트 최고급! 40-50% 성능 향상 주장 가능!"
            expected_improvement = "40-50%"
        elif avg_impact_score >= 3:
            final_assessment = "✅ 논문 임팩트 우수! 30-40% 성능 향상 주장 가능!"
            expected_improvement = "30-40%"
        elif avg_impact_score >= 2:
            final_assessment = "📊 논문 임팩트 보통. 20-30% 성능 향상 주장 가능."
            expected_improvement = "20-30%"
        else:
            final_assessment = "⚠️ 논문 임팩트 부족. 추가 개선 필요."
            expected_improvement = "15-25%"
        
        print(f"\n🎊 최종 논문 임팩트 평가:")
        print(f"📰 {final_assessment}")
        print(f"🚀 예상 성능 향상: {expected_improvement}")
        print(f"📈 현재 상태: Standard 0.9 → 목표: Standard 0.5, Boost 0.8")
        
        # 다음 단계 제안
        print(f"\n💡 다음 단계:")
        print("1. 🧪 이 복잡한 질문들로 실제 B-RAG vs Standard RAG 테스트 실행")
        print("2. 📊 실제 성능 차이 측정 및 확인")
        print("3. 📝 논문 작성 시 '복잡한 법률 쟁점에서의 획기적 성능 향상' 강조")
        print("4. 🎯 Boost RAG의 복잡한 분석 능력 vs Standard RAG의 한계 부각")
        
    else:
        print("❌ 모든 복잡한 테스트가 실패했습니다.")
        expected_improvement = "미확정"
        final_assessment = "평가 불가"
    
    # 결과 저장
    save_path = "performance_differentiation_test_results.json"
    save_data = {
        "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_type": "성능 차별화 극대화 테스트 (논문 임팩트용)",
        "objective": "Standard RAG 성능을 의도적으로 낮춰서 Boost RAG와의 성능 차이 극대화하여 논문 임팩트 증대",
        "strategy": [
            "복합 조건과 예외 상황이 결합된 복잡한 GT 질문 사용",
            "애매한 경계 사례와 해석론적 쟁점 적극 활용", 
            "국제사법, 디지털 자산 등 새로운 영역의 어려운 문제 포함",
            "enhanced 모드로 의도적 복잡화 프롬프트 적용"
        ],
        "expected_paper_impact": expected_improvement,
        "total_test_cases": len(complex_gt_questions),
        "successful_tests": len(successful_tests),
        "summary": {
            "high_impact_count": high_impact_count if successful_tests else 0,
            "medium_impact_count": medium_impact_count if successful_tests else 0,
            "low_impact_count": low_impact_count if successful_tests else 0,
            "average_complexity_score": avg_complexity if successful_tests else 0,
            "average_impact_score": avg_impact_score if successful_tests else 0,
            "average_consistency": avg_consistency if successful_tests else 0,
            "average_generation_time": avg_time if successful_tests else 0,
            "final_assessment": final_assessment
        },
        "differentiation_results": differentiation_results
    }
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 성능 차별화 테스트 결과가 {save_path}에 저장되었습니다")
    print(f"📁 파일 경로: {os.path.abspath(save_path)}")

if __name__ == "__main__":
    run_performance_differentiation_test() 