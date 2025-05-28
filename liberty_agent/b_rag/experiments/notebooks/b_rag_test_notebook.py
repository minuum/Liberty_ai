# B-RAG 프로젝트 테스트 노트북
# 복사-붙여넣기로 Jupyter 노트북에서 바로 사용 가능

# =============================================================================
# 셀 1: 환경 설정 및 경로 추가
# =============================================================================

import sys
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# 상위 디렉토리를 Python path에 추가
current_dir = Path.cwd()
print(f"현재 디렉토리: {current_dir}")

# B-RAG 프로젝트 루트 찾기
b_rag_dir = None
for parent in current_dir.parents:
    if (parent / "b_rag").exists():
        b_rag_dir = parent / "b_rag"
        break

if b_rag_dir is None:
    # 현재 디렉토리에서 b_rag 찾기
    if (current_dir / "b_rag").exists():
        b_rag_dir = current_dir / "b_rag"
    else:
        print("❌ b_rag 디렉토리를 찾을 수 없습니다.")
        print("현재 디렉토리 구조:")
        for item in current_dir.iterdir():
            print(f"  - {item.name}")

if b_rag_dir:
    sys.path.append(str(b_rag_dir))
    sys.path.append(str(b_rag_dir.parent))
    print(f"✅ B-RAG 디렉토리 추가: {b_rag_dir}")

# =============================================================================
# 셀 2: 모듈 Import 및 초기화
# =============================================================================

try:
    from core.question_generation.unified_yesno_question_generator import UnifiedYesNoQuestionGenerator
    from core.rag_system.yesno_rag_system import YesNoRAGSystem, YesNoRAGConfig
    from core.schemas.yesno_question_schemas import (
        YesNoAnswer, 
        TenLevelYesNoQuestions,
        LevelQuestion,
        SAMPLE_TARGET_AUDIENCES
    )
    from experiments.configs.experiment_config import BRAGConfig, get_config
    from experiments.b_rag_experiment_runner import BRAGExperimentRunner
    
    print("✅ 모든 모듈 import 성공!")
    
except ImportError as e:
    print(f"❌ Import 오류: {e}")
    print("\n현재 Python path:")
    for i, path in enumerate(sys.path):
        print(f"  {i}: {path}")
    
    print(f"\n📁 B-RAG 디렉토리 내용 ({b_rag_dir}):")
    if b_rag_dir and b_rag_dir.exists():
        for item in b_rag_dir.rglob("*.py"):
            print(f"  - {item.relative_to(b_rag_dir)}")
    
    # 개별 모듈 테스트
    print("\n🔍 개별 모듈 import 테스트:")
    modules_to_test = [
        "core.schemas.yesno_question_schemas",
        "core.question_generation.unified_yesno_question_generator", 
        "core.rag_system.yesno_rag_system",
        "experiments.configs.experiment_config"
    ]
    
    for module in modules_to_test:
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except ImportError as module_error:
            print(f"  ❌ {module}: {module_error}")

# =============================================================================
# 셀 3: 빠른 테스트 - 질문 생성기
# =============================================================================

print("🔧 질문 생성기 테스트")

# 질문 생성기 초기화
try:
    generator = UnifiedYesNoQuestionGenerator()
    print("✅ 질문 생성기 초기화 성공")
    
    # 테스트 데이터
    test_gt_question = "동업자가 채권의 준점유자에 해당하지 아니 한다고 할 수 있는가?"
    test_document = """
    대법원 1982. 11. 9. 선고 80다3135 판결
    
    【판시사항】
    동업자가 채권의 준점유자에 해당하지 아니한다고 할 수 있다.
    
    【판결요지】
    민법 제470조에 따르면 채권의 준점유자에게 변제한 경우에도 
    변제자가 선의이고 과실이 없으면 유효한 변제가 된다.
    그러나 단순한 동업관계만으로는 채권의 준점유자로 볼 수 없다.
    """
    
    # 질문 생성 테스트
    print("🔄 10개 레벨 질문 생성 중...")
    generated_questions = generator.generate_ten_level_questions(
        gt_question=test_gt_question,
        document_content=test_document,
        keywords_to_consider="법률 용어, 판례, 법리"
    )
    
    print(f"✅ 질문 생성 완료: {len(generated_questions.questions)}개")
    print(f"📈 일관성 비율: {generated_questions.get_consistency_rate():.2%}")
    
    # 생성된 질문들 출력
    print("\n📋 생성된 질문들:")
    for i, q in enumerate(generated_questions.questions, 1):
        print(f"  Level {q.level}: {q.question}")
        print(f"    대상: {q.target_audience}")
        print(f"    예상 답변: {q.expected_answer.value}")
        print(f"    확신도: {q.confidence:.2f}")
        print()

except Exception as e:
    print(f"❌ 질문 생성기 테스트 실패: {e}")

# =============================================================================
# 셀 4: RAG 시스템 테스트
# =============================================================================

print("🔧 RAG 시스템 테스트")

try:
    # RAG 시스템 초기화
    rag_config = YesNoRAGConfig(
        top_k=3,
        similarity_threshold=0.7
    )
    rag_system = YesNoRAGSystem(rag_config)
    print("✅ RAG 시스템 초기화 성공")
    
    # 테스트 질문들
    test_questions = [
        "동업자가 채권의 준점유자에 해당하지 아니 한다고 할 수 있는가?",
        "민법 제470조에 따라 변제자가 선의이고 과실이 없으면 유효한 변제가 되는가?"
    ]
    
    print("\n🔍 Standard RAG vs Boost RAG 비교 테스트")
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n📝 질문 {i}: {question}")
        
        # Standard RAG
        standard_result = rag_system.run_standard_rag(question)
        print(f"  Standard RAG:")
        print(f"    답변: {standard_result.answer}")
        print(f"    확신도: {standard_result.confidence:.3f}")
        print(f"    처리 시간: {standard_result.processing_time:.2f}초")
        
        # Boost RAG
        boost_result = rag_system.run_boost_rag(question, max_iterations=2)
        print(f"  Boost RAG:")
        print(f"    답변: {boost_result.answer}")
        print(f"    확신도: {boost_result.confidence:.3f}")
        print(f"    처리 시간: {boost_result.processing_time:.2f}초")
        
        # 개선도 계산
        confidence_improvement = boost_result.confidence - standard_result.confidence
        print(f"  📈 확신도 개선: {confidence_improvement:.3f}")

except Exception as e:
    print(f"❌ RAG 시스템 테스트 실패: {e}")

# =============================================================================
# 셀 5: 통합 실험 실행
# =============================================================================

print("🚀 B-RAG 통합 실험 실행")

try:
    # 빠른 테스트 설정 로드
    config = get_config("quick_test")
    print(f"✅ 설정 로드 완료: {config.pipeline.experiment_name}")
    
    # 실험 러너 생성
    runner = BRAGExperimentRunner(config)
    print("✅ 실험 러너 생성 완료")
    
    # 전체 실험 실행
    print("\n🎯 전체 실험 시작...")
    results = runner.run_full_experiment()
    
    print("✅ 실험 완료!")
    
    # 결과 요약 출력
    if "performance_analysis" in results and "overall_performance" in results["performance_analysis"]:
        perf = results["performance_analysis"]["overall_performance"]
        print(f"\n📊 실험 결과 요약:")
        print(f"  확신도 개선: {perf['confidence_improvement']:.3f}")
        print(f"  Yes 답변 증가: {perf['yes_answer_improvement']}개")
        print(f"  Standard RAG Yes 비율: {perf['standard_yes_rate']:.2%}")
        print(f"  Boost RAG Yes 비율: {perf['boost_yes_rate']:.2%}")

except Exception as e:
    print(f"❌ 통합 실험 실행 실패: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# 셀 6: 결과 시각화
# =============================================================================

print("📊 결과 시각화")

try:
    if 'results' in locals() and results:
        # 데이터 준비
        rag_results = results.get("rag_experiment_results", {})
        
        if "standard_rag" in rag_results and "boost_rag" in rag_results:
            standard_data = rag_results["standard_rag"]
            boost_data = rag_results["boost_rag"]
            
            # 확신도 비교 차트
            plt.figure(figsize=(12, 8))
            
            # 서브플롯 1: 확신도 비교
            plt.subplot(2, 2, 1)
            methods = ['Standard RAG', 'Boost RAG']
            confidences = [standard_data["avg_confidence"], boost_data["avg_confidence"]]
            colors = ['skyblue', 'lightcoral']
            
            bars = plt.bar(methods, confidences, color=colors)
            plt.title('평균 확신도 비교')
            plt.ylabel('확신도')
            plt.ylim(0, 1)
            
            # 값 표시
            for bar, conf in zip(bars, confidences):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{conf:.3f}', ha='center', va='bottom')
            
            # 서브플롯 2: Yes 답변 개수 비교
            plt.subplot(2, 2, 2)
            yes_counts = [standard_data["yes_count"], boost_data["yes_count"]]
            
            bars = plt.bar(methods, yes_counts, color=colors)
            plt.title('Yes 답변 개수 비교')
            plt.ylabel('Yes 답변 수')
            
            # 값 표시
            for bar, count in zip(bars, yes_counts):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        f'{count}', ha='center', va='bottom')
            
            # 서브플롯 3: 처리 시간 비교
            plt.subplot(2, 2, 3)
            standard_times = [r["processing_time"] for r in standard_data["results"]]
            boost_times = [r["processing_time"] for r in boost_data["results"]]
            
            avg_standard_time = sum(standard_times) / len(standard_times)
            avg_boost_time = sum(boost_times) / len(boost_times)
            
            times = [avg_standard_time, avg_boost_time]
            bars = plt.bar(methods, times, color=colors)
            plt.title('평균 처리 시간 비교')
            plt.ylabel('처리 시간 (초)')
            
            # 값 표시
            for bar, time in zip(bars, times):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{time:.2f}s', ha='center', va='bottom')
            
            # 서브플롯 4: 개선율 요약
            plt.subplot(2, 2, 4)
            improvements = [
                boost_data["avg_confidence"] - standard_data["avg_confidence"],
                boost_data["yes_count"] - standard_data["yes_count"]
            ]
            improvement_labels = ['확신도 개선', 'Yes 답변 증가']
            
            bars = plt.bar(improvement_labels, improvements, color=['green', 'orange'])
            plt.title('Boost RAG 개선 효과')
            plt.ylabel('개선량')
            
            # 값 표시
            for bar, imp in zip(bars, improvements):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                        f'{imp:.3f}' if abs(imp) < 1 else f'{imp:.0f}', 
                        ha='center', va='bottom')
            
            plt.tight_layout()
            plt.show()
            
            print("✅ 시각화 완료")
        else:
            print("⚠️ RAG 실험 결과가 없어 시각화를 건너뜁니다.")
    else:
        print("⚠️ 실험 결과가 없어 시각화를 건너뜁니다.")

except Exception as e:
    print(f"❌ 시각화 실패: {e}")

# =============================================================================
# 셀 7: 결과 분석 및 리포트
# =============================================================================

print("📄 결과 분석 및 리포트 생성")

try:
    if 'results' in locals() and results:
        # 상세 분석 리포트
        print("\n" + "="*60)
        print("📋 B-RAG 실험 상세 리포트")
        print("="*60)
        
        # 실험 정보
        exp_info = results["experiment_info"]
        print(f"실험명: {exp_info['name']}")
        print(f"실행 시간: {exp_info['duration_seconds']:.2f}초")
        
        # 질문 생성 결과
        if "question_generation_results" in results:
            qg_results = results["question_generation_results"]
            print(f"\n📝 질문 생성 결과:")
            print(f"  처리된 문서 수: {len(qg_results)}")
            
            for doc_key, doc_data in qg_results.items():
                if "consistency_rate" in doc_data:
                    print(f"  {doc_key} 일관성: {doc_data['consistency_rate']:.2%}")
        
        # RAG 성능 분석
        if "performance_analysis" in results and "overall_performance" in results["performance_analysis"]:
            perf = results["performance_analysis"]["overall_performance"]
            print(f"\n🔍 RAG 성능 분석:")
            print(f"  확신도 개선: {perf['confidence_improvement']:.3f}")
            print(f"  Yes 답변 증가: {perf['yes_answer_improvement']}개")
            print(f"  Standard RAG Yes 비율: {perf['standard_yes_rate']:.2%}")
            print(f"  Boost RAG Yes 비율: {perf['boost_yes_rate']:.2%}")
            
            # 목표 달성 여부
            if "target_achievement" in results["performance_analysis"]:
                target = results["performance_analysis"]["target_achievement"]
                print(f"\n🎯 목표 달성 여부:")
                print(f"  확신도 개선 목표: {'✅ 달성' if target['boost_improvement_achieved'] else '❌ 미달성'}")
                print(f"  Yes 답변 증가 목표: {'✅ 달성' if target['yes_increase_achieved'] else '❌ 미달성'}")
        
        # 레벨별 분석
        if "performance_analysis" in results and "level_analysis" in results["performance_analysis"]:
            level_analysis = results["performance_analysis"]["level_analysis"]
            print(f"\n📊 레벨별 성능 분석:")
            
            for level_key, level_data in level_analysis.items():
                level_num = level_key.split('_')[1]
                improvement = level_data["improvement"]
                print(f"  Level {level_num}: {improvement:+d}개 개선 "
                      f"({level_data['standard_yes_count']} → {level_data['boost_yes_count']})")
        
        print("\n✅ 리포트 생성 완료")
    else:
        print("⚠️ 분석할 실험 결과가 없습니다.")

except Exception as e:
    print(f"❌ 리포트 생성 실패: {e}")

# =============================================================================
# 셀 8: 추가 실험 및 커스터마이징
# =============================================================================

print("🔧 추가 실험 옵션")

# 사용자 정의 실험 함수
def run_custom_experiment(gt_question, document_content, experiment_name="custom"):
    """사용자 정의 실험 실행"""
    print(f"\n🎯 커스텀 실험 시작: {experiment_name}")
    
    try:
        # 질문 생성
        generator = UnifiedYesNoQuestionGenerator()
        generated_questions = generator.generate_ten_level_questions(
            gt_question=gt_question,
            document_content=document_content,
            keywords_to_consider="법률 용어, 판례, 법리"
        )
        
        print(f"✅ 질문 생성 완료: {len(generated_questions.questions)}개")
        
        # RAG 테스트
        rag_config = YesNoRAGConfig(top_k=3)
        rag_system = YesNoRAGSystem(rag_config)
        
        # GT 질문과 생성된 질문들로 RAG 테스트
        test_questions = [gt_question] + [q.question for q in generated_questions.questions]
        
        standard_results = []
        boost_results = []
        
        for question in test_questions:
            standard_result = rag_system.run_standard_rag(question)
            boost_result = rag_system.run_boost_rag(question, max_iterations=2)
            
            standard_results.append(standard_result)
            boost_results.append(boost_result)
        
        # 결과 분석
        standard_yes = sum(1 for r in standard_results if r.answer.strip().startswith("Yes"))
        boost_yes = sum(1 for r in boost_results if r.answer.strip().startswith("Yes"))
        
        print(f"📊 실험 결과:")
        print(f"  Standard RAG Yes 답변: {standard_yes}/{len(test_questions)}")
        print(f"  Boost RAG Yes 답변: {boost_yes}/{len(test_questions)}")
        print(f"  개선: {boost_yes - standard_yes}개")
        
        return {
            "generated_questions": generated_questions,
            "standard_results": standard_results,
            "boost_results": boost_results,
            "improvement": boost_yes - standard_yes
        }
        
    except Exception as e:
        print(f"❌ 커스텀 실험 실패: {e}")
        return None

# 예시 커스텀 실험
print("\n💡 커스텀 실험 예시:")
print("다음 코드를 수정하여 자신만의 실험을 실행해보세요:")
print("""
custom_gt = "계약 해지 시 손해배상청구권이 소멸시효에 걸리는가?"
custom_doc = '''
대법원 2020. 5. 14. 선고 2018다12345 판결

계약 해지로 인한 손해배상청구권은 일반 채권으로서 
민법 제162조 제1항에 따라 10년의 소멸시효에 걸린다.

다만, 불법행위로 인한 손해배상청구권과는 구별되며,
계약 해지 시점부터 시효가 진행된다.
'''

custom_result = run_custom_experiment(custom_gt, custom_doc, "소멸시효_실험")
""")

print("\n🎉 B-RAG 테스트 노트북 완료!")
print("위의 코드들을 Jupyter 노트북의 각 셀에 복사-붙여넣기하여 사용하세요.") 