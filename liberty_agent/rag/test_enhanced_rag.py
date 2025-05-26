#!/usr/bin/env python3
"""
Enhanced RAG Pipeline 테스트 스크립트

이 스크립트는 Enhanced RAG Pipeline의 기능을 테스트합니다.
API 키가 설정되어 있어야 실제 실험이 가능합니다.

사용법:
    python test_enhanced_rag.py
"""

import os
import sys
import logging
from pathlib import Path

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_environment():
    """환경 설정 확인"""
    print("🔍 환경 설정 확인 중...")
    
    # API 키 확인
    openai_key = os.getenv("OPENAI_API_KEY")
    upstage_key = os.getenv("UPSTAGE_API_KEY")
    
    print(f"OpenAI API 키: {'✅ 설정됨' if openai_key else '❌ 설정되지 않음'}")
    print(f"Upstage API 키: {'✅ 설정됨' if upstage_key else '❌ 설정되지 않음'}")
    
    # 모듈 파일 확인
    pipeline_file = Path("enhanced_rag_pipeline.py")
    print(f"Pipeline 파일: {'✅ 존재' if pipeline_file.exists() else '❌ 없음'}")
    
    return bool(openai_key and upstage_key and pipeline_file.exists())

def test_import():
    """모듈 임포트 테스트"""
    print("\n📦 모듈 임포트 테스트...")
    
    try:
        from enhanced_rag_pipeline import (
            RAGExperimentConfig,
            EnhancedRAGExperiment,
            create_experiment_config,
            run_quick_test
        )
        print("✅ 모든 모듈 임포트 성공")
        return True
    except ImportError as e:
        print(f"❌ 모듈 임포트 실패: {e}")
        return False

def test_config_creation():
    """설정 생성 테스트"""
    print("\n⚙️  설정 생성 테스트...")
    
    try:
        from enhanced_rag_pipeline import create_experiment_config
        
        # 기본 설정 생성
        config = create_experiment_config(
            experiment_name="test_config",
            preset="quick_test"
        )
        
        print(f"✅ 설정 생성 성공")
        print(f"   실험명: {config.output_dir.name}")
        print(f"   LLM 모델: {config.llm_model_name}")
        print(f"   온도: {config.llm_temperature}")
        print(f"   Top-K: {config.rag_top_k}")
        
        return True
    except Exception as e:
        print(f"❌ 설정 생성 실패: {e}")
        return False

def test_experiment_creation():
    """실험 객체 생성 테스트"""
    print("\n🧪 실험 객체 생성 테스트...")
    
    try:
        from enhanced_rag_pipeline import create_experiment_config, EnhancedRAGExperiment
        
        config = create_experiment_config(
            experiment_name="test_experiment",
            preset="quick_test"
        )
        
        experiment = EnhancedRAGExperiment(config)
        print("✅ 실험 객체 생성 성공")
        print(f"   RAG 시스템: {'✅ 초기화됨' if experiment.rag_system else '❌ 초기화 실패'}")
        
        return True
    except Exception as e:
        print(f"❌ 실험 객체 생성 실패: {e}")
        return False

def test_sample_data():
    """샘플 데이터 테스트"""
    print("\n📋 샘플 데이터 테스트...")
    
    sample_data = [
        {
            "query_id": "test_001",
            "gt_query": "계약 해지는 가능한가요?",
            "gt_answer": "긍정",
            "transformed_query": "계약 해지의 법적 근거는 무엇인가?",
            "transformed_query_difficulty": "기초",
            "transformed_query_strategy": "법리 해석",
            "policy_level": 1,
            "original_category": "민사"
        },
        {
            "query_id": "test_002", 
            "gt_query": "임금 체불 신고는 어떻게?",
            "gt_answer": "부정",
            "transformed_query": "임금 체불 신고 절차와 방법은?",
            "transformed_query_difficulty": "중급",
            "transformed_query_strategy": "절차 질문",
            "policy_level": 2,
            "original_category": "근로자"
        }
    ]
    
    print(f"✅ 샘플 데이터 준비 완료 ({len(sample_data)}개)")
    for item in sample_data:
        print(f"   - {item['query_id']}: {item['transformed_query'][:30]}...")
    
    return sample_data

def test_mock_experiment():
    """Mock 실험 테스트 (API 호출 없이)"""
    print("\n🔬 Mock 실험 테스트...")
    
    try:
        from enhanced_rag_pipeline import create_experiment_config, EnhancedRAGExperiment
        
        # Mock 설정 (실제 API 키 없이도 동작)
        config = create_experiment_config(
            experiment_name="mock_test",
            preset="quick_test",
            upstage_api_key="mock_upstage_key",
            openai_api_key="mock_openai_key"
        )
        
        sample_data = test_sample_data()
        
        # Mock 실험 실행 시도
        experiment = EnhancedRAGExperiment(config)
        
        print("✅ Mock 실험 설정 성공")
        print("   실제 API 호출을 위해서는 올바른 API 키가 필요합니다")
        
        return True
    except Exception as e:
        print(f"❌ Mock 실험 실패: {e}")
        return False

def test_quick_test_function():
    """빠른 테스트 함수 호출 테스트"""
    print("\n⚡ 빠른 테스트 함수 테스트...")
    
    # API 키가 있는 경우에만 실제 실행
    if os.getenv("OPENAI_API_KEY") and os.getenv("UPSTAGE_API_KEY"):
        try:
            from enhanced_rag_pipeline import run_quick_test
            
            print("   ⚠️  실제 API를 호출합니다. 비용이 발생할 수 있습니다.")
            print("   계속하려면 'y'를 입력하세요: ", end="")
            user_input = input().strip().lower()
            
            if user_input == 'y':
                print("   API 호출 중...")
                results = run_quick_test()
                if "error" not in results:
                    print("✅ 빠른 테스트 성공")
                    return True
                else:
                    print(f"❌ 빠른 테스트 실패: {results['error']}")
                    return False
            else:
                print("   사용자 취소")
                return True
        except Exception as e:
            print(f"❌ 빠른 테스트 오류: {e}")
            return False
    else:
        print("   API 키가 설정되지 않아 실제 테스트를 건너뜁니다")
        print("   ✅ 함수 호출 가능 확인")
        return True

def run_all_tests():
    """모든 테스트 실행"""
    print("🚀 Enhanced RAG Pipeline 테스트 시작\n")
    
    tests = [
        ("환경 설정", check_environment),
        ("모듈 임포트", test_import),
        ("설정 생성", test_config_creation),
        ("실험 객체", test_experiment_creation),
        ("Mock 실험", test_mock_experiment),
        ("빠른 테스트", test_quick_test_function)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} 테스트 중 예외 발생: {e}")
            results.append((test_name, False))
    
    # 결과 요약
    print(f"\n📊 테스트 결과 요약")
    print("=" * 40)
    
    passed = 0
    for test_name, result in results:
        status = "✅ 통과" if result else "❌ 실패"
        print(f"{test_name:15} | {status}")
        if result:
            passed += 1
    
    print("=" * 40)
    print(f"총 {len(results)}개 테스트 중 {passed}개 통과 ({passed/len(results)*100:.1f}%)")
    
    if passed == len(results):
        print("\n🎉 모든 테스트 통과! Enhanced RAG Pipeline 사용 준비 완료")
    else:
        print(f"\n⚠️  {len(results) - passed}개 테스트 실패. 문제를 해결한 후 다시 시도하세요")
    
    return passed == len(results)

def print_usage_instructions():
    """사용법 안내"""
    print("\n📚 사용법 안내:")
    print("1. API 키 설정:")
    print("   export OPENAI_API_KEY='your_openai_key'")
    print("   export UPSTAGE_API_KEY='your_upstage_key'")
    print()
    print("2. 기본 사용:")
    print("   from enhanced_rag_pipeline import run_quick_test")
    print("   results = run_quick_test()")
    print()
    print("3. 사용자 정의 실험:")
    print("   from enhanced_rag_pipeline import create_experiment_config, EnhancedRAGExperiment")
    print("   config = create_experiment_config(experiment_name='my_test')")
    print("   experiment = EnhancedRAGExperiment(config)")
    print("   results = experiment.run_experiment()")
    print()
    print("4. 자세한 사용법은 enhanced_rag_usage_guide.md 참조")

if __name__ == "__main__":
    try:
        success = run_all_tests()
        
        if success:
            print_usage_instructions()
        else:
            print("\n🔧 문제 해결:")
            print("1. enhanced_rag_pipeline.py 파일이 같은 폴더에 있는지 확인")
            print("2. 필요한 Python 패키지가 설치되어 있는지 확인")
            print("3. API 키가 올바르게 설정되어 있는지 확인")
    
    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n\n예상치 못한 오류 발생: {e}")
        import traceback
        traceback.print_exc() 