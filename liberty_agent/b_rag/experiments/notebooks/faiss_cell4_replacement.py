# ===== 셀 4: FAISS 기반 저장된 질문 로드 및 검증 =====
import json
import os
import time

print("\n🔄 FAISS 기반 저장된 질문 로드 및 검증 테스트")
print("-" * 50)

try:
    # FAISS 기반 결과 파일 경로 설정
    faiss_results_path = "faiss_based_natural_distribution_test_results.json"
    
    # JSON 파일에서 FAISS 기반 질문 로드
    if os.path.exists(faiss_results_path):
        with open(faiss_results_path, 'r', encoding='utf-8') as f:
            faiss_loaded_data = json.load(f)
        
        print(f"✅ {faiss_results_path}에서 FAISS 기반 질문 로드 성공")
        print(f"📝 테스트 타입: {faiss_loaded_data['test_type']}")
        print(f"⏰ 생성 시간: {faiss_loaded_data['test_timestamp']}")
        print(f"📁 FAISS 소스 경로: {faiss_loaded_data['faiss_source_path']}")
        print(f"📊 총 GT 케이스: {faiss_loaded_data['total_gt_cases_processed']}개")
        print(f"✅ 성공한 생성: {faiss_loaded_data['successful_generation_cases']}개")
        print(f"🏅 평균 품질 점수: {faiss_loaded_data['average_quality_score']:.2f}/5.0")
        
        # 성공한 테스트 중 첫 번째 사용
        successful_results = [r for r in faiss_loaded_data['results_per_gt_case'] if r.get('success', False)]
        
        if successful_results:
            first_success = successful_results[0]
            print(f"\n📝 대표 GT 질문: {first_success['gt_question']}")
            print(f"📊 분배: Yes {first_success['yes_count']}개, No {first_success['no_count']}개")
            print(f"🎯 일관성: {first_success['consistency_rate']:.1%}")
            print(f"🏅 품질: {first_success['quality_level']}")
            print(f"⏱️ 생성 시간: {first_success['generation_time']:.2f}초")
            
            # 문서 소스 정보 출력
            if 'source_info' in first_success:
                source_info = first_success['source_info']
                print(f"\n📄 문서 소스 정보:")
                print(f"  키워드: {source_info.get('keywords', 'N/A')}")
                print(f"  문서 요약: {source_info.get('document_summary', 'N/A')[:100]}...")
            
            # 전역 변수 업데이트 (FAISS 메모리 변수가 없는 경우)
            if 'generated_questions' not in locals() or generated_questions is None:
                print("\n⚠️ 메모리에 질문이 없어서 FAISS 파일에서 로드한 데이터를 사용합니다")
                test_gt_question = first_success['gt_question']
                
                # generated_questions 객체 재생성 (FAISS 기반)
                generated_questions = type('obj', (object,), {
                    'questions': [type('q', (object,), {
                        'level': q['level'],
                        'question': q['question'],
                        'expected_answer': type('ans', (object,), {'value': q['expected_answer']})()
                    })() for q in first_success['generated_questions_details']]
                })()
                
                print("✅ FAISS 기반 질문 객체 재생성 완료")
            else:
                print("✅ 메모리에 FAISS 기반 질문이 이미 존재합니다")
            
            # 로드된 질문 샘플 출력
            print("\n📝 FAISS 기반 로드된 질문 샘플 (처음 5개):")
            for i, q in enumerate(first_success['generated_questions_details'][:5]):
                print(f"  Level {q['level']} ({q['expected_answer']}): {q['question']}")
            
            # 추가 성공 케이스들 요약
            if len(successful_results) > 1:
                print(f"\n📊 추가 성공 케이스 {len(successful_results)-1}개 요약:")
                for i, result in enumerate(successful_results[1:], 2):
                    print(f"  케이스 {i}: {result['gt_question'][:30]}... (Yes:{result['yes_count']} No:{result['no_count']})")
        
        else:
            print("❌ 성공한 FAISS 기반 테스트 결과가 없습니다.")
            raise ValueError("성공한 테스트 결과 없음")
        
        print("\n✅ FAISS 기반 질문 로드 및 검증 완료")
        
        # 다음 셀을 위한 변수 상태 확인
        print(f"\n🔄 다음 셀을 위한 FAISS 기반 변수 상태:")
        print(f"  test_gt_question: {'설정됨' if 'test_gt_question' in locals() else '미설정'}")
        print(f"  generated_questions: {'설정됨' if 'generated_questions' in locals() and generated_questions is not None else '미설정'}")
        if 'generated_questions' in locals() and generated_questions is not None:
            print(f"  질문 개수: {len(generated_questions.questions)}개")
            
            # 분포 확인
            yes_count = sum(1 for q in generated_questions.questions if q.expected_answer.value == 'Yes')
            no_count = sum(1 for q in generated_questions.questions if q.expected_answer.value == 'No')
            print(f"  분포: Yes {yes_count}개, No {no_count}개")
        
        # 다음 RAG 비교 테스트를 위한 준비 완료 메시지
        print(f"\n🚀 FAISS 기반 RAG 비교 테스트 준비 완료!")
        print(f"   📋 GT 질문: {test_gt_question}")
        print(f"   📊 생성된 질문: {len(generated_questions.questions) if 'generated_questions' in locals() and generated_questions else 0}개")
        
    else:
        print(f"❌ {faiss_results_path} 파일이 없습니다.")
        print("⚠️ 먼저 셀 3 (FAISS DB 기반 자연스러운 분포 테스트)을 실행하세요.")
        
        # FAISS 기본값 설정
        test_gt_question = "동업자가 채권의 준점유자에 해당하지 아니 한다고 할 수 있는가?"
        generated_questions = None
        print("⚠️ FAISS 기본 GT 질문으로 설정했습니다.")
        
        # 대체 경로들 확인
        alternative_paths = [
            "improved_natural_distribution_test_results.json",
            "natural_distribution_test_results.json",
            "generated_questions.json"
        ]
        
        print("\n🔍 대체 파일 검색 중...")
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                print(f"📄 발견된 대체 파일: {alt_path}")
                try:
                    with open(alt_path, 'r', encoding='utf-8') as f:
                        alt_data = json.load(f)
                    print(f"✅ {alt_path}를 대체로 사용합니다.")
                    # 이전 형식에 대한 기본 처리는 생략하고 FAISS 우선 사용 권장
                    break
                except Exception as e:
                    print(f"⚠️ {alt_path} 로드 실패: {e}")
                    continue
        
except Exception as e:
    print(f"❌ FAISS 기반 질문 로드 실패: {e}")
    import traceback
    traceback.print_exc()
    
    # 오류 시 FAISS 기본값 설정
    test_gt_question = "동업자가 채권의 준점유자에 해당하지 아니 한다고 할 수 있는가?"
    generated_questions = None
    print("\n⚠️ 오류로 인해 FAISS 기본값으로 설정했습니다.")
    print("💡 해결 방법: 셀 3 (FAISS DB 기반 테스트)을 먼저 실행해 주세요.")

print(f"\n{'='*50}")
print(f"🏁 FAISS 기반 질문 로드 검증 완료")
print(f"📋 최종 GT 질문: {test_gt_question if 'test_gt_question' in locals() else '미설정'}")
print(f"📊 준비된 질문 수: {len(generated_questions.questions) if 'generated_questions' in locals() and generated_questions else 0}개")
print(f"{'='*50}") 