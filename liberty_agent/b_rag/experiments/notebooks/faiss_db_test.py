#!/usr/bin/env python3
"""
FAISS DB 기반 실제 문서 테스트 스크립트
목표: FAISS 벡터스토어에서 실제 문서를 로드하여 GT 질문 생성 및 테스트
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any
import random

# 프로젝트 루트 경로 추가
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from liberty_agent.b_rag.core.question_generation.unified_yesno_question_generator import UnifiedYesNoQuestionGenerator
    from langchain_community.vectorstores import FAISS
    from langchain_upstage import UpstageEmbeddings
except ImportError as e:
    print(f"❌ 모듈 import 실패: {e}")
    sys.exit(1)

def load_faiss_vectorstore():
    """FAISS 벡터스토어 로드"""
    print("🔄 FAISS 벡터스토어 로드 중...")
    
    # 임베딩 모델 초기화
    dense_embedder = UpstageEmbeddings(
        model="solar-embedding-1-large-query"
    )
    
    # FAISS 캐시 경로들 확인
    faiss_paths = [
        str(project_root / "liberty_agent" / "cached_vectors" / "balanced_json"),
        str(project_root / "liberty_agent" / "cached_vectors" / "retrievers" / "faiss"),
        "../cached_vectors/balanced_json/",
        "../cached_vectors/retrievers/faiss/"
    ]
    
    for cache_path in faiss_paths:
        faiss_index_path = Path(cache_path) / "index.faiss"
        if faiss_index_path.exists():
            print(f"📁 FAISS 인덱스 발견: {cache_path}")
            try:
                vectorstore = FAISS.load_local(
                    cache_path,
                    dense_embedder,
                    allow_dangerous_deserialization=True
                )
                print(f"✅ FAISS 벡터스토어 로드 성공: {cache_path}")
                return vectorstore, cache_path
            except Exception as e:
                print(f"⚠️ {cache_path} 로드 실패: {e}")
                continue
    
    raise FileNotFoundError("FAISS 인덱스를 찾을 수 없습니다.")

def extract_documents_from_faiss(vectorstore) -> List[Dict[str, Any]]:
    """FAISS DB에서 다양한 문서들을 추출하여 GT 질문 생성"""
    print("📋 FAISS DB에서 실제 문서 샘플 추출 중...")
    
    # 다양한 검색어로 문서들을 가져와서 GT 질문 생성
    search_queries = [
        "채권 준점유자",
        "계약 해제",
        "소유권 이전", 
        "손해배상",
        "시효 완성",
        "동업자",
        "법률 효력",
        "민법",
        "판례",
        "소송",
        "무효",
        "취소",
        "불법행위",
        "과실",
        "고의"
    ]
    
    # 검색을 통해 다양한 문서들 수집
    collected_docs = []
    doc_ids = set()  # 중복 방지
    
    for query in search_queries:
        try:
            docs = vectorstore.similarity_search(query, k=5)
            for doc in docs:
                # 문서 내용의 일부를 ID로 사용 (중복 방지)
                doc_id = doc.page_content[:100]
                if doc_id not in doc_ids and len(doc.page_content) > 100:
                    collected_docs.append(doc)
                    doc_ids.add(doc_id)
                    if len(collected_docs) >= 20:  # 충분한 샘플 수집
                        break
        except Exception as e:
            print(f"⚠️ 검색어 '{query}' 처리 중 오류: {e}")
            continue
        
        if len(collected_docs) >= 20:
            break
    
    print(f"✅ {len(collected_docs)}개의 고유 문서 수집 완료")
    
    # 수집된 문서들에서 GT 질문과 관련 정보 추출
    test_cases = []
    
    for i, doc in enumerate(collected_docs[:10]):  # 상위 10개 문서만 사용
        content = doc.page_content
        metadata = doc.metadata if hasattr(doc, 'metadata') else {}
        
        # 문서 내용 기반 GT 질문 생성
        gt_question = generate_gt_question_from_content(content)
        
        # 키워드 추출
        keywords = extract_keywords_from_content(content)
        
        test_cases.append({
            "question": gt_question,
            "document_content": content[:1000],  # 처음 1000자만 사용
            "keywords_to_consider": ", ".join(keywords[:5]),  # 상위 5개 키워드
            "expected_pattern": "FAISS DB 기반 자연스러운 분포",
            "description": f"FAISS 문서 {i+1}: {content[:50]}...",
            "source_metadata": metadata
        })
    
    print(f"✅ {len(test_cases)}개의 GT 질문 생성 완료")
    return test_cases

def generate_gt_question_from_content(content: str) -> str:
    """문서 내용을 기반으로 GT 질문 생성"""
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

def extract_keywords_from_content(content: str) -> List[str]:
    """문서 내용에서 법률 키워드 추출"""
    legal_keywords = [
        "채권", "준점유자", "계약", "해제", "취소", "소유권", "손해배상", "시효", 
        "효력", "책임", "민법", "판례", "무효", "불법행위", "과실", "고의",
        "청구권", "소멸", "완성", "성립", "인정", "변제", "이행", "의무",
        "권리", "법률행위", "의사표시", "합의", "당사자", "법원", "판결"
    ]
    
    keywords = []
    content_lower = content.lower()
    
    for keyword in legal_keywords:
        if keyword in content:
            keywords.append(keyword)
    
    return keywords

def run_faiss_based_test():
    """FAISS DB 기반 테스트 실행"""
    print("🧪 FAISS DB 기반 실제 문서 테스트 시작")
    print("=" * 70)
    
    try:
        # FAISS 벡터스토어 로드
        vectorstore, used_path = load_faiss_vectorstore()
        
        # 문서 추출 및 GT 질문 생성
        test_cases = extract_documents_from_faiss(vectorstore)
        
        if not test_cases:
            print("❌ 테스트 케이스가 생성되지 않았습니다.")
            return
        
        # 질문 생성기 초기화
        generator = UnifiedYesNoQuestionGenerator(
            model_name="gpt-4o-2024-08-06",
            temperature=0.1,
            prompt_mode="balanced"
        )
        print(f"✅ 질문 생성기 초기화 완료 (프롬프트 모드: {generator.prompt_mode})")
        
        # 테스트 결과 저장
        faiss_based_results = []
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n📝 테스트 케이스 {i}/{len(test_cases)}")
            print(f"GT 질문: {test_case['question']}")
            print(f"키워드: {test_case['keywords_to_consider']}")
            print(f"문서 요약: {test_case['description']}")
            print("-" * 60)
            
            try:
                start_time = time.time()
                result = generator.generate_ten_level_questions(
                    gt_question=test_case['question'],
                    document_content=test_case.get('document_content', ""),
                    keywords_to_consider=test_case.get('keywords_to_consider', "")
                )
                generation_time = time.time() - start_time
                
                if result and result.questions:
                    yes_count = sum(1 for q in result.questions if q.expected_answer.value == 'Yes')
                    no_count = sum(1 for q in result.questions if q.expected_answer.value == 'No')
                    consistency_rate = result.get_consistency_rate()
                    
                    print(f"✅ 생성 성공! (소요시간: {generation_time:.2f}초)")
                    print(f"📊 분포: Yes {yes_count}개, No {no_count}개")
                    print(f"🎯 일관성: {consistency_rate:.1%}")
                    
                    # 분포 평가
                    if yes_count >= 9 or no_count >= 9:
                        distribution_type = "🌟 매우 자연스러운 분포 (9:1)"
                        distribution_score = 5
                    elif yes_count >= 8 or no_count >= 8:
                        distribution_type = "✨ 자연스러운 분포 (8:2)"
                        distribution_score = 4
                    elif yes_count >= 7 or no_count >= 7:
                        distribution_type = "📊 적당한 분포 (7:3)"
                        distribution_score = 3
                    elif yes_count >= 6 or no_count >= 6:
                        distribution_type = "⚖️ 균형적 분포 (6:4)"
                        distribution_score = 2
                    else:
                        distribution_type = "🎯 완전 균형 분포 (5:5)"
                        distribution_score = 1
                    print(f"📈 분포 평가: {distribution_type}")

                    # 일관성 평가
                    if consistency_rate >= 0.9:
                        consistency_level = "🎯 매우 높은 일관성"
                        consistency_score = 5
                    elif consistency_rate >= 0.8:
                        consistency_level = "✅ 높은 일관성"
                        consistency_score = 4
                    elif consistency_rate >= 0.7:
                        consistency_level = "📊 적당한 일관성"
                        consistency_score = 3
                    elif consistency_rate >= 0.6:
                        consistency_level = "⚠️ 낮은 일관성"
                        consistency_score = 2
                    else:
                        consistency_level = "❌ 매우 낮은 일관성"
                        consistency_score = 1
                    print(f"🔍 일관성 평가: {consistency_level}")
                    
                    # 전체 품질 점수
                    total_score = (distribution_score + consistency_score) / 2
                    if total_score >= 4.5: quality_level = "🏆 최고 품질"
                    elif total_score >= 3.5: quality_level = "🥇 우수한 품질"
                    elif total_score >= 2.5: quality_level = "🥈 보통 품질"
                    elif total_score >= 1.5: quality_level = "🥉 개선 필요"
                    else: quality_level = "⚠️ 품질 문제"
                    print(f"🏅 전체 품질: {quality_level} (점수: {total_score:.1f}/5.0)")

                    print("\n📝 생성된 질문 샘플 (처음 3개):")
                    for q_idx, q_obj in enumerate(result.questions[:3]):
                        print(f"  Level {q_obj.level} ({q_obj.expected_answer.value}): {q_obj.question}")

                    test_result = {
                        "test_case_index": i,
                        "gt_question": test_case['question'],
                        "source_info": {
                            "document_summary": test_case['document_content'][:200] + "...",
                            "keywords": test_case.get('keywords_to_consider', "N/A"),
                            "metadata": test_case.get('source_metadata', {})
                        },
                        "yes_count": yes_count,
                        "no_count": no_count,
                        "consistency_rate": consistency_rate,
                        "distribution_type": distribution_type,
                        "quality_level": quality_level,
                        "total_quality_score": total_score,
                        "generation_time": generation_time,
                        "success": True,
                        "generated_questions_sample": [
                            {
                                "level": q.level,
                                "question": q.question[:80] + "..." if len(q.question) > 80 else q.question,
                                "expected_answer": q.expected_answer.value
                            }
                            for q in result.questions[:5]  # 처음 5개만
                        ]
                    }
                    faiss_based_results.append(test_result)
                else:
                    print("❌ 질문이 생성되지 않았습니다.")
                    faiss_based_results.append({
                        "test_case_index": i, 
                        "gt_question": test_case['question'], 
                        "success": False, 
                        "error": "질문 생성 실패"
                    })
            except Exception as e:
                print(f"❌ 테스트 케이스 {i} 처리 중 오류: {e}")
                faiss_based_results.append({
                    "test_case_index": i, 
                    "gt_question": test_case['question'], 
                    "success": False, 
                    "error": str(e)
                })

        # 전체 결과 요약
        print("\n" + "=" * 70)
        print("🏁 FAISS DB 기반 테스트 완료!")
        print("=" * 70)
        
        successful_tests = [r for r in faiss_based_results if r.get('success', False)]
        if successful_tests:
            print(f"✅ 성공한 테스트: {len(successful_tests)}/{len(test_cases)}")
            avg_consistency = sum(r['consistency_rate'] for r in successful_tests) / len(successful_tests)
            avg_total_score = sum(r['total_quality_score'] for r in successful_tests) / len(successful_tests)
            avg_time = sum(r['generation_time'] for r in successful_tests) / len(successful_tests)
            
            print(f"🎯 평균 일관성: {avg_consistency:.1%}")
            print(f"🏅 평균 품질 점수: {avg_total_score:.1f}/5.0")
            print(f"⏱️ 평균 생성 시간: {avg_time:.2f}초/GT 질문")
            print(f"📁 사용된 FAISS 경로: {used_path}")
            
            # 분포별 통계
            distribution_stats = {}
            for result in successful_tests:
                dist_type = result['distribution_type']
                if dist_type not in distribution_stats:
                    distribution_stats[dist_type] = 0
                distribution_stats[dist_type] += 1
            
            print("\n📊 분포 타입별 통계:")
            for dist_type, count in distribution_stats.items():
                percentage = (count / len(successful_tests)) * 100
                print(f"  {dist_type}: {count}개 ({percentage:.1f}%)")
        else:
            print("❌ 모든 테스트가 실패했습니다.")

        # 결과 저장
        faiss_test_save_path = "faiss_based_test_results.json"
        final_save_data = {
            "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_type": "FAISS DB 기반 실제 문서 테스트",
            "faiss_source_path": used_path,
            "total_gt_cases_processed": len(test_cases),
            "successful_generation_cases": len(successful_tests),
            "average_quality_score": sum(r['total_quality_score'] for r in successful_tests) / len(successful_tests) if successful_tests else 0,
            "results_per_gt_case": faiss_based_results
        }
        
        with open(faiss_test_save_path, 'w', encoding='utf-8') as f:
            json.dump(final_save_data, f, ensure_ascii=False, indent=2)
        print(f"\n💾 테스트 결과가 {faiss_test_save_path}에 저장되었습니다.")
        
        return faiss_based_results, successful_tests
        
    except Exception as e:
        print(f"❌ FAISS DB 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return [], []

if __name__ == "__main__":
    print("🚀 FAISS DB 기반 실제 문서 테스트")
    print("--------------------------------------------------")
    
    results, successful_tests = run_faiss_based_test()
    
    if successful_tests:
        print(f"\n🎉 테스트 완료! {len(successful_tests)}개 성공")
        print("이제 노트북에서 이 결과를 기반으로 RAG 비교 테스트를 실행할 수 있습니다.")
    else:
        print("\n❌ 테스트 실패. 설정을 확인해주세요.") 