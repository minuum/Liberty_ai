# ===== 셀 3: FAISS DB 카테고리별 다양한 문서 기반 균형 분포 테스트 =====
import json
import os
import sys
import time
import random
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_upstage import UpstageEmbeddings

print("\n🧪 FAISS DB 카테고리별 다양한 문서 기반 균형 분포 테스트")
print("-" * 70)

# --- FAISS 벡터스토어 로드 함수 ---
def load_faiss_vectorstore():
    """FAISS 벡터스토어 로드"""
    print("🔄 FAISS 벡터스토어 로드 중...")
    
    # 임베딩 모델 초기화
    dense_embedder = UpstageEmbeddings(
        model="solar-embedding-1-large-query"
    )
    
    # 현재 노트북 위치에서 상대적인 FAISS 캐시 경로들 확인
    faiss_paths = [
        "../../../../cached_vectors/balanced_json/",
        "../../../../cached_vectors/retrievers/faiss/",
        "../../../cached_vectors/balanced_json/",
        "../../../cached_vectors/retrievers/faiss/"
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

# --- GT 질문 생성 함수 ---
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

# --- 키워드 추출 함수 ---
def extract_keywords_from_content(content: str) -> list:
    """문서 내용에서 법률 키워드 추출"""
    legal_keywords = [
        "채권", "준점유자", "계약", "해제", "취소", "소유권", "손해배상", "시효", 
        "효력", "책임", "민법", "판례", "무효", "불법행위", "과실", "고의",
        "청구권", "소멸", "완성", "성립", "인정", "변제", "이행", "의무",
        "권리", "법률행위", "의사표시", "합의", "당사자", "법원", "판결"
    ]
    
    keywords = []
    for keyword in legal_keywords:
        if keyword in content:
            keywords.append(keyword)
    
    return keywords

# --- FAISS DB에서 카테고리별 문서 추출 및 GT 질문 생성 ---
try:
    # FAISS 벡터스토어 로드
    vectorstore, used_path = load_faiss_vectorstore()
    
    print("📋 FAISS DB에서 카테고리별 다양한 문서 샘플 추출 중...")
    
    # 카테고리별 특화 검색어 매핑
    category_search_mapping = {
        "민사": ["계약", "소유권", "손해배상", "채권", "불법행위"],
        "행정": ["행정처분", "행정소송", "행정절차", "취소소송", "무효확인"],
        "형사A(생활형)": ["절도", "폭행", "사기", "협박", "무고"],
        "형사B(일반형)": ["살인", "강도", "성범죄", "마약", "교통사고"],
        "금융조세": ["세무", "조세", "금융", "은행", "투자"],
        "근로자": ["근로", "임금", "해고", "노동", "산재"],
        "특허/저작권": ["특허", "저작권", "상표", "지적재산", "침해"],
        "기업": ["회사", "법인", "기업", "주주", "상법"],
        "가사": ["이혼", "양육", "재산분할", "가족", "상속"],
        "개인정보/ICT": ["개인정보", "정보통신", "데이터", "온라인", "사이버"]
    }
    
    # 카테고리별 GT 질문 템플릿
    category_gt_templates = {
        "민사": [
            "계약의 해제권이 발생하는가?",
            "소유권 이전의 효력이 인정되는가?", 
            "손해배상의 범위가 제한되는가?",
            "채권자의 권리가 보호되는가?",
            "불법행위 책임이 성립하는가?"
        ],
        "행정": [
            "행정처분이 위법하다고 할 수 있는가?",
            "취소소송의 요건이 충족되는가?",
            "행정절차가 적법하게 진행되었는가?",
            "행정청의 재량권이 인정되는가?",
            "행정행위가 무효라고 볼 수 있는가?"
        ],
        "형사A(생활형)": [
            "절도죄가 성립한다고 볼 수 있는가?",
            "폭행죄의 구성요건이 충족되는가?",
            "사기죄가 인정된다고 할 수 있는가?",
            "협박죄가 성립하는가?",
            "무고죄의 요건이 갖춰졌는가?"
        ],
        "형사B(일반형)": [
            "살인죄가 성립한다고 볼 수 있는가?",
            "강도죄의 구성요건이 충족되는가?",
            "성범죄가 인정되는가?",
            "마약사용죄가 성립하는가?",
            "교통사고 처벌법 위반이 인정되는가?"
        ],
        "금융조세": [
            "세무조사가 적법하게 진행되었는가?",
            "조세회피가 인정된다고 볼 수 있는가?",
            "금융거래의 효력이 인정되는가?",
            "은행의 책임이 발생하는가?",
            "투자손실에 대한 책임이 있는가?"
        ],
        "근로자": [
            "부당해고가 인정되는가?",
            "임금체불이 성립한다고 볼 수 있는가?",
            "산업재해 인정 요건이 충족되는가?",
            "노동조합 활동이 보장되는가?",
            "근로시간 위반이 인정되는가?"
        ],
        "특허/저작권": [
            "특허침해가 인정되는가?",
            "저작권 침해가 성립한다고 볼 수 있는가?",
            "상표권 침해가 인정되는가?",
            "지적재산권이 보호되는가?",
            "침해금지청구권이 인정되는가?"
        ],
        "기업": [
            "이사의 선관주의의무 위반이 인정되는가?",
            "주주총회 결의가 유효한가?",
            "회사의 법인격이 부인되는가?",
            "기업결합이 제한되는가?",
            "상법상 책임이 발생하는가?"
        ],
        "가사": [
            "이혼사유가 인정되는가?",
            "양육권이 인정될 수 있는가?",
            "재산분할 청구권이 발생하는가?",
            "상속권이 인정되는가?",
            "가족관계가 성립한다고 볼 수 있는가?"
        ],
        "개인정보/ICT": [
            "개인정보보호법 위반이 인정되는가?",
            "정보통신망법 위반이 성립하는가?",
            "데이터 처리가 적법한가?",
            "온라인상 책임이 발생하는가?",
            "사이버 침해가 인정되는가?"
        ]
    }
    
    # 카테고리별 문서 수집 (균등 분배)
    target_docs_per_category = 2  # 카테고리당 2개씩 수집 (더 다양성 확보)
    collected_docs = []
    doc_ids = set()  # 중복 방지
    
    print("🔍 카테고리별 문서 수집 중...")
    category_stats = {}
    
    # 랜덤 시드 설정으로 매번 다른 문서 선택
    random.seed()  # 매번 다른 시드 사용
    
    for category, search_terms in category_search_mapping.items():
        category_docs = []
        
        # 해당 카테고리의 검색어들로 문서 검색
        for search_term in search_terms:
            try:
                docs = vectorstore.similarity_search(search_term, k=15)  # 더 많이 가져와서 선택의 여지 확보
                
                # 카테고리 매칭 문서들만 수집
                matching_docs = []
                for doc in docs:
                    # 메타데이터에서 카테고리 확인
                    doc_category = doc.metadata.get('category', 'unknown') if hasattr(doc, 'metadata') else 'unknown'
                    
                    # 해당 카테고리 문서만 수집 & 중복 방지
                    doc_id = doc.page_content[:150]  # 더 긴 ID로 중복 방지 강화
                    if (doc_category == category and 
                        doc_id not in doc_ids and 
                        len(doc.page_content) > 200):  # 최소 길이 증가
                        
                        matching_docs.append(doc)
                        doc_ids.add(doc_id)
                
                # 랜덤하게 섞어서 다양성 확보
                random.shuffle(matching_docs)
                category_docs.extend(matching_docs)
                
                if len(category_docs) >= target_docs_per_category:
                    break
                    
            except Exception as e:
                print(f"⚠️ 카테고리 '{category}', 검색어 '{search_term}' 처리 중 오류: {e}")
                continue
        
        # 수집된 문서 통계 (중복 제거 후 최종 선택)
        final_category_docs = category_docs[:target_docs_per_category]
        if final_category_docs:
            collected_docs.extend(final_category_docs)
            category_stats[category] = len(final_category_docs)
            print(f"📁 {category}: {len(final_category_docs)}개 문서 수집")
            
            # 선택된 문서의 간단한 정보 출력
            for idx, doc in enumerate(final_category_docs):
                content_preview = doc.page_content[:50].replace('\n', ' ')
                print(f"   📄 {idx+1}: {content_preview}...")
        else:
            print(f"⚠️ {category}: 문서를 찾을 수 없음")
            category_stats[category] = 0
    
    # 카테고리별 수집 결과 요약
    print(f"\n✅ 카테고리별 문서 수집 완료:")
    total_collected = sum(category_stats.values())
    for category, count in category_stats.items():
        print(f"  📊 {category}: {count}개")
    print(f"📋 총 수집된 문서: {total_collected}개")
    
    # 수집된 문서들에서 GT 질문과 관련 정보 추출
    test_cases = []
    
    for i, doc in enumerate(collected_docs[:10]):  # 상위 10개 문서만 사용
        content = doc.page_content
        metadata = doc.metadata if hasattr(doc, 'metadata') else {}
        doc_category = metadata.get('category', 'unknown')
        
        # 카테고리별 특화 GT 질문 생성
        if doc_category in category_gt_templates:
            # 해당 카테고리의 템플릿 중에서 내용에 맞는 것 선택
            gt_templates = category_gt_templates[doc_category]
            gt_question = gt_templates[i % len(gt_templates)]  # 순환 선택
        else:
            # 일반적인 GT 질문 생성
            gt_question = generate_gt_question_from_content(content)
        
        # 키워드 추출
        keywords = extract_keywords_from_content(content)
        
        # 카테고리별 키워드 추가
        if doc_category in category_search_mapping:
            category_keywords = category_search_mapping[doc_category][:3]  # 상위 3개 키워드 추가
            keywords.extend(category_keywords)
        
        test_cases.append({
            "question": gt_question,
            "document_content": content[:1000],  # 처음 1000자만 사용
            "keywords_to_consider": ", ".join(list(set(keywords))[:5]),  # 중복 제거 후 상위 5개 키워드
            "expected_pattern": "FAISS DB 기반 카테고리별 균형 분포",
            "description": f"[{doc_category}] {content[:30]}...",
            "source_metadata": metadata,
            "document_category": doc_category
        })
    
    print(f"✅ {len(test_cases)}개의 카테고리별 GT 질문 생성 완료")
    print(f"📁 사용된 FAISS 경로: {used_path}")
    
    # 카테고리별 테스트 케이스 분포 출력
    print(f"\n📊 테스트 케이스 카테고리 분포:")
    test_category_stats = {}
    for test_case in test_cases:
        category = test_case.get('document_category', 'unknown')
        test_category_stats[category] = test_category_stats.get(category, 0) + 1
    
    for category, count in test_category_stats.items():
        print(f"  🏷️ {category}: {count}개")

except Exception as e:
    print(f"❌ FAISS DB 로드 실패: {e}")
    print("⚠️ 샘플 질문으로 대체합니다.")
    
    # 대체 샘플 질문 (FAISS 로드 실패 시)
    test_cases = [
        {
            "question": "동업자가 채권의 준점유자에 해당하지 아니 한다고 할 수 있는가?",
            "document_content": "민법 제470조는 채권의 준점유자에 대한 변제는 변제자가 선의이며 과실없는 때에 한하여 효력이 있다고 규정하고 있다...",
            "keywords_to_consider": "채권, 준점유자, 민법 제470조, 동업자",
            "expected_pattern": "부정 질문 → 균형 분포 (6-7:3-4)",
            "description": "[민사] 부정형 질문 - 아니하다는 것이 맞다",
            "source_metadata": {"source": "샘플"},
            "document_category": "민사"
        },
        {
            "question": "계약이 유효하다고 할 수 있는가?",
            "document_content": "계약의 유효성은 의사표시의 합치, 법률행위의 목적, 내용 등을 종합적으로 고려하여 판단된다...",
            "keywords_to_consider": "계약, 유효성, 의사표시, 법률행위",
            "expected_pattern": "긍정 질문 → 균형 분포 (6-7:3-4)",
            "description": "[민사] 긍정형 질문 - 유효하다는 것이 맞다",
            "source_metadata": {"source": "샘플"},
            "document_category": "민사"
        }
    ]
    print(f"⚠️ 샘플 테스트 케이스 {len(test_cases)}개를 사용합니다.")
    used_path = "샘플 데이터"

# FAISS DB 기반 카테고리별 균형 분포 테스트 결과 저장
faiss_based_results = []

try:
    # 질문 생성기 초기화 (balanced 모드 사용)
    generator = UnifiedYesNoQuestionGenerator(
        model_name="gpt-4o-2024-08-06",
        temperature=0.1,
        prompt_mode="balanced"
    )
    print(f"✅ 질문 생성기 초기화 완료 (프롬프트 모드: {generator.prompt_mode})")
    print(f"🎯 목표 분포: 65% Yes (6-7:3-4) - 극단 편향 해결")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 테스트 케이스 {i}/{len(test_cases)}")
        print(f"카테고리: [{test_case.get('document_category', 'unknown')}]")
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
                
                # 개선된 분포 평가 (65% 목표 기준)
                yes_ratio = yes_count / 10
                if 0.6 <= yes_ratio <= 0.7:
                    distribution_type = "🎯 목표 균형 분포 (6-7:3-4)"
                    distribution_score = 5
                elif yes_ratio == 0.5:
                    distribution_type = "⚖️ 완전 균형 분포 (5:5)"
                    distribution_score = 4
                elif 0.7 < yes_ratio <= 0.8:
                    distribution_type = "📊 적당한 분포 (8:2)"
                    distribution_score = 3
                elif 0.8 < yes_ratio <= 0.9:
                    distribution_type = "✨ 자연스러운 분포 (9:1)"
                    distribution_score = 2
                else:
                    distribution_type = "⚠️ 극단 편향 분포"
                    distribution_score = 1
                print(f"📈 분포 평가: {distribution_type} (비율: {yes_ratio:.1%})")

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
                
                # 개선 효과 평가 (균형화 기준)
                if 0.6 <= yes_ratio <= 0.7:
                    improvement_status = "🏆 최적 균형 달성 (목표 범위)"
                    improvement_score = 5
                elif yes_ratio == 0.5:
                    improvement_status = "🎯 완전 균형 달성"
                    improvement_score = 4
                elif 0.7 < yes_ratio <= 0.8:
                    improvement_status = "📊 적당한 균형"
                    improvement_score = 3
                elif 0.4 <= yes_ratio < 0.6:
                    improvement_status = "📊 적당한 균형 (역방향)"
                    improvement_score = 3
                elif 0.8 < yes_ratio <= 0.9:
                    improvement_status = "⚠️ 편향 경향"
                    improvement_score = 2
                else:
                    improvement_status = "❌ 극단 편향"
                    improvement_score = 1
                print(f"🚀 개선 효과: {improvement_status}")

                # 전체 품질 점수
                total_score = (distribution_score + consistency_score + improvement_score) / 3
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
                    "document_category": test_case.get('document_category', 'unknown'),
                    "source_info": {
                        "document_summary": test_case['document_content'][:200] + "..." if test_case.get('document_content') else "N/A",
                        "keywords": test_case.get('keywords_to_consider', "N/A"),
                        "metadata": test_case.get('source_metadata', {})
                    },
                    "yes_count": yes_count,
                    "no_count": no_count,
                    "yes_ratio": yes_ratio,
                    "consistency_rate": consistency_rate,
                    "distribution_type": distribution_type,
                    "quality_level": quality_level,
                    "total_quality_score": total_score,
                    "generation_time": generation_time,
                    "success": True,
                    "generated_questions_details": [
                        {
                            "level": q.level,
                            "question": q.question,
                            "expected_answer": q.expected_answer.value,
                        }
                        for q in result.questions
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
    print("🏁 FAISS DB 기반 카테고리별 균형 분포 테스트 완료!")
    print("=" * 70)
    
    successful_tests = [r for r in faiss_based_results if r.get('success', False)]
    if successful_tests:
        print(f"✅ 성공한 테스트: {len(successful_tests)}/{len(test_cases)}")
        avg_consistency = sum(r['consistency_rate'] for r in successful_tests) / len(successful_tests)
        avg_total_score = sum(r['total_quality_score'] for r in successful_tests) / len(successful_tests)
        avg_time = sum(r['generation_time'] for r in successful_tests) / len(successful_tests)
        avg_yes_ratio = sum(r['yes_ratio'] for r in successful_tests) / len(successful_tests)
        
        print(f"🎯 평균 일관성: {avg_consistency:.1%}")
        print(f"🏅 평균 품질 점수: {avg_total_score:.1f}/5.0")
        print(f"⏱️ 평균 생성 시간: {avg_time:.2f}초/GT 질문")
        print(f"📊 평균 Yes 비율: {avg_yes_ratio:.1%} (목표: 60-70%)")
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
            
        # 카테고리별 성공률
        category_success = {}
        for result in successful_tests:
            category = result.get('document_category', 'unknown')
            if category not in category_success:
                category_success[category] = 0
            category_success[category] += 1
        
        print("\n🏷️ 카테고리별 성공 분포:")
        for category, count in category_success.items():
            percentage = (count / len(successful_tests)) * 100
            print(f"  {category}: {count}개 ({percentage:.1f}%)")
    else:
        print("❌ 모든 테스트가 실패했습니다.")

    # 결과 저장
    faiss_test_save_path = "faiss_based_natural_distribution_test_results.json"
    final_save_data = {
        "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_type": "FAISS DB 기반 카테고리별 균형 분포 테스트",
        "prompt_version": "balanced v3 (분포 편향 해결)",
        "target_distribution": "60-70% Yes (6-7:3-4)",
        "faiss_source_path": used_path,
        "total_gt_cases_processed": len(test_cases),
        "successful_generation_cases": len(successful_tests),
        "average_quality_score": sum(r['total_quality_score'] for r in successful_tests) / len(successful_tests) if successful_tests else 0,
        "average_yes_ratio": sum(r['yes_ratio'] for r in successful_tests) / len(successful_tests) if successful_tests else 0,
        "results_per_gt_case": faiss_based_results
    }
    
    with open(faiss_test_save_path, 'w', encoding='utf-8') as f:
        json.dump(final_save_data, f, ensure_ascii=False, indent=2)
    print(f"\n💾 테스트 결과가 {faiss_test_save_path}에 저장되었습니다.")
    
    # 다음 셀을 위한 전역 변수 설정
    if successful_tests:
        # 첫 번째 성공한 테스트의 결과를 사용하여 다음 셀에서 사용할 변수들 설정
        best_test = successful_tests[0]
        
        # generated_questions 설정 (다음 셀에서 사용)
        generated_questions = best_test['generated_questions_details']
        test_gt_question = best_test['gt_question']
        
        print(f"✅ FAISS DB 기반 카테고리별 테스트 완료. 다음 셀에서 {len(generated_questions)}개 질문으로 RAG 비교 테스트를 진행할 수 있습니다.")
        print(f"📋 선택된 GT 질문: {test_gt_question}")
        print(f"🏷️ 문서 카테고리: {best_test.get('document_category', 'unknown')}")

except Exception as e:
    print(f"❌ 전체 테스트 프로세스 실패: {e}")
    import traceback
    traceback.print_exc() 