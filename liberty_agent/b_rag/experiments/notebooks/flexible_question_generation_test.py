"""
🎯 확장 가능한 B-RAG 질문 생성 및 테스트 시스템
- 질문 개수 자유 설정 (5개, 10개, 20개 등)
- FAISS DB에서 여러 케이스 선택
- 확신도 기준 조정 가능
- 배치 처리 지원
"""

import importlib
import sys
import time
import random
from pathlib import Path
from typing import List, Dict, Any, Optional

# 모듈 강제 reload
module_name = 'liberty_agent.b_rag.core.question_generation.configurable_question_generator'
if module_name in sys.modules:
    print("🔄 ConfigurableQuestionGenerator 모듈 reload 중...")
    importlib.reload(sys.modules[module_name])

from liberty_agent.b_rag.core.question_generation.configurable_question_generator import (
    ConfigurableQuestionGenerator,
    GenerationConfig,
    TestCase,
    create_standard_config,
    create_quick_config,
    create_comprehensive_config
)

# FAISS 관련 import
from langchain_community.vectorstores import FAISS
from langchain_upstage import UpstageEmbeddings

def load_faiss_vectorstore():
    """FAISS 벡터스토어 로드"""
    print("🔄 FAISS 벡터스토어 로드 중...")
    
    # 임베딩 모델 초기화
    dense_embedder = UpstageEmbeddings(model="solar-embedding-1-large-query")
    
    # FAISS 캐시 경로들 확인
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

def extract_test_cases_from_faiss(vectorstore, num_cases: int = 5) -> List[TestCase]:
    """FAISS DB에서 다양한 테스트 케이스 추출"""
    print(f"📋 FAISS DB에서 {num_cases}개의 테스트 케이스 추출 중...")
    
    # 카테고리별 검색어
    category_search_mapping = {
        "민사": ["계약", "채권", "손해배상", "소유권"],
        "행정": ["행정처분", "행정소송", "취소소송"],
        "형사": ["절도", "사기", "폭행", "무고"],
        "금융": ["세무", "조세", "금융", "투자"],
        "근로": ["근로", "임금", "해고", "노동"],
        "기업": ["회사", "법인", "상법", "주주"],
        "가사": ["이혼", "양육", "상속", "가족"],
        "특허": ["특허", "저작권", "지적재산"]
    }
    
    # 다양한 문서 수집
    collected_docs = []
    doc_ids = set()
    
    categories = list(category_search_mapping.keys())
    random.shuffle(categories)  # 카테고리 순서 랜덤화
    
    for category in categories:
        if len(collected_docs) >= num_cases:
            break
            
        search_terms = category_search_mapping[category]
        for search_term in search_terms:
            try:
                docs = vectorstore.similarity_search(search_term, k=3)
                for doc in docs:
                    doc_id = doc.page_content[:100]
                    if (doc_id not in doc_ids and 
                        len(doc.page_content) > 200 and 
                        len(collected_docs) < num_cases):
                        
                        collected_docs.append(doc)
                        doc_ids.add(doc_id)
                        
                if len(collected_docs) >= num_cases:
                    break
            except Exception as e:
                print(f"⚠️ 검색 오류 ({search_term}): {e}")
                continue
    
    # TestCase 객체로 변환
    test_cases = []
    for i, doc in enumerate(collected_docs):
        content = doc.page_content
        metadata = doc.metadata if hasattr(doc, 'metadata') else {}
        doc_category = metadata.get('category', 'unknown')
        
        # 문서 내용 기반 GT 질문 생성
        gt_question = generate_gt_question_from_content(content, doc_category)
        
        # 키워드 추출
        keywords = extract_keywords_from_content(content)
        
        test_case = TestCase(
            name=f"케이스_{i+1}_{doc_category}",
            gt_question=gt_question,
            document_content=content[:1000],
            keywords=", ".join(keywords[:5]),
            metadata={
                "category": doc_category,
                "source": "FAISS_DB",
                "case_index": i+1
            }
        )
        test_cases.append(test_case)
    
    print(f"✅ {len(test_cases)}개의 테스트 케이스 생성 완료")
    return test_cases

def generate_gt_question_from_content(content: str, category: str) -> str:
    """문서 내용과 카테고리 기반 GT 질문 생성"""
    # 카테고리별 질문 템플릿
    templates = {
        "민사": [
            "계약이 유효하다고 할 수 있는가?",
            "채권자의 권리가 인정되는가?",
            "손해배상 책임이 성립하는가?",
            "소유권 이전이 유효한가?"
        ],
        "행정": [
            "행정처분이 적법하다고 할 수 있는가?",
            "취소소송이 인용될 수 있는가?",
            "행정절차가 위법하다고 볼 수 있는가?"
        ],
        "형사": [
            "범죄가 성립한다고 할 수 있는가?",
            "구성요건이 충족되는가?",
            "정당방위가 인정되는가?"
        ],
        "기업": [
            "이사의 책임이 인정되는가?",
            "주주권이 침해되었다고 할 수 있는가?",
            "회사의 결정이 유효한가?"
        ],
        "default": [
            "법적 권리가 인정되는가?",
            "해당 조치가 적법한가?",
            "책임이 성립한다고 할 수 있는가?"
        ]
    }
    
    category_templates = templates.get(category, templates["default"])
    return random.choice(category_templates)

def extract_keywords_from_content(content: str) -> List[str]:
    """문서 내용에서 키워드 추출"""
    # 간단한 키워드 추출 (실제로는 더 정교한 방법 사용 가능)
    keywords = []
    
    # 법률 관련 주요 키워드들
    legal_terms = [
        "민법", "형법", "상법", "행정법", "계약", "채권", "소유권", 
        "손해배상", "불법행위", "과실", "고의", "책임", "권리", "의무",
        "법원", "판결", "소송", "취소", "무효", "효력", "성립", "인정"
    ]
    
    for term in legal_terms:
        if term in content:
            keywords.append(term)
    
    return keywords[:10]  # 상위 10개만 반환

def run_flexible_question_generation_test():
    """유연한 질문 생성 테스트 실행"""
    print("🚀 확장 가능한 B-RAG 질문 생성 테스트 시작")
    print("=" * 80)
    
    # === 설정 옵션들 ===
    print("📋 설정 옵션:")
    print("1. 빠른 테스트 (5개 질문)")
    print("2. 표준 테스트 (10개 질문)")  
    print("3. 포괄적 테스트 (20개 질문)")
    print("4. 사용자 정의")
    
    choice = input("선택하세요 (1-4): ").strip()
    
    if choice == "1":
        config = create_quick_config(num_questions=5)
        num_test_cases = 3
    elif choice == "2":
        config = create_standard_config(num_questions=10)
        num_test_cases = 5
    elif choice == "3":
        config = create_comprehensive_config(num_questions=20)
        num_test_cases = 3
    elif choice == "4":
        num_questions = int(input("질문 개수 (5-50): "))
        confidence_threshold = float(input("확신도 임계값 (0.7-0.99): "))
        num_test_cases = int(input("테스트 케이스 수 (1-10): "))
        
        config = GenerationConfig(
            num_questions=num_questions,
            min_level=1,
            max_level=10,
            difficulty_mode="enhanced",
            confidence_threshold=confidence_threshold,
            temperature=0.2
        )
    else:
        config = create_standard_config()
        num_test_cases = 5
    
    print(f"\n✅ 설정 완료:")
    print(f"   - 질문 개수: {config.num_questions}개")
    print(f"   - 확신도 임계값: {config.confidence_threshold}")
    print(f"   - 테스트 케이스 수: {num_test_cases}개")
    
    try:
        # FAISS DB 로드
        vectorstore, faiss_path = load_faiss_vectorstore()
        
        # 테스트 케이스 추출
        test_cases = extract_test_cases_from_faiss(vectorstore, num_test_cases)
        
        # 질문 생성기 초기화
        generator = ConfigurableQuestionGenerator(config)
        
        # 배치 처리 실행
        print(f"\n🔄 {len(test_cases)}개 케이스에 대해 배치 처리 시작...")
        results = generator.batch_generate_questions(
            test_cases=test_cases,
            save_individual=True,
            output_dir="flexible_generation_results"
        )
        
        # 결과 요약 출력
        print(f"\n📊 전체 결과 요약:")
        print(f"   성공한 케이스: {results['successful_cases']}/{results['total_cases']}")
        print(f"   총 생성된 질문: {results['total_questions_generated']}개")
        print(f"   평균 확신도: {results['average_confidence']:.3f}")
        print(f"   평균 일관성: {results['average_consistency']:.1%}")
        print(f"   총 소요시간: {results['total_processing_time']:.1f}초")
        
        # 개별 케이스 결과
        print(f"\n📝 개별 케이스 결과:")
        for i, case_result in enumerate(results['case_results']):
            if case_result['success']:
                stats = case_result['result'].get_statistics()
                print(f"   케이스 {i+1}: ✅ {stats['total_questions']}개 질문 | "
                      f"Yes {stats['yes_count']}개 | "
                      f"일관성 {stats['consistency_rate']:.1%}")
            else:
                print(f"   케이스 {i+1}: ❌ 실패 - {case_result.get('error', '알 수 없는 오류')}")
        
        print(f"\n💾 결과 파일 저장 위치: flexible_generation_results/")
        
    except Exception as e:
        print(f"❌ 테스트 실행 중 오류 발생: {e}")
        print("⚠️ FAISS DB 연결을 확인하거나 설정을 조정해주세요.")

if __name__ == "__main__":
    run_flexible_question_generation_test() 