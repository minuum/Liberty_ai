from datetime import datetime
import traceback
from langchain_upstage import UpstageEmbeddings, UpstageGroundednessCheck
from transformers import AutoModel, AutoTokenizer
import torch
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import secrets
from tqdm.auto import tqdm
import time
import logging
from langchain_teddynote.community.pinecone import PineconeKiwiHybridRetriever
from pathlib import Path
from langchain.schema import Document
# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LegalSearchEngine:
    def __init__(
        self, 
        retrievers: Dict,
        sparse_encoder,
        pinecone_index,
        namespace: str = None,
        cache_dir: Optional[str] = "./cached_vectors/search_engine"
    ):
        self.retrievers = retrievers
        self.sparse_encoder = sparse_encoder
        self.index = pinecone_index
        self.namespace = namespace
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.context_window = []
        
        # 검증기 초기화
        self.upstage_checker = UpstageGroundednessCheck()
        self._setup_kobert()
        
        # 검색 결과 캐시 초기화
        self.result_cache = {}
        
    def _setup_kobert(self):
        """KoBERT 모델 및 토크나이저 설정"""
        self.kobert_model = AutoModel.from_pretrained("monologg/kobert")
        self.kobert_tokenizer = AutoTokenizer.from_pretrained(
            "monologg/kobert",
            trust_remote_code=True
        )
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.kobert_model = self.kobert_model.to(self.device)
        
    def _generate_hash(self) -> str:
        """고유 해시 생성"""
        return secrets.token_hex(8)
        
    def process_batch(
        self,
        batch: List[int],
        contents: List[str],
        metadatas: Dict[str, List],
        sparse_encoder,
        namespace: str
    ) -> Optional[Dict]:
        """
        배치 단위로 문서를 처리하여 Pinecone에 업로드
        
        Args:
            batch: 현재 처리할 문서들의 인덱스 리스트
            contents: 전체 문서 내용 리스트
            metadatas: 전체 문서의 메타데이터
            sparse_encoder: 희소 임베딩 인코더
            namespace: Pinecone 네임스페이스
        """
        try:
            # 현재 배치 데이터 추출
            context_batch = [contents[i] for i in batch]
            
            # 임베딩 생성
            dense_vectors = self.dense_embedder.embed_documents(context_batch)
            sparse_vectors = sparse_encoder.encode_documents(context_batch)
            
            # 벡터 생성
            ids = [self._generate_hash() for _ in range(len(batch))]
            
            vectors = [
                {
                    "id": id_,
                    "values": dense,
                    "sparse_values": sparse,
                    "metadata": {
                        key: metadatas[key][i] 
                        for key in metadatas.keys()
                    }
                }
                for id_, dense, sparse, i in zip(
                    ids, dense_vectors, sparse_vectors, batch
                )
            ]
            
            # Pinecone 업로드
            return self.index.upsert(
                vectors=vectors,
                namespace=namespace,
                async_req=False
            )
            
        except Exception as e:
            logger.error(f"배치 ���리 중 오류 발생: {str(e)}")
            return None
            
    def batch_upload(
        self,
        contents: List[str],
        metadatas: Dict[str, List],
        sparse_encoder,
        namespace: str,
        start_idx: int = 0
    ):
        """
        문서를 배치 단위로 Pinecone에 업로드
        
        Args:
            contents: 전체 문서 내용 리스트
            metadatas: 전체 문서의 메타데이터
            sparse_encoder: 희소 임베딩 인코더
            namespace: Pinecone 네임스페이스
            start_idx: 시작 인덱스
        """
        # 데이터 슬라이싱
        contents_subset = contents[start_idx:]
        metadata_subset = {
            key: values[start_idx:] 
            for key, values in metadatas.items()
        }
        
        # 배치 생성
        batches = [
            range(i, min(i + self.batch_size, len(contents_subset)))
            for i in range(0, len(contents_subset), self.batch_size)
        ]
        
        # 진행 상황 표시
        with tqdm(total=len(contents_subset), desc="문서 업로드 중") as pbar:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [
                    executor.submit(
                        self.process_batch,
                        batch,
                        contents_subset,
                        metadata_subset,
                        sparse_encoder,
                        namespace
                    )
                    for batch in batches
                ]
                
                start_time = time.time()
                processed_docs = 0
                
                for idx, future in enumerate(as_completed(futures)):
                    result = future.result()
                    processed_docs += len(batches[idx])
                    
                    # 진행 상황 업데이트
                    elapsed_time = time.time() - start_time
                    progress = processed_docs / len(contents_subset)
                    estimated_total = elapsed_time / progress if progress > 0 else 0
                    remaining_time = estimated_total - elapsed_time
                    
                    pbar.set_postfix({
                        'progress': f'{processed_docs}/{len(contents_subset)}',
                        'remaining': f'{remaining_time:.1f}s'
                    })
                    pbar.update(len(batches[idx]))
                    
    def setup_hybrid_retriever(self, sparse_encoder):
        """하이브리드 검색기 설정"""
        logger.info("하이브리드 검색기 초기화 중...")
        try:
            pinecone_params = {
                "index": self.index,
                "namespace": self.namespace,
                "embeddings": self.dense_embedder,
                "sparse_encoder": sparse_encoder
            }
            self.hybrid_retriever = PineconeKiwiHybridRetriever(**pinecone_params)
            logger.info("하이브리드 검색기 초기화 완료")
            return self.hybrid_retriever
        except Exception as e:
            logger.error(f"하이브리드 검색기 초기화 실패: {str(e)}")
            raise
            
    def hybrid_search(self, query: str, top_k: int = 5) -> List[Document]:
        """개선된 하이브리드 검색"""
        try:
            # 1. 쿼리 의도 분석
            query_intent = self._analyze_query_intent(query)
            logger.info(f"쿼리 의도 분석 결과: {query_intent}")
            
            # 2. 검색 파라미터 설정
            search_params = self._create_search_params(query_intent)
            
            # 3. 메인 검색 실행
            try:
                results = self.retrievers['pinecone'].invoke(
                    query,
                    search_kwargs=search_params
                )
                
                # 4. 결과 품질 평가
                if results:
                    quality_score = self._evaluate_result_quality(results, query_intent)
                    logger.info(f"검색 결과 품질 점수: {quality_score}")
                    
                    if quality_score >= 0.5:  # 품질 임계값
                        return self._post_process_results(results, query)
                
                # 5. 폴백 검색 실행
                logger.info("메인 검색 실패, 폴백 검색 시작")
                fallback_results = self._get_fallback_results(query, query_intent)
                
                return fallback_results
                
            except Exception as search_error:
                logger.error(f"검색 실행 중 오류: {str(search_error)}")
                return self._get_fallback_results(query, query_intent)
                
        except Exception as e:
            logger.error(f"하이브리드 검색 중 오류: {str(e)}")
            return [Document(
                page_content="시스템 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
                metadata={"source": "error", "reliability": "low"}
            )]

    def _expand_legal_query(self, query: str) -> str:
        """법률 도메인 특화 쿼리 확장"""
        legal_terms = {
            '전세': ['임대차', '임차인', '보증금', '임대인'],
            '사기': ['기망', '편취', '배임', '횡령'],
            '계약': ['약정', '합의', '채무', '이행'],
            '소송': ['민사', '형사', '고소', '고발']
        }
        
        expanded_terms = []
        for key, values in legal_terms.items():
            if key in query:
                expanded_terms.extend(values)
        
        return f"{query} {' '.join(expanded_terms)}"

    def _post_process_results(self, results: List[Document]) -> List[Document]:
        """검색 결과 후처리"""
        for doc in results:
            base_score = doc.metadata.get('score', 0)
            
            # 1. 판례 최신성 가중치
            year = self._extract_year(doc.page_content)
            recency_weight = self._calculate_recency_weight(year)
            
            # 2. 판례
            #  중요도 가중치
            importance_weight = self._calculate_importance_weight(doc)
            
            # 최종 점수 계산
            doc.metadata['adjusted_score'] = base_score * recency_weight * importance_weight
            
        return results

    def _merge_and_rerank(self, results: Dict[str, List[Document]], query: str) -> List[Document]:
        """결과 통합 및 재순위화"""
        all_results = []
        
        # 카테고리별 가중치
        weights = {
            'criminal': 1.2,
            'civil': 1.0,
            'procedure': 0.8
        }
        
        for category, docs in results.items():
            for doc in docs:
                weight = weights.get(category, 1.0)
                doc.metadata['adjusted_score'] *= weight
                all_results.append(doc)
        
        # 컨텍스트 기반 가중치 적용
        if self.context_window:
            all_results = self._apply_context_weights(all_results, query)
        
        return sorted(all_results, 
                     key=lambda x: x.metadata['adjusted_score'], 
                     reverse=True)

    def _update_context(self, query: str, results: List[Document]):
        """컨텍스트 업데이트"""
        # 컨텍스트 윈도우 크기 제한
        if len(self.context_window) >= 5:
            self.context_window.pop(0)
        
        # 컨텍스트 윈도우에 결과 추가
        self.context_window.append(results)

    def _apply_context_weights(self, results: List[Document], query: str) -> List[Document]:
        """컨텍스트 기반 가중치 적용"""
        # 컨텍스트 윈도우 크기 제한
        if len(self.context_window) > 5:
            self.context_window = self.context_window[-5:]
        
        # 컨텍스트 윈도우에 따른 가중치 계산
        for doc in results:
            context_weight = 1.0
            for context in self.context_window:
                context_weight *= self._calculate_context_similarity(query, doc.page_content)
            
            doc.metadata['adjusted_score'] *= context_weight
        
        return results

    def _calculate_context_similarity(self, query: str, context: str) -> float:
        """컨텍스트 유사도 계산"""
        # 컨텍스트 유사도 계산 로직 구현
        pass

    def _extract_year(self, text: str) -> int:
        """판례 연도 추출"""
        # 판례 연도 추출 로직 구현
        pass

    def _calculate_recency_weight(self, year: int) -> float:
        """례 최신성 가중치 계산"""
        # 판례 최신성 가중치 계산 로직 구현
        pass

    def _calculate_importance_weight(self, doc: Document) -> float:
        """판례 중요도 가중치 계산"""
        # 판례 중요도 가중치 계산 로직 구현
        pass

    def validate_answer(
        self,
        context: List[Document],
        answer: str,
        upstage_weight: float = 0.3,
        kobert_weight: float = 0.7
    ) -> float:
        """
        답변의 신뢰도 검증
        """
        logger.info(f"""
        === 답변 신뢰도 검증 시작 ===
        컨텍스트 수: {len(context)}
        답변 길이: {len(answer)}
        """)
        
        try:
            # Upstage 검증
            upstage_response = self.upstage_checker.run(
                {"context": context, "answer": answer}
            )   
            logger.info(f"Upstage 검증 결과: {upstage_response}")
            
            # KoBERT 검증
            kobert_score = self._kobert_check(context, answer)
            logger.info(f"KoBERT 유사도 점수: {kobert_score:.3f}")
            
            # Upstage 점수 변환
            upstage_score = {
                "grounded": 1.0,
                "notGrounded": 0.0,
                "notSure": 0.33
            }.get(upstage_response, 0.33)
            logger.info(f"Upstage 변환 점수: {upstage_score:.3f}")
            
            # 결합 점수 계산
            final_score = (upstage_weight * upstage_score) + (kobert_weight * kobert_score)
            
            logger.info(f"""
            === 최종 신뢰도 점수 ===
            Upstage 가중치: {upstage_weight} × 점수: {upstage_score:.3f} = {upstage_weight * upstage_score:.3f}
            KoBERT 가중치: {kobert_weight} × 점수: {kobert_score:.3f} = {kobert_weight * kobert_score:.3f}
            최종 결합 점수: {final_score:.3f}
            """)
            
            return final_score
            
        except Exception as e:
            logger.error(f"신뢰도 검증 중 오류 발생: {str(e)}")
            raise
        
    def _kobert_check(self, context: List[Document], answer: str) -> float:
        """KoBERT를 사용한 관련성 점수 계산"""
        logger.info("=== KoBERT 관련성 검사 시작 ===")
        
        try:
            # Document 객체에서 텍스트 추출
            context_text = [doc.page_content for doc in context]
            
            # 컨텍스트가 비어있으면 기본값 반환
            if not context_text or not answer:
                logger.warning("컨텍스트 또는 답변이 비어있음")
                return 0.0
                
            # 컨텍스트 텍스트 결합
            combined_context = " ".join(context_text)
            
            # 토크나이즈
            context_inputs = self.kobert_tokenizer(
                combined_context,  # str 타입으로 변환된 컨텍스트
                return_tensors="pt",
                truncation=True,
                max_length=512,  # 최대 길이 제한
                padding=True
            )
            
            answer_inputs = self.kobert_tokenizer(
                answer,  # 이미 str 타입
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            logger.info(f"""
            토큰화 결과:
            컨텍스트 토큰 수: {context_inputs['input_ids'].shape[1]}
            답변 토큰 수: {answer_inputs['input_ids'].shape[1]}
            """)
            
            # 디바이스 이동
            context_inputs = {
                k: v.to(self.device) for k, v in context_inputs.items()
            }
            answer_inputs = {
                k: v.to(self.device) for k, v in answer_inputs.items()
            }
            
            # 임베딩 생성
            with torch.no_grad():
                context_outputs = self.kobert_model(**context_inputs)
                answer_outputs = self.kobert_model(**answer_inputs)
            
            # 평균 풀링
            context_embedding = context_outputs.last_hidden_state.mean(dim=1)
            answer_embedding = answer_outputs.last_hidden_state.mean(dim=1)
            
            # 코사인 유사도 계산
            similarity = torch.nn.functional.cosine_similarity(
                context_embedding,
                answer_embedding
            )
            
            # 점수 변환 (0~1 범위)
            final_score = (similarity.item() + 1) / 2
            
            logger.info(f"""
            === KoBERT 점수 계산 완료 ===
            원본 코사인 유사도: {similarity.item():.3f}
            정규화된 최종 점수: {final_score:.3f}
            """)
            
            return final_score
            
        except Exception as e:
            logger.error(f"KoBERT 검사 중 오류: {str(e)}")
            return 0.0

    def _preprocess_query(self, query: str) -> str:
        """쿼리 전처리"""
        try:
            # 1. 법률 용어 정규화
            legal_terms = {
                "이혼": ["이혼", "혼인관계해소", "혼인파탄"],
                "소유권": ["소유권", "물권", "소유권이전"],
                "귀책사유": ["귀책사유", "책임", "과실"],
                # ... 추가 법률 용어 매핑
            }
            
            # 2. 쿼리 확장
            expanded_terms = []
            for term, synonyms in legal_terms.items():
                if term in query:
                    expanded_terms.extend(synonyms)
            
            # 3. 메타데이터 키워드 추출
            metadata_fields = [
                "caseNm",      # 사건명
                "courtType",   # 법원 유형
                "class_name",  # 분류명
                "keyword"      # 키워드 태그
            ]
            
            # 4. 쿼리 재구성
            processed_query = f"""
            원본 쿼리: {query}
            법률 용어: {' '.join(expanded_terms)}
            검색 범위: {' '.join(metadata_fields)}
            """
            
            logger.info(f"""
            === 쿼리 전처리 결과 ===
            원본: {query}
            확장: {expanded_terms}
            메타데이터 필드: {metadata_fields}
            """)
            
            return processed_query
            
        except Exception as e:
            logger.error(f"쿼리 전처리 중 오류: {str(e)}")
            return query  # 오류 발생 시 원본 쿼리 반환

    def _create_dynamic_filters(self, query_intent: Dict, top_k: int = 5) -> Dict:
        """동적 검색 파라미터 생성"""
        search_params = {
            "top_k": top_k,
            "alpha": 0.7,
            "namespace": self.namespace,
            "filter": {}
        }
        
        # 기본 필터 완화
        search_params["filter"] = {
            "class_name": {
                "$in": ["민사", "가사", "형사A", "형사B", "행정", "기업", "근로자", 
                       "특허.저작권", "금융조세", "개인정보/ICT"]
            }
        }
        
        # 가사 사건 관련 키워드가 있을 경우 필터 조정
        if any(kw in query_intent.get("keywords", []) for kw in ["이혼", "양육", "친권", "상속"]):
            search_params["filter"]["class_name"] = {"$in": ["가사", "민사"]}
            search_params["alpha"] = 0.8  # 정확도 가중치 증가
        
        return search_params

    def _analyze_query_intent(self, query: str) -> Dict:
        logger.info("=== 쿼리 의도 분석 시작 ===")
        """
        쿼리 의도를 단계적으로 분석하여 검색 범위를 좁혀가는 로직
        """
        intent = {
            "legal_areas": [],        # 법률 분야
            "sub_categories": [],     # 세부 카테고리
            "case_types": [],         # 사건 유형
            "temporal_info": {},      # 시간 정보
            "court_levels": [],       # 법원 단계
            "importance": "normal",   # 중요도
            "specificity": "general", # 구체성
            "keywords": set()         # 추출된 키워드
        }

        # 1단계: 기본 법률 분야 분류
        PRIMARY_AREAS = {
            "상속��": ["상속", "상속인", "상속순위", "상속권", "유산"],
            "가족법": ["가족", "친족", "혼인", "이혼", "입양"],
            "민사": ["계약", "손해배상", "소유권", "채권", "부동산", "임대차", "보증", "매매"],
            "가사": ["이혼", "상속", "친권", "양육", "혼인", "부양", "가족"],
            "형사A": ["폭행", "절도", "사기", "강도", "폭력", "협박", "모욕"],
            "형사B": ["살인", "횡령", "배임", "조직범죄", "마약", "뇌물"],
            "행정": ["인허가", "행정처분", "과태료", "영업정지", "취소"],
            "기업": ["회사", "주주", "이사회", "합병", "주식"],
            "근로자": ["해고", "임금", "퇴직금", "산재", "근로"],
            "특허": ["특허", "저작권", "상표", "영업비밀"],
            "금융조세": ["세금", "금융", "보험", "조세", "납세"],
            "개인정보": ["개인정보", "정보보호", "사이버", "전자"]
        }

        # 2단계: 세부 카테고리 분류
        SUB_CATEGORIES = {
            "상속법": {
                "법정상속": ["상속순위", "상속인", "법정상속인"],
                "유언상속": ["유언", "유언장", "유증"],
                "상속포기": ["상속포기", "상속승인", "한정승인"]
            },
            "민사": {
                "계약관계": ["계약", "약정", "합의", "위약"],
                "손해배상": ["손해", "배상", "보상", "책임"],
                "부동산": ["등기", "소유권", "임대차", "전세"],
                "금전관계": ["대여금", "채무", "변제", "이자"]
            },
            "형사A": {
                "폭력": ["폭행", "상해", "협박", "폭력"],
                "재산범죄": ["절도", "사기", "횡령", "배임"],
                "명예훼손": ["명예훼손", "모욕", "비방"]
            },
            "가사": {
                "이혼": ["이혼", "협의이혼", "재판이혼", "위자료"],
                "양육": ["양육권", "친권", "양육비", "면접교섭"],
                "상속": ["상속", "유산", "상속포기", "유류분"],
                "가정폭력": ["가정폭력", "접근금지", "보호명령", "피해자보호"]
            },
            "형사B": {
                "중범죄": ["살인", "강도", "강간", "방화"],
                "경제범죄": ["횡령", "배임", "사기", "탈세"],
                "조직범죄": ["조직폭력", "마약", "밀수", "사채"]
            },
            "행정": {
                "인허가": ["허가", "신고", "등록", "취소"],
                "행정처분": ["과태료", "영업정지", "시정명령"],
                "행정심판": ["행정심판", "행정소송", "취소소송"]
                }
        }

        # 3단계: 사건 처리 단계
        CASE_STAGES = {
            "1심": ["1심", "제1심", "지방법원", "단독판사"],
            "항소심": ["2심", "항소", "고등법원"],
            "상고심": ["3심", "상고", "대법원", "파기환송"]
        }

        def _extract_temporal_info(query: str) -> dict:
            """시간 정보 추출 개선"""
            temporal = {}
            year_pattern = r'\d{4}년|\d{2}년'
            month_pattern = r'\d{1,2}개월'
            
            # 기간 정보 추출
            if "이내" in query or "전부터" in query:
                temporal["period_type"] = "within"
            elif "이후" in query or "이래" in query:
                temporal["period_type"] = "after"
            
            # 구체적 시점 추출
            if "최근" in query:
                temporal["recency"] = True
            
            return temporal

        def _analyze_importance(query: str) -> str:
            """중요도 분석 개선"""
            importance_patterns = {
                "high": ["중요", "핵심", "대표", "필수", "전원합의체", "판례변경"],
                "low": ["일반", "통상", "참고", "유사", "관련"],
                "urgent": ["긴급", "즉시", "급한", "시급"]
            }
            
            for level, patterns in importance_patterns.items():
                if any(p in query for p in patterns):
                    return level
            return "normal"
        # 쿼리 분석 실행
        try:
            # 1. 기본 법률 분야 확인
            for area, keywords in PRIMARY_AREAS.items():
                if any(kw in query for kw in keywords):
                    intent["legal_areas"].append(area)
                    intent["keywords"].update(set(kw for kw in keywords if kw in query))

            # 2. 선택된 분야의 세부 카테고리 분석
            for area in intent["legal_areas"]:
                if area in SUB_CATEGORIES:
                    for sub_cat, keywords in SUB_CATEGORIES[area].items():
                        if any(kw in query for kw in keywords):
                            intent["sub_categories"].append(sub_cat)
                            intent["keywords"].update(set(kw for kw in keywords if kw in query))

            # 3. 사건 처리 단계 분석
            for stage, keywords in CASE_STAGES.items():
                if any(kw in query for kw in keywords):
                    intent["court_levels"].append(stage)

            # 4. 시간 정보 분석
            intent["temporal_info"] = _extract_temporal_info(query)

            # 5. 중요도 분석
            intent["importance"] = _analyze_importance(query)

            # 6. 구체성 분석 개선
            specificity_score = len(intent["keywords"]) + \
                              len(intent["legal_areas"]) * 2 + \
                              len(intent["sub_categories"]) * 3
            
            intent["specificity"] = "specific" if specificity_score >= 5 else \
                                  "moderate" if specificity_score >= 3 else "general"

            logger.info(f"""
            === 쿼리 의도 분석 결과 ===
            법률 분야: {intent['legal_areas']}
            세부 카테고리: {intent['sub_categories']}
            법원 단계: {intent['court_levels']}
            시간 정보: {intent['temporal_info']}
            중요도: {intent['importance']}
            구체성: {intent['specificity']}
            추출 키워드: {intent['keywords']}
            """)

            return intent

        except Exception as e:
            logger.error(f"쿼리 의도 분석 중 오류 발생: {str(e)}")
            return {"error": str(e)}
    def _create_search_params(self, intent: Dict) -> Dict:
        """검색 파라미터 생성 함수"""
        params = {
            "filter": {},
            "weights": {
                "dense": 0.7,
                "sparse": 0.3
            },
            "top_k": 10,  # 기본값
            "score_threshold": 0.5
        }
        
        # 1. 법률 분야 필터
        if intent["legal_areas"]:
            params["filter"]["category"] = {"$in": intent["legal_areas"]}
            
            # 가사 사건 특별 처리
            if "가사" in intent["legal_areas"]:
                if any(kw in intent["keywords"] for kw in ["이혼", "양육", "가정폭력"]):
                    params["weights"]["sparse"] = 0.5  # 키워드 중요도 증가
        
        # 2. 법원 단계 필터
        if intent["court_levels"]:
            params["filter"]["court_level"] = {"$in": intent["court_levels"]}
        
        # 3. 시간 정보 필터
        if intent["temporal_info"]:
            if "recency" in intent["temporal_info"]:
                params["filter"]["judgment_year"] = {
                    "$gte": datetime.now().year - 5  # 최근 5년
                }
            if "period_type" in intent["temporal_info"]:
                if intent["temporal_info"]["period_type"] == "within":
                    params["filter"]["judgment_year"] = {
                        "$gte": datetime.now().year - 3  # 3년 이내
                    }
        
        # 4. 가중치 동적 조정
        if intent["specificity"] == "specific":
            params["weights"]["sparse"] = 0.5  # 키워드 매칭 중요도 증가
            params["top_k"] = 7  # 더 적은 결과
        elif intent["specificity"] == "general":
            params["weights"]["dense"] = 0.8  # 의미적 유사도 중요도 증가
            params["top_k"] = 15  # 더 많은 결과
        
        # 5. 중요도에 따른 조정
        if intent["importance"] == "high":
            params["score_threshold"] = 0.7  # 더 높은 임계값
        elif intent["importance"] == "urgent":
            params["top_k"] = 5  # 더 적은, 정확한 결과
        
        return params
    
    def _get_fallback_results(self, query: str, intent: Dict) -> List[Document]:
        """검색 결과가 없을 때의 개선된 폴백 로직"""
        try:
            fallback_results = []
            
            # 1. 기본 법률 정보 데이터베이스
            DEFAULT_LEGAL_INFO = {
                "상속": {
                    "content": """
                    상속 순위에 대한 기본 정보:
                    1순위: 직계비속 (자녀, 손자녀 등)
                    2순위: 직계존속 (부모, 조부모 등)
                    3순위: 형제자매
                    4순위: 4촌 이내의 방계혈족
                    
                    관련 법령: 민법 제1000조, 제1003조
                    자세한 상담은 법률구조공단을 통해 받으실 수 있습니다.
                    """,
                    "metadata": {
                        "source": "default_response",
                        "category": "상속법",
                        "reliability": "high"
                    }
                },
                "이혼": {
                    "content": """
                    이혼의 종류:
                    1. 협의이혼: 부부 간의 합의로 이루어지는 이혼
                    2. 재판이혼: 법원의 판결로 이루어지는 이혼
                    
                    필요 서류:
                    - 협의이혼: 협의이혼의사확인신청서, 혼인관계증명서 등
                    - 재판이혼: 이혼청구서, 증거자료 등
                    
                    자세한 상담은 가까운 법원 또는 법률구조공단을 방문하시기 바랍니다.
                    """,
                    "metadata": {
                        "source": "default_response",
                        "category": "가족법",
                        "reliability": "high"
                    }
                }
                # 다른 주요 법률 분야 추가...
            }
            
            # 2. 키워드 기반 폴백 검색
            if intent["keywords"]:
                basic_query = " ".join(intent["keywords"])
                keyword_results = self.retrievers['pinecone'].invoke(
                    basic_query,
                    search_kwargs={
                        "top_k": 3,
                        "filter": {"reliability": {"$gte": 0.7}}
                    }
                )
                if keyword_results:
                    fallback_results.extend(keyword_results)
            
            # 3. 법률 분야 기반 기본 정보 제공
            for area in intent["legal_areas"]:
                area_lower = area.replace("법", "")  # "상속법" -> "상속"
                if area_lower in DEFAULT_LEGAL_INFO:
                    info = DEFAULT_LEGAL_INFO[area_lower]
                    fallback_results.append(
                        Document(
                            page_content=info["content"],
                            metadata=info["metadata"]
                        )
                    )
            
            # 4. 일반적인 법률 정보 문서 (아무 결과도 없을 때)
            if not fallback_results:
                fallback_results.append(Document(
                    page_content="""
                    죄송합니다. 귀하의 질문에 대한 정확한 정보를 찾지 못했습니다.
                    보다 정확한 법률 상담을 위해서는:
                    1. 대한법률구조공단 (국번없이 132)
                    2. 가까운 법률구조공단 지부
                    3. 법률구조공단 홈페이지 (https://www.klac.or.kr/)
                    를 이용해주시기 바랍니다.
                    """,
                    metadata={
                        "source": "default_response",
                        "reliability": "medium",
                        "category": "general"
                    }
                ))
            
            return fallback_results
                
        except Exception as e:
            logger.error(f"폴백 검색 중 오류: {str(e)}")
            # 최소한의 응답 보장
            return [Document(
                page_content="죄송합니다. 일시적인 시스템 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
                metadata={"source": "error", "reliability": "low"}
            )]
    def _adjust_search_weights(self, intent: Dict, initial_results: List[Document]) -> Dict:
        """검색 결과에 따른 가중치 동적 조정"""
        weights = {
            "dense": 0.7,
            "sparse": 0.3,
            "title": 0.1,
            "recency": 0.05
        }
        
        try:
            # 1. 결과 수에 따른 조정
            if len(initial_results) < 3:
                weights["dense"] -= 0.1  # 더 넓은 검색
                weights["sparse"] += 0.1
            
            # 2. 결과 품질에 따른 조정
            avg_score = sum(doc.metadata.get('score', 0) for doc in initial_results) / len(initial_results) if initial_results else 0
            if avg_score < 0.5:
                weights["dense"] += 0.1  # 의미적 유사도 강화
            
            # 3. 쿼리 특성에 따른 조정
            if len(intent["keywords"]) > 3:
                weights["sparse"] += 0.15  # 키워드 매칭 강화
            
            # 4. 시간 정보에 따른 조정
            if intent["temporal_info"].get("recency"):
                weights["recency"] += 0.1
            
            # 가중치 정규화
            total = sum(weights.values())
            return {k: v/total for k, v in weights.items()}
            
        except Exception as e:
            logger.error(f"가중치 조정 중 오류: {str(e)}")
            return weights

    def _evaluate_result_quality(self, results: List[Document], intent: Dict) -> float:
        """검색 결과 품질 평가 개선"""
        try:
            if not results:
                return 0.0
            
            quality_scores = []
            
            for doc in results:
                score = 0.0
                
                # 1. 기본 검색 점수 (가중치 상향)
                score += doc.metadata.get('score', 0) * 0.5  # 0.4 -> 0.5
                
                # 2. 키워드 매칭 (정확도 향상)
                content_lower = doc.page_content.lower()
                matched_keywords = sum(1 for kw in intent["keywords"] 
                                    if kw.lower() in content_lower)
                keyword_score = matched_keywords / max(1, len(intent["keywords"]))
                score += keyword_score * 0.3
                
                # 3. 법률 분야 일치도 (더 세밀한 매칭)
                doc_category = doc.metadata.get('category', '').lower()
                legal_area_match = any(
                    area.lower() in doc_category or 
                    doc_category in area.lower() 
                    for area in intent["legal_areas"]
                )
                if legal_area_match:
                    score += 0.2
                
                quality_scores.append(score)
            
            # 최소 품질 점수 설정
            avg_quality = sum(quality_scores) / len(quality_scores)
            return max(0.4, min(1.0, avg_quality))  # 최소 0.4 보장
            
        except Exception as e:
            logger.error(f"결과 품질 평가 중 오류: {str(e)}")
            return 0.4  # 오류 시 기본값 반환

    def _calculate_search_weight(self, query_intent: Dict) -> float:
        """검색 가중치 계산"""
        base_alpha = 0.7
        
        # 중요도에 따른 조정
        if query_intent["importance"] == "high":
            base_alpha += 0.1
        
        # 구체성에 따른 조정
        if query_intent["specificity"] == "specific":
            base_alpha += 0.1
        
        return min(base_alpha, 0.9)  # 최대 0.9로 제한
    




    def optimize_hybrid_search(self, query: str, top_k: int = 5) -> List[Document]:
        """최적화된 하이브리드 검색"""
        try:
            # 1. 쿼리 전처리
            processed_query = self._preprocess_query(query)
            logger.info(f"""
            === 쿼리 전처리 결과 ===
            원본: {query}
            처리됨: {processed_query}
            """)
            
            # 2. 쿼리 의도 분석
            query_intent = self._analyze_query_intent(processed_query)
            logger.info(f"""
            === 쿼리 의도 분석 결과 ===
            의도: {query_intent}
            """)
            
            # 3. 동적 가중치 계산
            alpha = self._calculate_dynamic_weights(processed_query)
            logger.info(f"동적 가중치(alpha): {alpha}")
            
            # 4. 검색 파라미터 최적화
            search_params = {
                "top_k": min(top_k * 3, 20),
                "alpha": alpha,
                "filter": self._create_metadata_filters(query_intent),
                "include_metadata": True
            }
            
            # 5. 검색 실행
            try:
                results = self.retrievers['pinecone'].invoke(
                    processed_query,
                    search_kwargs=search_params
                )
                
                if not results:
                    logger.warning("검색 결과 없음 - 폴백 결과 반환")
                    return self._get_fallback_results(query, query_intent)
                
                # 6. 결과 후처리 및 품질 검증
                processed_results = self._post_process_results(results, query)
                quality_score = self._evaluate_result_quality(processed_results, query_intent)
                
                logger.info(f"""
                === 검색 품질 평가 ===
                품질 점수: {quality_score}
                처리된 결과 수: {len(processed_results)}
                """)
                
                if quality_score < 0.4:
                    logger.warning(f"품질 점수 미달({quality_score}) - 폴백 결과 반환")
                    return self._get_fallback_results(query, query_intent)
                
                return processed_results[:top_k]
                
            except Exception as search_error:
                logger.error(f"검색 실행 중 오류: {str(search_error)}")
                return self._get_fallback_results(query, query_intent)
                
        except Exception as e:
            logger.error(f"하이브리드 검색 중 오류: {str(e)}")
            return []

    def _calculate_dynamic_weights(self, query: str) -> float:
        """개선된 동적 가중치 계산"""
        try:
            # 기본 dense 가중치
            alpha = 0.65  # 기본값 조정
            
            # 1. 쿼리 길이 기반 조정
            query_length = len(query.split())
            if query_length <= 3:
                alpha -= 0.15  # sparse 가중치 더 강화
            elif query_length >= 8:
                alpha += 0.15
            
            # 2. 전문 용어 가중치
            legal_terms = {
                "강": 0.1, "중": 0.15, "약": 0.2,
                "강도": ["판례", "법원", "소송"],
                "중도": ["계약", "위반", "책임"],
                "약도": ["문의", "질문", "상담"]
            }
            
            term_weights = []
            for term_type, terms in legal_terms.items():
                if isinstance(terms, list):
                    if any(term in query for term in terms):
                        term_weights.append(legal_terms[term_type[0]])
            
            if term_weights:
                alpha -= sum(term_weights) / len(term_weights)
            
            return max(0.2, min(0.8, alpha))  # 범위 조정
            
        except Exception as e:
            logger.error(f"가중치 계산 중 오류: {str(e)}")
            return 0.65  # 오류 시 기본값

    def _create_metadata_filters(self, query_intent: Dict) -> Dict:
        """메타데이터 필터 생성"""
        filters = {}
        
        # 1. 시간 기반 필터
        if "year" in query_intent:
            filters["judgment_year"] = {
                "$gte": query_intent["year"] - 5,
                "$lte": query_intent["year"] + 5
            }
        
        # 2. 법원 단계 필터
        if "court_level" in query_intent:
            filters["court_level"] = query_intent["court_level"]
        
        # 3. 사건 유형 필터
        if "case_type" in query_intent:
            filters["case_type"] = query_intent["case_type"]
        
        return filters

    def _post_process_results(self, results: List[Document], query: str) -> List[Document]:
        """검색 결과 후처리 및 재순위화"""
        for doc in results:
            score = doc.metadata.get('score', 0)
            
            # 1. 제목 유사도 보너스
            if 'title' in doc.metadata:
                title_similarity = self._calculate_similarity(query, doc.metadata['title'])
                score += title_similarity * 0.1
            
            # 2. 최신성 보너스
            if 'judgment_year' in doc.metadata:
                recency_bonus = self._calculate_recency_bonus(doc.metadata['judgment_year'])
                score += recency_bonus * 0.05
            
            # 3. 인용 횟수 보너스
            if 'citation_count' in doc.metadata:
                citation_bonus = min(doc.metadata['citation_count'] * 0.01, 0.1)
                score += citation_bonus
            
            doc.metadata['adjusted_score'] = score
        
        # 재정렬
        results.sort(key=lambda x: x.metadata.get('adjusted_score', 0), reverse=True)
        return results

    def search(self, query: str) -> List[Document]:
        """검색 실행"""
        try:
            logger.info(f"""
            === SEARCH 시작 ===
            쿼리: {query}
            """)
            
            # 하이브리드 검색 실행 (쿼리 의도 분석 포함)
            results = self.optimize_hybrid_search(query)
            logger.info(f"""
            === 하이브리드 검색 결과 ===
            결과 개수: {len(results)}
            스코어 범위: {min([doc.metadata.get('score', 0) for doc in results]) if results else 'N/A'} 
                        ~ {max([doc.metadata.get('score', 0) for doc in results]) if results else 'N/A'}
            """)

            return results
            
        except Exception as e:
            logger.error(f"""
            === SEARCH 오류 ===
            쿼리: {query}
            오류 메시지: {str(e)}
            스택 트레이스: {traceback.format_exc()}
            """)
            return []

    def _get_fallback_results(self, query: str, query_intent: dict) -> List[Document]:
        """폴백 결과 생성"""
        try:
            # 기본 응답 생성
            fallback_doc = Document(
                page_content="죄송합니다. 관련 판례를 찾을 수 없습니다.",
                metadata={
                    "score": 0.0,
                    "source": "fallback",
                    "query": query,
                    "intent": query_intent
                }
            )
            return [fallback_doc]
        except Exception as e:
            logger.error(f"폴백 결과 생성 중 오류: {str(e)}")
            return []