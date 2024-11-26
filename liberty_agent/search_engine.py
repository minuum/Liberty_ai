from datetime import datetime
import traceback
from langchain_upstage import UpstageEmbeddings, UpstageGroundednessCheck
from transformers import AutoModel, AutoTokenizer
import torch
from typing import Dict, List, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import secrets
from tqdm.auto import tqdm
import time
import logging
from langchain_teddynote.community.pinecone import PineconeKiwiHybridRetriever
from pathlib import Path
from langchain.schema import Document
import re
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
        try:
            self.kobert_model = AutoModel.from_pretrained("monologg/kobert")
            self.kobert_tokenizer = AutoTokenizer.from_pretrained(
                "monologg/kobert",
                trust_remote_code=True
            )
            
            # MPS 디바이스 처리 개선
            if torch.backends.mps.is_available():
                self.device = torch.device("cpu")  # MPS 대신 CPU 사용
                logger.info("MPS 가용하나 안정성을 위해 CPU 사용")
            else:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
            self.kobert_model = self.kobert_model.to(self.device)
            logger.info(f"KoBERT 모델 로드 완료 (device: {self.device})")
            
        except Exception as e:
            logger.error(f"KoBERT 설정 중 오류: {str(e)}")
            raise
        
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
            logger.error(f"배치 리 중 오류 발생: {str(e)}")
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
        # 컨스트 윈도우 크기 제한
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
        upstage_weight: float = 0.2,
        kobert_weight: float = 0.8
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
            
            # 코사인 유사도 계산ㅋ
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

    def _create_dynamic_filters(self, query_intent: Dict, top_k: int = 20) -> Dict:
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
            "이혼/가족": [
                "이혼", "협의이혼", "재판이혼", "위자료", "양육권", "친권", "양육비",
                "재산분할", "혼인", "가족관계", "가정폭력", "면접교섭", "부부상담"
            ],
            "상속": [
                "상속", "상속순위", "유류분", "상속포기", "유언장", "상속인", "법정상속",
                "유언", "한정승인", "상속재산", "상속세", "유증", "상속권"
            ],
            "민사/계약": [
                "계약서", "계약해지", "손해배상", "보증", "채무", "계약금", "위약금",
                "임대차", "매매", "해제", "해지", "불이행", "채권", "보증인", "민사소송"
            ],
            "부동산": [
                "매매", "임대차", "전세", "등기", "재개발", "재건축", "분양",
                "소유권", "임대인", "임차인", "보증금", "권리금", "부동산중개"
            ],
            "형사": [
                "고소", "고발", "변호사", "형사절차", "보석", "구속", "기소",
                "합의", "불기소", "형사고소", "고소장", "형사처벌", "선고", "폭행",
                "절도", "사기", "강도", "폭력", "협박", "모욕", "살인", "횡령", "배임"
            ],
            "행정": [
                "인허가", "행정처분", "과태료", "영업정지", "취소", "행정심판",
                "행정소송", "행정규제", "인가", "허가", "신고", "등록", "행정명령"
            ],
            "기업/회사": [
                "회사", "주주", "이사회", "합병", "주식", "법인", "기업회생",
                "회사설립", "증자", "감자", "주주총회", "이사", "감사", "경영권"
            ],
            "근로/노동": [
                "해고", "임금", "퇴직금", "산재", "근로", "노동", "연차",
                "직장내괴롭힘", "근로계약", "최저임금", "야근수당", "노동조합"
            ],
            "지식재산": [
                "특허", "저작권", "상표", "영업비밀", "디자인", "발명", "특허침해",
                "라이선스", "지식재산권", "특허등록", "상표등록", "저작물"
            ],
            "금융/세무": [
                "세금", "금융", "보험", "조세", "납세", "금융거래", "세무조사",
                "소득세", "부가가치세", "상속세", "증여세", "보험금", "대출"
            ],
            "개인정보/IT": [
                "개인정보", "정보보호", "사이버", "전자", "온라인", "데이터",
                "프라이버시", "해킹", "정보유출", "전자상거래", "SNS"
            ]
        }
        # 2단계: 세부 카테고리 분류 확장
        SUB_CATEGORIES = {
            "이혼/가족": {
                "이혼절차": ["이혼절차", "협의이혼", "재판이혼", "이혼절차", "이혼신청"],
                "위자료": ["위자료", "위자료청구", "위자료산정", "위자료소송"],
                "양육권": ["양육권", "친권", "양육비", "면접교섭권"],
                "재산분할": ["재산분할", "재산분할청구", "재산분할비율", "혼인재산"]
            },
            "상속": {
                "상속순위": ["상속순위", "법정상속인", "상속권", "상속인"],
                "유류분": ["유류분", "유류분청구", "유류분반환", "유류분권"],
                "상속포기": ["상속포기", "한정승인", "단순승인", "포기신고"],
                "유언장": ["유언장", "유언", "공정증서", "자필증서"]},
            "계약": {
                "계약작성": ["계약서작성", "표준계약서", "특약사항"],
                "계약해지": ["계약해지", "해지통보", "위약금", "해지사유"],
                "손해배상": ["손해배상", "배상책임", "손해액", "배상범위"],
                "보증": ["보증계약", "연대보증", "보증인", "보증책임"]},
            "부동산": {
                "매매": ["부동산매매", "매매계약", "소유권이전", "잔금"],
                "임대차": ["임대차계약", "전세계약", "보증금", "월세"],
                "등기": ["소유권등기", "등기신청", "등기부등본", "등기절차"],
                "재개발": ["재개발", "재건축", "조합원", "분양권"]},
            "형사": {
                "고소/고발": ["고소장", "고발장", "고소기간", "고소취하"],
                "변호사선임": ["변호사선임", "국선변호", "변호인", "수임료"],
                "형사절차": ["수사절차", "공판절차", "구속영장", "기소"],
                "보석": ["보석신청", "보석금", "구속취소", "보석심문"],
            },
            "행정": {
                "인허가": ["인허가신청", "영업허가", "허가취소", "등록신청"],
                "행정처분": ["행정처분", "시정명령", "과태료", "영업정지"],
                "행정심판": ["행정심판", "행정소송", "집행정지", "취소소송"],
                "행정규제": ["규제위반", "행정명령", "시정조치", "이행강제"]
            },
            "기업/회사": {
                "회사설립": ["법인설립", "설립등기", "정관작성", "자본금"],
            "주주관계": ["주주권", "주주총회", "의결권", "배당금"],
            "이사/임원": ["이사회", "대표이사", "감사", "등기이사"],
            "기업구조": ["합병", "분할", "청산", "회생절차"]
        },
        "근로/노동": {
            "근로계약": ["근로계약서", "취업규칙", "근로조건", "계약해지"],
            "임금/수당": ["임금체불", "퇴직금", "연장수당", "4대보험"],
            "산업재해": ["산재보상", "산재신청", "요양급여", "장해급여"],
            "근로환경": ["직장내괴롭힘", "차별", "근로시간", "휴가"]
            },
            "지식재산": {
                "특허": ["특허출원", "특허등록", "특허심사", "특허침해"],
                "상표/디자인": ["상표등록", "디자인권", "상표침해", "브랜드"],
                "저작권": ["저작권등록", "저작물이용", "저작권침해", "라이선스"],
                "영업비밀": ["비밀유지", "기술유출", "영업비밀침해", "비밀관리"]
            },
            "금융/세무": {
                "세금": ["세금신고", "세무조사", "세금체납", "세금감면"],
                "금융거래": ["대출", "투자", "예금", "금융상품"],
                "보험": ["보험계약", "보험금청구", "보험사고", "보험금지급"],
                "조세불복": ["조세불복", "경정청구", "과세전적부", "조세심판"]
            },
            "개인정보/IT": {
                "개인정보보호": ["정보수집", "정보유출", "동의철회", "프라이버시"],
            "전자상거래": ["전자계약", "온라인판매", "환불", "청약철회"],
            "사이버범죄": ["해킹", "피싱", "명예훼손", "스미싱"],
            "플랫폼분쟁": ["온라인분쟁", "플랫폼책임", "리뷰분쟁", "중개책임"]
            }
        }   

        # 3단계: 중요도 분석 개선
        def _analyze_importance(query: str) -> str:
            importance_patterns = {
                "high": [
                    "긴급", "즉시", "급한", "중요", "필수", "반드시",
                    "시급", "지금", "당장", "빨리", "신속"
                ],
                "normal": [
                    "문의", "상담", "알고싶다", "궁금", "어떻게",
                    "방법", "절차", "과정"
                ],
                "low": [
                    "참고", "일반", "보통", "관련", "기타", "혹시",
                    "가능한지", "고민"
                ]
            }
            
            for level, patterns in importance_patterns.items():
                if any(p in query for p in patterns):
                    return level
            return "normal"

        try:
            # 1. 기본 법률 분야 확인
            for area, keywords in PRIMARY_AREAS.items():
                if any(kw in query for kw in keywords):
                    intent["legal_areas"].append(area)
                    intent["keywords"].update(set(kw for kw in keywords if kw in query))

            # 2. 세부 카테고리 분석
            for area in intent["legal_areas"]:
                if area in SUB_CATEGORIES:
                    for sub_cat, keywords in SUB_CATEGORIES[area].items():
                        if any(kw in query for kw in keywords):
                            intent["sub_categories"].append(sub_cat)
                            intent["keywords"].update(set(kw for kw in keywords if kw in query))

            # 3. 구체성 분석 개선
            specificity_score = (
                len(intent["keywords"]) * 1.5 +
                len(intent["legal_areas"]) * 2.0 +
                len(intent["sub_categories"]) * 2.5
            )
            
            intent["specificity"] = (
                "specific" if specificity_score >= 6 else
                "moderate" if specificity_score >= 3 else
                "general"
            )

            # 4. 시간 정보 분석
            intent["temporal_info"] = self._extract_temporal_info(query)

            # 5. 중요도 분석
            intent["importance"] = _analyze_importance(query)

            logger.info(f"""
            === 쿼리 의도 분석 결과 ===
            법률 분야: {intent['legal_areas']}
            세부 카테고리: {intent['sub_categories']}
            구체성: {intent['specificity']} (점수: {specificity_score:.1f})
            중요도: {intent['importance']}
            키워드 수: {len(intent['keywords'])}
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

    def evaluate_context_quality(self, context: List[Document], question: str) -> float:
        """컨텍스트 품질 평가 (quick_filter용)"""
        weights = {
                    'semantic': 0.8,  # 의미적 유사도 강화
                    'keyword': 0.5,
                    'metadata': 0.2
                }
        
        try:
            # 빠른 필터링을 위한 기준값 설정
            keyword_score = self._calculate_keyword_match(context, question)
            semantic_score = self._calculate_semantic_similarity(context, question)
            metadata_score = self._calculate_metadata_reliability(context)
            
            final_score = (
                keyword_score * weights['keyword'] +
                semantic_score * weights['semantic'] +
                metadata_score * weights['metadata']
            )
            
            logger.info(f"""
            === 컨텍스트 품질 평가 ===
            키워드 점수: {keyword_score:.3f} × {weights['keyword']} = {keyword_score * weights['keyword']:.3f}
            의미 점수: {semantic_score:.3f} × {weights['semantic']} = {semantic_score * weights['semantic']:.3f}
            메타데���터 점수: {metadata_score:.3f} × {weights['metadata']} = {metadata_score * weights['metadata']:.3f}
            최종 점수: {final_score:.3f}
            """)
            
            return final_score
            
        except Exception as e:
            logger.error(f"컨텍스트 품질 평가 중 오류: {str(e)}")
            return 0.0

    def _calculate_keyword_match(self, context: List[Document], question: str) -> float:
        try:
            # 법률 전문 용어 사전 활용
            legal_terms = self._get_legal_terms()
            
            # 동의어/유사어 확장
            expanded_keywords = self._expand_keywords(question)
            
            scores = []
            for doc in context:
                content = self._safe_get_content(doc)
                
                # 1. 기본 키워드 매칭
                basic_score = self._basic_keyword_match(content, expanded_keywords)
                
                # 2. 법률 용어 매칭
                legal_score = self._legal_term_match(content, legal_terms)
                
                # 3. 문맥 기반 매칭
                context_score = self._context_match(content, question)
                
                final_score = (
                    basic_score * 0.4 +
                    legal_score * 0.4 +
                    context_score * 0.2
                )
                scores.append(final_score)
            
            return sum(scores) / len(scores)
            
        except Exception as e:
            logger.error(f"키워드 매칭 계산 중 오류: {str(e)}")
            return 0.0
    def _get_legal_terms(self) -> Dict:
        """법률 전문 용어 사전 로드"""
        # 법률 전문 용어 사전 로드 로직 구현
        pass
    def _expand_keywords(self, question: str) -> List[str]:
        """동의어/유사어 확장"""
        try:
            # 기본 키워드 추출
            keywords = [w for w in question.split() if len(w) > 1]
            
            # 기본 법률 동의어
            legal_synonyms = {
                "소송": ["재판", "심판", "법적절차"],
                "계약": ["약정", "계약서", "합의"],
                "위반": ["불이행", "위배", "위법"],
                "책임": ["의무", "책무", "책무"]
            }
            
            # 키워드 확장
            expanded = set(keywords)
            for kw in keywords:
                if kw in legal_synonyms:
                    expanded.update(legal_synonyms[kw])
                
            return list(expanded)
            
        except Exception as e:
            logger.error(f"키워드 확장 중 오류: {str(e)}")
            return []
    def _basic_keyword_match(self, content: str, keywords: List[str]) -> float:
        """기본 키워드 매칭"""
        try:
            if not content or not keywords:
                return 0.0
            
            content_lower = content.lower()
            matched = sum(1 for kw in keywords if kw.lower() in content_lower)
            return matched / len(keywords)
            
        except Exception as e:
            logger.error(f"기본 키워드 매칭 중 오류: {str(e)}")
            return 0.0
    def _legal_term_match(self, content: str, legal_terms: Dict) -> float:
        """법률 용어 매칭"""
        try:
            if not content or not legal_terms:
                return 0.0
            
            content_lower = content.lower()
            basic_terms = {
                "법원": 0.3,
                "판례": 0.3,
                "법률": 0.2,
                "조항": 0.2,
                "소송": 0.2,
                "계약": 0.2,
                "위반": 0.2,
                "책임": 0.2
            }
            
            score = 0.0
            for term, weight in basic_terms.items():
                if term in content_lower:
                    score += weight
            
            return min(1.0, score)
            
        except Exception as e:
            logger.error(f"법률 용어 매칭 중 오류: {str(e)}")
            return 0.0
    def _context_match(self, content: str, question: str) -> float:
        """문맥 기반 매칭"""
        try:
            if not content or not question:
                return 0.0
            
            # 1. 길이 기반 점수
            length_score = min(len(content.split()) / 100, 1.0) * 0.3
            
            # 2. 문장 구조 유사성
            question_words = set(question.split())
            content_words = set(content.split())
            overlap = len(question_words & content_words)
            structure_score = (overlap / len(question_words)) * 0.7
            
            return length_score + structure_score
            
        except Exception as e:
            logger.error(f"문맥 매칭 중 오류: {str(e)}")
            return 0.0
    def _safe_get_content(self, doc: Union[Document, str]) -> str:
        """문서 내용 안전하게 가져오기"""
        try:
            if isinstance(doc, Document):
                return doc.page_content
            return str(doc)
        except Exception as e:
            logger.error(f"문서 내용 가져오기 실패: {str(e)}")
            return ""
    def _calculate_semantic_similarity(self, context: List[Document], question: str) -> float:
        """의미적 유사도 계산"""
        try:
            # KoBERT 모델 활용 (기존 validate_answer 로직 활용)
            if not context:
                return 0.0
            
            context_text = [self._safe_get_content(doc) for doc in context]
            combined_context = " ".join(context_text)
            
            # KoBERT 토크나이저 및 모델 사용
            inputs = self.kobert_tokenizer(
                [question, combined_context],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            with torch.no_grad():
                outputs = self.kobert_model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
            
            # 코사인 유사도 계산
            similarity = torch.nn.functional.cosine_similarity(
                embeddings[0].unsqueeze(0),
                embeddings[1].unsqueeze(0)
            )
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"의미적 유사도 계산 중 오류: {str(e)}")
            return 0.0

    def _calculate_metadata_reliability(self, context: List[Document]) -> float:
        """메타데이터 신뢰도 계산"""
        try:
            if not context:
                return 0.0
            
            reliability_scores = []
            for doc in context:
                score = 0.0
                metadata = getattr(doc, 'metadata', {}) or {}  # None인 경우 빈 딕셔너리 반환
                
                # 1. 소스 신뢰도
                source = metadata.get('source', '').lower() if metadata.get('source') else ''
                if source == 'court_decision':
                    score += 0.4
                elif source == 'law_firm':
                    score += 0.3
                elif source == 'legal_article':
                    score += 0.2
                
                # 2. 시간적 관련성
                try:
                    year = int(metadata.get('judgment_year', datetime.now().year))
                except (ValueError, TypeError):
                    year = datetime.now().year
                    
                years_diff = datetime.now().year - year
                time_score = max(0, 1 - (years_diff / 10))  # 10년 이상 차이나면 0점
                score += time_score * 0.3
                
                # 3. 법원 단계
                court_level = metadata.get('court_level', '').lower() if metadata.get('court_level') else ''
                if court_level == 'supreme':
                    score += 0.3
                elif court_level == 'high':
                    score += 0.2
                elif court_level == 'district':
                    score += 0.1
                
                reliability_scores.append(score)
            
            final_score = sum(reliability_scores) / len(reliability_scores) if reliability_scores else 0.0
            logger.info(f"메타데이터 신뢰도 점수: {final_score:.3f}")
            return final_score
            
        except Exception as e:
            logger.error(f"메타데이터 신뢰도 계산 중 오류: {str(e)}")
            return 0.0

    def _extract_temporal_info(self, query: str) -> Dict:
        """
        쿼리에서 시간 관련 정보를 추출
        """
        temporal_info = {
            "is_recent": False,      # 최신 정보 요청 여부
            "specific_date": None,   # 특정 날짜
            "time_range": None,      # 기간
            "temporal_keywords": []  # 발견된 시간 관련 키워드
        }

        try:
            # 1. 최신 정보 관련 키워드 체크
            recent_keywords = ["최신", "최근", "새로운", "현재", "요즘"]
            if any(keyword in query for keyword in recent_keywords):
                temporal_info["is_recent"] = True
                temporal_info["temporal_keywords"].extend(
                    [k for k in recent_keywords if k in query]
                )

            # 2. 기간 관련 패턴 체크
            time_patterns = {
                "년": r"(\d+)년",
                "개월": r"(\d+)개월",
                "주": r"(\d+)주",
                "일": r"(\d+)일",
            }
            
            for unit, pattern in time_patterns.items():
                matches = re.findall(pattern, query)
                if matches:
                    if temporal_info["time_range"] is None:
                        temporal_info["time_range"] = {}
                    temporal_info["time_range"][unit] = int(matches[0])
                    temporal_info["temporal_keywords"].append(f"{matches[0]}{unit}")

            # 3. 특정 시점 체크
            time_points = {
                "작년": datetime.now().year - 1,
                "올해": datetime.now().year,
                "내년": datetime.now().year + 1
            }
            
            for keyword, year in time_points.items():
                if keyword in query:
                    temporal_info["specific_date"] = year
                    temporal_info["temporal_keywords"].append(keyword)

            logger.info(f"""
            === 시간 정보 추출 결과 ===
            최신 정보 요청: {temporal_info['is_recent']}
            특정 날짜: {temporal_info['specific_date']}
            기간: {temporal_info['time_range']}
            발견된 키워드: {temporal_info['temporal_keywords']}
            """)

            return temporal_info

        except Exception as e:
            logger.error(f"시간 정보 추출 중 오류: {str(e)}")
            return temporal_info