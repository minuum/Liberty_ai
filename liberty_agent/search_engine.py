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

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LegalSearchEngine:
    def __init__(
        self, 
        pinecone_index,
        batch_size: int = 100,
        max_workers: int = 30,
        use_combined_check: bool = True,
        namespace: str = None
    ):
        """
        법률 검색 엔진 초기화
        
        Args:
            pinecone_index: Pinecone 인덱스 객체
            batch_size: 배치 처리 크기
            max_workers: 스레드 풀 작업자 수
            use_combined_check: KoBERT와 Upstage 결합 검증 사용 여부
            namespace: Pinecone 네임스페이스
        """
        self.index = pinecone_index
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.use_combined_check = use_combined_check
        self.namespace = namespace
        
        # 임베딩 모델 초기화
        self.dense_embedder = UpstageEmbeddings(
            model="solar-embedding-1-large-query"
        )
        self.upstage_checker = UpstageGroundednessCheck()
        
        # KoBERT 모델 초기화
        self._setup_kobert()
        
        # Hybrid Retriever 초기화
        self.hybrid_retriever = None
        
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
            logger.error(f"배치 처리 중 오류 발생: {str(e)}")
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
            
    def hybrid_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """하이브리드 검색 수행"""
        try:
            # 1. 쿼리 전처리
            processed_query = self._preprocess_query(query)
            
            # 2. 실제 사용 가능한 검색 파라미터
            search_params = {
                "top_k": top_k,  # 반환할 문서 수
                "alpha": 0.7,    # dense-sparse 가중치 (0.0 ~ 1.0)
                "namespace": self.namespace,  # Pinecone 네임스페이스
                
                # 메타데이터 필터 (Pinecone 지원 형식)
                "filter": {
                    "class_name": {"$in": ["민사", "가사", "형사"]},
                    "courtType": {"$in": ["판례(대법원)", "판례(하급심)"]}
                }
            }
            
            logger.info(f"""
            === 검색 파라미터 ===
            쿼리: {query}
            처리된 쿼리: {processed_query}
            top_k: {top_k}
            alpha: {search_params['alpha']}
            namespace: {self.namespace}
            filter: {search_params['filter']}
            """)
            
            # 3. 검색 실행
            results = self.hybrid_retriever.invoke(
                processed_query,
                search_kwargs=search_params
            )
            
            return results
            
        except Exception as e:
            logger.error(f"하이브리드 검색 중 오류: {str(e)}")
            raise
                    
    def validate_answer(
        self,
        context: str,
        answer: str,
        upstage_weight: float = 0.6,
        kobert_weight: float = 0.4
    ) -> float:
        """
        답변의 신뢰도 검증
        """
        logger.info(f"""
        === 답변 신뢰도 검증 시작 ===
        컨텍스트 길이: {len(context)}
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
        
    def _kobert_check(self, context: str, answer: str) -> float:
        """KoBERT를 사용한 관련성 점수 계산"""
        logger.info("=== KoBERT 관련성 검사 시작 ===")
        
        try:
            # 토크나이즈
            context_inputs = self.kobert_tokenizer(
                context,
                return_tensors="pt",
                truncation=True,
                padding=True
            )
            answer_inputs = self.kobert_tokenizer(
                answer,
                return_tensors="pt",
                truncation=True,
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
            logger.error(f"KoBERT 검사 중 오류 발생: {str(e)}")
            raise

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
