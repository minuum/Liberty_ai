from langchain_upstage import UpstageEmbeddings
from langchain_teddynote.community.pinecone import (
    create_sparse_encoder,
    load_sparse_encoder,
    fit_sparse_encoder
)
from langchain_teddynote.korean import stopwords
from pinecone import Pinecone
import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
import glob
from tqdm import tqdm
import os
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from liberty_agent.retrievers.kiwi_bm25 import CustomKiwiBM25Retriever
from langchain_teddynote.community.pinecone import PineconeKiwiHybridRetriever
from liberty_agent.retrievers.hybrid import HybridRetriever
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
import hashlib
import pickle
import concurrent.futures
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import secrets
# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
# httpx 로깅 레벨 조정
logging.getLogger("httpx").setLevel(logging.WARNING)  # 또는 logging.ERROR

logger = logging.getLogger(__name__)
@staticmethod
def generate_hash() -> str:
    """24자리 무작위 hex 값을 생성하고 6자리씩 나누어 '-'로 연결합니다."""
    random_hex = secrets.token_hex(12)
    return "-".join(random_hex[i: i + 6] for i in range(0, 24, 6))
    
@staticmethod
def chunks(iterable, size):
    """이터러블을 지정된 크기의 청크로 분할"""
    it = iter(iterable)
    chunk = list(itertools.islice(it, size))
    while chunk:
        yield chunk
        chunk = list(itertools.islice(it, size))

class LegalDataProcessor:
    def __init__(
        self, 
        pinecone_api_key: str,
        index_name: str,
        encoder_path: Optional[str] = None,
        cache_dir: Optional[str] = "./liberty_agent/cached_vectors",
        cache_mode: bool = True,
    ):
        logger.info(f"======================= DataProcessor 초기화 시작 =======================")
        # Pinecone 초기화
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index_name = index_name
        self.pinecone_index = self.pc.Index(index_name)
        #self.namespace = "liberty-db-namespace-legal-agent"
        self.namespace = "liberty-db-namespace-legal-agent-241122"
        # 캐시 설정
        self.cache_dir = Path(cache_dir)
        self.cache_mode = cache_mode
        self.retriever_cache_dir = self.cache_dir / "retrievers"
        logger.info(f"캐시 디렉토리: {self.cache_dir}")
        logger.info(f"리트리버 캐시 디렉토리: {self.retriever_cache_dir}")
        # 임베딩 모델 초기화
        self.dense_embedder = UpstageEmbeddings(
            model="solar-embedding-1-large-query"
        )
        
        # Sparse 인코더 초기화
        self.encoder_path = encoder_path
        self.sparse_encoder =None
        
        # 캐시 디렉토리 생성
        if self.cache_mode:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.retriever_cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _initialize_sparse_encoder(self, load_encoder: bool, contents: List[str] = []):
        """Sparse Encoder 초기화"""
        if load_encoder:
            if not self.encoder_path or not Path(self.encoder_path).exists():
                logger.error(f"유효하지 않은 encoder_path: {self.encoder_path}")
                raise ValueError("유효한 encoder_path가 필요합니다.")
            self.sparse_encoder = load_sparse_encoder(self.encoder_path)
            logger.info(f"기존 Sparse Encoder 로드 완료: {self.encoder_path}")
        else:
            # 먼저 sparse encoder 생성
            self.sparse_encoder = create_sparse_encoder(stopwords(), mode="kiwi")
            if contents:  # contents가 있는 경우에만 학습 및 저장
                self.save_sparse_encoder(self.encoder_path, contents)
            logger.info("새로운 Sparse Encoder 생성 완료")
        return self.sparse_encoder

    def save_sparse_encoder(self, save_path: Optional[str] = None, contents: List[str] = []):
        """현재 Sparse Encoder 저장"""
        save_path = save_path or self.encoder_path
        if not save_path:
            logger.error("저장 경로가 지정되지 않음")
            raise ValueError("저장 경로가 지정되지 않았습니다.")
        
        if self.sparse_encoder is None:
            logger.error("Sparse Encoder가 초기화되지 않았습니다.")
            raise ValueError("Sparse Encoder가 초기화되지 않았습니다.")
        
        saved_path = fit_sparse_encoder(
            sparse_encoder=self.sparse_encoder,
            contents=contents,
            save_path=save_path
        )
        logger.info(f"Sparse Encoder 저장 완료: {saved_path}")
        return saved_path

    def fit_sparse_encoder(self, contents: List[str], save_path: Optional[str] = None):
        """Sparse Encoder 학습 및 저장"""
        logger.info("Sparse Encoder 학습 시작")
        save_path = save_path or self.encoder_path
        if not save_path:
            logger.error("저장 경로가 지정되지 않음")
            raise ValueError("저 경로가 지정되지 않았습니다.")
            
        saved_path = fit_sparse_encoder(
            sparse_encoder=self.sparse_encoder,
            contents=contents,
            save_path=save_path
        )
        logger.info(f"Sparse Encoder 학습 및 저장 완료: {saved_path}")
        return saved_path
    
    def process_json_data(
        self, 
        file_path: str,
        include_summary: bool = False
    ) -> List[Document]:
        """JSON 파일 처리 및 필요한 정보만 추출"""
        processed_docs = []
        
        try:
            folders = self._get_tl_folders(file_path)
            total_files = sum(len(glob.glob(str(Path(folder) / "*.json"))) for folder in folders)
            
            with tqdm(total=total_files, desc="전체 파일 처리") as pbar:
                for folder in folders:
                    json_files = [f for f in glob.glob(str(Path(folder) / "*.json"))
                             if not f.endswith('.zip')]
                    
                    logger.info(f"{Path(folder).name}에서 {len(json_files)}개의 JSON 파일을 찾았습니다.")
                    
                    for json_file in json_files:
                        try:
                            with open(json_file, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                
                            # 핵심 메타데이터 추출
                            metadata = {
                            "case_no": data['info']['caseNo'],
                            "court": data['info']['courtNm'],
                            "date": data['info']['judmnAdjuDe'],
                            "category": data['Class_info']['class_name'],
                            "subcategory": data['Class_info']['instance_name'],
                            "reference_rules": data['Reference_info'].get('reference_rules', ''),
                            "reference_court_case": data['Reference_info'].get('reference_court_case', '')
                            }
                            
                            # 본문 내용 구성 (우선순위에 따라)
                            content_parts = []
                            
                            # 1. 기본 정보 추가
                            content_parts.append(f"제목: {data['info']['caseTitle']}")
                            content_parts.append(f"사건: {data['info']['caseNm']}")
                            
                            # 2. 본문 내용 (summ_contxt 우선, 없으면 jdgmn 사용)
                            if data['Summary'] and data['Summary'][0].get('summ_contxt'):
                                content_parts.append(f"내용: {data['Summary'][0] ['summ_contxt']}")
                            elif data.get('jdgmn'):
                                content_parts.append(f"내용: {data['jdgmn']}")
                            
                            # 3. 질의/응답 정보 추가
                            if data['jdgmnInfo']:
                                for qa in data['jdgmnInfo']:
                                    content_parts.append(f"질문: {qa['question']}")
                                    content_parts.append(f"답변: {qa['answer']}")
                            
                                    # Document 생성
                                doc = Document(
                                        page_content='\n\n'.join(content_parts),
                                        metadata=metadata
                                    )
                                processed_docs.append(doc)
                                
                            pbar.update(1)
                            pbar.set_postfix({'현재 폴더': Path(folder).name})
                            
                        except Exception as e:
                            logger.error(f"파일 처리 중 오류 발생 ({json_file}): {str(e)}")
                            continue
                        
            return processed_docs
            
        except Exception as e:
            logger.error(f"데이터 리 중 오류 발생: {str(e)}")
            raise

    def _get_tl_folders(self, base_path: str) -> List[str]:
        """TL_ 폴더 목록 반환"""
        tl_folders = [
            f for f in glob.glob(str(Path(base_path) / "TL_*"))
            if not f.endswith('.zip') and Path(f).is_dir()
        ]
        logger.info(f"{len(tl_folders)}개의 TL 폴더를 찾았습니다.")
        return tl_folders

    def process_batch(
        self,
        batch: List[int],
        documents: List[Document],
        start_idx: int
    ) -> Optional[Dict]:
        """배치 단위로 문서를 처리하여 Pinecone에 업로드"""
        try:
            # 현재 배치의 문서와 메타데이터 추출
            batch_docs = [documents[i] for i in batch]
            batch_contents = [doc.page_content for doc in batch_docs]
            
            # 임베딩 생성
            dense_embeds = self.dense_embedder.embed_documents(batch_contents)
            sparse_embeds = self.sparse_encoder.encode_documents(batch_contents)
            
            # 벡터 ID 생성
            vector_ids = [generate_hash() for _ in range(len(batch_docs))]
            
            # Pinecone 업로드용 벡터 생성
            vectors = [
                {
                    "id": id_,
                    "values": dense_embed,
                    "sparse_values": sparse_embed,
                    "metadata": {
                        **doc.metadata,  # 원본 메타데이터 모두 포함
                        "context": content[:1000]  # 컨텍스트 길이 제한
                    }
                }
                for id_, dense_embed, sparse_embed, doc, content in zip(
                    vector_ids, dense_embeds, sparse_embeds, batch_docs, batch_contents
                )
            ]
            
            logger.info(f"업로드할 벡터 수: {len(vectors)}")
            
            try:
                result = self.pinecone_index.upsert(
                    vectors=vectors,
                    namespace=self.namespace,
                    async_req=False
                )
                logger.info(f"배치 업로드 성공: {result}")
                return result
            except Exception as e:
                logger.error(f"Upsert 중 오류 발생: {e}")
                return None
                
        except Exception as e:
            logger.error(f"배치 처리 중 오류: {str(e)}")
            return None

    def create_embeddings(
        self, 
        documents: List[Document],
        start_batch: int = 0,
        end_batch: Optional[int] = None,  # end_batch 파라미터 추가
        batch_size: int = 100,  # 배치 크기 증가
        max_workers: int = 30,  # 워커 수 증가
        show_progress: bool = True
    ) -> None:
        """멀티스레딩을 사용한 임베딩 생성 및 Pinecone 업로드"""
        try:
            start_idx = start_batch * batch_size
            
            # end_batch가 지정된 경우 해당 배치까지만 처리
            if end_batch is not None:
                end_idx = min(end_batch * batch_size, len(documents))
                documents_subset = documents[start_idx:end_idx]
            else:
                documents_subset = documents[start_idx:]
                
            total_docs = len(documents_subset)
            
            logger.info(f"임베딩 생성 시작: 총 {total_docs}개 문서 (배치 {start_batch}부터 {end_batch if end_batch else '끝'}까지)")
            
            # 초기 상태 확인
            initial_stats = self.pinecone_index.describe_index_stats()
            initial_count = initial_stats['total_vector_count']
            
            # 배치 생성
            batches = list(chunks(range(total_docs), batch_size))
            
            results = []
            if show_progress:
                pbar = tqdm(total=total_docs, desc="임베딩 생성 중")
            
            start_time = time.time()
            processed_docs = 0
            
            # 멀티스레딩 처리
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        self.process_batch,
                        batch,
                        documents_subset,
                        start_idx
                    ) for batch in batches
                ]
                
                for idx, future in enumerate(as_completed(futures)):
                    result = future.result()
                    if result:
                        results.append(result)
                    
                    if show_progress:
                        processed_docs += len(batches[idx])
                        elapsed_time = time.time() - start_time
                        progress = processed_docs / total_docs
                        est_total_time = elapsed_time / progress if progress > 0 else 0
                        remaining_time = est_total_time - elapsed_time
                        
                        pbar.set_postfix({
                            'progress': f'{processed_docs}/{total_docs}',
                            'remaining': f'{remaining_time:.1f}s'
                        })
                        pbar.update(len(batches[idx]))
            
            if show_progress:
                pbar.close()
            
            # 결과 확인
            total_upserted = sum(result.upserted_count for result in results if result)
            final_stats = self.pinecone_index.describe_index_stats()
            vectors_added = final_stats['total_vector_count'] - initial_count
            
            logger.info(f"총 {total_upserted}개의 벡터가 업로드되었습니다.")
            logger.info(f"실제 추가된 벡터 수: {vectors_added}")
            if vectors_added < total_docs:
                logger.warning(f"예상보다 적은 벡터가 추가됨: {vectors_added}/{total_docs}")
            
        except Exception as e:
            logger.error(f"임베딩 생성 중 오류 발생: {str(e)}")
            raise

    def get_encoder(self):
        """현재 sparse encoder 반환"""
        return self.sparse_encoder

    def load_test_documents(self, data_path: str) -> List[str]:
        """테스트용 문서 로드"""
        documents = []
        for root, _, files in os.walk(data_path):
            for file in files:
                if file.endswith('.json'):
                    with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # 판례 내용 추출
                        content = data.get('case_text', '')
                        if content:
                            documents.append(content)
        return documents

    def get_relevant_documents(self, query: str) -> List[str]:
        """
        쿼리에 대한 관련 문서 반환
        이 부분은 실제 데이터셋에 맞게 구현 필요
        """
        # 예시 구현:
        # 1. 수동으로 레이블링된 데이터 사용
        # 2. 특정 키워드나 규칙 기반 매칭
        # 3. 전문가가 증한 결과 사용
        pass

    def _check_index_exists(self):
        """Pinecone 인덱스 존재 여부 확인"""
        try:
            indexes = self.pc.list_indexes()
            if self.index_name not in indexes:
                logger.warning(f"인덱스 '{self.index_name}'가 존재하지 않습니다.")
                return False
            return True
        except Exception as e:
            logger.error(f"인덱스 확인 중 오류: {str(e)}")
            return False

    def _check_index_status(self):
        """인덱스 상태 확인"""
        try:
            stats = self.pinecone_index.describe_index_stats()
            logger.info(f"인덱스 상태: {stats}")
            return stats
        except Exception as e:
            logger.error(f"인덱스 상태 확인 중 오류: {str(e)}")
            return None

    def save_retrievers(self, 
                       faiss_store: FAISS = None, 
                       kiwi_retriever: CustomKiwiBM25Retriever = None,
                       save_dir: Optional[str] = None,
                       index_name: str = "legal_index") -> None:
        """FAISS와 KiwiBM25Retriever를 로컬에 저장"""
        save_path = Path(save_dir or self.retriever_cache_dir)
        try:
            save_path.mkdir(parents=True, exist_ok=True)
            
            if faiss_store is not None:
                logger.info(f"FAISS 인덱스 저장 중... ({save_path})")
                faiss_store.save_local(str(save_path))
                
            if kiwi_retriever is not None:
                logger.info(f"KiwiBM25 인덱스 저장 중... ({save_path})")
                kiwi_retriever.save_local(str(save_path), index_name)
                
        except Exception as e:
            logger.error(f"리트리버 저장 중 오류 발: {str(e)}")
            raise

    def load_retrievers(self,
                       load_dir: Optional[str] = None,
                       index_name: str = "legal_index",
                       allow_dangerous: bool = True) -> tuple:
        """저장된 FAISS와 KiwiBM25Retriever를 로드"""
        load_path = Path(load_dir or self.retriever_cache_dir)
        
        try:
            faiss_store = None
            kiwi_retriever = None
            
            faiss_index_path = load_path / "index.faiss"
            if faiss_index_path.exists():
                #logger.info(f"FAISS 인덱스 로드 중... ({load_path})")
                faiss_store = FAISS.load_local(
                    str(load_path), 
                    self.dense_embedder,
                    allow_dangerous_deserialization=allow_dangerous
                )
                
            kiwi_path = load_path / f"{index_name}_vectorizer.pkl"
            if kiwi_path.exists():
                #logger.info(f"KiwiBM25 인덱스 로드 중... ({load_path})")
                kiwi_retriever = CustomKiwiBM25Retriever.load_local(
                    str(load_path), 
                    index_name
                )
            logger.info(f"KiwiBM25 인덱스 로드 완료 ({load_path})")
            return faiss_store, kiwi_retriever
            
        except Exception as e:
            logger.error(f"리트리버 로드 중 오류 발생: {str(e)}")
            raise

    def create_hybrid_retriever(
        self,
        documents: Optional[List[Document]] = None,
        cache_dir: Optional[str] = None,
        k: int = 20,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        use_cache: bool = True
    ) -> Optional[HybridRetriever]:
        """FAISS와 KiwiBM25를 결합한 하이브리드 검색기 생성"""
        try:
            cache_dir = cache_dir or self.retriever_cache_dir
            cache_mode = 'load' if use_cache else 'create'
            
            # 기본 검색기 생성
            retrievers, _ = self.create_retrievers(
                documents=documents,
                use_faiss=True,
                use_kiwi=True,
                use_pinecone=False,
                cache_mode=cache_mode
            )
            
            if not retrievers or 'faiss' not in retrievers or 'kiwi' not in retrievers:
                logger.error("필요한 검색기를 찾을 수 없습니다.")
                return None
                
            # Dense 검색기 설정
            dense_retriever = retrievers['faiss'].as_retriever(
                search_type="similarity",
                search_kwargs={"k": k}
            )
            
            # 하이브리드 검색기 생성
            hybrid = HybridRetriever(
                dense_retriever=dense_retriever,
                sparse_retriever=retrievers['kiwi'],
                k=k,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight
            )
            
            logger.info("하이브리드 검색기 생성 완료")
            return hybrid
            
        except Exception as e:
            logger.error(f"하이브리드 검색기 생성 중 오류: {str(e)}")
            return None

    def create_vectorstore(self, 
                          documents: List[Document],
                          cache_mode: str = 'load',
                          local_db: str = "./cached_vectors/") -> FAISS:
        """
        FAISS 벡터 저장소 생성 또는 로드
        
        Args:
            documents: 문서 리스트
            cache_mode: 'store' (새로 생성), 'load' (로컬에서 로드), 'create' (캐시 없이 생성)
            local_db: 로컬 저장소 경로
            
        Returns:
            FAISS: FAISS 벡터 저장소
        """
        try:
            if not documents:
                logger.warning("문서가 비어있습니다.")
                cache_mode = 'load'
                
            if cache_mode == 'store':
                logger.info(f"로컬({local_db})에 새로운 FAISS 벡터 저장소 생성 중...")
                
                # 저장 디렉토리 생성
                os.makedirs(local_db, exist_ok=True)
                
                # 벡터 저장소 생성
                vectorstore = FAISS.from_documents(
                    documents=documents,
                    embedding=self.dense_embedder
                )
                
                # 로컬에 저장
                vectorstore.save_local(local_db)
                logger.info("FAISS 벡터 저장 생성 및 저장 완료")
                
                return vectorstore
                
            elif cache_mode == 'load':
                logger.info(f"로컬({local_db})에서 FAISS 벡터 저장소 로드 중...")
                
                if not os.path.exists(os.path.join(local_db, "index.faiss")):
                    logger.error(f"저된 FAISS 인덱스를 찾을 수 없습니다: {local_db}")
                    raise FileNotFoundError(f"FAISS 인덱스 파일이 없습니다: {local_db}")
                    
                vectorstore = FAISS.load_local(
                    local_db,
                    self.dense_embedder,
                    allow_dangerous_deserialization=True
                )
                logger.info("FAISS 벡터 저장소 로드 완료")
                
                return vectorstore
                
            else:  # 'create'
                logger.info("메모리에 FAISS 벡터 저장소 생성 중...")
                vectorstore = FAISS.from_documents(
                    documents=documents,
                    embedding=self.dense_embedder
                )
                logger.info("FAISS 벡터 저장소 생성 완료")
                
                return vectorstore
                
        except Exception as e:
            logger.error(f"FAISS 벡터 저장소 처리 중 오류 발생: {str(e)}")
            raise




    def manage_embeddings(
        self,
        documents: List[Document],
        embedding_cache_dir: str = "./cached_vectors/embeddings"
    ) -> Dict[str, Any]:
        """문서 임베딩을 관리하고 캐시합니다."""
        try:
            # 임베딩 캐시 저장소 초기화
            embedding_store = LocalFileStore(embedding_cache_dir)
            
            # 네임스페이스 생성 (모델명 사용)
            namespace = f"upstage_{self.dense_embedder.model}"
            
            cached_embedder = CacheBackedEmbeddings.from_bytes_store(
                underlying_embeddings=self.dense_embedder,
                document_embedding_cache=embedding_store,
                namespace=namespace
            )

            # # CacheBackedEmbeddings를 직접 사용하여 임베딩 처리
            # logger.info("임베딩 처리 중...")
            # with tqdm(total=len(documents), desc="임베딩 생성") as pbar:
            #     embeddings = []
            #     for i in range(0, len(documents), self.batch_size):
            #         batch = documents[i:min(i+self.batch_size, len(documents))]
            #         batch_embeddings = cached_embedder.embed_documents([doc.page_content for doc in batch])
            #         embeddings.extend(batch_embeddings)
            #         pbar.update(len(batch))
            #         pbar.set_postfix({'처리': f'{i+len(batch)}/{len(documents)}'})

            return cached_embedder
            

        except Exception as e:
            logger.error(f"임베딩 관리 중 오류 발생: {str(e)}")
            raise

    def create_retrievers(
        self,
        documents: Optional[List[Document]],
        use_faiss: bool = True,
        use_kiwi: bool = True,
        use_pinecone: bool = True,
        use_hybrid: bool = False,  # 하이브리드 검색기 사용 여부
        cache_mode: str = 'load',
        k: Optional[int] = 10,  # 하이브리드 검색기용 파라미터
        dense_weight: Optional[float] = 0.7,
        sparse_weight: Optional[float] = 0.3
    ) -> Tuple[Dict[str, Any], Any]:
        """통합 검색기 생성"""
        try:
            retrievers = {}
            # Sparse 인코더 초기화
            sparse_encoder = self._initialize_sparse_encoder(load_encoder=True)
            
            # FAISS 검색기
            if use_faiss:
                faiss_dir = self.retriever_cache_dir / "faiss" 
                if cache_mode == 'load':
                    faiss_index_path = faiss_dir / "index.faiss"
                    if faiss_index_path.exists():
                        retrievers['faiss'] = FAISS.load_local(
                            str(faiss_dir),
                            self.dense_embedder,
                            allow_dangerous_deserialization=True
                        )
                        logger.info(f"FAISS 인덱스 로드 완료 ({faiss_dir})")
                    else:
                        logger.warning(f"저장된 FAISS 인덱스를 찾을 수 없습니다.{faiss_index_path}")
        
            # KiwiBM25 검색기
            if use_kiwi:
                kiwi_dir = self.retriever_cache_dir / "kiwi"
                if cache_mode == 'load':
                    kiwi_vectorizer_path = kiwi_dir / "legal_kiwi_vectorizer.pkl"
                    if kiwi_vectorizer_path.exists():

                        retrievers['kiwi'] = CustomKiwiBM25Retriever.load_local(
                            str(kiwi_dir), 
                            "legal_kiwi"
                        )
                        logger.info(f"KiwiBM25 검색기 로드 완료 ({kiwi_dir})")
                    else:
                        logger.warning(f"저장된 KiwiBM25 검색기를 찾을 수 없습니다.{kiwi_vectorizer_path}")
        
            # Pinecone 하이브리드 검색기
            if use_pinecone and sparse_encoder is not None:
                pinecone_params = {
                    "index": self.pinecone_index,
                    "namespace": self.namespace,
                    "embeddings": self.dense_embedder,
                    "sparse_encoder": sparse_encoder
                }
                retrievers['pinecone'] = PineconeKiwiHybridRetriever(**pinecone_params)
                logger.info("Pinecone KiwiBM25 하이브리드 검색기 초기화 완료")

            # FAISS-Kiwi 하이브리드 검색기 추가
            if use_hybrid and 'faiss' in retrievers and 'kiwi' in retrievers:
                dense_retriever = retrievers['faiss'].as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": k}
                )
                
                hybrid = HybridRetriever(
                    dense_retriever=dense_retriever,
                    sparse_retriever=retrievers['kiwi'],
                    k=k,
                    dense_weight=dense_weight,
                    sparse_weight=sparse_weight
                )
                retrievers['hybrid'] = hybrid
                logger.info("FAISS-Kiwi 하이브리드 검색기 생성 완료")
        
            return retrievers, sparse_encoder
            
        except Exception as e:
            logger.error(f"검색기 생성 중 오류 발생: {str(e)}")
            return {}, None  # 에러 발생 시 빈 딕셔너리와 None 반환
