from langchain_upstage import UpstageEmbeddings
from langchain_teddynote.community.pinecone import (
    create_sparse_encoder,
    load_sparse_encoder,
    fit_sparse_encoder
)
from langchain_teddynote.korean import stopwords
from pinecone import Pinecone
import json
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging
import glob
from tqdm import tqdm

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LegalDataProcessor:
    def __init__(
        self, 
        pinecone_api_key: str, 
        index_name: str,
        encoder_path: Optional[str] = None,
        batch_size: int = 100,
        load_encoder: bool = False  # 기존 encoder를 로드할지 여부
    ):
        """
        법률 데이터 처리기 초기화
        
        Args:
            pinecone_api_key: Pinecone API 키
            index_name: Pinecone 인덱스 이름
            encoder_path: encoder 저장/로드 경로
            batch_size: 배치 처리 크기
            load_encoder: True면 기존 encoder 로드, False면 새로 생성
        """
        logger.info(f"LegalDataProcessor 초기화 - index_name: {index_name}")
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index_name = index_name
        self.batch_size = batch_size
        self.encoder_path = encoder_path
        
        # Dense Embedder 초기화
        logger.info("Dense Embedder 초기화 중...")
        self.dense_embedder = UpstageEmbeddings(
            model="solar-embedding-1-large-query"
        )
        
        # Sparse Encoder 초기화
        logger.info("Sparse Encoder 초기화 중...")
        self.sparse_encoder = self._initialize_sparse_encoder(load_encoder)
        
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
            raise ValueError("저장 경로가 지정되지 않았습니다.")
            
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
        include_summary: bool = False,
        metadata_keys: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        JSON 파일 또는 디렉토리 내의 모든 JSON 파일들을 처리
        
        Args:
            file_path: JSON 파일 또는 디렉토리 경로
            include_summary: 요약문 포함 여부
            metadata_keys: 추출할 메타데이터 키 목록
        """
        logger.info(f"JSON 파일 처리 시작: {file_path}")
        processed_docs = []
        
        # 디렉토리인 경우 TL_ 폴더들을 찾아서 처리
        if Path(file_path).is_dir():
            # TL_ 로 시작하는 모든 폴더 찾기 (.zip이 포함된 폴더명 제외)
            tl_folders = [f for f in glob.glob(str(Path(file_path) / "TL_*")) 
                         if '.zip' not in Path(f).name]
            logger.info(f"{len(tl_folders)}개의 TL 폴더를 찾았습니다.")
            
            for folder in tl_folders:
                if Path(folder).is_file():  # 파일인 경우 건너뛰기
                    continue
                    
                json_files = glob.glob(str(Path(folder) / "*.json"))
                logger.info(f"{Path(folder).name}에서 {len(json_files)}개의 JSON 파일을 찾았습니다.")
                
                for json_file in tqdm(json_files, desc=f"Processing {Path(folder).name}"):
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            
                        # 메타데이터 추출
                        metadata = {
                            'case_id': data.get('info', {}).get('id'),
                            'case_name': data.get('info', {}).get('caseNm'),
                            'court_type': data.get('info', {}).get('courtType'),
                            'court_name': data.get('info', {}).get('courtNm'),
                            'judgment_date': data.get('info', {}).get('judmnAdjuDe'),
                            'case_no': data.get('info', {}).get('caseNo'),
                            'class_name': data.get('Class_info', {}).get('class_name'),
                            'instance_name': data.get('Class_info', {}).get('instance_name'),
                            'keywords': [k.get('keyword') for k in data.get('keyword_tagg', [])],
                            'reference_rules': data.get('Reference_info', {}).get('reference_rules'),
                            'reference_cases': data.get('Reference_info', {}).get('reference_court_case')
                        }
                        
                        # 요약문 처리
                        if include_summary and "Summary" in data:
                            for summary in data["Summary"]:
                                processed_docs.append({
                                    "text": summary["summ_contxt"],
                                    "metadata": {**metadata, "type": "summary"}
                                })
                        
                        # 판결문 처리
                        if "jdgmn" in data:
                            processed_docs.append({
                                "text": data["jdgmn"],
                                "metadata": {**metadata, "type": "judgment"}
                            })
                            
                    except Exception as e:
                        logger.error(f"파일 처리 중 오류 발생 - {json_file}: {str(e)}")
                        continue
        
        logger.info(f"JSON 파일 처리 완료: {len(processed_docs)}개 문서 처리됨")    
        return processed_docs
    
    def create_embeddings(
        self, 
        docs: List[Dict[str, Any]],
        show_progress: bool = True
    ):
        """
        임베딩 생성 및 Pinecone 업로드
        
        Args:
            docs: 처리할 문서 리스트
            show_progress: 진행률 표시 여부
        """
        from tqdm import tqdm
        
        logger.info(f"임베딩 생성 시작: 총 {len(docs)}개 문서")
        total_batches = (len(docs) + self.batch_size - 1) // self.batch_size
        batch_iterator = range(0, len(docs), self.batch_size)
        
        if show_progress:
            batch_iterator = tqdm(batch_iterator, total=total_batches)
            
        for i in batch_iterator:
            batch = docs[i:i + self.batch_size]
            texts = [doc["text"] for doc in batch]
            
            try:
                # 임베딩 생성
                logger.debug(f"배치 {i//self.batch_size + 1} Dense 임베딩 생성 중...")
                dense_vectors = self.dense_embedder.embed_documents(texts)
                logger.debug(f"배치 {i//self.batch_size + 1} Sparse 임베딩 생성 중...")
                sparse_vectors = self.sparse_encoder.encode_documents(texts)
                
                # Pinecone 업로드용 벡터 생성
                vectors = [
                    {
                        "id": f"doc_{i+j}",
                        "values": dense,
                        "sparse_values": sparse,
                        "metadata": batch[j]["metadata"]
                    }
                    for j, (dense, sparse) in enumerate(zip(dense_vectors, sparse_vectors))
                ]
                
                # Pinecone 업로드
                logger.debug(f"배치 {i//self.batch_size + 1} Pinecone 업로드 중...")
                self.pc.Index(self.index_name).upsert(vectors=vectors)
                logger.info(f"배치 {i//self.batch_size + 1}/{total_batches} 처리 완료")
                
            except Exception as e:
                logger.error(f"배치 {i//self.batch_size + 1} 처리 중 오류 발생: {str(e)}")
                continue
    
    def get_encoder(self):
        """현재 sparse encoder 반환"""
        return self.sparse_encoder
