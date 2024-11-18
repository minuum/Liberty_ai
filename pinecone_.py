import os
import time
import pickle
import secrets
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm
import glob
from dotenv import load_dotenv
from pinecone import Index, init, Pinecone
from langchain_upstage import UpstageEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from typing import List, Dict, Any
from langchain_teddynote.community.kiwi_tokenizer import KiwiBM25Tokenizer

# 환경 변수 로드
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "liberty-ai-index")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east1-aws")
def generate_hash() -> str:
    """24자리 무작위 hex 값을 생성하고 6자리씩 나누어 '-'로 연결합니다."""
    random_hex = secrets.token_hex(12)
    return "-".join(random_hex[i: i + 6] for i in range(0, 24, 6))

from pinecone import Index, init



def preprocess_documents(
    split_docs: List[Any], metadata_keys: List[str] = ["source", "page", "author"], min_length: int = 5, use_basename: bool = True
):
    """문서를 전처리하고 내용과 메타데이터를 반환합니다."""
    contents = []
    metadatas = {key: [] for key in metadata_keys}
    for doc in tqdm(split_docs):
        content = doc.page_content.strip()
        if content and len(content) >= min_length:
            contents.append(content)
            for k in metadata_keys:
                value = doc.metadata.get(k)
                if k == "source" and use_basename:
                    value = os.path.basename(value)
                try:
                    metadatas[k].append(int(value))
                except (ValueError, TypeError):
                    metadatas[k].append(value)
    return contents, metadatas

class PineconeRetrievalChain:
    def __init__(self, index_name=PINECONE_INDEX_NAME, api_key=PINECONE_API_KEY, embeddings="solar-embedding-1-large-query", tokenizer="kiwi"):
        # Embeddings 초기화
        self.passage_embeddings = UpstageEmbeddings(model=embeddings)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
        
        # Pinecone 초기화 및 인덱스 설정
        self.pc=Pinecone(api_key=PINECONE_API_KEY) 
        self.index=self.pc.Index(index_name)
        self.index_name = index_name
        self.api_key = api_key
        self.pinecone_params = self.Pinecone_init(index_name)
        
    def Pinecone_init(self, index_name):
        """Pinecone 인덱스를 초기화하거나 존재하지 않으면 생성합니다."""
        host = "https://liberty-index-hwsbh8f.svc.aped-4627-b74a.pinecone.io"
        index = self.create_index(api_key=PINECONE_API_KEY, host=host, index_name=index_name)
        return {"index": index, "namespace": index_name + "-namespace-02"}
    
    def load_documents(self, data_dir="./data", mode="pdf"):
        split_docs = []
        if mode == "pdf":
            files = sorted(glob.glob(os.path.join(data_dir, "*.pdf")))
            print(files)
            for filename in files:
                loader = PDFPlumberLoader(filename)
                split_docs.extend(loader.load_and_split(self.text_splitter))
        if mode == "json":
            import json
            with open(data_dir, 'r', encoding='utf-8') as f:
                try:
                    json_data = json.load(f)
                    
                    # data 배열에서 문서 추출
                    if isinstance(json_data, dict) and 'data' in json_data:
                        for item in json_data['data']:
                            if isinstance(item, dict):
                                # Document 객체 생성
                                from langchain.schema import Document
                                doc = Document(
                                    page_content=item.get('text', ''),
                                    metadata={
                                        'book_id': item.get('book_id'),
                                        'category': item.get('category'),
                                        'popularity': item.get('popularity'),
                                        'keyword': item.get('keyword', []),
                                        'word_segment': item.get('word_segment', []),
                                        'publication_ymd': item.get('publication_ymd')
                                    }
                                )
                                split_docs.append(doc)
                    else:
                        print("JSON 데이터가 예상된 형식이 아닙니다.")
                        print("데이터 구조:", json_data.keys() if isinstance(json_data, dict) else type(json_data))
                        
                except json.JSONDecodeError as e:
                    print(f"JSON 파일 파싱 중 오류가 발생했습니다: {e}")
                    
        return split_docs
    
    def preprocess_documents(self,
        split_docs: List[Any], metadata_keys: List[str] = ["source", "page", "author"], min_length: int = 5, use_basename: bool = True
    ):
        """문서를 전처리하고 내용과 메타데이터를 반환합니다."""
        contents = []
        metadatas = {key: [] for key in metadata_keys}
        print(split_docs)
        for doc in tqdm(split_docs):
            content = doc.page_content.strip()
            if content and len(content) >= min_length:
                contents.append(content)
                for k in metadata_keys:
                    value = doc.metadata.get(k)
                    if k == "source" and use_basename:
                        value = os.path.basename(value)
                    try:
                        metadatas[k].append(int(value))
                    except (ValueError, TypeError):
                        metadatas[k].append(value)
        return contents, metadatas

    def create_sparse_encoder(self, stopwords: List[str]) -> Any:
        """BM25Encoder를 생성하고 반환합니다."""
        from pinecone_text.sparse import BM25Encoder
        bm25 = BM25Encoder(language="english")
        bm25._tokenizer = KiwiBM25Tokenizer(stop_words=stopwords)
        return bm25
    

    def load_sparse_encoder(self, file_path: str) -> Any:
        """저장된 스파스 인코더를 로드합니다."""
        try:
            with open(file_path, "rb") as f:
                loaded_file = pickle.load(f)
            print(f"Loaded Sparse Encoder from: {file_path}")
            return loaded_file
        except Exception as e:
            print(f"Error loading sparse encoder: {e}")
            return None

    def fit_sparse_encoder(self, sparse_encoder: Any, contents: List[str], save_path: str) -> str:
        """Sparse Encoder를 학습하고 저장합니다."""
        sparse_encoder.fit(contents)
        with open(save_path, "wb") as f:
            pickle.dump(sparse_encoder, f)
        print(f"Saved Sparse Encoder to: {save_path}")
        return save_path
    
    def Pinecone_upsert(self, contents, metadatas):
        vectors = []
        for i, content in enumerate(contents):
            # 여기에 벡터 생성 로직 추가
            metadata = {key: metadatas[key][i] for key in metadatas}
            vector = {
                "id": generate_hash(),
                "values": self.passage_embeddings.embed_documents([content])[0],
                "metadata": metadata
            }
            vectors.append(vector)

        # Pinecone 인덱스에 업서트
        self.pinecone_params["index"].upsert(vectors=vectors, namespace=self.pinecone_params["namespace"])
        print(f"Upserted {len(contents)} documents to Pinecone")
    
    def upsert_documents_parallel(self, contents, metadatas, sparse_encoder, embedder=UpstageEmbeddings(model="solar-embedding-1-large-query"), batch_size=100, max_workers=30):
        keys = list(metadatas.keys())

        def chunks(iterable, size):
            it = iter(iterable)
            chunk = list(itertools.islice(it, size))
            while chunk:
                yield chunk
                chunk = list(itertools.islice(it, size))

        def process_batch(batch):
            context_batch = [contents[i] for i in batch]
            metadata_batches = {key: [metadatas[key][i] for i in batch] for key in keys}

            batch_result = [
                {
                    "context": context[:1000],
                    **{key: metadata_batches[key][j] for key in keys},
                } for j, context in enumerate(context_batch)
            ]

            ids = [generate_hash() for _ in range(len(batch))]
            dense_embeds = self.passage_embeddings.embed_documents(context_batch)
            sparse_embeds = sparse_encoder.encode_documents(context_batch)

            vectors = [
                {
                    "id": _id,
                    "sparse_values": sparse,
                    "values": dense,
                    "metadata": metadata,
                }
                for _id, sparse, dense, metadata in zip(ids, sparse_embeds, dense_embeds, batch_result)
            ]

            try:
                return self.pinecone_params["index"].upsert(vectors=vectors, namespace=self.pinecone_params["namespace"], async_req=False)
            
            except Exception as e:
                print(f"Upsert 중 오류 발생: {e}")
                return None

        batches = list(chunks(range(len(contents)), batch_size))
        print(batches)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_batch, batch) for batch in batches]
            print(futures)
            results = []
            for future in tqdm(as_completed(futures), total=len(futures), desc="문서 Upsert 중"):
                result = future.result()
                if result:
                    results.append(result)

        total_upserted = sum(result.upserted_count for result in results if result)
        print(f"총 {total_upserted}개의 Vector가 Upsert 되었습니다.")
        print(f"{self.pinecone_params['index'].describe_index_stats()}")
    def create_index(self,api_key: str=PINECONE_API_KEY, index_name: str=PINECONE_INDEX_NAME, dimension: int = 768, metric: str = "dotproduct", host="https://liberty-index-hwsbh8f.svc.aped-4627-b74a.pinecone.io"):
        """Pinecone 인덱스를 생성하고 반환합니다."""
        # Pinecone 클라이언트를 API 키로 초기화
        #init(api_key=api_key,environment=PINECONE_ENVIRONMENT)
        
        try:
            # 이미 존재하는 인덱스를 가져옴
            index = Index(index_name, host=host)
            print(f"인덱스 '{index_name}'가 이미 존재합니다.")
        except Exception as e:
            # 인덱스가 존재하지 않으면 새로 생성
            print(f"인덱스 '{index_name}'가 존재하지 않아서 새로 생성합니다.")
            from pinecone import create_index
            create_index(
                name=index_name,
                dimension=dimension,  # 임베딩 차원수 (모델에 맞게 설정)
                metric=metric
            )
            index = Index(index_name, host=host)
            print(f"새로운 인덱스 '{index_name}' 생성 완료")
        
        return index


