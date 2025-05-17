import logging
from typing import List, Tuple
from pathlib import Path
import numpy as np
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_upstage import UpstageEmbeddings

logger = logging.getLogger(__name__)

DEFAULT_PROMPTS_ROOT_DIR = Path(__file__).parent / "prompts"

# 기본 프롬프트 내용 정의 (파일 로드 실패 시 또는 internal 모드 시 사용)
DEFAULT_INTERNAL_PROMPTS = {
    "rag_answer_system": "당신은 법률 전문가입니다. 주어진 컨텍스트를 바탕으로 사용자의 질문에 정확하고 간결하게 답변해주세요. 컨텍스트에 없는 내용은 언급하지 마세요.",
    "rag_answer_human": "컨텍스트:\n{context}\n\n질문: {query}\n\n답변:"
}

class SimpleRAGSystem:
    def __init__(self, 
                 faiss_cache_dir: str, 
                 embedding_model: UpstageEmbeddings,
                 llm_model_name: str = "gpt-4o-2024-08-06",
                 llm_temperature: float = 0.1,
                 llm_api_key: str = None,
                 prompt_mode: str = "internal", # "internal", "koo", "harin", "minu"
                 prompts_root_dir: Path = DEFAULT_PROMPTS_ROOT_DIR
                ):
        self.llm = ChatOpenAI(
            model_name=llm_model_name, 
            temperature=llm_temperature,
            api_key=llm_api_key
        )
        self.dense_embedder = embedding_model
        self.prompt_mode = prompt_mode
        
        if self.prompt_mode in ["koo", "harin", "minu"]:
            self.prompts_dir = prompts_root_dir / self.prompt_mode
        else: # internal 또는 기타
             self.prompts_dir = prompts_root_dir # 사용 안함, 내부 프롬프트로 대체

        try:
            logger.info(f"FAISS 벡터 저장소 로드 중... ({faiss_cache_dir})")
            self.vectorstore = FAISS.load_local(
                faiss_cache_dir,
                self.dense_embedder,
                allow_dangerous_deserialization=True
            )
            logger.info("FAISS 벡터 저장소 로드 완료.")
        except Exception as e:
            logger.error(f"FAISS 벡터 저장소 로드 중 오류 발생: {e}")
            self.vectorstore = None
            raise

    def _load_prompt_content(self, file_key: str) -> str:
        """지정된 키에 해당하는 프롬프트 내용을 로드합니다."""
        if self.prompt_mode in ["koo", "harin", "minu"]:
            prompt_file_path = self.prompts_dir / f"{file_key}.txt"
            try:
                content = prompt_file_path.read_text(encoding="utf-8")
                logger.info(f"외부 프롬프트 로드 성공 (멤버: {self.prompt_mode}): {prompt_file_path}")
                return content
            except FileNotFoundError:
                logger.warning(f"외부 프롬프트 파일 없음 (멤버: {self.prompt_mode}): {prompt_file_path}. 내부 기본값 사용 시도.")
            except Exception as e:
                logger.error(f"외부 프롬프트 로드 중 오류 (멤버: {self.prompt_mode}, {prompt_file_path}): {e}. 내부 기본값 사용 시도.")
        
        logger.debug(f"내부 기본 프롬프트 사용: key='{file_key}'")
        return DEFAULT_INTERNAL_PROMPTS.get(file_key, "")

    def retrieve(self, query: str, top_k: int = 3) -> Tuple[List[Document], List[float]]:
        """FAISS 벡터 저장소를 사용하여 유사한 문서를 검색하고 L2 거리를 반환합니다."""
        if self.vectorstore is None:
            logger.warning("FAISS 벡터 저장소가 로드되지 않았습니다. 빈 결과를 반환합니다.")
            return [], []
            
        try:
            results_with_scores = self.vectorstore.similarity_search_with_score(
                query,
                k=top_k
            )
            retrieved_docs = [doc for doc, score in results_with_scores]
            l2_distances = [score for doc, score in results_with_scores]
            
            logger.info(f"'{query}'에 대해 {len(retrieved_docs)}개의 문서 검색됨.")
            return retrieved_docs, l2_distances
        except Exception as e:
            logger.error(f"문서 검색 중 오류 발생: {e}")
            return [], []

    def generate_answer_with_context(self, query: str, retrieved_docs: List[Document]) -> str:
        """검색된 문맥을 바탕으로 LLM을 사용하여 답변을 생성합니다."""
        if not retrieved_docs:
            logger.warning("제공된 컨텍스트가 없습니다. 컨텍스트 없이 답변 생성 시도.")
            return "관련 정보를 찾을 수 없어 답변을 생성할 수 없습니다."

        context_str = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        system_content = self._load_prompt_content("rag_answer_system")
        human_template = self._load_prompt_content("rag_answer_human")

        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_content),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
        
        chain = prompt_template | self.llm
        response = chain.invoke({"context": context_str, "query": query})
        
        logger.info(f"'{query}'에 대한 답변 생성 완료.")
        return response.content 