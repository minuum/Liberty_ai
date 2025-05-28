import logging
import json
from typing import List, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from langchain_upstage import UpstageEmbeddings

# 생성된 legal_schemas.py와 rag_system.py에서 필요한 클래스들을 가져옵니다.
from legal_schemas import QAExample, RAGOutput, DifficultyLevel
from rag_system import SimpleRAGSystem # SimpleRAGSystem 임포트 추가

logger = logging.getLogger(__name__)

def load_qa_dataset(filepath: str) -> List[QAExample]:
    """QA 데이터셋을 JSON 파일에서 로드합니다."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # difficulty 필드를 DifficultyLevel 타입으로 변환 시도
        # return [QAExample(**item) for item in data]
        # QAExample의 difficulty가 DifficultyLevel enum을 직접 사용하도록 수정되었으므로, 별도 변환 없이 바로 로드
        examples = []
        for item in data:
            try:
                # strategy 필드가 enum에 없는 값일 경우를 대비 (선택적)
                # if 'strategy' in item and item['strategy'] not in LegalQuestion.model_fields['strategy'].metadata[0].enum:
                #     item['strategy'] = "종합 분석" # 또는 기본값
                examples.append(QAExample(**item))
            except Exception as e:
                logger.warning(f"데이터 항목 로드 중 오류: {item}, 오류: {e}")
                continue # 문제가 있는 항목은 건너<0xEB><0x9B><0x84>
        return examples

    except FileNotFoundError:
        logger.error(f"QA 데이터셋 파일({filepath})을 찾을 수 없습니다.")
        return []
    except json.JSONDecodeError:
        logger.error(f"QA 데이터셋 파일({filepath})이 올바른 JSON 형식이 아닙니다.")
        return []
    except Exception as e:
        logger.error(f"QA 데이터셋 로드 중 예상치 못한 오류: {e}")
        return []

def calculate_user_cosine_similarity(query_embedding: np.ndarray, doc_embedding: np.ndarray) -> float:
    """코사인 유사도 계산 (사용자 제공 함수)"""
    query_tensor = torch.from_numpy(np.array(query_embedding)).float()
    doc_tensor = torch.from_numpy(np.array(doc_embedding)).float()
    
    if len(query_tensor.shape) == 1:
        query_tensor = query_tensor.unsqueeze(0)
    if len(doc_tensor.shape) == 1:
        doc_tensor = doc_tensor.unsqueeze(0)
        
    similarity = F.cosine_similarity(query_tensor, doc_tensor)
    return float(similarity.mean().item())

def run_rag_evaluation_pipeline(
    qa_dataset: List[QAExample], 
    rag_system: SimpleRAGSystem,
    answer_embedding_model: UpstageEmbeddings
) -> pd.DataFrame:
    evaluation_results: List[RAGOutput] = []

    for i, qa_example in enumerate(qa_dataset):
        logger.info(f"평가 진행 중 ({i+1}/{len(qa_dataset)}): {qa_example.question}")
        current_query = qa_example.question

        retrieved_docs, l2_scores = rag_system.retrieve(current_query, top_k=3)
        retrieved_contexts_str = [doc.page_content for doc in retrieved_docs]
        generated_answer = rag_system.generate_answer_with_context(current_query, retrieved_docs)
        
        similarity_score = None
        if generated_answer and qa_example.reference_answer:
            try:
                gen_answer_embedding = np.array(answer_embedding_model.embed_documents([generated_answer])[0])
                ref_answer_embedding = np.array(answer_embedding_model.embed_documents([qa_example.reference_answer])[0])
                similarity_score = calculate_user_cosine_similarity(gen_answer_embedding, ref_answer_embedding)
            except Exception as e:
                logger.error(f"답변 유사도 계산 중 오류: {e}")
        
        output = RAGOutput(
            input_query=current_query,
            retrieved_contexts=retrieved_contexts_str,
            generated_answer=generated_answer,
            reference_answer=qa_example.reference_answer,
            cosine_similarity_score=similarity_score,
            l2_distances=l2_scores,
            original_difficulty=qa_example.difficulty
        )
        evaluation_results.append(output)

    results_df = pd.DataFrame([res.dict() for res in evaluation_results])
    return results_df

def save_and_visualize_results(results_df: pd.DataFrame, output_dir: str = "rag_evaluation_output"):
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    csv_filepath = output_path / "rag_faiss_evaluation_results.csv"
    results_df.to_csv(csv_filepath, index=False, encoding='utf-8-sig')
    logger.info(f"평가 결과가 {csv_filepath}에 저장되었습니다.")
    
    if 'cosine_similarity_score' in results_df.columns and 'original_difficulty' in results_df.columns:
        results_df_cleaned = results_df.dropna(subset=['cosine_similarity_score'])
        if not results_df_cleaned.empty:
            # DifficultyLevel enum의 모든 값을 순서대로 사용
            difficulty_order = [level.value for level in DifficultyLevel] # Literal 사용시 .value 필요없음
            # difficulty_order = list(DifficultyLevel.__args__) # Literal의 멤버 가져오기
            difficulty_order = DifficultyLevel.__args__ # Python 3.8+ 에서 Literal의 값들

            # DataFrame의 original_difficulty를 category 타입으로 변환하여 순서 지정
            results_df_cleaned['original_difficulty'] = pd.Categorical(
                results_df_cleaned['original_difficulty'], 
                categories=difficulty_order, 
                ordered=True
            )
            
            avg_similarity_by_difficulty = results_df_cleaned.groupby('original_difficulty', observed=False)['cosine_similarity_score'].mean().reindex(difficulty_order)
            avg_similarity_by_difficulty = avg_similarity_by_difficulty.dropna() # 평균 계산 후 NaN인 그룹 제거
                            
            if not avg_similarity_by_difficulty.empty:
                plt.figure(figsize=(10, 6))
                avg_similarity_by_difficulty.plot(kind='bar')
                plt.title("난이도별 평균 답변 유사도 (코사인)")
                plt.xlabel("질문 난이도")
                plt.ylabel("평균 코사인 유사도")
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                plot_filepath = output_path / "avg_similarity_by_difficulty_faiss.png"
                plt.savefig(plot_filepath)
                logger.info(f"시각화 결과가 {plot_filepath}에 저장되었습니다.")
                # plt.show() # 스크립트 환경에서는 주석 처리
            else:
                logger.info("시각화할 유효한 평균 유사도 데이터가 없습니다.")
        else:
            logger.info("코사인 유사도 점수가 모두 NaN이거나 데이터가 없어 시각화를 건너<0xEB><0x9A><0x85>니다.")
    else:
        logger.warning("결과 DataFrame에 'cosine_similarity_score' 또는 'original_difficulty' 컬럼이 없어 시각화를 건너<0xEB><0x9A><0x85>니다.") 