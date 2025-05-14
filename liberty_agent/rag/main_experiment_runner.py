import logging
from langchain_upstage import UpstageEmbeddings
import pandas as pd

# 생성된 모듈들에서 필요한 함수와 클래스를 가져옵니다.
from legal_schemas import QAExample # QAExample만 필요할 수 있음, 나머지는 각 모듈 내부에서 사용
from advanced_question_generator import AdvancedQuestionGenerator # 정책 기반 질문 생성 시 필요
from rag_system import SimpleRAGSystem
from evaluation_pipeline import load_qa_dataset, run_rag_evaluation_pipeline, save_and_visualize_results, calculate_user_cosine_similarity
import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_experiment_input_data(filepath: str) -> List[Dict[str, Any]]:
    """
    qa_experiment_input.json과 같은 실험 입력 파일을 로드합니다.
    이 파일은 GT Q, GT A, 변형된 Q, 정책 레벨 등을 포함합니다.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        logger.error(f"실험 입력 파일({filepath})을 찾을 수 없습니다.")
        return []
    except json.JSONDecodeError:
        logger.error(f"실험 입력 파일({filepath})이 올바른 JSON 형식이 아닙니다.")
        return []
    except Exception as e:
        logger.error(f"실험 입력 파일 로드 중 예상치 못한 오류: {e}")
        return []

def run_faiss_rag_experiment_with_generated_queries():
    """FAISS 기반 RAG 실험 실행 함수 (생성/변형된 질문 사용)"""
    # 0. 설정
    experiment_input_filepath = 'qa_experiment_input.json' # qa_dataset_generator_from_faiss.py의 출력 파일
    faiss_cache_dir = "../cached_vectors/balanced_json" 
    output_dir = "rag_faiss_experiment_results_generated_q" # 결과 디렉토리명 변경
    
    logger.info("실험 설정값 로드 완료.")
    logger.info(f"실험 입력 데이터 경로: {experiment_input_filepath}")
    logger.info(f"FAISS 캐시 디렉토리: {faiss_cache_dir}")
    logger.info(f"결과 저장 디렉토리: {output_dir}")

    # 임베딩 모델 초기화
    try:
        embedder = UpstageEmbeddings(model="solar-embedding-1-large-query") 
        logger.info("UpstageEmbeddings 모델 초기화 완료.")
    except Exception as e:
        logger.error(f"UpstageEmbeddings 모델 초기화 중 오류: {e}. 스크립트를 종료합니다.")
        return

    # 1. 실험 입력 데이터 로드
    logger.info(f"실험 입력 데이터 로드 시도: {experiment_input_filepath}")
    experiment_data_list = load_experiment_input_data(experiment_input_filepath)
    if not experiment_data_list:
        logger.error("실험 입력 데이터 로드에 실패했거나 데이터가 없습니다. 스크립트를 종료합니다.")
        return
    logger.info(f"실험 입력 데이터 로드 완료: {len(experiment_data_list)}개 항목.")

    # 3. RAG 시스템 초기화 (FAISS 사용)
    try:
        logger.info("RAG 시스템 초기화 시도...")
        rag_sys = SimpleRAGSystem(faiss_cache_dir=faiss_cache_dir, embedding_model=embedder)
        logger.info("RAG 시스템 초기화 완료.")
    except Exception as e:
        logger.error(f"SimpleRAGSystem 초기화 실패: {e}. 스크립트를 종료합니다.")
        return

    # 4. 평가 파이프라인 실행 (수정된 입력에 맞춰 호출)
    logger.info("RAG 평가 파이프라인 실행 시작...")
    
    # run_rag_evaluation_pipeline을 직접 호출하는 대신, 여기서 루프를 돌며 RAGOutput을 만듭니다.
    # 또는 run_rag_evaluation_pipeline의 입력을 List[Dict[str, Any]]로 받고 내부에서 처리하도록 수정할 수 있습니다.
    # 여기서는 직접 루프를 도는 예시를 보여드립니다.
    
    evaluation_outputs: List[Dict[str, Any]] = [] # RAGOutput 모델 대신 dict로 우선 저장
    for i, exp_input in enumerate(experiment_data_list):
        current_query = exp_input["transformed_query"]
        gt_answer = exp_input["gt_answer"] # Ground Truth 답변
        
        logger.info(f"  평가 진행 ({i+1}/{len(experiment_data_list)}): Query: {current_query[:50]}..., Policy: {exp_input['policy_level']}")

        retrieved_docs, l2_scores = rag_sys.retrieve(current_query, top_k=3)
        retrieved_contexts_str = [doc.page_content for doc in retrieved_docs]
        generated_answer_by_rag = rag_sys.generate_answer_with_context(current_query, retrieved_docs)
        
        similarity_score = None
        if generated_answer_by_rag and gt_answer:
            try:
                gen_ans_emb = np.array(embedder.embed_documents([generated_answer_by_rag])[0])
                gt_ans_emb = np.array(embedder.embed_documents([gt_answer])[0])
                similarity_score = calculate_user_cosine_similarity(gen_ans_emb, gt_ans_emb)
            except Exception as e:
                logger.error(f"    답변 유사도 계산 중 오류: {e}")
        
        # RAGOutput 스키마에 맞게 결과 저장 (또는 유사한 dict 구조)
        eval_output_data = {
            "input_query": current_query, # 변형된 질문
            "gt_query": exp_input["gt_query"], # 원본 GT 질문
            "retrieved_contexts": retrieved_contexts_str,
            "generated_answer": generated_answer_by_rag,
            "reference_answer": gt_answer, # GT 답변
            "cosine_similarity_score": similarity_score,
            "l2_distances": l2_scores,
            "policy_level": exp_input["policy_level"],
            "original_difficulty": exp_input.get("transformed_query_difficulty"), # 변형된 질문의 난이도
            "original_category": exp_input.get("original_category"),
            # 필요시 다른 정보 추가
        }
        evaluation_outputs.append(eval_output_data)

    evaluation_df = pd.DataFrame(evaluation_outputs)
    
    if not evaluation_df.empty:
        logger.info(f"평가 완료. 결과 DataFrame 생성: {evaluation_df.shape[0]} 행")
        # 5. 결과 저장 및 시각화
        # save_and_visualize_results는 'original_difficulty'를 DifficultyLevel 타입으로 기대할 수 있으므로,
        # DataFrame 컬럼 타입을 맞추거나, 시각화 함수를 좀 더 유연하게 만들어야 합니다.
        # 여기서는 컬럼명을 그대로 사용합니다.
        save_and_visualize_results(evaluation_df, output_dir=output_dir)
        logger.info("RAG (FAISS & Generated Q) 실험 완료.")
    else:
        logger.info("평가 결과가 비어있습니다.")

if __name__ == "__main__":
    run_faiss_rag_experiment_with_generated_queries() 