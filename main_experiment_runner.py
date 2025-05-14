def run_faiss_rag_experiment():
    """FAISS 기반 RAG 실험 실행 함수"""
    # 0. 설정
    qa_dataset_filepath = 'qa_dataset.json' 
    faiss_cache_dir = "../cached_vectors/balanced_json" # 경로 수정
    output_dir = "rag_faiss_experiment_results_py" 