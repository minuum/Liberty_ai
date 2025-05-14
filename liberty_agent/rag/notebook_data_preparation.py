# notebook_data_preparation.py
# 이 코드를 05_query_gereneration_test.ipynb 파일의 상단부
# (기본 import 및 초기 설정 직후, 주요 시스템 초기화 전)에 추가하세요.

import os
import sys
import logging
from pathlib import Path
import json
import random

# --------------------------------------------------------------------------
# 노트북의 기존 import 문들이 이 아래에 이미 있다고 가정합니다.
# 예시:
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from langchain_openai import ChatOpenAI
# from langchain_upstage import UpstageEmbeddings
# ... 등등
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# 노트북의 기존 경로 및 API 키 설정이 이 아래에 이미 있다고 가정합니다.
# 예시:
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")
# UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY", "YOUR_UPSTAGE_API_KEY")
# FAISS_CACHE_DIR = "../cached_vectors/balanced_json"
# QA_DATASET_FILEPATH = "qa_experiment_input_notebook.json"
# OUTPUT_DIR = "experiment_results_notebook"
# Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
# ... 등등
# --------------------------------------------------------------------------

# === 이 아래부터가 새로 추가될 데이터셋 준비 로직입니다. ===

# 필요한 클래스를 여기서 다시 임포트 (노트북의 셀 실행 순서 독립성 확보)
# 만약 노트북 상단에 이미 아래 임포트가 있다면, 중복될 수 있으나 안전을 위해 포함
try:
    from legal_schemas import QAExample, DocumentAnalysis, LegalQuestion, QuestionSet, DifficultyLevel
    from advanced_question_generator import AdvancedQuestionGenerator
    from rag_system import SimpleRAGSystem # FAISS 로드용
    from langchain_upstage import UpstageEmbeddings # FAISS 로드용 임베딩
except ImportError:
    # liberty_agent.rag 폴더가 sys.path에 있는지 확인 필요
    # 예를 들어, 노트북이 liberty_agent/rag 폴더에 있다면:
    # module_path = os.path.abspath(os.path.join('.'))
    # if module_path not in sys.path:
    #     sys.path.append(module_path)
    # logger.info(f"Path 추가: {module_path}")
    # from legal_schemas import ...
    logger.error("CRITICAL: 데이터셋 준비에 필요한 모듈(schemas, generator, rag_system)을 임포트할 수 없습니다. 경로를 확인하세요.")
    # 이 경우, 아래 로직이 실패하므로 적절한 오류 처리나 노트북 중단이 필요할 수 있습니다.

# 로거 설정 (노트북 상단에 이미 있다면 이 부분은 생략 가능)
if not logging.getLogger().hasHandlers(): # 중복 로거 방지
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
logger = logging.getLogger(__name__)


# 데이터셋 생성 관련 설정값 (노트북 상단 설정값 사용 또는 여기서 명시적 재정의)
# 아래 값들은 노트북 상단 설정 부분에서 가져온다고 가정합니다.
# NUM_DOCS_TO_SAMPLE_FROM_FAISS = 10
# NUM_QUESTIONS_PER_POLICY_LEVEL = 1
# POLICY_LEVELS_FOR_STANDARD_QA = [1, 2, 3, 4]
# QA_DATASET_FILEPATH = "qa_experiment_input_notebook.json"
# FAISS_CACHE_DIR = "../cached_vectors/balanced_json"
# UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")

logger.info("=== 1. 실험용 QA 데이터셋 준비 시작 ===")
qa_input_data_for_experiment = []
qa_dataset_path = Path(QA_DATASET_FILEPATH)

if not qa_dataset_path.exists():
    logger.info(f"'{QA_DATASET_FILEPATH}' 파일이 없습니다. FAISS DB에서 새로운 QA 데이터셋을 생성합니다.")

    # FAISS 로드 및 문서 샘플링을 위한 임시 RAG 시스템 및 질문 생성기 초기화
    temp_rag_for_docs = None
    question_generator_for_setup = None
    embedding_model_for_faiss = None

    try:
        if not UPSTAGE_API_KEY or UPSTAGE_API_KEY == "YOUR_UPSTAGE_API_KEY":
            raise ValueError("Upstage API 키가 설정되지 않았습니다.")
        embedding_model_for_faiss = UpstageEmbeddings(model="solar-embedding-1-large", api_key=UPSTAGE_API_KEY)
        
        temp_rag_for_docs = SimpleRAGSystem(
            faiss_cache_dir=FAISS_CACHE_DIR,
            embedding_model=embedding_model_for_faiss,
            llm_model_name="gpt-3.5-turbo" # 답변 생성용 LLM은 아니므로 임시 설정
        )
        logger.info(f"데이터셋 생성을 위한 FAISS 벡터 저장소 접근 준비 완료: {FAISS_CACHE_DIR}")
        
        question_generator_for_setup = AdvancedQuestionGenerator(
            model_name=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-2024-08-06"), # 질문 생성 LLM
            temperature=0.1
        )
        logger.info("데이터셋 생성을 위한 AdvancedQuestionGenerator 준비 완료.")

    except Exception as e:
        logger.error(f"데이터셋 생성을 위한 시스템 초기화 중 오류: {e}", exc_info=True)
        # 이 경우, 데이터셋 생성 불가

    if temp_rag_for_docs and temp_rag_for_docs.vectorstore and question_generator_for_setup:
        all_doc_ids = list(temp_rag_for_docs.vectorstore.docstore._dict.keys())
        if not all_doc_ids:
            logger.error("FAISS DB에 문서가 없습니다. 데이터셋을 생성할 수 없습니다.")
        else:
            logger.info(f"FAISS에서 총 {len(all_doc_ids)}개의 문서를 찾았습니다.")
            
            num_to_sample = min(NUM_DOCS_TO_SAMPLE_FROM_FAISS, len(all_doc_ids))
            sampled_doc_ids = random.sample(all_doc_ids, num_to_sample)
            logger.info(f"{num_to_sample}개의 문서를 샘플링하여 Standard 질문을 생성합니다.")

            for doc_idx, doc_id in enumerate(sampled_doc_ids):
                try:
                    original_doc = temp_rag_for_docs.vectorstore.docstore._dict[doc_id]
                    original_doc_content = original_doc.page_content
                    original_doc_metadata = original_doc.metadata
                    
                    # gt_query: 원본 문서 메타데이터의 'question' 필드 사용 가정
                    gt_query_from_meta = original_doc_metadata.get("question")
                    if not gt_query_from_meta:
                        logger.warning(f"문서 ID {doc_id}의 메타데이터에 'question' 필드가 없어 gt_query를 설정할 수 없습니다. 이 문서는 건너<0xEB><0x9A><0x89>니다.")
                        continue # gt_query 없이는 QA 쌍 구성 어려움

                    # gt_answer: 원본 문서 메타데이터의 'answer' 필드 사용 가정
                    # 이 부분이 불확실할 수 있으며, 프로젝트 상황에 맞게 조정 필요
                    gt_answer_from_meta = original_doc_metadata.get("answer", "원본 답변 정보 없음")
                    
                    original_category = original_doc_metadata.get("category", "미분류")

                    logger.info(f"  샘플 {doc_idx+1}/{num_to_sample}: 문서 ID {doc_id} (카테고리: {original_category}) 처리 중...")
                    logger.info(f"    원본 GT Query (메타데이터): {gt_query_from_meta[:80]}...")

                    # 문서 분석 (Standard 질문 생성에 활용)
                    analysis_result = question_generator_for_setup.analyze_document(original_doc_content)

                    for policy_level in POLICY_LEVELS_FOR_STANDARD_QA:
                        logger.info(f"      정책 레벨 {policy_level}에 대한 Standard 질문 생성 시도...")
                        standard_questions = question_generator_for_setup.generate_standard_questions_with_policy(
                            document=original_doc_content,
                            analysis=analysis_result,
                            num_questions=NUM_QUESTIONS_PER_POLICY_LEVEL,
                            hybrid_policy_level=policy_level
                        )

                        for std_q_obj in standard_questions: # LegalQuestion 객체
                            qa_item = {
                                "gt_query": gt_query_from_meta,
                                "gt_answer": gt_answer_from_meta, # 메타데이터 기반 또는 "정보 없음"
                                "transformed_query": std_q_obj.question,
                                "transformed_query_reasoning": std_q_obj.reasoning,
                                "transformed_query_difficulty": std_q_obj.difficulty,
                                "transformed_query_strategy": std_q_obj.strategy,
                                "transformed_query_keywords": std_q_obj.keywords,
                                "policy_level": policy_level,
                                "original_category": original_category,
                                "original_document_metadata": original_doc_metadata
                                # 필요한 경우 query_id도 여기서 생성 (예: f"std_q_generated_{doc_id}_{policy_level}_{idx}")
                            }
                            qa_input_data_for_experiment.append(qa_item)
                    
                except Exception as e:
                    logger.error(f"문서 ID {doc_id} 처리 중 오류 발생: {e}", exc_info=True)
                    continue
            
            if qa_input_data_for_experiment:
                try:
                    with open(QA_DATASET_FILEPATH, 'w', encoding='utf-8') as f:
                        json.dump(qa_input_data_for_experiment, f, ensure_ascii=False, indent=4)
                    logger.info(f"새로운 QA 데이터셋 생성 완료: '{QA_DATASET_FILEPATH}' ({len(qa_input_data_for_experiment)}개 항목)")
                except Exception as e:
                    logger.error(f"QA 데이터셋 파일 저장 중 오류: {e}")
            else:
                logger.warning("생성된 QA 데이터가 없어 파일을 저장하지 않습니다. FAISS 샘플링 또는 질문 생성 과정을 확인하세요.")
    else:
        logger.error("FAISS 접근 또는 질문 생성기 준비 실패로 데이터셋 생성을 건너<0xEB><0x9A><0x89>니다.")

else: # qa_dataset_path.exists() == True
    logger.info(f"기존 QA 데이터셋 파일 '{QA_DATASET_FILEPATH}'을 사용합니다.")
    try:
        with open(QA_DATASET_FILEPATH, 'r', encoding='utf-8') as f:
            qa_input_data_for_experiment = json.load(f)
        logger.info(f"기존 QA 데이터셋 로드 완료: {len(qa_input_data_for_experiment)}개 항목")
    except Exception as e:
        logger.error(f"기존 QA 데이터셋 파일 ('{QA_DATASET_FILEPATH}') 로드 중 오류: {e}. 빈 리스트로 시작합니다.")
        qa_input_data_for_experiment = []

# 이후 노트북의 나머지 부분에서는 'qa_input_data_for_experiment' 변수를 사용합니다.
if not qa_input_data_for_experiment:
    logger.critical("CRITICAL: 실험을 위한 QA 입력 데이터가 없습니다. 노트북 실행에 문제가 발생할 수 있습니다.")
    # 필요시 raise ValueError("실험 데이터가 없어 진행할 수 없습니다.")

logger.info("=== 1. 실험용 QA 데이터셋 준비 완료/로드 완료 ===")