# rag_exp_setup_with_summary.py
# 이 코드를 rag_exp.ipynb 노트북의 최상단 셀들에 나눠서 추가하세요.

# --- 셀 1: 기본 라이브러리 임포트 및 경로 설정 ---
import os
import sys
import logging
from pathlib import Path
import json
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # 요약에서는 직접 안쓰지만, 노트북 전체적으로 필요
import seaborn as sns # 요약에서는 직접 안쓰지만, 노트북 전체적으로 필요

# LangChain 및 외부 라이브러리 (필요시점에 따라 추가/조정)
from langchain_openai import ChatOpenAI
from langchain_upstage import UpstageEmbeddings
from langchain_core.documents import Document

# 현재 노트북 파일이 위치한 디렉토리를 기준으로 모듈 경로 설정
# rag_exp.ipynb 파일이 liberty_agent/rag/ 폴더에 있다고 가정합니다.
module_path = os.path.abspath(os.getcwd())
if module_path not in sys.path:
    sys.path.append(module_path)
    # logger.info(f"Added to sys.path: {module_path}") # 로거 초기화 전에 호출될 수 있으므로 print 사용 가능

# 커스텀 모듈 임포트
try:
    from legal_schemas import QAExample, DocumentAnalysis, LegalQuestion, QuestionSet, DifficultyLevel
    from advanced_question_generator import AdvancedQuestionGenerator
    from rag_system import SimpleRAGSystem
    from evaluation_pipeline import load_qa_dataset # 다른 함수는 필요시점에 임포트
except ImportError as e:
    print(f"CRITICAL: 커스텀 모듈 임포트 중 오류 발생: {e}")
    print("rag_exp.ipynb 파일이 liberty_agent/rag/ 폴더에 있고,")
    print("해당 폴더 내에 legal_schemas.py, advanced_question_generator.py 등이 있는지 확인하세요.")
    print(f"현재 sys.path: {sys.path}")
    # 이 오류 발생 시, 이후 코드 실행이 어려울 수 있습니다.

# --- 셀 2: 로깅 및 기본 설정값 정의 ---

# 로깅 설정 (이미 설정되어 있다면 중복 실행 방지)
if not logging.getLogger().hasHandlers() or not logging.getLogger(__name__).hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
logger = logging.getLogger(__name__) # 노트북 전체에서 사용할 로거

logger.info("=== Liberty AI RAG Experiment Notebook Setup ===")

# API 키 설정 (환경 변수에서 로드 또는 직접 입력)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-2024-08-06")
UPSTAGE_EMBEDDING_MODEL_NAME = os.getenv("UPSTAGE_EMBEDDING_MODEL_NAME", "solar-embedding-1-large")

if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
if not UPSTAGE_API_KEY:
    logger.warning("UPSTAGE_API_KEY 환경 변수가 설정되지 않았습니다.")

# 주요 경로 설정
BASE_DIR = Path(module_path)
FAISS_CACHE_DIR = BASE_DIR / ".." / "cached_vectors" / "balanced_json"
QA_DATASET_FILENAME = "qa_experiment_input_notebook.json"
QA_DATASET_FILEPATH = BASE_DIR / QA_DATASET_FILENAME
OUTPUT_DIR_NAME = "experiment_results_rag_exp"
OUTPUT_DIR = BASE_DIR / OUTPUT_DIR_NAME

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
logger.info(f"FAISS DB 경로: {FAISS_CACHE_DIR}")
logger.info(f"QA 데이터셋 경로: {QA_DATASET_FILEPATH}")
logger.info(f"결과 저장 디렉토리: {OUTPUT_DIR}")

# 데이터셋 생성 관련 설정
NUM_DOCS_TO_SAMPLE_FROM_FAISS = 10
NUM_QUESTIONS_PER_POLICY_LEVEL = 1
POLICY_LEVELS_FOR_STANDARD_QA = [1, 2, 3, 4]

NUM_BASE_ITEMS_FOR_BOOST_SAMPLING = min(NUM_DOCS_TO_SAMPLE_FROM_FAISS * 2, 20)
NUM_BOOSTED_QUESTIONS_PER_DOC = 2

logger.info(f"Standard QA 생성 시 샘플링 문서 수: {NUM_DOCS_TO_SAMPLE_FROM_FAISS}")
logger.info(f"Standard QA 생성 시 정책 레벨당 질문 수: {NUM_QUESTIONS_PER_POLICY_LEVEL}")
logger.info(f"Boost QA 생성 시 기반 샘플 수: {NUM_BASE_ITEMS_FOR_BOOST_SAMPLING}")
logger.info(f"Boost QA 생성 시 문서당 질문 수: {NUM_BOOSTED_QUESTIONS_PER_DOC}")

# --- 셀 3: 실험용 QA 데이터셋 준비 (qa_experiment_input_notebook.json) ---
logger.info(f"--- 실험용 QA 데이터셋 '{QA_DATASET_FILENAME}' 준비 시작 ---")
qa_input_data_for_experiment = []

if not QA_DATASET_FILEPATH.exists():
    logger.info(f"'{QA_DATASET_FILEPATH}' 파일이 없습니다. FAISS DB에서 새로운 QA 데이터셋을 생성합니다.")
    temp_rag_for_docs = None
    question_generator_for_setup = None
    embedding_model_for_faiss = None
    try:
        if not UPSTAGE_API_KEY: raise ValueError("Upstage API 키가 설정되지 않아 FAISS 로드가 불가능합니다.")
        embedding_model_for_faiss = UpstageEmbeddings(model=UPSTAGE_EMBEDDING_MODEL_NAME, api_key=UPSTAGE_API_KEY)
        temp_rag_for_docs = SimpleRAGSystem(
            faiss_cache_dir=str(FAISS_CACHE_DIR),
            embedding_model=embedding_model_for_faiss,
            llm_model_name="gpt-3.5-turbo"
        )
        logger.info(f"FAISS 벡터 저장소 접근 준비 완료: {FAISS_CACHE_DIR}")
        if not OPENAI_API_KEY: raise ValueError("OpenAI API 키가 설정되지 않아 질문 생성이 불가능합니다.")
        question_generator_for_setup = AdvancedQuestionGenerator(model_name=OPENAI_MODEL_NAME, temperature=0.1)
        logger.info("AdvancedQuestionGenerator 준비 완료.")
    except Exception as e:
        logger.error(f"데이터셋 생성을 위한 시스템 초기화 중 치명적 오류: {e}", exc_info=True)

    if temp_rag_for_docs and temp_rag_for_docs.vectorstore and question_generator_for_setup:
        if hasattr(temp_rag_for_docs.vectorstore.docstore, '_dict'):
            all_doc_ids = list(temp_rag_for_docs.vectorstore.docstore._dict.keys())
        else:
            logger.error("FAISS docstore._dict 접근 불가. FAISS 문서 ID를 가져올 수 없습니다.")
            all_doc_ids = []
        if not all_doc_ids:
            logger.error("FAISS DB에 문서가 없거나 접근할 수 없습니다. 데이터셋을 생성할 수 없습니다.")
        else:
            logger.info(f"FAISS에서 총 {len(all_doc_ids)}개의 문서를 찾았습니다.")
            num_to_sample = min(NUM_DOCS_TO_SAMPLE_FROM_FAISS, len(all_doc_ids))
            if num_to_sample == 0 and len(all_doc_ids) > 0:
                 num_to_sample = 1
                 logger.warning(f"NUM_DOCS_TO_SAMPLE_FROM_FAISS가 0으로 설정되어 최소 1개로 조정합니다.")
            if num_to_sample > 0:
                sampled_doc_ids = random.sample(all_doc_ids, num_to_sample)
                logger.info(f"{num_to_sample}개의 문서를 샘플링하여 Standard 질문을 생성합니다.")
                for doc_idx, doc_id in enumerate(sampled_doc_ids):
                    try:
                        original_doc = temp_rag_for_docs.vectorstore.docstore._dict[doc_id]
                        original_doc_content = original_doc.page_content
                        original_doc_metadata = original_doc.metadata
                        gt_query_from_meta = original_doc_metadata.get("question")
                        if not gt_query_from_meta:
                            logger.warning(f"문서 ID {doc_id} 메타데이터 'question' 필드 부재. 건너<0xEB><0x9A><0x89>니다.")
                            continue
                        gt_answer_from_meta = original_doc_metadata.get("answer", "원본 답변 정보 없음")
                        original_category = original_doc_metadata.get("category", "미분류")
                        logger.info(f"  샘플 {doc_idx+1}/{num_to_sample}: 문서 ID {doc_id} (카테고리: {original_category})")
                        logger.info(f"    L 원본 GT Query (메타데이터): {str(gt_query_from_meta)[:80]}...")
                        analysis_result = question_generator_for_setup.analyze_document(original_doc_content)
                        for policy_level in POLICY_LEVELS_FOR_STANDARD_QA:
                            logger.info(f"      L 정책 레벨 {policy_level} Standard 질문 생성 중...")
                            standard_questions = question_generator_for_setup.generate_standard_questions_with_policy(
                                document=original_doc_content,
                                analysis=analysis_result,
                                num_questions=NUM_QUESTIONS_PER_POLICY_LEVEL,
                                hybrid_policy_level=policy_level
                            )
                            for q_idx, std_q_obj in enumerate(standard_questions):
                                query_id = f"std_q_gen_{doc_idx}_{policy_level}_{q_idx}"
                                qa_item = {
                                    "query_id": query_id, "gt_query": gt_query_from_meta,
                                    "gt_answer": gt_answer_from_meta, "transformed_query": std_q_obj.question,
                                    "transformed_query_reasoning": std_q_obj.reasoning,
                                    "transformed_query_difficulty": std_q_obj.difficulty,
                                    "transformed_query_strategy": std_q_obj.strategy,
                                    "transformed_query_keywords": std_q_obj.keywords,
                                    "policy_level": policy_level, "original_category": original_category,
                                    "original_document_metadata": original_doc_metadata
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
                else: logger.warning("생성된 QA 데이터가 없어 파일을 저장하지 않습니다.")
            else: logger.warning("샘플링할 문서 수가 0입니다. 데이터셋 생성을 건너<0xEB><0x9A><0x89>니다.")
    else: logger.error("FAISS 접근 또는 질문 생성기 준비 실패로 데이터셋 생성을 건너<0xEB><0x9A><0x89>니다.")
else:
    logger.info(f"기존 QA 데이터셋 파일 '{QA_DATASET_FILEPATH}'을 사용합니다.")
    try:
        with open(QA_DATASET_FILEPATH, 'r', encoding='utf-8') as f:
            qa_input_data_for_experiment = json.load(f)
        logger.info(f"기존 QA 데이터셋 로드 완료: {len(qa_input_data_for_experiment)}개 항목")
    except Exception as e:
        logger.error(f"기존 QA 데이터셋 파일 ('{QA_DATASET_FILEPATH}') 로드 중 오류: {e}. 빈 리스트로 시작합니다.")
        qa_input_data_for_experiment = []

if not qa_input_data_for_experiment:
    logger.critical("CRITICAL: 실험을 위한 QA 입력 데이터가 없습니다.")
else:
    logger.info(f"--- 실험용 QA 데이터셋 준비 완료/로드 완료: 총 {len(qa_input_data_for_experiment)}개 항목 ---")
    if qa_input_data_for_experiment:
        logger.info("--- 로드된/생성된 QA 데이터셋 요약 ---")
        df_summary = pd.DataFrame(qa_input_data_for_experiment)
        summary_output = [f"총 질문(항목) 수: {len(df_summary)}"]
        if 'policy_level' in df_summary.columns:
            summary_output.append("\n[정책 수준 (policy_level) 분포]")
            policy_counts = df_summary['policy_level'].value_counts().sort_index()
            for level, count in policy_counts.items(): summary_output.append(f"  - Level {level}: {count} 개")
        if 'original_category' in df_summary.columns:
            summary_output.append("\n[원본 카테고리 (original_category) 분포]")
            category_counts = df_summary['original_category'].value_counts().sort_index()
            for category, count in category_counts.items(): summary_output.append(f"  - {category}: {count} 개")
        if 'transformed_query_difficulty' in df_summary.columns:
            summary_output.append("\n[생성된 질문 난이도 (transformed_query_difficulty) 분포]")
            difficulty_order = ["입문", "기초", "중급", "고급", "전문가", "미지정"]
            difficulty_counts = df_summary['transformed_query_difficulty'].value_counts().reindex(difficulty_order, fill_value=0)
            for difficulty, count in difficulty_counts.items():
                if count > 0: summary_output.append(f"  - {difficulty}: {count} 개")
        if 'transformed_query_strategy' in df_summary.columns:
            summary_output.append("\n[생성된 질문 전략 (transformed_query_strategy) 분포]")
            strategy_counts = df_summary['transformed_query_strategy'].value_counts().sort_index()
            for strategy, count in strategy_counts.items(): summary_output.append(f"  - {strategy}: {count} 개")
        if 'gt_query' in df_summary.columns:
            unique_gt_queries = df_summary['gt_query'].nunique()
            summary_output.append(f"\n고유한 원본 질문 (gt_query) 수: {unique_gt_queries}")
        print("\n" + "="*50 + "\nQA Dataset Summary:\n" + "="*50)
        for line in summary_output: print(line)
        print("="*50 + "\n")
        logger.info("샘플 데이터 (처음 3개 항목):")
        print(df_summary.head(3).to_string())
        logger.info("--- QA 데이터셋 요약 완료 ---")

logger.info(f"--- 노트북 초기 설정 및 데이터셋 준비 완료 ---")