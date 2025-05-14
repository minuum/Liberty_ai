# qa_dataset_generator_from_faiss.py

import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import re # 질문/답변 추출을 위한 정규 표현식 라이브러리

from langchain_upstage import UpstageEmbeddings
# 다른 필요한 모듈 임포트
# from legal_schemas import QAExample # 최종 출력 형태는 다를 수 있음
from advanced_question_generator import AdvancedQuestionGenerator 
# SimpleRAGSystem은 FAISS 로드 및 전체 문서 접근용으로만 사용 (검색X)
# rag_system.py에서 FAISS 로드 부분만 가져오거나, 간단한 로더를 만들 수도 있습니다.
# 여기서는 vectorstore를 직접 다루는 것으로 가정하고, 필요시 rag_system의 일부를 활용합니다.
from langchain_community.vectorstores import FAISS 

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def extract_qa_from_content(content: str) -> Optional[Dict[str, str]]:
    """
    주어진 page_content에서 '질문:'과 '답변:'을 파싱하여 추출합니다.
    LegalDocumentProcessor.process_balanced_json의 로직을 참고.
    """
    # 더 정교한 정규 표현식이 필요할 수 있습니다.
    # 예: 여러 줄의 질문/답변, 다양한 포맷 고려
    question_match = re.search(r"질문:\s*(.*?)\s*답변:", content, re.DOTALL)
    answer_match = re.search(r"답변:\s*(.*)", content, re.DOTALL)

    if question_match and answer_match:
        question = question_match.group(1).strip()
        answer = answer_match.group(1).strip()
        if question and answer: # 질문과 답변 모두 내용이 있어야 함
            return {"question": question, "answer": answer}
    return None

def generate_experimental_dataset(
    faiss_cache_dir: str,
    embedding_model: UpstageEmbeddings,
    output_filepath: str,
    num_samples_per_category: int = 1, # 카테고리별로 몇 개의 문서를 샘플링할지
    num_policy_questions: int = 2 # 각 원본 질문에 대해 정책별로 생성할 변형 질문 수
):
    """
    FAISS DB에서 문서를 로드하고, GT Q/A 추출, 변형 질문 생성을 통해
    실험용 데이터셋을 JSON 파일로 저장합니다.
    """
    logger.info(f"FAISS DB 로드 시도: {faiss_cache_dir}")
    try:
        vectorstore = FAISS.load_local(
            faiss_cache_dir,
            embedding_model,
            allow_dangerous_deserialization=True
        )
        logger.info("FAISS DB 로드 성공.")
    except Exception as e:
        logger.error(f"FAISS DB 로드 실패: {e}")
        return

    # FAISS DB의 모든 문서를 가져옵니다.
    # vectorstore.index.ntotal 로 전체 문서 수를 알 수 있습니다.
    # docstore에서 모든 id를 가져와 문서를 로드합니다.
    all_doc_ids = list(vectorstore.docstore._dict.keys())
    all_documents = [vectorstore.docstore.search(id) for id in all_doc_ids]
    # None인 경우 필터링
    all_documents = [doc for doc in all_documents if doc is not None]

    logger.info(f"FAISS에서 총 {len(all_documents)}개의 문서 로드 완료.")

    # 카테고리별 문서 분류
    categorized_docs: Dict[str, List[Dict[str, Any]]] = {}
    for doc in all_documents:
        category = doc.metadata.get("category", "unknown")
        # 사용자확인: FAISS에 저장된 문서의 page_content에서 GT-Q, GT-A를 추출하는 로직입니다.
        # LegalDocumentProcessor.process_balanced_json 에서 content에 "질문:", "답변:" 형태로 저장했다면 아래 로직이 유효합니다.
        # 만약 metadata에 이미 파싱된 question이 있다면 그것을 우선적으로 사용하고, answer만 content에서 추출할 수 있습니다.
        
        gt_qa_pair = None
        if "question" in doc.metadata and doc.metadata["question"]: # 메타데이터에 질문이 있다면 사용
            # 답변은 content에서 추출 시도
            answer_match = re.search(r"답변:\s*(.*)", doc.page_content, re.DOTALL)
            if answer_match and answer_match.group(1).strip():
                gt_qa_pair = {"question": doc.metadata["question"], "answer": answer_match.group(1).strip()}
        
        if not gt_qa_pair: # 메타데이터에 질문이 없거나 답변 추출 실패 시 content 전체에서 파싱
            gt_qa_pair = extract_qa_from_content(doc.page_content)

        if gt_qa_pair:
            if category not in categorized_docs:
                categorized_docs[category] = []
            # page_content 전체도 함께 저장하여 AdvancedQuestionGenerator의 analyze_document에 사용
            categorized_docs[category].append({
                "gt_query": gt_qa_pair["question"],
                "gt_answer": gt_qa_pair["answer"],
                "original_document_content": doc.page_content, # 문서 분석용
                "original_metadata": doc.metadata
            })
        else:
            logger.warning(f"문서에서 Q/A 추출 실패 (ID: {doc.metadata.get('id', 'N/A') if hasattr(doc, 'metadata') else 'N/A'}): {doc.page_content[:100]}...")


    # AdvancedQuestionGenerator 초기화
    # 사용자확인: AdvancedQuestionGenerator의 모델명, temperature 등 파라미터 확인 및 필요시 조정
    q_generator = AdvancedQuestionGenerator(model_name="gpt-4o-2024-08-06", temperature=0.1)
    
    experimental_data: List[Dict[str, Any]] = []
    
    for category, docs_in_category in categorized_docs.items():
        logger.info(f"카테고리 '{category}' 처리 중 ({len(docs_in_category)}개 문서)")
        # 각 카테고리에서 num_samples_per_category 만큼 샘플링
        # 사용자확인: 실제 샘플링 전략을 적용할 수 있습니다 (예: 랜덤 샘플링). 여기서는 앞에서부터 순차적으로 가져옵니다.
        samples_to_process = docs_in_category[:num_samples_per_category]
        
        for sample_doc_info in samples_to_process:
            gt_query = sample_doc_info["gt_query"]
            gt_answer = sample_doc_info["gt_answer"]
            original_content = sample_doc_info["original_document_content"]
            original_meta = sample_doc_info["original_metadata"]

            logger.info(f"  GT 질문 분석 및 변형 질문 생성: {gt_query[:50]}...")
            try:
                # 문서 분석 (원본 문서 내용 전체 또는 GT 답변을 분석 대상으로 할지 결정)
                # 사용자확인: analyze_document의 입력으로 original_content 전체를 사용할지, gt_answer만 사용할지, 혹은 gt_query+gt_answer를 사용할지 결정 필요합니다.
                # 현재는 original_content를 사용합니다.
                analysis_result = q_generator.analyze_document(original_content)
                
                # 정책 레벨별로 변형 질문 생성
                for policy_level in [1, 2, 3, 4]: # "실험 구현 방법론"의 4단계 정책
                    # 사용자확인: base_keywords를 analysis_result.keywords로 할지, 아니면 gt_query에서 추출한 키워드로 할지 등 결정 가능
                    transformed_questions = q_generator.generate_questions(
                        document=original_content, # 또는 gt_answer
                        analysis=analysis_result,
                        num_questions=num_policy_questions,
                        hybrid_policy_level=policy_level,
                        base_keywords=analysis_result.keywords 
                    )
                    
                    for t_question in transformed_questions:
                        experimental_data.append({
                            "gt_query": gt_query,
                            "gt_answer": gt_answer,
                            "transformed_query": t_question.question, # LegalQuestion 객체의 question 필드
                            "transformed_query_reasoning": t_question.reasoning,
                            "transformed_query_difficulty": t_question.difficulty,
                            "transformed_query_strategy": t_question.strategy,
                            "policy_level": policy_level,
                            "original_category": category,
                            "original_document_metadata": original_meta 
                            # 필요시 여기에 analysis_result.keywords, t_question.keywords 등 추가 정보 저장
                        })
            except Exception as e:
                logger.error(f"    GT 질문 처리 중 오류 발생 ({gt_query[:50]}...): {e}")
                continue
                
    logger.info(f"총 {len(experimental_data)}개의 실험 데이터 포인트 생성 완료.")
    
    # JSON 파일로 저장
    output_p = Path(output_filepath)
    output_p.parent.mkdir(parents=True, exist_ok=True) # output 디렉토리 생성
    with open(output_p, 'w', encoding='utf-8') as f:
        json.dump(experimental_data, f, ensure_ascii=False, indent=4)
    logger.info(f"실험용 데이터셋 저장 완료: {output_filepath}")


if __name__ == '__main__':
    # 사용자확인: FAISS DB 경로 및 Upstage API 키, 출력 파일명 확인
    faiss_db_path = "../cached_vectors/balanced_json" # main_experiment_runner.py와 동일한 상대 경로
    output_json_path = "qa_experiment_input.json" # main_experiment_runner.py가 읽을 파일
    
    # 사용자확인: UpstageEmbeddings 모델명 확인
    # "solar-embedding-1-large-query"는 질의용. 문서 내용을 임베딩할 때는 passage/document용 모델이 더 적합할 수 있으나,
    # FAISS 로드 시에는 DB 생성 시 사용된 임베딩 모델과 동일한 것을 사용해야 합니다.
    # 여기서는 FAISS 로드만을 위해 사용하므로, DB 생성시 사용된 모델과 일치시키면 됩니다.
    try:
        embed_model = UpstageEmbeddings(model="solar-embedding-1-large-query")
    except Exception as e:
        logger.error(f"UpstageEmbeddings 모델 초기화 실패: {e}. API 키 설정을 확인하세요.")
        exit() # 모델 초기화 실패 시 종료

    # 카테고리별로 1개의 문서를 샘플링하고, 각 문서에 대해 정책별로 1개의 변형 질문 생성
    # 이 값들을 늘리면 더 많은 실험 데이터가 생성됩니다.
    generate_experimental_dataset(
        faiss_cache_dir=faiss_db_path,
        embedding_model=embed_model,
        output_filepath=output_json_path,
        num_samples_per_category=1, 
        num_policy_questions=1 
    )







'''
qa_dataset_generator_from_faiss.py 주요 내용 및 #사용자확인 포인트:
FAISS DB에서 모든 문서를 로드합니다.
extract_qa_from_content 함수: page_content에서 "질문:", "답변:" 패턴으로 GT-Q, GT-A를 추출합니다.
#사용자확인: FAISS에 저장된 Document의 metadata에 question 키가 이미 존재하고 유효하다면, 해당 값을 우선적으로 사용하고 page_content에서는 "답변:" 부분만 추출하는 로직으로 개선할 수 있습니다. 현재 코드는 메타데이터에 질문이 있을 경우 이를 사용하고, 없으면 page_content에서 파싱을 시도합니다. 실제 데이터 포맷을 보시고 이 부분을 조정해야 할 수 있습니다.
카테고리별로 지정된 수만큼 문서를 샘플링합니다.
#사용자확인: num_samples_per_category 값으로 샘플링 수를 조절합니다. 현재는 단순 순차 샘플링인데, 필요시 랜덤 샘플링 등으로 변경 가능합니다.
샘플링된 각 GT (Q,A)에 대해 AdvancedQuestionGenerator를 사용:
analyze_document: GT 문서 내용(또는 GT 답변)을 분석합니다.
#사용자확인: analyze_document의 입력으로 원본 문서 전체(original_content)를 사용할지, 추출된 gt_answer 또는 gt_query+gt_answer를 사용할지 결정해야 합니다. 이는 생성될 질문의 품질에 영향을 줄 수 있습니다. 현재 코드는 original_content를 사용합니다.
generate_questions: 정책 레벨(1~4)별로 변형된 질문(transformed_query)을 생성합니다.
#사용자확인: num_policy_questions로 정책당 생성할 질문 수를 조절합니다.
#사용자확인: base_keywords를 analysis_result.keywords로 할지, gt_query 자체의 키워드를 사용할지 등도 고려해볼 수 있습니다.
최종적으로 gt_query, gt_answer, transformed_query, policy_level 등의 정보를 포함하는 JSON 파일을 생성합니다.
'''