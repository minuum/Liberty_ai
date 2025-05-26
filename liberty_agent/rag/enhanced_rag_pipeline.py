"""
Enhanced RAG Experiment Pipeline
프롬프트 방법론을 구체화하고 쉽게 테스트할 수 있는 RAG 실험 파이프라인

작성자: Liberty AI Team
작성일: 2024년
"""

import os
import sys
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, HTML

# LangChain 및 커스텀 모듈
from langchain_openai import ChatOpenAI
from langchain_upstage import UpstageEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class RAGExperimentConfig:
    """RAG 실험을 위한 통합 설정 클래스"""
    
    # API 및 모델 설정
    upstage_api_key: str
    openai_api_key: str
    embedding_model_name: str = "solar-embedding-1-large"
    llm_model_name: str = "gpt-4o-2024-08-06"
    llm_temperature: float = 0.1
    
    # 경로 설정
    faiss_cache_dir: Path = None
    input_data_path: Path = None
    output_dir: Path = None
    
    # RAG 파라미터
    rag_top_k: int = 3
    interim_save_interval: int = 10
    
    # 실험 설정
    run_standard_rag: bool = True
    run_boost_rag: bool = True
    run_analysis: bool = True
    run_visualization: bool = True
    
    # Boost RAG 설정
    num_boosted_questions_per_doc: int = 2
    boost_strategy: str = "robustness_set"
    
    # 프롬프트 방법론 설정
    prompt_templates: Dict[str, str] = field(default_factory=lambda: {
        "standard_rag_system_prompt": """
당신은 법률 전문가입니다. 주어진 법률 문서를 바탕으로 정확하고 명확한 답변을 제공해주세요.

답변 지침:
1. 주어진 컨텍스트를 기반으로만 답변하세요
2. 법률 용어는 정확하게 사용하세요
3. 답변이 불분명한 경우, "주어진 정보만으로는 판단하기 어렵습니다"라고 명시하세요
4. 예/아니오 질문의 경우 명확히 "긍정" 또는 "부정"으로 시작하세요
5. 답변은 간결하되 충분한 근거를 제시하세요

전문성 수준:
- 법률 전문용어를 정확히 사용
- 판례와 법령을 구분하여 인용
- 법리적 논증을 체계적으로 전개
""",
        "standard_rag_user_prompt": """
다음 법률 문서들을 참고하여 질문에 답변해주세요:

===== 참고 문서 =====
{context}

===== 질문 =====
{question}

===== 답변 =====
""",
        "boost_rag_system_prompt": """
당신은 고급 법률 전문가입니다. 복잡하고 도전적인 법률 질문에 대해 심층적이고 정확한 답변을 제공해주세요.

고급 답변 지침:
1. 주어진 컨텍스트를 철저히 분석하고 다각도로 검토하세요
2. 법률 쟁점을 체계적으로 분류하고 우선순위를 매기세요
3. 판례와 법령, 학설을 구분하여 종합적으로 설명하세요
4. 반대 의견이나 예외 상황도 함께 언급하세요
5. 불확실한 부분은 명시적으로 언급하고 추가 검토가 필요한 사항을 제시하세요
6. 예/아니오 질문의 경우 결론을 먼저 제시하고 상세한 근거를 설명하세요

전문성 강화:
- 법리적 쟁점의 핵심을 파악하여 체계적으로 논증
- 유사 판례와의 비교 분석
- 실무적 적용 가능성과 한계 제시
- 법적 안정성과 예측가능성 고려
""",
        "accuracy_evaluation_prompt": """
다음 두 답변이 실질적으로 동일한 의미인지 법률 전문가 관점에서 평가해주세요:

정답: {gt_answer}
생성된 답변: {generated_answer}

평가 기준:
1. 법률적 결론이 동일한가?
2. 핵심 법리가 일치하는가?
3. "긍정"/"부정" 질문의 경우 최종 판단이 일치하는가?
4. 법적 근거의 타당성이 유사한가?

답변: (일치/불일치) - 간단한 사유와 함께
""",
        "enhanced_context_prompt": """
다음은 법률 문서 검색 결과입니다. 각 문서의 관련성과 신뢰도를 고려하여 답변하세요:

{context_with_scores}

질문: {question}

답변 시 고려사항:
1. 문서별 관련성 점수를 참고하세요
2. 여러 문서 간 내용이 상충하는 경우 이를 명시하세요
3. 가장 관련성이 높은 문서를 우선적으로 참조하세요
""",
        "prompt_comparison_template": """
[실험: {experiment_name}]
[모델: {model_name}]
[온도: {temperature}]
[프롬프트 유형: {prompt_type}]

{base_prompt}

=== 성능 비교를 위한 추가 지침 ===
- 일관성 있는 답변 형식 유지
- 핵심 내용 우선 제시
- 근거 명확히 제시
"""
    })

class EnhancedRAGExperiment:
    """Enhanced RAG 실험을 위한 메인 클래스"""
    
    def __init__(self, config: RAGExperimentConfig):
        self.config = config
        self.rag_system = None
        self.question_generator = None
        self._setup_experiment()
    
    def _setup_experiment(self):
        """실험 환경 설정"""
        # 출력 디렉토리 생성
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # RAG 시스템 초기화
        self._initialize_rag_system()
        
        # 질문 생성기 초기화 (Boost RAG용)
        if self.config.run_boost_rag:
            self._initialize_question_generator()
    
    def _initialize_rag_system(self):
        """RAG 시스템 초기화"""
        try:
            # 여기서는 커스텀 모듈을 직접 임포트할 수 없으므로 플레이스홀더 사용
            # 실제 사용 시에는 아래와 같이 임포트:
            # from rag_system import SimpleRAGSystem
            
            embedding_model = UpstageEmbeddings(
                model=self.config.embedding_model_name,
                api_key=self.config.upstage_api_key
            )
            
            # SimpleRAGSystem 대신 플레이스홀더 클래스 사용
            self.rag_system = MockRAGSystem(
                faiss_cache_dir=str(self.config.faiss_cache_dir),
                embedding_model=embedding_model,
                llm_model_name=self.config.llm_model_name,
                llm_temperature=self.config.llm_temperature,
                llm_api_key=self.config.openai_api_key,
                prompt_templates=self.config.prompt_templates
            )
            
            logger.info(f"RAG 시스템 초기화 완료 (Model: {self.config.llm_model_name})")
            
        except Exception as e:
            logger.error(f"RAG 시스템 초기화 중 오류: {e}")
            raise
    
    def _initialize_question_generator(self):
        """질문 생성기 초기화"""
        try:
            # 실제로는 AdvancedQuestionGenerator를 임포트해야 함
            self.question_generator = MockQuestionGenerator(
                model_name=self.config.llm_model_name,
                temperature=0.1
            )
            logger.info("질문 생성기 초기화 완료")
        except Exception as e:
            logger.error(f"질문 생성기 초기화 중 오류: {e}")
            raise
    
    def run_experiment(self, input_data: Optional[List[Dict]] = None) -> Dict[str, pd.DataFrame]:
        """전체 실험 파이프라인 실행"""
        logger.info("=== Enhanced RAG 실험 파이프라인 시작 ===")
        
        results = {}
        
        # 입력 데이터 로드
        if input_data is None:
            input_data = self._load_input_data()
        
        # Standard RAG 실험
        if self.config.run_standard_rag:
            logger.info("Standard RAG 실험 시작...")
            standard_results = self.run_standard_rag_experiment(input_data)
            results['standard_rag'] = standard_results
        
        # Boost RAG 실험
        if self.config.run_boost_rag and self.question_generator:
            logger.info("Boost RAG 실험 시작...")
            boost_results = self.run_boost_rag_experiment(input_data)
            results['boost_rag'] = boost_results
        
        # 결과 분석
        if self.config.run_analysis:
            logger.info("결과 분석 시작...")
            for experiment_name, df in results.items():
                if df is not None:
                    analyzed_df = self.analyze_results(df, experiment_name)
                    results[f"{experiment_name}_analyzed"] = analyzed_df
        
        # 시각화
        if self.config.run_visualization:
            logger.info("결과 시각화 시작...")
            for experiment_name, df in results.items():
                if df is not None and 'analyzed' in experiment_name:
                    self.visualize_results(df, experiment_name)
        
        logger.info("=== Enhanced RAG 실험 파이프라인 완료 ===")
        return results
    
    def _load_input_data(self) -> List[Dict]:
        """입력 데이터 로드"""
        if not self.config.input_data_path.exists():
            raise FileNotFoundError(f"입력 데이터 파일이 없습니다: {self.config.input_data_path}")
        
        with open(self.config.input_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"입력 데이터 로드 완료: {len(data)}개 항목")
        return data
    
    def run_standard_rag_experiment(self, input_data: List[Dict]) -> pd.DataFrame:
        """Standard RAG 실험 수행"""
        results = []
        total_queries = len(input_data)
        
        logger.info(f"총 {total_queries}개 질문에 대한 Standard RAG 실험 시작")
        
        for i, qa_item in enumerate(input_data):
            try:
                query_id = qa_item.get("query_id", f"std_query_{i}")
                transformed_query = qa_item.get("transformed_query", "")
                
                if not transformed_query:
                    logger.warning(f"Query ID '{query_id}'에 질문이 없습니다. 건너뜁니다.")
                    continue
                
                gt_answer = qa_item.get("gt_answer", "")
                
                # 문서 검색 및 답변 생성
                retrieved_docs, distances = self.rag_system.retrieve(
                    transformed_query, 
                    top_k=self.config.rag_top_k
                )
                
                generated_answer = self.rag_system.generate_answer_with_context(
                    transformed_query, 
                    retrieved_docs,
                    prompt_type="standard"
                )
                
                # 결과 저장
                result_item = {
                    "query_id": query_id,
                    "gt_query": qa_item.get("gt_query"),
                    "gt_answer": gt_answer,
                    "transformed_query": transformed_query,
                    "transformed_query_difficulty": qa_item.get("transformed_query_difficulty"),
                    "transformed_query_strategy": qa_item.get("transformed_query_strategy"),
                    "policy_level": qa_item.get("policy_level"),
                    "original_category": qa_item.get("original_category"),
                    "generated_answer_standard_rag": generated_answer,
                    "retrieved_contexts": [
                        {
                            "page_content": doc.page_content,
                            "metadata": doc.metadata,
                            "distance": float(dist)
                        }
                        for doc, dist in zip(retrieved_docs, distances)
                    ],
                    "experiment_method": "Enhanced Standard RAG",
                    "model_name": self.config.llm_model_name,
                    "temperature": self.config.llm_temperature,
                    "top_k": self.config.rag_top_k
                }
                results.append(result_item)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"  진행률: {i+1}/{total_queries} ({(i+1)/total_queries*100:.1f}%)")
                
            except Exception as e:
                logger.error(f"Query ID '{query_id}' 처리 중 오류: {e}")
                continue
        
        df_results = pd.DataFrame(results)
        
        # 결과 저장
        output_path = self.config.output_dir / "enhanced_standard_rag_results.json"
        df_results.to_json(output_path, orient='records', force_ascii=False, indent=4)
        logger.info(f"Standard RAG 결과 저장: {output_path}")
        
        return df_results
    
    def run_boost_rag_experiment(self, input_data: List[Dict]) -> pd.DataFrame:
        """Boost RAG 실험 수행"""
        # 이 부분은 실제 AdvancedQuestionGenerator와 연동 필요
        logger.info("Boost RAG 실험은 아직 구현 중입니다.")
        return pd.DataFrame()
    
    def analyze_results(self, results_df: pd.DataFrame, experiment_name: str) -> pd.DataFrame:
        """결과 분석"""
        if results_df.empty:
            logger.warning(f"{experiment_name} 결과가 비어있습니다.")
            return results_df
        
        df_analyzed = results_df.copy()
        
        # 정확도 계산
        if 'gt_answer' in df_analyzed.columns and 'generated_answer_standard_rag' in df_analyzed.columns:
            df_analyzed['correct_standard_rag'] = df_analyzed.apply(
                lambda row: self._calculate_accuracy(
                    row.get('gt_answer', ''),
                    row.get('generated_answer_standard_rag', '')
                ), axis=1
            )
            
            overall_accuracy = df_analyzed['correct_standard_rag'].mean()
            logger.info(f"{experiment_name} 전체 정확도: {overall_accuracy:.2%}")
            
            # 그룹별 분석
            self._analyze_by_groups(df_analyzed)
        
        # 분석 결과 저장
        analysis_output_path = self.config.output_dir / f"{experiment_name}_analysis.json"
        df_analyzed.to_json(analysis_output_path, orient='records', force_ascii=False, indent=4)
        
        return df_analyzed
    
    def _calculate_accuracy(self, gt_answer: str, generated_answer: str) -> bool:
        """정확도 계산"""
        gt = str(gt_answer).strip()
        generated = str(generated_answer).strip()
        
        if not gt or not generated:
            return False
        
        # 긍정/부정 답변 처리
        if gt in ["긍정", "부정"]:
            if gt in generated:
                return True
            
            positive_synonyms = ["긍정", "예", "맞습니다", "그렇습니다", "가능합니다"]
            negative_synonyms = ["부정", "아니오", "틀렸습니다", "그렇지 않습니다", "불가능합니다"]
            
            if gt == "긍정":
                return any(syn in generated for syn in positive_synonyms)
            elif gt == "부정":
                return any(syn in generated for syn in negative_synonyms)
        
        # 기본 매칭
        return (gt in generated or 
                generated.startswith(gt) or 
                gt.lower() in generated.lower())
    
    def _analyze_by_groups(self, df_analyzed: pd.DataFrame):
        """그룹별 분석"""
        analysis_groups = {
            "정책 수준": "policy_level",
            "원본 카테고리": "original_category", 
            "질문 난이도": "transformed_query_difficulty",
            "질문 전략": "transformed_query_strategy"
        }
        
        for display_name, col_name in analysis_groups.items():
            if col_name in df_analyzed.columns:
                accuracy_by_group = df_analyzed.groupby(col_name, observed=True)['correct_standard_rag'].agg(['mean', 'count'])
                logger.info(f"\n{display_name}별 정확도:")
                for idx, row in accuracy_by_group.iterrows():
                    logger.info(f"  {idx}: {row['mean']:.2%} ({int(row['count'])}개)")
    
    def visualize_results(self, analyzed_df: pd.DataFrame, experiment_name: str):
        """결과 시각화"""
        if analyzed_df.empty or 'correct_standard_rag' not in analyzed_df.columns:
            logger.warning(f"{experiment_name} 시각화를 위한 데이터가 부족합니다.")
            return
        
        # 한글 폰트 설정
        plt.rcParams['font.family'] = 'AppleGothic'
        plt.rcParams['axes.unicode_minus'] = False
        
        # 전체 정확도
        overall_accuracy = analyzed_df['correct_standard_rag'].mean()
        
        # 시각화 설정
        visualization_configs = [
            {
                "title": "정책 수준별 정확도",
                "column": "policy_level",
                "color": "skyblue"
            },
            {
                "title": "원본 카테고리별 정확도", 
                "column": "original_category",
                "color": "lightcoral"
            },
            {
                "title": "질문 난이도별 정확도",
                "column": "transformed_query_difficulty", 
                "color": "mediumseagreen"
            }
        ]
        
        for viz_config in visualization_configs:
            col_name = viz_config["column"]
            
            if col_name not in analyzed_df.columns:
                continue
            
            # 데이터 준비
            accuracy_data = analyzed_df.groupby(col_name, observed=True)['correct_standard_rag'].agg(['mean', 'count'])
            
            if accuracy_data.empty:
                continue
            
            # 플롯 생성
            plt.figure(figsize=(12, 8))
            bars = plt.bar(range(len(accuracy_data)), accuracy_data['mean'], 
                          color=viz_config["color"], alpha=0.7)
            
            # 플롯 꾸미기
            plt.title(f"{viz_config['title']} - {experiment_name}", fontsize=16, fontweight='bold')
            plt.xlabel(col_name.replace("_", " ").title(), fontsize=14)
            plt.ylabel('평균 정확도', fontsize=14)
            plt.xticks(range(len(accuracy_data)), accuracy_data.index, rotation=45, ha='right')
            plt.ylim(0, 1.05)
            
            # 값 라벨 추가
            for i, (idx, row) in enumerate(accuracy_data.iterrows()):
                plt.text(i, row['mean'] + 0.02, 
                        f"{row['mean']:.1%}\n({int(row['count'])}개)", 
                        ha='center', va='bottom', fontsize=10)
            
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # 저장
            plot_path = self.config.output_dir / f"{experiment_name}_{col_name}_accuracy.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.show()
        
        logger.info(f"{experiment_name} 시각화 완료")


class MockRAGSystem:
    """RAG 시스템 목업 클래스 (실제 SimpleRAGSystem 대체용)"""
    
    def __init__(self, faiss_cache_dir, embedding_model, llm_model_name, 
                 llm_temperature, llm_api_key, prompt_templates):
        self.faiss_cache_dir = faiss_cache_dir
        self.embedding_model = embedding_model
        self.llm_model_name = llm_model_name
        self.llm_temperature = llm_temperature
        self.llm_api_key = llm_api_key
        self.prompt_templates = prompt_templates
        
        # LLM 초기화
        self.llm = ChatOpenAI(
            model=llm_model_name,
            temperature=llm_temperature,
            api_key=llm_api_key
        )
    
    def retrieve(self, query: str, top_k: int = 3) -> Tuple[List[Document], List[float]]:
        """문서 검색 (목업)"""
        # 실제로는 FAISS에서 검색해야 함
        dummy_docs = [
            Document(
                page_content=f"이는 '{query}'와 관련된 법률 문서 내용 {i+1}입니다. 실제 법률 조항과 판례 내용이 여기에 포함됩니다.",
                metadata={"source": f"legal_doc_{i+1}.txt", "category": "민사"}
            )
            for i in range(top_k)
        ]
        distances = [0.1 * (i + 1) for i in range(top_k)]
        return dummy_docs, distances
    
    def generate_answer_with_context(self, query: str, context_docs: List[Document], 
                                   prompt_type: str = "standard") -> str:
        """컨텍스트 기반 답변 생성"""
        # 컨텍스트 준비
        context = "\n\n".join([f"문서 {i+1}: {doc.page_content}" 
                              for i, doc in enumerate(context_docs)])
        
        # 프롬프트 선택
        if prompt_type == "standard":
            system_prompt = self.prompt_templates["standard_rag_system_prompt"]
            user_prompt = self.prompt_templates["standard_rag_user_prompt"]
        else:
            system_prompt = self.prompt_templates["boost_rag_system_prompt"]
            user_prompt = self.prompt_templates["standard_rag_user_prompt"]
        
        # 프롬프트 생성
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt.format(context=context, question=query)}
        ]
        
        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"답변 생성 중 오류: {e}")
            return f"답변 생성 중 오류가 발생했습니다: {str(e)}"


class MockQuestionGenerator:
    """질문 생성기 목업 클래스"""
    
    def __init__(self, model_name, temperature):
        self.model_name = model_name
        self.temperature = temperature
    
    def generate_legal_questions_from_text(self, text_content: str, num_questions: int, 
                                         method: str = "boost", strategy_for_boost: str = "robustness_set"):
        """부스트 질문 생성 (목업)"""
        # 실제로는 AdvancedQuestionGenerator 로직 필요
        return []


def create_experiment_config(
    experiment_name: str = "enhanced_rag_test",
    preset: str = None,
    custom_prompts: Optional[Dict[str, str]] = None,
    **kwargs
) -> RAGExperimentConfig:
    """실험 설정 생성 헬퍼 함수"""
    
    # 기본값 설정
    base_config = {
        "upstage_api_key": os.getenv("UPSTAGE_API_KEY", "YOUR_UPSTAGE_KEY"),
        "openai_api_key": os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_KEY"),
        "embedding_model_name": "solar-embedding-1-large",
        "llm_model_name": "gpt-4o-2024-08-06",
        "llm_temperature": 0.1,
        "faiss_cache_dir": Path("../cached_vectors/balanced_json"),
        "input_data_path": Path("qa_experiment_input_notebook.json"),
        "output_dir": Path("./experiment_results") / experiment_name,
        "rag_top_k": 3,
        "interim_save_interval": 10,
    }
    
    # 프리셋 적용
    presets = {
        "quick_test": {
            "rag_top_k": 2,
            "interim_save_interval": 5,
            "run_boost_rag": False,
            "run_visualization": False
        },
        "full_standard": {
            "rag_top_k": 3,
            "run_boost_rag": False,
            "run_analysis": True,
            "run_visualization": True
        },
        "comprehensive": {
            "rag_top_k": 5,
            "run_standard_rag": True,
            "run_boost_rag": True,
            "run_analysis": True,
            "run_visualization": True,
            "num_boosted_questions_per_doc": 3
        },
        "prompt_comparison": {
            "rag_top_k": 3,
            "run_boost_rag": False,
            "run_analysis": True,
            "llm_temperature": 0.0
        }
    }
    
    if preset and preset in presets:
        base_config.update(presets[preset])
    
    # 사용자 정의 설정 적용
    base_config.update(kwargs)
    
    # 설정 객체 생성
    config = RAGExperimentConfig(**base_config)
    
    # 사용자 정의 프롬프트 적용
    if custom_prompts:
        config.prompt_templates.update(custom_prompts)
    
    return config


def run_quick_test():
    """빠른 테스트 실행"""
    logger.info("=== 빠른 테스트 시작 ===")
    
    # 설정 생성
    config = create_experiment_config(
        experiment_name="quick_test",
        preset="quick_test"
    )
    
    # 실험 실행
    experiment = EnhancedRAGExperiment(config)
    
    # 테스트 데이터 생성
    test_data = [
        {
            "query_id": "test_1",
            "gt_query": "계약 해지 시 위약금은 어떻게 처리되나요?",
            "gt_answer": "긍정",
            "transformed_query": "계약 해지 시 위약금 규정은 어떻게 적용되는가?",
            "transformed_query_difficulty": "중급",
            "transformed_query_strategy": "법리 해석",
            "policy_level": 2,
            "original_category": "민사"
        },
        {
            "query_id": "test_2", 
            "gt_query": "임금 체불 시 대응 방법은?",
            "gt_answer": "부정",
            "transformed_query": "임금 체불 발생 시 근로자의 권리 구제 방법은?",
            "transformed_query_difficulty": "기초",
            "transformed_query_strategy": "절차질문",
            "policy_level": 1,
            "original_category": "근로자"
        }
    ]
    
    # 실험 실행
    results = experiment.run_experiment(test_data)
    
    logger.info("=== 빠른 테스트 완료 ===")
    return results


def run_prompt_comparison_experiment():
    """프롬프트 비교 실험"""
    logger.info("=== 프롬프트 비교 실험 시작 ===")
    
    # 다양한 프롬프트 버전 정의
    prompt_variations = {
        "standard": {
            "standard_rag_system_prompt": """
당신은 법률 전문가입니다. 주어진 법률 문서를 바탕으로 정확하고 명확한 답변을 제공해주세요.

답변 지침:
1. 주어진 컨텍스트를 기반으로만 답변하세요
2. 법률 용어는 정확하게 사용하세요
3. 예/아니오 질문의 경우 명확히 "긍정" 또는 "부정"으로 시작하세요
"""
        },
        "detailed": {
            "standard_rag_system_prompt": """
당신은 고급 법률 전문가입니다. 주어진 법률 문서를 바탕으로 상세하고 체계적인 답변을 제공해주세요.

답변 지침:
1. 주어진 컨텍스트를 철저히 분석하세요
2. 법률 용어와 개념을 정확히 사용하세요
3. 관련 판례나 법령을 구체적으로 인용하세요
4. 예/아니오 질문의 경우 결론을 먼저 제시하고 상세한 근거를 설명하세요
5. 불확실한 부분은 명시적으로 언급하세요
"""
        },
        "concise": {
            "standard_rag_system_prompt": """
당신은 법률 전문가입니다. 간결하고 정확한 답변을 제공해주세요.

답변 지침:
1. 핵심 내용만 간결하게 답변하세요
2. 예/아니오 질문은 "긍정" 또는 "부정"으로 명확히 답변하세요
3. 주요 법적 근거 1-2개만 제시하세요
"""
        }
    }
    
    results = {}
    
    for prompt_name, custom_prompts in prompt_variations.items():
        logger.info(f"프롬프트 '{prompt_name}' 실험 중...")
        
        config = create_experiment_config(
            experiment_name=f"prompt_comparison_{prompt_name}",
            preset="prompt_comparison",
            custom_prompts=custom_prompts
        )
        
        experiment = EnhancedRAGExperiment(config)
        
        # 동일한 테스트 데이터 사용
        test_data = [
            {
                "query_id": "prompt_test_1",
                "gt_query": "계약 해지 시 위약금은 어떻게 처리되나요?",
                "gt_answer": "긍정",
                "transformed_query": "계약 해지 시 위약금 규정은 어떻게 적용되는가?",
                "transformed_query_difficulty": "중급",
                "transformed_query_strategy": "법리 해석",
                "policy_level": 2,
                "original_category": "민사"
            }
        ]
        
        prompt_results = experiment.run_experiment(test_data)
        results[prompt_name] = prompt_results
    
    logger.info("=== 프롬프트 비교 실험 완료 ===")
    return results


if __name__ == "__main__":
    # 실행 예시
    print("Enhanced RAG Pipeline이 로드되었습니다.")
    print("\n사용 가능한 함수:")
    print("1. run_quick_test() - 빠른 테스트")
    print("2. run_prompt_comparison_experiment() - 프롬프트 비교 실험")
    print("3. create_experiment_config() - 사용자 정의 실험 설정")
    
    # 빠른 테스트 실행 (주석 해제하여 사용)
    # results = run_quick_test() 