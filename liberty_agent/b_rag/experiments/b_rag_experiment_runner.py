"""
B-RAG 프로젝트 통합 실험 러너
질문 생성론 검증을 위한 전체 파이프라인 실행
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

# 상위 디렉토리를 Python path에 추가
current_dir = Path(__file__).parent
b_rag_dir = current_dir.parent
liberty_dir = b_rag_dir.parent
sys.path.append(str(b_rag_dir))
sys.path.append(str(liberty_dir))

from core.question_generation.unified_yesno_question_generator import UnifiedYesNoQuestionGenerator
from core.rag_system.yesno_rag_system import YesNoRAGSystem, YesNoRAGConfig
from core.schemas.yesno_question_schemas import TenLevelYesNoQuestions
from experiments.configs.experiment_config import BRAGConfig, get_config

class BRAGExperimentRunner:
    """B-RAG 프로젝트 통합 실험 러너"""
    
    def __init__(self, config: BRAGConfig):
        """
        초기화
        
        Args:
            config: B-RAG 실험 설정
        """
        self.config = config
        self.experiment_start_time = datetime.now()
        
        # 결과 저장 디렉토리 설정
        self.output_dir = Path(config.pipeline.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 컴포넌트 초기화
        self.question_generator = None
        self.rag_system = None
        
        print(f"🚀 B-RAG 실험 러너 초기화 완료")
        print(f"   실험명: {config.pipeline.experiment_name}")
        print(f"   출력 디렉토리: {self.output_dir}")
    
    def _initialize_components(self):
        """실험 컴포넌트 초기화"""
        print("🔧 실험 컴포넌트 초기화 중...")
        
        # 질문 생성기 초기화
        if self.config.pipeline.run_question_generation:
            self.question_generator = UnifiedYesNoQuestionGenerator(
                model_name=self.config.question_generation.model_name
            )
            print("✅ 질문 생성기 초기화 완료")
        
        # RAG 시스템 초기화
        if self.config.pipeline.run_standard_rag or self.config.pipeline.run_boost_rag:
            rag_config = YesNoRAGConfig(
                embedding_model=self.config.rag_experiment.embedding_model,
                llm_model=self.config.rag_experiment.llm_model,
                llm_temperature=self.config.rag_experiment.llm_temperature,
                top_k=self.config.rag_experiment.top_k,
                similarity_threshold=self.config.rag_experiment.similarity_threshold
            )
            self.rag_system = YesNoRAGSystem(rag_config)
            print("✅ RAG 시스템 초기화 완료")
    
    def run_full_experiment(self) -> Dict[str, Any]:
        """전체 B-RAG 실험 실행"""
        print(f"\n🎯 B-RAG 전체 실험 시작: {self.config.pipeline.experiment_name}")
        print("=" * 60)
        
        # 컴포넌트 초기화
        self._initialize_components()
        
        experiment_results = {
            "experiment_info": {
                "name": self.config.pipeline.experiment_name,
                "start_time": self.experiment_start_time.isoformat(),
                "config": self.config.to_dict()
            },
            "question_generation_results": {},
            "rag_experiment_results": {},
            "performance_analysis": {}
        }
        
        # 1단계: 질문 생성
        if self.config.pipeline.run_question_generation:
            print("\n📝 1단계: 10개 레벨 Yes/No 질문 생성")
            question_results = self._run_question_generation()
            experiment_results["question_generation_results"] = question_results
        
        # 2단계: RAG 실험
        if self.config.pipeline.run_standard_rag or self.config.pipeline.run_boost_rag:
            print("\n🔍 2단계: RAG 실험 실행")
            rag_results = self._run_rag_experiments(
                experiment_results.get("question_generation_results", {})
            )
            experiment_results["rag_experiment_results"] = rag_results
        
        # 3단계: 성능 분석
        if self.config.pipeline.run_comparison_analysis:
            print("\n📊 3단계: 성능 분석 및 비교")
            analysis_results = self._run_performance_analysis(
                experiment_results["rag_experiment_results"]
            )
            experiment_results["performance_analysis"] = analysis_results
        
        # 실험 완료 시간 기록
        experiment_end_time = datetime.now()
        experiment_results["experiment_info"]["end_time"] = experiment_end_time.isoformat()
        experiment_results["experiment_info"]["duration_seconds"] = (
            experiment_end_time - self.experiment_start_time
        ).total_seconds()
        
        # 결과 저장
        self._save_experiment_results(experiment_results)
        
        # 요약 출력
        self._print_experiment_summary(experiment_results)
        
        return experiment_results
    
    def _run_question_generation(self) -> Dict[str, Any]:
        """질문 생성 실험 실행"""
        question_results = {}
        
        for i, (gt_question, document) in enumerate(
            zip(self.config.pipeline.gt_questions, self.config.pipeline.sample_documents), 1
        ):
            print(f"\n📋 문서 {i}: GT 질문 처리 중...")
            print(f"   GT 질문: {gt_question[:50]}...")
            
            try:
                # 10개 레벨 질문 생성
                generated_questions = self.question_generator.generate_ten_level_questions(
                    gt_question=gt_question,
                    document_content=document,
                    keywords_to_consider="법률 용어, 판례, 법리"
                )
                
                # 결과 저장
                question_results[f"document_{i}"] = {
                    "gt_question": gt_question,
                    "document_summary": generated_questions.document_summary,
                    "generated_questions": [
                        {
                            "level": q.level,
                            "question": q.question,
                            "target_audience": q.target_audience,
                            "expected_answer": q.expected_answer.value,
                            "confidence": q.confidence
                        }
                        for q in generated_questions.questions
                    ],
                    "consistency_rate": generated_questions.get_consistency_rate(),
                    "semantic_consistency": generated_questions.semantic_consistency
                }
                
                print(f"   ✅ 생성 완료: {len(generated_questions.questions)}개 질문")
                print(f"   📈 일관성 비율: {generated_questions.get_consistency_rate():.2%}")
                
            except Exception as e:
                print(f"   ❌ 질문 생성 오류: {e}")
                question_results[f"document_{i}"] = {
                    "gt_question": gt_question,
                    "error": str(e),
                    "generated_questions": []
                }
        
        return question_results
    
    def _run_rag_experiments(self, question_results: Dict[str, Any]) -> Dict[str, Any]:
        """RAG 실험 실행"""
        rag_results = {
            "standard_rag": {},
            "boost_rag": {},
            "comparison": {}
        }
        
        # 모든 생성된 질문 수집
        all_questions = []
        question_metadata = []
        
        for doc_key, doc_data in question_results.items():
            if "generated_questions" in doc_data:
                gt_question = doc_data["gt_question"]
                all_questions.append(gt_question)  # GT 질문 추가
                question_metadata.append({
                    "document": doc_key,
                    "level": 0,  # GT 질문은 레벨 0
                    "question_type": "gt",
                    "original_question": gt_question
                })
                
                # 생성된 질문들 추가
                for q_data in doc_data["generated_questions"]:
                    all_questions.append(q_data["question"])
                    question_metadata.append({
                        "document": doc_key,
                        "level": q_data["level"],
                        "question_type": "generated",
                        "original_question": gt_question,
                        "expected_answer": q_data["expected_answer"],
                        "confidence": q_data["confidence"]
                    })
        
        print(f"📊 총 {len(all_questions)}개 질문으로 RAG 실험 진행")
        
        # Standard RAG 실험
        if self.config.pipeline.run_standard_rag:
            print("\n🔍 Standard RAG 실험...")
            standard_results = []
            
            for i, question in enumerate(all_questions):
                print(f"   질문 {i+1}/{len(all_questions)}: {question[:30]}...")
                result = self.rag_system.run_standard_rag(question)
                
                standard_results.append({
                    "question": question,
                    "answer": result.answer,
                    "confidence": result.confidence,
                    "processing_time": result.processing_time,
                    "metadata": question_metadata[i]
                })
            
            rag_results["standard_rag"] = {
                "results": standard_results,
                "total_questions": len(standard_results),
                "avg_confidence": sum(r["confidence"] for r in standard_results) / len(standard_results),
                "yes_count": sum(1 for r in standard_results if r["answer"].strip().startswith("Yes"))
            }
        
        # Boost RAG 실험
        if self.config.pipeline.run_boost_rag:
            print("\n🚀 Boost RAG 실험...")
            boost_results = []
            
            for i, question in enumerate(all_questions):
                print(f"   질문 {i+1}/{len(all_questions)}: {question[:30]}...")
                result = self.rag_system.run_boost_rag(
                    question, 
                    max_iterations=self.config.rag_experiment.max_boost_iterations
                )
                
                boost_results.append({
                    "question": question,
                    "answer": result.answer,
                    "confidence": result.confidence,
                    "processing_time": result.processing_time,
                    "metadata": question_metadata[i]
                })
            
            rag_results["boost_rag"] = {
                "results": boost_results,
                "total_questions": len(boost_results),
                "avg_confidence": sum(r["confidence"] for r in boost_results) / len(boost_results),
                "yes_count": sum(1 for r in boost_results if r["answer"].strip().startswith("Yes"))
            }
        
        return rag_results
    
    def _run_performance_analysis(self, rag_results: Dict[str, Any]) -> Dict[str, Any]:
        """성능 분석 실행"""
        analysis = {}
        
        if "standard_rag" in rag_results and "boost_rag" in rag_results:
            standard_data = rag_results["standard_rag"]
            boost_data = rag_results["boost_rag"]
            
            # 기본 성능 비교
            confidence_improvement = boost_data["avg_confidence"] - standard_data["avg_confidence"]
            yes_answer_improvement = boost_data["yes_count"] - standard_data["yes_count"]
            
            analysis["overall_performance"] = {
                "confidence_improvement": confidence_improvement,
                "yes_answer_improvement": yes_answer_improvement,
                "standard_yes_rate": standard_data["yes_count"] / standard_data["total_questions"],
                "boost_yes_rate": boost_data["yes_count"] / boost_data["total_questions"]
            }
            
            # 레벨별 분석
            level_analysis = {}
            for level in range(1, 11):
                standard_level_results = [
                    r for r in standard_data["results"] 
                    if r["metadata"]["level"] == level
                ]
                boost_level_results = [
                    r for r in boost_data["results"] 
                    if r["metadata"]["level"] == level
                ]
                
                if standard_level_results and boost_level_results:
                    standard_yes = sum(1 for r in standard_level_results if r["answer"].strip().startswith("Yes"))
                    boost_yes = sum(1 for r in boost_level_results if r["answer"].strip().startswith("Yes"))
                    
                    level_analysis[f"level_{level}"] = {
                        "standard_yes_count": standard_yes,
                        "boost_yes_count": boost_yes,
                        "improvement": boost_yes - standard_yes,
                        "total_questions": len(standard_level_results)
                    }
            
            analysis["level_analysis"] = level_analysis
            
            # 목표 달성 여부 확인
            target_metrics = self.config.pipeline.target_metrics
            analysis["target_achievement"] = {
                "boost_improvement_achieved": confidence_improvement >= target_metrics.get("boost_rag_improvement", 0.1),
                "yes_increase_achieved": yes_answer_improvement >= target_metrics.get("yes_answer_increase", 2),
                "confidence_improvement_actual": confidence_improvement,
                "yes_improvement_actual": yes_answer_improvement
            }
        
        return analysis
    
    def _save_experiment_results(self, results: Dict[str, Any]):
        """실험 결과 저장"""
        timestamp = self.experiment_start_time.strftime("%Y%m%d_%H%M%S")
        
        # JSON 결과 저장
        json_path = self.output_dir / f"{self.config.pipeline.experiment_name}_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 실험 결과 저장 완료: {json_path}")
        
        # 요약 리포트 저장
        summary_path = self.output_dir / f"{self.config.pipeline.experiment_name}_summary_{timestamp}.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(self._generate_summary_report(results))
        
        print(f"📄 요약 리포트 저장 완료: {summary_path}")
    
    def _generate_summary_report(self, results: Dict[str, Any]) -> str:
        """요약 리포트 생성"""
        report = f"""
# B-RAG 실험 결과 요약 리포트

## 실험 정보
- 실험명: {results['experiment_info']['name']}
- 시작 시간: {results['experiment_info']['start_time']}
- 종료 시간: {results['experiment_info']['end_time']}
- 실행 시간: {results['experiment_info']['duration_seconds']:.2f}초

## 질문 생성 결과
"""
        
        if "question_generation_results" in results:
            qg_results = results["question_generation_results"]
            total_docs = len(qg_results)
            avg_consistency = sum(
                doc_data.get("consistency_rate", 0) 
                for doc_data in qg_results.values() 
                if "consistency_rate" in doc_data
            ) / total_docs if total_docs > 0 else 0
            
            report += f"- 처리된 문서 수: {total_docs}\n"
            report += f"- 평균 일관성 비율: {avg_consistency:.2%}\n"
        
        if "performance_analysis" in results and "overall_performance" in results["performance_analysis"]:
            perf = results["performance_analysis"]["overall_performance"]
            report += f"""
## RAG 성능 비교
- 확신도 개선: {perf['confidence_improvement']:.3f}
- Yes 답변 증가: {perf['yes_answer_improvement']}개
- Standard RAG Yes 비율: {perf['standard_yes_rate']:.2%}
- Boost RAG Yes 비율: {perf['boost_yes_rate']:.2%}
"""
        
        if "performance_analysis" in results and "target_achievement" in results["performance_analysis"]:
            target = results["performance_analysis"]["target_achievement"]
            report += f"""
## 목표 달성 여부
- 확신도 개선 목표 달성: {'✅' if target['boost_improvement_achieved'] else '❌'}
- Yes 답변 증가 목표 달성: {'✅' if target['yes_increase_achieved'] else '❌'}
"""
        
        return report
    
    def _print_experiment_summary(self, results: Dict[str, Any]):
        """실험 요약 출력"""
        print("\n" + "=" * 60)
        print("🎉 B-RAG 실험 완료!")
        print("=" * 60)
        
        duration = results["experiment_info"]["duration_seconds"]
        print(f"⏱️ 총 실행 시간: {duration:.2f}초")
        
        if "performance_analysis" in results and "overall_performance" in results["performance_analysis"]:
            perf = results["performance_analysis"]["overall_performance"]
            print(f"📈 확신도 개선: {perf['confidence_improvement']:.3f}")
            print(f"📊 Yes 답변 증가: {perf['yes_answer_improvement']}개")
            
            if "target_achievement" in results["performance_analysis"]:
                target = results["performance_analysis"]["target_achievement"]
                print(f"🎯 목표 달성률: {sum(target.values()[:2])}/2")

def main():
    """메인 실행 함수"""
    print("🚀 B-RAG 실험 시작")
    
    # 설정 선택
    config_type = input("실험 설정을 선택하세요 (quick_test/full_experiment/performance_test): ").strip()
    if not config_type:
        config_type = "quick_test"
    
    # 설정 로드
    config = get_config(config_type)
    
    # 실험 러너 생성 및 실행
    runner = BRAGExperimentRunner(config)
    results = runner.run_full_experiment()
    
    print("\n✅ 모든 실험이 완료되었습니다!")
    return results

if __name__ == "__main__":
    main() 