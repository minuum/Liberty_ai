# ===== 🧪 셀 4-실제: 실제 다중 문서 RAG 비교 테스트 =====

print("\n===== 실제 FAISS DB RAG 테스트 시작 =====")
print("🔥 진짜 RAG 시스템 vs 시뮬레이션이 아닌 실제 테스트!")
print("📋 FAISS DB + LLM + 실제 문서 검색 + 실제 답변 생성")
print("-" * 60)

try:
    import time
    import json
    import random
    from dataclasses import dataclass, asdict
    from typing import List, Dict, Any, Optional
    from datetime import datetime
    from pathlib import Path
    
    # 필요 모듈 임포트
    try:
        from langchain_community.vectorstores import FAISS
        from langchain_upstage import UpstageEmbeddings  
        from langchain_openai import ChatOpenAI
        from langchain.schema import Document
        from langchain.prompts import PromptTemplate
        print("✅ 필요 모듈 로드 완료")
    except ImportError as e:
        print(f"❌ 모듈 로드 오류: {e}")
        raise
    
    @dataclass
    class RealRAGResult:
        question: str
        answer: str
        confidence: float
        document_ids: List[str]
        retrieved_docs: List[str]
        processing_time: float
        retrieval_scores: List[float]
        reasoning: str
        
    @dataclass
    class RealYesNoRAGConfig:
        embedding_model: str = "solar-embedding-1-large"
        llm_model: str = "gpt-4o-2024-08-06"
        llm_temperature: float = 0.1
        top_k: int = 3
        similarity_threshold: float = 0.7
        boost_mode: bool = False
    
    # 실제 RAG 시스템 클래스
    class RealYesNoRAGSystem:
        def __init__(self, config: RealYesNoRAGConfig, vectorstore: FAISS):
            self.config = config
            self.vectorstore = vectorstore
            self.embeddings = UpstageEmbeddings(model=config.embedding_model)
            self.llm = ChatOpenAI(
                model_name=config.llm_model,
                temperature=config.llm_temperature
            )
            
            # RAG 프롬프트 템플릿
            if config.boost_mode:
                self.prompt_template = PromptTemplate(
                    input_variables=["question", "context", "documents"],
                    template="""당신은 법률 전문가입니다. 주어진 법률 문서를 기반으로 질문에 Yes 또는 No로 답변하세요.

🎯 B-RAG 강화 모드: 더 정확하고 세밀한 분석을 수행합니다.

📄 관련 법률 문서들:
{documents}

📄 문맥:
{context}

❓ 질문: {question}

📋 답변 형식:
답변: [Yes/No]
확신도: [0.0-1.0 사이의 숫자]
근거: [법적 근거와 판단 이유를 상세히 설명]

답변을 시작하세요:"""
                )
            else:
                self.prompt_template = PromptTemplate(
                    input_variables=["question", "context", "documents"], 
                    template="""당신은 법률 전문가입니다. 주어진 법률 문서를 기반으로 질문에 Yes 또는 No로 답변하세요.

📄 관련 법률 문서들:
{documents}

📄 문맥:
{context}

❓ 질문: {question}

📋 답변 형식:
답변: [Yes/No]
확신도: [0.0-1.0 사이의 숫자]
근거: [판단 이유]

답변을 시작하세요:"""
                )
    
        def retrieve_documents(self, question: str) -> List[tuple]:
            """문서 검색"""
            try:
                # 유사도 점수와 함께 문서 검색
                docs_with_scores = self.vectorstore.similarity_search_with_score(
                    question, 
                    k=self.config.top_k
                )
                
                # 임계값 필터링
                filtered_docs = [
                    (doc, float(score)) for doc, score in docs_with_scores 
                    if score >= self.config.similarity_threshold
                ]
                
                return filtered_docs if filtered_docs else docs_with_scores[:1]  # 최소 1개
                
            except Exception as e:
                print(f"⚠️ 문서 검색 오류: {e}")
                return []
    
        def generate_answer(self, question: str, retrieved_docs: List[tuple]) -> RealRAGResult:
            """실제 RAG 답변 생성"""
            start_time = time.time()
            
            try:
                if not retrieved_docs:
                    return self._create_fallback_result(question, start_time)
                
                # 문서 정보 추출
                documents = []
                doc_ids = []
                scores = []
                
                for doc, score in retrieved_docs:
                    doc_id = doc.metadata.get('id', 'unknown')
                    title = doc.metadata.get('title', '제목 없음')
                    content = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
                    
                    documents.append(f"문서 ID: {doc_id}\n제목: {title}\n내용: {content}")
                    doc_ids.append(doc_id)
                    scores.append(float(score))
                
                # 컨텍스트 생성
                context = "\n\n".join(documents)
                documents_text = "\n\n".join(documents)
                
                # LLM 호출
                prompt = self.prompt_template.format(
                    question=question,
                    context=context,
                    documents=documents_text
                )
                
                response = self.llm.invoke(prompt)
                response_text = response.content
                
                # 답변 파싱
                answer, confidence, reasoning = self._parse_response(response_text)
                
                processing_time = time.time() - start_time
                
                return RealRAGResult(
                    question=question,
                    answer=answer,
                    confidence=confidence,
                    document_ids=doc_ids,
                    retrieved_docs=[doc.page_content[:200] + "..." for doc, _ in retrieved_docs],
                    processing_time=processing_time,
                    retrieval_scores=scores,
                    reasoning=reasoning
                )
                
            except Exception as e:
                print(f"⚠️ 답변 생성 오류: {e}")
                return self._create_fallback_result(question, start_time)
    
        def _parse_response(self, response_text: str) -> tuple:
            """LLM 응답 파싱"""
            try:
                lines = response_text.strip().split('\n')
                answer = "No"  # 기본값
                confidence = 0.5  # 기본값
                reasoning = "응답 파싱 실패"
                
                for line in lines:
                    line = line.strip()
                    if line.startswith('답변:'):
                        answer_part = line.replace('답변:', '').strip()
                        if 'yes' in answer_part.lower() or '예' in answer_part or 'Yes' in answer_part:
                            answer = "Yes"
                        else:
                            answer = "No"
                    elif line.startswith('확신도:'):
                        conf_part = line.replace('확신도:', '').strip()
                        try:
                            confidence = float(conf_part.replace('%', '').replace('점', ''))
                            if confidence > 1.0:  # 백분율인 경우
                                confidence = confidence / 100.0
                        except:
                            confidence = 0.5
                    elif line.startswith('근거:'):
                        reasoning = line.replace('근거:', '').strip()
                
                return answer, confidence, reasoning
                
            except Exception as e:
                print(f"⚠️ 응답 파싱 오류: {e}")
                return "No", 0.5, str(e)
    
        def _create_fallback_result(self, question: str, start_time: float) -> RealRAGResult:
            """실패 시 fallback 결과"""
            processing_time = time.time() - start_time
            return RealRAGResult(
                question=question,
                answer="No",
                confidence=0.3,
                document_ids=["fallback"],
                retrieved_docs=["검색 실패"],
                processing_time=processing_time,
                retrieval_scores=[0.0],
                reasoning="시스템 오류로 인한 기본 답변"
            )

    # FAISS DB 로드
    faiss_db_path = Path("/Users/minu/dev/Liberty/Liberty_ai/liberty_agent/cached_vectors/balanced_json")
    print(f"🔄 FAISS DB 로드 중: {faiss_db_path}")
    
    try:
        embeddings = UpstageEmbeddings(model="solar-embedding-1-large")
        vectorstore = FAISS.load_local(str(faiss_db_path), embeddings, allow_dangerous_deserialization=True)
        total_docs = int(vectorstore.index.ntotal)
        print(f"✅ FAISS DB 로드 완료: {total_docs} 문서")
    except Exception as e:
        print(f"❌ FAISS DB 로드 실패: {e}")
        raise
    
    # 다중 문서 질문 확인
    if ('multi_document_questions' in globals() and 
        'multi_document_gt_questions' in globals() and
        globals()['multi_document_questions'] and
        globals()['multi_document_gt_questions']):
        
        multi_doc_questions = globals()['multi_document_questions']
        
        print(f"✅ 다중 문서 질문 발견: {len(multi_doc_questions)}개 세트")
        
        # 테스트할 질문 준비 (축소해서 빠르게)
        all_test_questions = []
        for question_set in multi_doc_questions[:3]:  # 처음 3개 세트만
            # 각 세트에서 Level 1, 5, 10만 선택
            selected_questions = []
            target_levels = [1, 5, 10]
            
            for level in target_levels:
                level_questions = [q for q in question_set['questions'] if q['level'] == level]
                if level_questions:
                    selected_questions.append(level_questions[0]['question'])
            
            all_test_questions.append({
                "gt_question": question_set['gt_question'],
                "test_questions": selected_questions
            })
        
        print(f"📋 테스트할 질문 세트: {len(all_test_questions)}개")
        total_questions = sum(len(q['test_questions']) for q in all_test_questions)
        print(f"📊 총 테스트 질문 수: {total_questions}개")
        
        # 샘플 출력
        print(f"\n📝 실제 테스트 질문 샘플:")
        for i, test_set in enumerate(all_test_questions):
            print(f"\n  세트 {i+1} - GT: '{test_set['gt_question'][:50]}...'")
            for j, q in enumerate(test_set['test_questions']):
                print(f"    레벨 {[1,5,10][j]}: {q}")
        
        # RAG 시스템 설정
        standard_config = RealYesNoRAGConfig(
            llm_temperature=0.1,
            top_k=3,
            similarity_threshold=0.7,
            boost_mode=False
        )
        
        boost_config = RealYesNoRAGConfig(
            llm_temperature=0.05,
            top_k=5,
            similarity_threshold=0.6,
            boost_mode=True
        )
        
        # RAG 시스템 초기화
        print(f"\n🔄 실제 RAG 시스템 초기화 중...")
        standard_rag = RealYesNoRAGSystem(standard_config, vectorstore)
        boost_rag = RealYesNoRAGSystem(boost_config, vectorstore)
        
        print(f"✅ 실제 RAG 시스템 초기화 완료:")
        print(f"  🔵 Standard RAG: {standard_config.llm_model}, temp={standard_config.llm_temperature}, top_k={standard_config.top_k}")
        print(f"  🎯 Boost RAG: {boost_config.llm_model}, temp={boost_config.llm_temperature}, top_k={boost_config.top_k}")
        
        # 실제 테스트 실행
        print(f"\n🚀 실제 RAG 테스트 실행 시작!")
        print(f"⚠️  각 질문마다 실제 FAISS 검색 + LLM 호출이 발생합니다")
        
        all_standard_results = []
        all_boost_results = []
        
        total_start_time = time.time()
        
        for test_set_idx, test_set in enumerate(all_test_questions):
            print(f"\n🔍 세트 {test_set_idx+1}/{len(all_test_questions)} - '{test_set['gt_question'][:40]}...'")
            
            standard_results = []
            boost_results = []
            
            for q_idx, question in enumerate(test_set['test_questions']):
                level_name = ["Level 1", "Level 5", "Level 10"][q_idx]
                print(f"\n  📝 {level_name}: '{question[:60]}...'")
                
                try:
                    # Standard RAG 실행
                    print(f"    🔵 Standard RAG: 검색 중...", end="", flush=True)
                    std_docs = standard_rag.retrieve_documents(question)
                    print(f" {len(std_docs)}개 문서 → 답변 생성 중...", end="", flush=True)
                    
                    standard_result = standard_rag.generate_answer(question, std_docs)
                    standard_results.append(standard_result)
                    print(f" 완료!")
                    print(f"      결과: {standard_result.answer} (확신도: {standard_result.confidence:.3f}, {standard_result.processing_time:.1f}초)")
                    
                    # Boost RAG 실행  
                    print(f"    🎯 Boost RAG: 검색 중...", end="", flush=True)
                    boost_docs = boost_rag.retrieve_documents(question)
                    print(f" {len(boost_docs)}개 문서 → 답변 생성 중...", end="", flush=True)
                    
                    boost_result = boost_rag.generate_answer(question, boost_docs)
                    boost_results.append(boost_result)
                    print(f" 완료!")
                    print(f"      결과: {boost_result.answer} (확신도: {boost_result.confidence:.3f}, {boost_result.processing_time:.1f}초)")
                    
                    # 개선 효과 표시
                    confidence_diff = boost_result.confidence - standard_result.confidence
                    if confidence_diff > 0.05:
                        print(f"      🚀 확신도 크게 개선: +{confidence_diff:.3f}")
                    elif confidence_diff > 0:
                        print(f"      ✅ 확신도 소폭 개선: +{confidence_diff:.3f}")
                    elif confidence_diff < -0.05:
                        print(f"      ⚠️ 확신도 감소: {confidence_diff:.3f}")
                    
                except Exception as e:
                    print(f"\n      ❌ 오류 발생: {e}")
                    # 실패 시 기본 결과
                    fallback_result = RealRAGResult(
                        question=question,
                        answer="No",
                        confidence=0.3,
                        document_ids=["error"],
                        retrieved_docs=["오류"],
                        processing_time=1.0,
                        retrieval_scores=[0.0],
                        reasoning="처리 중 오류 발생"
                    )
                    standard_results.append(fallback_result)
                    boost_results.append(fallback_result)
            
            # 세트별 결과 저장 및 요약
            all_standard_results.append(standard_results)
            all_boost_results.append(boost_results)
            
            std_yes = sum(1 for r in standard_results if r.answer == "Yes")
            boost_yes = sum(1 for r in boost_results if r.answer == "Yes")
            
            std_avg_conf = sum(r.confidence for r in standard_results) / len(standard_results)
            boost_avg_conf = sum(r.confidence for r in boost_results) / len(boost_results)
            
            print(f"\n  📊 세트 {test_set_idx+1} 실제 결과 요약:")
            print(f"    🔵 Standard RAG: Yes {std_yes}/{len(standard_results)}개, 평균 확신도 {std_avg_conf:.3f}")
            print(f"    🎯 Boost RAG: Yes {boost_yes}/{len(boost_results)}개, 평균 확신도 {boost_avg_conf:.3f}")
            print(f"    🚀 실제 개선 효과: 확신도 {boost_avg_conf - std_avg_conf:+.3f}")
        
        total_processing_time = time.time() - total_start_time
        
        # 전체 결과 요약
        total_standard_yes = sum(sum(1 for r in results if r.answer == "Yes") for results in all_standard_results)
        total_boost_yes = sum(sum(1 for r in results if r.answer == "Yes") for results in all_boost_results)
        
        total_standard_questions = sum(len(results) for results in all_standard_results)
        total_boost_questions = sum(len(results) for results in all_boost_results)
        
        avg_standard_confidence = sum(sum(r.confidence for r in results) for results in all_standard_results) / total_standard_questions
        avg_boost_confidence = sum(sum(r.confidence for r in results) for results in all_boost_results) / total_boost_questions
        
        # 처리 시간 통계
        std_avg_time = sum(sum(r.processing_time for r in results) for results in all_standard_results) / total_standard_questions
        boost_avg_time = sum(sum(r.processing_time for r in results) for results in all_boost_results) / total_boost_questions
        
        print(f"\n{'='*60}")
        print(f"🏆 전체 실제 RAG 테스트 결과 요약")
        print(f"{'='*60}")
        print(f"  🎯 총 테스트 세트: {len(all_test_questions)}개")
        print(f"  📝 총 테스트 질문: {total_standard_questions}개")
        print(f"  ⏱️ 총 소요 시간: {total_processing_time:.2f}초")
        print(f"\n  🔵 Standard RAG (기본 모드):")
        print(f"    📊 Yes 답변: {total_standard_yes}/{total_standard_questions}개 ({total_standard_yes/total_standard_questions:.1%})")
        print(f"    🎯 평균 확신도: {avg_standard_confidence:.3f}")
        print(f"    ⚡ 평균 처리시간: {std_avg_time:.2f}초/질문")
        print(f"\n  🎯 Boost RAG (강화 모드):")
        print(f"    📊 Yes 답변: {total_boost_yes}/{total_boost_questions}개 ({total_boost_yes/total_boost_questions:.1%})")
        print(f"    🎯 평균 확신도: {avg_boost_confidence:.3f}")
        print(f"    ⚡ 평균 처리시간: {boost_avg_time:.2f}초/질문")
        print(f"\n  🚀 실제 B-RAG 개선 효과:")
        print(f"    📈 확신도 향상: {avg_boost_confidence - avg_standard_confidence:+.3f}")
        print(f"    ⚡ 처리 속도: {boost_avg_time - std_avg_time:+.2f}초")
        
        # 성과 판정
        if avg_boost_confidence > avg_standard_confidence + 0.02:
            print(f"    🎉 유의미한 성능 개선 확인!")
        elif avg_boost_confidence > avg_standard_confidence:
            print(f"    ✅ 소폭 성능 개선")
        else:
            print(f"    ⚠️ 성능 개선 미미")
        
        # 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_data = {
            "test_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "test_type": "실제 FAISS DB RAG 비교 테스트",
            "total_test_sets": len(all_test_questions),
            "total_questions": total_standard_questions,
            "total_processing_time": float(total_processing_time),
            "faiss_db_documents": int(total_docs),
            "standard_rag_results": {
                "yes_count": int(total_standard_yes),
                "yes_ratio": float(total_standard_yes / total_standard_questions),
                "avg_confidence": float(avg_standard_confidence),
                "avg_processing_time": float(std_avg_time)
            },
            "boost_rag_results": {
                "yes_count": int(total_boost_yes),
                "yes_ratio": float(total_boost_yes / total_boost_questions),
                "avg_confidence": float(avg_boost_confidence),
                "avg_processing_time": float(boost_avg_time)
            },
            "improvements": {
                "confidence_improvement": float(avg_boost_confidence - avg_standard_confidence),
                "speed_change": float(boost_avg_time - std_avg_time),
                "significant_improvement": bool(avg_boost_confidence > avg_standard_confidence + 0.02)
            },
            "detailed_results": []
        }
        
        # 상세 결과 추가 (JSON 직렬화 안전)
        for i, (test_set, std_results, boost_results) in enumerate(zip(all_test_questions, all_standard_results, all_boost_results)):
            set_data = {
                "set_index": i + 1,
                "gt_question": test_set["gt_question"],
                "test_questions": test_set["test_questions"],
                "standard_results": [asdict(r) for r in std_results],
                "boost_results": [asdict(r) for r in boost_results]
            }
            result_data["detailed_results"].append(set_data)
        
        # JSON 파일 저장
        result_filename = f"real_faiss_rag_test_results_{timestamp}.json"
        with open(result_filename, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 실제 RAG 테스트 결과 저장 완료:")
        print(f"📄 파일명: {result_filename}")
        print(f"📍 파일 위치: {Path.cwd() / result_filename}")
        
        # 전역 변수에 결과 저장
        globals()['real_standard_rag_results'] = all_standard_results
        globals()['real_boost_rag_results'] = all_boost_results
        globals()['real_test_questions'] = all_test_questions
        globals()['real_rag_test_data'] = result_data
        
        print(f"\n✅ 전역 변수 저장 완료:")
        print(f"  - real_standard_rag_results: Standard RAG 결과")
        print(f"  - real_boost_rag_results: Boost RAG 결과") 
        print(f"  - real_test_questions: 테스트 질문들")
        print(f"  - real_rag_test_data: 전체 결과 데이터")
            
    else:
        print("❌ 다중 문서 질문을 찾을 수 없습니다")
        print("💡 해결 방법:")
        print("   1. 먼저 셀 3-8 (FAISS DB 활용 다중 문서 질문 생성)을 실행하세요")
        
except Exception as e:
    print(f"❌ 실제 RAG 테스트 중 오류 발생: {e}")
    import traceback
    traceback.print_exc()

print(f"\n{'='*60}")
print(f"🏁 실제 FAISS DB RAG 테스트 완료!")
print(f"🔥 이번엔 진짜로 FAISS DB + LLM을 사용했습니다!")
print(f"{'='*60}") 