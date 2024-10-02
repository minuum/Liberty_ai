# Liberty(Agent 기반 사용자 맞춤 법률 답변 서비스)

### Team

| 역할 | 학번 | 이름 |
| --- | --- | --- |
| PM, AI Agent, UI/UX Design | 202204249 | 이민우 |
| DL, Fine-tuning | 202204248 | 이동현 |

<img width="100" alt="SCR-20241002-jarm" src="https://github.com/user-attachments/assets/17172682-f88a-4417-9057-a3aa3e4379652639">



## Abstract

본 프로젝트에서는 **AI** **법률 답변** **챗봇 서비스**를 **Langgraph**를 이용한 **AI Agent**로 구현하려고 합니다. 판례 답변에 대한 데이터는 2022년 구축된 aihub **의료, 법률 전문 서적 말뭉치**를 이용하였으며 법률에 대한 데이터는 **국회법률정보시스템의 민,형법 자료(PDF)**를 자료 기반으로 삼아 검색 증강 생성(RAG) 기술을 적용시켰습니다. **Langgraph**를 이용해 검색 엔진, DB 검색기, 답변 평가 등의 과정을 거치며 기존 RAG에 비해 정확한 답변을 가능케 하는 AI Agent를 구현하였습니다. 이를 통해 법률 상담, 답변에 대한 준수한 답변 성능을 제공하는 AI 서비스를 만들고자 합니다.

## 1. 개발 배경 및 필요성

- **개발 목적**: 일반 민간인 대상의 법률 자문이나 답변에 대한 비용이나 접근성 및 편의성이 좋지 않음.
- **개발 동기**: 민간인 대상 법률 판례 AI가 많이 존재하지 않기 때문이며, 누구나 쉽게 법의 구제를 받으며 불합리하게 당하는 사건들을 미연에 방지할 수 있는 하나의 플랫폼을 만들고 싶다는 생각이 들었음.

## 2. 개발 목표

- AI Agent 기술을 이용하여 사용자에게 접근 용이한 AI 법률 답변 서비스를 만들고자 함.
- 파인-튜닝을 이용하여 데이터를 정형화하여 보다 확실한 답변을 하게끔 하고자 함.
- 특정 분야의 법률이 아닌 많은 분야의 법의 내용을 다뤄 이용자의 대답의 자유성을 넓혀 주고자 함.

## 3. 관련 연구

- **lawandsearch**: 생성형 AI 기술을 활용한 법률 챗봇 서비스로, 변호사 전문단 구축,
법률지식 특화, 높은 한국어 이해도, 환각현상 방지 등의 특징을 가짐.

https://lawandsearch.ai/

- **차별점**: Langgraph를 통해 데이터를 검색하고 평가하는 과정에서 정확도를 상승시키는 것.

https://github.com/kyopark2014/langgraph-agent

## 4. 예상 결과물 UI

- **UI 특징**: 간결하고 직관적인 디자인, 채팅 형식의 인터페이스, 추천 질문 기능, 법률 자료 버튼 제공.
- **UI 구현 도구**: [Streamlit](https://streamlit.io/), [creatie.ai](http://creatie.ai/)



## 5. 주요 기능

| 주요 기능 명칭 | 주요 기능 세부 설명 |
| --- | --- |
| 사용자 질문 분석 | Langchain의 LLMChain을 사용하여 KoBERT 모델로 사용자의 법률 질문을 분석하고 의도를 파악 |
| 법률 정보 검색 | Langchain의 VectorStore와 Retriever를 활용하여 관련 법률 조문, 판례, 전문 서적 등에서 질문과 관련된 정보를 효율적으로 검색 |
| 맥락 기반 답변 생성 | Langchain의 LLMChain을 이용해 GPT-4o 모델을 기반으로 검색된 법률 정보와 대화 맥락을 고려한 정확하고 상세한 법률 답변을 생성 |
| 답변 품질 평가 | Langchain의 AgentExecutor를 사용하여 생성된 답변의 정확성, 관련성, 완전성을 평가 |
| 대화 흐름 관리 | Langgraph를 활용하여 복잡한 법률 상담 과정을 여러 단계로 나누어 관리 |
| 법률 용어 설명 | Langchain의 Tool 기능을 이용해 답변 중 어려운 법률 용어나 개념이 나오면 자동으로 설명을 추가 |
| 관련 법조문 참조 | Langchain의 SQLDatabaseChain을 활용하여 답변과 관련된 정확한 법조문을 데이터베이스에서 찾아 참조로 제시 |
| 사용자 피드백 처리 | Langchain의 HumanInputRun을 통해 사용자의 추가 질문이나 명확화 요청을 처리 |

## 6. 시스템 구조(Architecture)

- **구성요소**: Agent Node, KoBERT LLM 판단 모델, 검색 도구, GPT-4 생성 모델, 답변 평가 Agent, Rewrite Node, 데이터베이스, 웹 서버, 프론트엔드.
- **구현 방법**: Python, Langgraph, PyTorch, Elasticsearch, OpenAI API, PostgreSQL, Django/FastAPI, React.js, Tailwind CSS.

#### [시스템 구조(figma)](https://www.figma.com/board/0LivhrgVnLyiOM9qjRxVR9/Liberty_Constructure?node-id=0-1&node-type=canvas&t=ixMVKGxF9RGzXS04-0)

## 7. 개발 방법론

- **구현환경**: UI는 Streamlit을 이용하여 Python 환경에서 구현.
- **사용 예정 데이터** https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71487
  <img width="800" alt="SCR-20241002-jlge" src="https://github.com/user-attachments/assets/30c2b4a1-ca47-4989-8f13-10d933e52639">
- **학습 방법**: 비 가공 데이터를 가공한 후, 파인-튜닝을 이용하여 학습.

## 8. 팀원 역할분담(R&R)

## 9. 일정(소요예산, 요청사항)

### Sprint 1 - 계획서 발표

- 1주차: 팀 조직
- 2주차: 주제 선정 및 기획서 초안 작성
- 3주차: 기획서 확정 및 개발환경 구
- 4주차: 개발 환경 구성

### Sprint 2 - 8주차 중간평가(1st 프로토타입 제작)

- 5주차: 데이터 수집 및 정제
- 6주차: 법률 데이터 수집 및 전처리
- 7주차: 모델 선택 및 기초 학습
- 8주차: 모델 성능 평가 및 1차 수정

### Sprint 3 - 12주 중간점검(2nd 프로토타입 제작)

- 9주차: 법률 상담용 UI/UX 디자인 및 개발
- 10주차: 모델 피드백 반영 및 재학습
- 11주차: 법률 답변 시나리오 확립
- 12주차: 2nd 프로토타입 기능 테스트 및 개선

### Sprint 4 (최종 배포 준비)

- 13주차: 시스템 통합 테스트
- 14주차: 법률 규정 및 데이터 윤리 검토
- 15주차: 최종 프로토타입 배포 및 발표 준비
![image.png](https://github.com/user-attachments/assets/d9cdf0b0-3f75-45ce-bff8-bccaf1d756e4)


## 10. 기대 효과 및 활용 분야

법률 답변 챗봇 서비스 Liberty는 일반인들에게 접근성 높고 비용 효율적인 기초 법률 정보를 제공합니다. 이 서비스는 복잡한 법률 개념을 이해하기 쉽게 설명하여 대중의 법률 이해도를 높이고 잘못된 법률 상식을 바로잡는 데 기여합니다. 또한, 필요시 적절한 전문 법률 서비스로 연결해주는 중개 역할을 수행하여, 법률 세계와 일반인 사이의 가교 역할을 합니다.
