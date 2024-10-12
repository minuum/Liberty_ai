# Liberty LangGraph QA 챗봇

Liberty LangGraph QA 챗봇은 LangChain과 LangGraph를 활용하여 PDF 문서에서 정보를 추출하고 사용자의 질문에 답변하는 고급 질의응답 시스템입니다. Streamlit을 통해 사용자 친화적인 인터페이스를 제공합니다.

## 주요 기능

- PDF 문서 기반 정보 추출 및 검색
- LLM을 활용한 동적 질문 답변 생성
- 답변의 관련성 검사 및 질문 재작성 기능
- KoBERT와 Upstage Groundedness Check를 결합한 하이브리드 관련성 검증
- 대화형 채팅 인터페이스

## 기술 스택

- **프레임워크**: Streamlit, LangChain, LangGraph
- **언어 모델**: OpenAI GPT-4
- **임베딩 모델**: KoBERT
- **관련성 검사**: Upstage Groundedness Check
- **문서 처리**: PyPDF2 (추정)

## 설치 및 실행

1. 저장소 클론:
   ```
   git clone https://github.com/your-repo/liberty-langgraph-qa.git
   cd liberty-langgraph-qa
   ```

2. 가상 환경 생성 및 활성화:
   ```
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. 의존성 패키지 설치:
   ```
   pip install -r requirements.txt
   ```

4. 환경 변수 설정:
   `.env` 파일을 생성하고 필요한 API 키를 설정하세요:
   ```
   OPENAI_API_KEY=your_openai_api_key
   UPSTAGE_API_KEY=your_upstage_api_key
   ```

5. 애플리케이션 실행:
   ```
   streamlit run app.py
   ```

## 사용 방법

1. 애플리케이션을 실행하면 Streamlit 인터페이스가 브라우저에서 열립니다.
2. 채팅 입력창에 질문을 입력하세요.
3. 시스템이 PDF 문서에서 관련 정보를 검색하고 답변을 생성합니다.
4. 답변의 관련성이 낮다고 판단되면 자동으로 질문을 재작성하고 프로세스를 반복합니다.
5. 최종 답변과 질문 재작성 횟수가 표시됩니다.

## 프로젝트 구조

```
liberty-langgraph-qa/
│
├── app.py                 # 메인 Streamlit 애플리케이션
├── rag/
│   ├── pdf.py             # PDF 처리 및 검색 기능
│   └── utils.py           # 유틸리티 함수
├── data/
│   └── Minbub Selected Provisions.pdf  # 샘플 PDF 문서
├── requirements.txt       # 프로젝트 의존성
├── .env                   # 환경 변수 (git에서 제외)
└── README.md              # 프로젝트 문서
```

## 기여하기

프로젝트 기여는 언제나 환영합니다. 버그 리포트, 기능 제안 또는 풀 리퀘스트를 통해 참여해 주세요.

## 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 연락처

프로젝트 관련 문의: minwool0357@gmail.com

---

© 2024 Liberty LangGraph QA 챗봇. All rights reserved.