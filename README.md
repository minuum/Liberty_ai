# Liberty LangGraph QA 챗봇

이 프로젝트는 Streamlit을 사용하여 구현된 QA 챗봇입니다. LangChain과 LangGraph를 활용하여 PDF 문서에서 정보를 추출하고 질문에 답변합니다.

## 설치 및 실행

1. 필요한 패키지 설치:
   ```
   pip install -r requirements.txt
   ```

2. 환경 변수 설정:
   `.env` 파일을 생성하고 필요한 API 키를 설정하세요.

3. 애플리케이션 실행:
   ```
   streamlit run app.py
   ```

## 주요 기능

- PDF 문서에서 정보 추출
- 사용자 질문에 대한 답변 생성
- 답변의 관련성 검사 및 질문 재작성
- 대화형 채팅 인터페이스

## 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다.