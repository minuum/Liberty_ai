# AI Agent Liberty - 랭그래프 기반 질의응답 시스템

<p align="center">
  <img src="(https://github.com/user-attachments/assets/4c13647e-0358-4b91-9abd-36ad93201795)" width="200">
</p>

## 프로젝트 소개

AI Agent Liberty는 랭그래프(LangGraph) 기법을 활용한 고급 질의응답 시스템입니다. 이 프로젝트는 사용자의 질문에 대해 정확하고 관련성 높은 답변을 제공하기 위해 설계되었습니다.

## 시스템 구조도

AI Agent Liberty의 시스템 구조를 자세히 보려면 아래 Figma 링크를 참조하세요:

[AI Agent Liberty 시스템 구조도](https://www.figma.com/board/0LivhrgVnLyiOM9qjRxVR9/Liberty_Constructure?node-id=0-1&node-type=canvas&t=v77ZqJiqpJL8Std6-0)

## 주요 기능

- 사용자 질문 입력 및 처리
- 랭그래프를 이용한 동적 질문 재작성
- 관련 문서 검색 및 답변 생성
- 답변의 관련성 검사
- 채팅 기록 관리

## 기술 스택

<p align="center">
  <img src="https://github.com/user-attachments/assets/f16b12c4-a786-40f2-86ff-8267cebd0027" alt="Python" width="200">
  <img src="https://github.com/user-attachments/assets/fca98822-0741-4356-b5da-bb392cbefcfb" alt="Streamlit" width="200">
  <img src="https://github.com/user-attachments/assets/0de82bd5-8bd2-4c5d-9111-06bddcc99479" alt="LangChain" width="200">
  <img src="https://github.com/user-attachments/assets/5190d84a-c348-47c5-bb18-3ce47915e3cb" alt="LangGraph" width="200">
</p>

- Python
- Streamlit
- LangChain
- LangGraph

## 설치 방법

1. 저장소를 클론합니다:
   ```
   git clone https://github.com/your-username/ai-agent-liberty.git
   ```

2. 프로젝트 디렉토리로 이동합니다:
   ```
   cd ai-agent-liberty
   ```

3. 필요한 패키지를 설치합니다:
   ```
   pip install -r requirements.txt
   ```

## 사용 방법

1. Streamlit 앱을 실행합니다:
   ```
   streamlit run app.py
   ```

2. 웹 브라우저에서 표시된 URL로 접속합니다.

3. 질문을 입력하고 AI Agent Liberty의 답변을 확인합니다.

## 프로젝트 구조

- `app.py`: 메인 애플리케이션 파일
- `pages/`: Streamlit 멀티페이지 구조를 위한 디렉토리
  - `home.py`: 홈 페이지
  - `settings.py`: 설정 페이지
  - `profile.py`: 프로필 페이지

## 랭그래프 기법 설명

<p align="center">
  <img src="https://github.com/user-attachments/assets/930fca9e-8d22-4a6d-87b1-0bc67d53e51a" alt="LangGraph 프로세스" width="600">
</p>

AI Agent Liberty는 랭그래프 기법을 사용하여 다음과 같은 프로세스를 구현합니다:

1. 사용자 질문 입력
2. 관련 문서 검색
3. LLM을 이용한 답변 생성
4. 답변의 관련성 검사
5. 필요시 질문 재작성 및 프로세스 반복

이 접근 방식을 통해 더 정확하고 관련성 높은 답변을 제공할 수 있습니다.

## 기여 방법

프로젝트에 기여하고 싶으시다면 다음 단계를 따라주세요:

1. 이 저장소를 포크합니다.
2. 새 브랜치를 생성합니다 (`git checkout -b feature/AmazingFeature`).
3. 변경 사항을 커밋합니다 (`git commit -m 'Add some AmazingFeature'`).
4. 브랜치에 푸시합니다 (`git push origin feature/AmazingFeature`).
5. Pull Request를 생성합니다.

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 연락처

프로젝트 관리자 - [@mniuum](https://github.com/dashboard) - minwool0357@gmail.com

프로젝트 링크: [https://github.com/minuum/Liberty_ai](https://github.com/minuum/Liberty_ai)
