# 법률 RAG 파이프라인 실험 프로젝트

## 1. 프로젝트 개요

본 프로젝트는 법률 분야의 질의응답(QA) 성능 향상을 위한 RAG(Retrieval-Augmented Generation) 파이프라인을 실험하고 평가하는 것을 목표로 합니다. 다양한 질문 유형과 정책 수준에 따른 모델의 Robustness를 측정하고, 답변 근거의 질을 평가하여 실제 법률 상황에 적용 가능한 AI 시스템을 구축하고자 합니다.

## 2. 주요 디렉토리 및 파일 구조

-   **`liberty_agent/rag/rag_exp.ipynb`**: 메인 실험 코드가 포함된 Jupyter Notebook 파일입니다. 실험의 각 단계(데이터 로드, 질문 생성, RAG 실행, 평가, 시각화)를 셀별로 실행하고 중간 결과를 확인할 수 있습니다.
-   **`data/`**: 원본 법률 데이터(판례 원문, 라벨링 데이터 등)가 저장된 디렉토리입니다.
    -   특히 `data/115.법률-규정 텍스트 분석 데이터_고도화_상황에 따른 판례 데이터/3.개방데이터/1.데이터/Training/02.라벨링데이터/` 경로 내의 JSON 파일들은 "질문-긍정/부정답변-답변근거"가 라벨링 되어 있어 실험의 핵심 데이터로 활용됩니다.
-   **`liberty_agent/cached_vectors/balanced_json/`**: 사전 구축된 FAISS DB 파일(`index.faiss`, `index.pkl`)이 위치합니다. RAG 시스템의 Retriever가 문서를 검색하는 데 사용됩니다.
-   **`pyproject.toml`**: 프로젝트 의존성 관리를 위한 Poetry 설정 파일입니다.
-   **`liberty_agent/rag/qa_experiment_input_notebook.json`**: `rag_exp.ipynb` 실행 시 생성 또는 로드되는 실험용 QA 데이터셋입니다. FAISS DB의 원본 문서를 기반으로 다양한 정책 수준의 변형 질문과 GT 답변, 관련 메타데이터를 포함합니다.

## 3. 실험 목표 및 중요 사항

-   **"긍정/부정" Ground Truth (GT) 유지 및 근거 평가:** 현재 실험은 판례의 주요 쟁점에 대한 답변을 "긍정" 또는 "부정"으로 설정하여 GT를 구축합니다.
-   **답변 근거의 중요성:** 단순 "긍정/부정" 정답률 외에, 모델이 해당 답변을 도출한 **근거(`summ_contxt`, `summ_pass` 필드 참고)**의 타당성을 함께 평가하는 것이 핵심입니다. 이를 통해 모델의 실제 법리 이해도와 Robustness를 심층적으로 분석합니다.
-   **질문 변형을 통한 Robustness 측정:** 동일한 원본 법률 쟁점에 대해 다양한 표현과 난이도의 질문(`transformed_query`)을 생성하여, 모델이 얼마나 일관되고 정확하게 답변 및 근거를 제시하는지 평가합니다.

## 4. 환경 설정 (윈도우 기준)

본 프로젝트는 Python 3.11 버전을 기준으로 하며, `poetry`를 사용하여 의존성을 관리합니다. 윈도우 환경에서 Python 버전을 관리하기 위한 두 가지 주요 방법을 안내합니다. 둘 중 선호하는 방식을 선택하여 진행할 수 있습니다.

**옵션 1: `pyenv-win` 사용 (Python 버전 집중 관리)**

`pyenv-win`은 여러 Python 버전을 쉽게 전환하며 사용할 수 있도록 돕는 도구입니다.

**4.1.1. `pyenv-win` 설치**

   -   PowerShell을 관리자 권한으로 실행합니다.
   -   아래 명령어를 순서대로 입력하여 `pyenv-win`을 설치합니다:
        ```powershell
        Invoke-WebRequest -UseBasicParsing -Uri "https://raw.githubusercontent.com/pyenv-win/pyenv-win/master/pyenv-win/install-pyenv-win.ps1" -OutFile "./install-pyenv-win.ps1"; &"./install-pyenv-win.ps1"
        ```
   -   설치가 완료되면 PowerShell을 다시 시작하거나 새 창을 엽니다.
   -   환경 변수 설정을 위해 다음 명령어를 실행합니다 (필요시 경로 수정):
        ```powershell
        [System.Environment]::SetEnvironmentVariable('PYENV',$env:USERPROFILE + "\.pyenv\pyenv-win\","User")
        [System.Environment]::SetEnvironmentVariable('PYENV_HOME',$env:USERPROFILE + "\.pyenv\pyenv-win\","User")
        [System.Environment]::SetEnvironmentVariable('PYENV_ROOT',$env:USERPROFILE + "\.pyenv\pyenv-win\","User")
        # Path 환경 변수에 pyenv 경로 추가
        $oldPath = [System.Environment]::GetEnvironmentVariable('Path', "User")
        $newPath = $env:USERPROFILE + "\.pyenv\pyenv-win\bin;" + $env:USERPROFILE + "\.pyenv\pyenv-win\shims;" + $oldPath
        [System.Environment]::SetEnvironmentVariable('Path', $newPath, "User")
        ```
   -   PowerShell을 다시 시작합니다.

**4.1.2. Python 3.11 설치 (`pyenv-win` 사용 시)**

   -   사용 가능한 Python 버전 목록을 확인합니다:
        ```bash
        pyenv install -l
        ```
   -   Python 3.11 버전 중 하나를 선택하여 설치합니다 (예: 3.11.10):
        ```bash
        pyenv install 3.11.10
        ```
   -   프로젝트에서 사용할 전역 Python 버전을 설정합니다:
        ```bash
        pyenv global 3.11.10
        ```
   -   또는 특정 프로젝트 폴더에서만 해당 버전을 사용하려면, 프로젝트 루트 디렉토리에서 다음을 실행합니다:
        ```bash
        pyenv local 3.11.10
        ```
   -   Python 버전이 올바르게 설정되었는지 확인합니다:
        ```bash
        python --version
        ```

**옵션 2: Anaconda / Miniconda 사용 (패키지 관리 및 환경 분리 중심)**

Anaconda (또는 경량 버전인 Miniconda)는 Python 및 R 배포판으로, 가상 환경 생성 및 패키지 관리에 강력한 기능을 제공합니다. 특히 복잡한 의존성을 가진 과학 계산 라이브러리 설치에 유용합니다.

**4.2.1. Anaconda 또는 Miniconda 설치**

   -   [Anaconda 공식 홈페이지](https://www.anaconda.com/products/distribution) 또는 [Miniconda 공식 홈페이지](https://docs.conda.io/en/latest/miniconda.html)에서 윈도우용 설치 파일을 다운로드하여 실행합니다.
   -   설치 과정에서 "Add Anaconda (or Miniconda) to my PATH environment variable" 옵션은 **체크하지 않는 것을 권장**합니다. 대신 Anaconda Prompt를 사용하거나, VSCode 등에서 인터프리터를 지정하여 사용합니다.

**4.2.2. Conda 가상 환경 생성 및 Python 3.11 설치**

   -   Anaconda Prompt (또는 Miniconda Prompt)를 실행합니다.
   -   Python 3.11 버전을 사용하는 새 가상 환경을 생성합니다. (예: 환경 이름 `rag_env`)
        ```bash
        conda create -n rag_env python=3.11
        ```
   -   생성된 가상 환경을 활성화합니다:
        ```bash
        conda activate rag_env
        ```
   -   이제 이 터미널에서는 `rag_env` 가상 환경이 사용됩니다.
   -   Python 버전이 올바르게 설정되었는지 확인합니다:
        ```bash
        python --version
        ```
    - (참고) 프로젝트 종료 후 가상 환경을 비활성화하려면:
        ```bash
        conda deactivate
        ```

**4.3. Poetry 설치 (의존성 관리 도구)**

   -   `pyenv-win`을 사용했다면 해당 환경에서, Anaconda/Miniconda를 사용했다면 위에서 활성화한 `conda` 가상 환경 터미널에서 다음을 진행합니다.
   -   PowerShell (또는 Anaconda Prompt)에서 다음 명령어를 실행하여 Poetry를 설치합니다:
        ```powershell
        (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
        ```
   -   설치 후, 터미널을 다시 시작하거나 새 터미널을 열어 환경 변수가 적용되도록 합니다.
   -   Poetry가 올바르게 설치되었는지 확인합니다:
        ```bash
        poetry --version
        ```

**4.4. 프로젝트 의존성 설치 (Poetry 사용)**

   -   프로젝트의 루트 디렉토리( `pyproject.toml` 파일이 있는 위치)로 이동합니다.
   -   현재 활성화된 Python 환경 (`pyenv-win`으로 설정된 버전 또는 활성화된 `conda` 환경)에서 다음 명령어를 실행하여 `pyproject.toml`에 명시된 의존성들을 설치합니다. Poetry는 해당 Python 환경 내에 라이브러리들을 설치하거나, 별도의 가상 환경을 관리할 수 있습니다.
        ```bash
        poetry install
        ```
   -   **`faiss-cpu` 설치 관련:**
        -   `poetry install` 과정에서 `faiss-cpu` 설치에 실패하는 경우가 있습니다.
        -   **Anaconda/Miniconda 환경을 사용하는 경우:** `poetry install` 전에 `conda install -c pytorch faiss-cpu -n rag_env` (여기서 `rag_env`는 사용자가 생성한 conda 환경 이름) 명령으로 먼저 `faiss-cpu`를 설치한 후 `poetry install`을 시도하면 성공률이 높아집니다.
        -   **`pyenv-win` 환경을 사용하는 경우:** `faiss-cpu`의 윈도우 빌드가 까다로울 수 있습니다. C++ 빌드 도구(Visual Studio Build Tools)가 필요할 수 있으며, 경우에 따라서는 [비공식 휠 파일](https://www.lfd.uci.edu/~gohlke/pythonlibs/) 등을 수동으로 설치해야 할 수도 있습니다.
   -   Poetry가 자체적으로 가상 환경을 생성하여 관리하는 경우 (기본 동작), 해당 환경을 활성화해야 합니다.

**4.5. Poetry 가상 환경 활성화 (Poetry가 자체 환경을 생성한 경우)**

   -   만약 `poetry config virtualenvs.in-project true` 설정을 하지 않았다면 Poetry는 프로젝트 외부에 가상 환경을 만듭니다. 이 경우, 프로젝트 루트 디렉토리에서 다음 명령어를 사용하여 Poetry가 생성한 가상 환경을 활성화합니다:
        ```bash
        poetry shell
        ```
   -   이제 이 터미널 세션에서는 프로젝트의 Poetry 가상 환경에서 Python과 라이브러리들이 실행됩니다.
   -   (Anaconda/Miniconda 환경 내에서 Poetry를 사용하고 `poetry install`을 실행했다면, 일반적으로는 이미 활성화된 Conda 환경 내에 패키지가 설치되므로 `poetry shell`을 필수로 실행할 필요는 없을 수 있습니다. 하지만 `poetry shell`을 사용하면 Poetry가 관리하는 경로를 명시적으로 사용하게 됩니다.)

## 5. 실험 실행 방법

1.  **Jupyter Notebook 실행:**
    -   위에서 활성화한 가상 환경 터미널 (`pyenv-win` + `poetry shell` 또는 `conda activate rag_env` 이후 `poetry shell` 또는 직접 Conda 환경에서)에서 Jupyter Notebook을 실행합니다:
        ```bash
        jupyter notebook
        ```
    -   웹 브라우저에서 Jupyter 인터페이스가 열립니다.
2.  **`rag_exp.ipynb` 파일 열기:**
    -   Jupyter 탐색기에서 `liberty_agent/rag/rag_exp.ipynb` 파일을 찾아 엽니다.
3.  **API 키 설정:**
    -   노트북 내에서 OpenAI API 키 등 필요한 API 키를 설정하는 부분이 있는지 확인하고, 본인의 키로 설정합니다. (주로 `.env` 파일을 사용하거나, 노트북 셀 내에서 직접 입력)
4.  **셀 실행:**
    -   노트북의 셀들을 순서대로 실행합니다. "Shift + Enter"를 사용하거나 상단 메뉴의 실행 버튼을 사용합니다.
    -   **데이터셋 생성/로드 (셀 3 부근):** `LOAD_FROM_FILE` 변수 등을 통해 기존에 생성된 `qa_experiment_input_notebook.json` 파일을 로드할지, 아니면 FAISS DB로부터 새로 생성할지 결정할 수 있습니다.
    -   **Standard RAG 실행 (셀 4 부근):** 변형된 질문들을 바탕으로 RAG 시스템을 통해 답변을 생성하고 관련 정보를 수집합니다.
    -   **결과 분석 및 평가 (셀 5 이후):** 생성된 답변과 GT를 비교하여 정확도를 계산하고, 다양한 기준으로 그룹화하여 성능을 분석 및 시각화합니다.

## 6. (선택) VSCode 연동

-   VSCode를 사용하는 경우, Python 인터프리터를 프로젝트 가상 환경의 것으로 설정하면 VSCode 내에서 바로 Jupyter Notebook을 실행하고 디버깅할 수 있습니다.
    1.  VSCode에서 프로젝트 폴더를 엽니다.
    2.  `Ctrl+Shift+P` (또는 `Cmd+Shift+P` on Mac)를 눌러 커맨드 팔레트를 엽니다.
    3.  `Python: Select Interpreter`를 검색하고 선택합니다.
    4.  목록에서 Poetry 가상 환경 또는 Conda 가상 환경 (`rag_env` 등)에 해당하는 인터프리터를 선택합니다.

## 7. 기타

-   실험 과정에서 생성되는 중간 결과물(`standard_rag_results_intermediate_*.json` 등)은 `liberty_agent/rag/experiment_results/` 디렉토리에 저장될 수 있습니다.
-   자세한 실험 방법론 및 각 코드 셀의 역할은 `rag_exp.ipynb` 노트북 내의 주석 및 마크다운 설명을 참고하십시오.

---

이제 `pyenv-win`과 Anaconda/Miniconda 두 가지 옵션을 포함하여 환경 설정을 안내하도록 수정했습니다. 팀원들이 각자의 환경에 더 익숙하거나 선호하는 방식을 선택하여 프로젝트를 진행할 수 있을 것입니다.