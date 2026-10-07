# RAG Chatbot

Python 3.11과 LangChain, OpenAI, Streamlit을 사용하는 RAG 챗봇 프로젝트입니다.

## 시작하기

```powershell
# .env가 없을 때만 생성합니다. 기존 API 키를 덮어쓰지 않습니다.
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
uv run streamlit run app.py
```

`.env`에 실제 OpenAI API 키를 입력하세요.

## Streamlit Community Cloud 배포

1. [Streamlit Community Cloud](https://share.streamlit.io/)에 GitHub 계정으로 로그인합니다.
2. **Create app**에서 저장소 `jonijonii/chatbot`, 브랜치 `main`, 진입 파일 `app.py`를 선택합니다.
3. **Advanced settings**에서 Python 버전을 **3.11**로 고릅니다.
4. **Secrets**에는 다음 형식으로 실제 키를 입력합니다. 값은 GitHub 파일에 넣지 않습니다.

   ```toml
   OPENAI_API_KEY = "실제 API 키"
   ```

5. **Deploy**를 누릅니다. `uv.lock`을 사용해 Python 패키지를 설치합니다.

현재 대화 기록은 서버의 `.chat_history/messages.json` 한 파일에 저장됩니다. 공개 배포에서
여러 사람이 접속하면 대화가 공유될 수 있고, 클라우드 서버 재시작 시 기록이 사라질 수
있습니다. 개인별로 계속 보존하려면 사용자 인증과 외부 데이터베이스가 필요합니다.

대화와 출처는 `.chat_history/messages.json`에 자동 저장됩니다. 새로고침, 서버 재시작,
코드·문서 변경 후에도 복원되며 사이드바의 **대화 초기화** 버튼으로 저장 기록을 지웁니다.
이 저장소는 현재 로컬 개인용 프로젝트에서 공유하는 하나의 대화 기록입니다.
`.chat_history/`는 Git에서 제외됩니다. 이미 지워진 과거 대화는 복구할 수 없습니다.

## 검색 및 검증

- PDF 표의 열 배치와 셀을 보존하고 반복 점선이 있는 목차를 검색에서 제외합니다.
- 의미 검색과 BM25를 RRF로 결합합니다. 짧은 정확 일치 검색어도 활용합니다.
- 국가명 등 희귀어와 지급구분표를 검색하고 앞뒤 페이지를 함께 읽습니다.
- 최대 24페이지, 85,000자와 추가 검색 1회로 검색 범위를 제한합니다.
- 모델이 원문 ID를 선택하면 별도의 모델 호출로 직급·금액·조건을 대조합니다.
- 명확한 일반직 지급구분표와 금액표는 원문 셀을 코드로 연결해 검산합니다. 국가·등급·금액은 원문에서 읽으며 답을 하드코딩하지 않습니다.
- 이 검산 경로에서는 모델의 자유 서술 대신 검산한 원문 값으로 최종 답변을 구성합니다. 복잡한 직종·조건·불명확한 표는 일반 검색 및 모델 검증 경로로 처리합니다.
- 사이드바의 개발용 검색 진단에서 검색어, 페이지, 후보 답변, 검증 결과를 확인할 수 있습니다.

```powershell
uv run python -W error -m unittest test_rag
# 실제 OpenAI API 비용이 발생하는 네 질문 회귀 검사
uv run python -W error verify_rag.py
uv run python -W error::DeprecationWarning -W error::FutureWarning verify_ui.py
```

`rag_verification.json`에는 답변과 근거가 저장됩니다. API 키는 저장하지 않습니다.
인덱스는 메모리에 있으므로 서버 재시작이나 문서/코드 변경 후 다시 임베딩합니다.
지원 파일은 PDF, TXT, MD입니다. 스캔 PDF는 OCR 처리가 필요합니다.
표 추출과 모델의 원문 대조는 완전한 정확성을 보장하지 않습니다. 중요한 금액은 표시된 원문으로 확인하세요.
