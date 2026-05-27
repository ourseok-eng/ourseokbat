# 경마 현장 분석 에이전트 FINAL

스마트폰 + Streamlit Cloud 사용을 전제로 만든 최종 단순 구조입니다.

## 핵심 원칙
- API는 과거 통계/참고 데이터 보강용입니다.
- 현장 당일 정보는 경마잡지/예상지/배당판/마체중표 사진 업로드로 반영합니다.
- 사진에서 추출한 정보와 실제 결과를 계속 저장해서 통계가 업데이트됩니다.

## 파일
- app.py
- requirements.txt
- packages.txt
- sample_race.csv
- README.md
- secrets.example.toml

## Streamlit Secrets 예시
아래 내용을 Streamlit Cloud의 Manage app → Settings → Secrets에 넣습니다.
인증키는 절대 GitHub 파일에 직접 넣지 마세요.

```toml
OPENAI_API_KEY = ""
OPENAI_VISION_MODEL = "gpt-4.1-mini"
DATA_GO_KR_SERVICE_KEY = ""

# 자동 통계 저장을 원할 때만 사용합니다.
GITHUB_TOKEN = ""
GITHUB_REPO = "본인아이디/저장소명"
GITHUB_BRANCH = "main"
STATS_FILE_PATH = "race_memory.json"
```

## 사용 흐름
1. 데이터 탭: 샘플 또는 CSV 업로드
2. 사진분석 탭: 예상지/배당판/마체중표 사진 업로드 후 분석
3. 최종분석 탭: 추천 배팅안 확인
4. 결과/통계 탭: 실제 도착순서 입력 후 저장
5. 저장소 탭: 통계 파일을 GitHub 또는 다운로드 파일로 유지
