# 경마 사진출전표 + 공공데이터 + 결과학습 에이전트

## 핵심
이 버전은 출전마 명단도 사진에서 읽습니다. CSV는 필수가 아닙니다.

## 흐름
1. 사진에서 출전마 추출
2. 공공데이터로 말별 과거통계 보강
3. 현장 사진으로 배당/마체중/전문가표 보정
4. 최종 배팅안 확인
5. 결과 기록
6. 통계 JSON 저장/불러오기

## GitHub 업로드 파일
- app.py
- requirements.txt
- packages.txt
- sample_race.csv
- README.md

## Streamlit Secrets
아래 3줄만 넣어도 됩니다.

```toml
OPENAI_API_KEY = "너의 OpenAI API 키"
OPENAI_VISION_MODEL = "gpt-4.1-mini"
DATA_GO_KR_SERVICE_KEY = "너의 공공데이터포털 Decoding 인증키"
```

API URL은 app.py 안에 기본값으로 들어 있습니다.
