# 경마 공공데이터 제대로 버전

## 이번 버전의 차이
- 경주 전 API와 경주 후 API를 분리했습니다.
- 경주 전 자동 로딩은 출전마현황 + 출전마체중 + 출전등록현황만 사용합니다.
- 경주결과/확정배당/코너순위 API는 사후 통계용으로 분리했습니다.
- 각 API는 필요한 요청변수가 다르므로 한 번에 동일 파라미터로 호출하지 않습니다.

## GitHub 업로드 파일
- app.py
- requirements.txt
- packages.txt
- sample_race.csv
- README.md

## Secrets 예시
```toml
OPENAI_API_KEY = "너의 OpenAI API 키"
OPENAI_VISION_MODEL = "gpt-4.1-mini"
DATA_GO_KR_SERVICE_KEY = "너의 공공데이터포털 Decoding 인증키"
```

## 사용 순서
1. 공공데이터로 출전마 불러오기 탭에서 경마장/날짜/경주번호 입력
2. 출전마/체중 API로 불러오기
3. 사진 분석 탭에서 예상지/마체중/배당판 사진 업로드
4. 최종 분석 확인
5. 경주 후 결과 저장
