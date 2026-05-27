# 경마 PDF출전표 + 공공데이터 + 선명사진 분석

## 핵심
- 출전마 명단을 PDF로 업로드합니다.
- PDF가 스캔이어도 고해상도 이미지로 렌더링해서 읽습니다.
- 사진/PDF 미리보기를 크게 표시해서 흐릿하게 보이지 않게 했습니다.

## Streamlit Secrets
```toml
OPENAI_API_KEY = "너의 OpenAI API 키"
OPENAI_VISION_MODEL = "gpt-4.1-mini"
DATA_GO_KR_SERVICE_KEY = "너의 공공데이터포털 Decoding 인증키"
```

## GitHub 업로드 파일
- app.py
- requirements.txt
- packages.txt
- sample_race.csv
- README.md
