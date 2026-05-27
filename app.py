
import base64
import json
import math
import re
from datetime import datetime
from io import BytesIO, StringIO

import pandas as pd
import requests
import streamlit as st
from PIL import Image

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

APP_TITLE = "🏇 경마 사진출전표 + 공공데이터 + 결과학습 에이전트"
DEFAULT_MODEL = "gpt-4.1-mini"

st.set_page_config(page_title=APP_TITLE, layout="wide")

SAMPLE_CSV = """horse_no,horse_name,gate_no,distance,odds_win,odds_place,recent_rank_avg,distance_score,jockey_score,trainer_score,weight_change,api_score
1,인사이드킹,1,1200,4.5,1.7,3.2,77,70,68,-1,0
2,스피드퀸,3,1200,2.6,1.3,2.1,86,82,80,2,0
3,막판추입,7,1200,8.8,2.4,4.6,73,71,70,-3,0
4,가속흐름,4,1200,5.5,1.9,3.1,81,77,75,1,0
5,파워런,8,1200,12.0,3.2,5.4,65,69,67,0,0
6,선행강자,2,1200,6.5,2.0,3.8,79,76,72,4,0
7,꾸준한말,5,1200,15.0,4.0,5.8,61,63,64,-2,0
8,복병후보,6,1200,18.0,4.8,6.2,60,66,62,3,0
"""

DEFAULT_API_URLS = {
    "KRA_HORSE_RECORD_API_URL": "https://apis.data.go.kr/B551015/API145",
    "KRA_RACE_INFO_API_URL": "https://apis.data.go.kr/B551015/API187",
    "KRA_SALES_ODDS_API_URL": "https://apis.data.go.kr/B551015/API179_1",
    "KRA_SECTION_PACE_API_URL": "https://apis.data.go.kr/B551015/API303",
    "KRA_FINAL_ODDS_API_URL": "https://apis.data.go.kr/B551015/API301",
    "KRA_RACE_RESULT_API_URL": "https://apis.data.go.kr/B551015/API299",
}

def sec(name, default=""):
    try:
        return str(st.secrets.get(name, default) or "")
    except Exception:
        return default

def api_url(name):
    return sec(name, DEFAULT_API_URLS.get(name, ""))

def safe_float(x, default=0.0):
    try:
        if pd.isna(x):
            return default
        s = str(x).replace(",", "").replace("%", "").strip()
        if s in ("", "None", "null"):
            return default
        return float(s)
    except Exception:
        return default

def safe_int(x, default=0):
    try:
        if pd.isna(x):
            return default
        return int(float(str(x).replace(",", "").strip()))
    except Exception:
        return default

def normalize_no(x):
    try:
        return str(int(float(str(x).strip())))
    except Exception:
        return str(x).strip()

def load_sample():
    return pd.read_csv(StringIO(SAMPLE_CSV))

def ensure_columns(df):
    if df is None or len(df) == 0:
        df = load_sample()
    df = df.copy()
    n = len(df)
    defaults = {
        "horse_no": list(range(1, n + 1)),
        "horse_name": [f"{i}번마" for i in range(1, n + 1)],
        "gate_no": 0,
        "distance": 1200,
        "odds_win": 0.0,
        "odds_place": 0.0,
        "recent_rank_avg": 5.0,
        "distance_score": 65.0,
        "jockey_score": 65.0,
        "trainer_score": 65.0,
        "weight_change": 0.0,
        "api_score": 0.0,
    }
    for k, v in defaults.items():
        if k not in df.columns:
            df[k] = v
    df["horse_no"] = df["horse_no"].apply(normalize_no)
    return df

def init_state():
    if "race_df" not in st.session_state:
        st.session_state.race_df = None
    if "race_meta" not in st.session_state:
        st.session_state.race_meta = {}
    if "photo_signals" not in st.session_state:
        st.session_state.photo_signals = {}
    if "memory" not in st.session_state:
        st.session_state.memory = {
            "version": "photo-first-publicdata-v1",
            "updated_at": "",
            "races": [],
            "horse_stats": {},
            "signal_stats": {
                "api_strong": {"total": 0, "top3": 0},
                "expert_pick": {"total": 0, "top3": 0},
                "condition_good": {"total": 0, "top3": 0},
                "dark_horse": {"total": 0, "top3": 0},
                "risk_flag": {"total": 0, "top3": 0},
            },
        }

init_state()

# ---------------- 사진/비전 ----------------
def pil_to_data_url(img):
    img = img.convert("RGB")
    max_side = 1600
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1:
        img = img.resize((int(w * scale), int(h * scale)))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

def call_vision(files, prompt, api_key, model):
    if not api_key:
        return {"error": "OPENAI_API_KEY 없음"}
    if OpenAI is None:
        return {"error": "openai 패키지를 불러오지 못했습니다."}
    content = [{"type": "input_text", "text": prompt}]
    for f in files:
        img = Image.open(f)
        content.append({"type": "input_image", "image_url": pil_to_data_url(img)})
    try:
        client = OpenAI(api_key=api_key)
        res = client.responses.create(
            model=model or DEFAULT_MODEL,
            input=[{"role": "user", "content": content}],
            temperature=0.1,
        )
        text = res.output_text.strip()
        text = re.sub(r"^```json\s*|\s*```$", "", text, flags=re.S).strip()
        return json.loads(text)
    except Exception as e:
        return {"error": f"사진 분석 실패: {e}"}

def extract_entries_from_photos(files, api_key, model):
    prompt = """
너는 한국 경마 출전표/예상지/경마잡지 사진에서 오늘 경주 출전마 기본자료를 추출하는 에이전트다.

사진에서 보이는 정보를 최대한 읽어서 JSON만 출력해라. 설명문 금지.

형식:
{
  "race_meta": {
    "racecourse": "서울/부산/제주/불명",
    "race_no": "",
    "race_date": "",
    "distance": "",
    "track_condition": "",
    "weather": ""
  },
  "horses": [
    {
      "horse_no": "1",
      "horse_name": "마명",
      "gate_no": "",
      "distance": "",
      "odds_win": null,
      "odds_place": null,
      "recent_rank_avg": null,
      "distance_score": null,
      "jockey_score": null,
      "trainer_score": null,
      "weight_change": null,
      "jockey": "",
      "trainer": "",
      "evidence": "사진에서 읽은 근거"
    }
  ],
  "warnings": []
}

규칙:
- horse_no와 horse_name은 반드시 최우선으로 추출해라.
- 숫자가 안 보이면 null 또는 빈 문자열로 둬라.
- 배당, 마체중, 전문가표가 같은 사진에 있으면 함께 읽어도 된다.
- 흐리면 추측하지 말고 warnings에 적어라.
"""
    return call_vision(files, prompt, api_key, model)

def analyze_field_photos(files, race_df, api_key, model):
    horses = race_df[["horse_no", "horse_name"]].to_dict("records")
    prompt = f"""
너는 한국 경마 예상지/배당판/마체중표/전문가 추천표를 분석하는 에이전트다.

출전마 목록:
{json.dumps(horses, ensure_ascii=False)}

사진에서 말번호별 현장 보정 정보를 추출해라. JSON만 출력해라.

형식:
{{
  "summary": "전체 요약",
  "horses": {{
    "1": {{
      "horse_name": "",
      "expert_count": 0,
      "condition_score": 0,
      "distance_score_delta": 0,
      "photo_odds_win": null,
      "photo_odds_place": null,
      "weight_change": null,
      "pace_note": "",
      "risk_flags": [],
      "dark_horse": false,
      "photo_score": 0,
      "evidence": "근거"
    }}
  }},
  "global_notes": []
}}

점수 기준:
condition_score -10~10
distance_score_delta -5~5
photo_score -20~20
전문가 지목 수가 보이면 expert_count에 숫자로 넣어라.
복병/주의/구멍 표시가 있으면 dark_horse=true.
마체중 급변, 거리불안, 고중량, 인기과열, 컨디션불안은 risk_flags에 넣어라.
"""
    return call_vision(files, prompt, api_key, model)

# ---------------- 공공데이터 ----------------
def parse_public_response(text):
    text = (text or "").strip()
    if not text:
        return [], "응답 없음"
    try:
        data = json.loads(text)
        lists = []
        def walk(x):
            if isinstance(x, list):
                lists.append(x)
            elif isinstance(x, dict):
                for v in x.values():
                    walk(v)
        walk(data)
        if lists:
            longest = max(lists, key=len)
            return longest, longest[:3]
        return [data], data
    except Exception:
        pass

    items = re.findall(r"<item>(.*?)</item>", text, flags=re.S)
    if items:
        out = []
        for block in items:
            d = {}
            for tag, val in re.findall(r"<([^/][^>]*)>(.*?)</\\1>", block, flags=re.S):
                d[tag.strip()] = re.sub(r"\s+", " ", val).strip()
            out.append(d or {"raw": block[:300]})
        return out, out[:3]

    msg = re.sub(r"<.*?>", " ", text)
    return [], msg[:500]

def api_get(url, service_key, params, timeout=12):
    if not service_key:
        return {"ok": False, "status": "서비스키 없음", "rows": 0, "items": [], "preview": "", "url": url}
    if not url:
        return {"ok": False, "status": "URL 없음", "rows": 0, "items": [], "preview": "", "url": url}

    q = {"serviceKey": service_key, "pageNo": 1, "numOfRows": 30, "_type": "json"}
    for k, v in params.items():
        if v not in ("", None):
            q[k] = v

    try:
        r = requests.get(url, params=q, timeout=timeout)
        items, preview = parse_public_response(r.text)
        if r.status_code >= 500:
            status = f"HTTP {r.status_code}: 서버/요청 변수 오류 가능"
        elif r.status_code >= 400:
            status = f"HTTP {r.status_code}: 요청 오류"
        elif len(items) == 0:
            status = "응답 있음/데이터 없음"
        else:
            status = "성공"
        return {"ok": r.status_code < 400 and len(items) > 0, "status": status, "rows": len(items), "items": items, "preview": preview, "url": r.url}
    except Exception as e:
        return {"ok": False, "status": f"연결 오류: {str(e)[:120]}", "rows": 0, "items": [], "preview": "", "url": url}

def value_from_keys(d, patterns):
    for k, v in d.items():
        lk = str(k).lower()
        if any(p in lk for p in patterns):
            return v
    return None

def rank_from_item(item):
    v = value_from_keys(item, ["rank", "ord", "plc", "chaksun", "착순", "순위"])
    return safe_float(v, None)

def score_horse_record(items):
    if not items:
        return 0.0, None
    ranks = []
    for it in items[:20]:
        if isinstance(it, dict):
            r = rank_from_item(it)
            if r is not None and 0 < r < 30:
                ranks.append(r)
    if ranks:
        avg = sum(ranks) / len(ranks)
        score = max(0, min(20, 22 - avg * 3))
        return score, avg
    return min(8, len(items) * 1.2), None

def enrich_public_data(df, service_key, meet_code="", race_date="", race_no=""):
    df = ensure_columns(df)
    logs = []

    # 경주정보 1회 조회
    race_params = {
        "meet": meet_code,
        "rc_date": race_date,
        "rc_no": race_no,
        "rc_year": race_date[:4] if len(race_date) >= 4 else "",
        "rc_month": race_date[4:6] if len(race_date) >= 6 else "",
    }
    race_res = api_get(api_url("KRA_RACE_INFO_API_URL"), service_key, race_params)
    logs.append({"target": "경주정보", **{k: race_res[k] for k in ["status", "rows", "url", "preview"]}})
    race_bonus = 2.0 if race_res["ok"] else 0.0

    # 말별 1년 전적: 마명 기준 개별 조회
    scores = []
    avgs = []
    for _, row in df.iterrows():
        horse_name = str(row.get("horse_name", "")).strip()
        params_list = [
            {"hr_name": horse_name, "rccrs_cd": meet_code},
            {"hr_name": horse_name},
            {"hr_no": str(row.get("horse_no", "")).strip(), "rccrs_cd": meet_code},
        ]
        best = None
        for params in params_list:
            res = api_get(api_url("KRA_HORSE_RECORD_API_URL"), service_key, params)
            if best is None or res["rows"] > best["rows"]:
                best = res
            if res["ok"]:
                break
        score, avg = score_horse_record(best["items"])
        scores.append(score + race_bonus)
        avgs.append(avg)
        logs.append({"target": f"{row['horse_no']}번 {horse_name}", **{k: best[k] for k in ["status", "rows", "url", "preview"]}})

    df["api_score"] = scores
    for i, avg in enumerate(avgs):
        if avg is not None and safe_float(df.iloc[i].get("recent_rank_avg"), 5.0) == 5.0:
            df.loc[df.index[i], "recent_rank_avg"] = avg

    return df, logs

# ---------------- 점수 ----------------
def memory_bonus(no, memory):
    hs = memory.get("horse_stats", {}).get(str(no), {})
    total = hs.get("total", 0)
    if total < 2:
        return 0.0
    return (hs.get("top3", 0) / total - 0.33) * 10

def compute_scores(df, signals, memory):
    df = ensure_columns(df)
    rows = []
    for _, r in df.iterrows():
        no = str(r["horse_no"])
        sig = signals.get(no, {})

        odds_win = safe_float(sig.get("photo_odds_win"), safe_float(r["odds_win"], 0))
        odds_place = safe_float(sig.get("photo_odds_place"), safe_float(r["odds_place"], 0))

        # 사진에서 마체중 변화 읽으면 덮어쓰기
        weight_change = safe_float(sig.get("weight_change"), safe_float(r["weight_change"], 0))

        recent = max(0, 20 - safe_float(r["recent_rank_avg"], 5) * 2.2)
        dist = safe_float(r["distance_score"], 65) / 5
        jockey = safe_float(r["jockey_score"], 65) / 7
        trainer = safe_float(r["trainer_score"], 65) / 8
        gate = max(0, 6 - abs(safe_int(r["gate_no"], 6) - 3) * 0.6)
        weight_penalty = -4 if abs(weight_change) >= 14 else (-2 if abs(weight_change) >= 10 else 0)
        api_score = safe_float(r["api_score"], 0)

        photo_score = safe_float(sig.get("photo_score", 0), 0)
        expert_bonus = min(8, safe_int(sig.get("expert_count", 0)) * 2)
        cond_bonus = safe_float(sig.get("condition_score", 0), 0)
        dist_delta = safe_float(sig.get("distance_score_delta", 0), 0)
        risk_penalty = -3 * len(sig.get("risk_flags", []) or [])
        dark_bonus = 3 if sig.get("dark_horse") else 0
        mem_bonus = memory_bonus(no, memory)

        raw = recent + dist + jockey + trainer + gate + weight_penalty + api_score + photo_score + expert_bonus + cond_bonus + dist_delta + risk_penalty + dark_bonus + mem_bonus
        d = r.to_dict()
        d.update({
            "odds_win": odds_win,
            "odds_place": odds_place,
            "weight_change": weight_change,
            "photo_score": photo_score,
            "expert_bonus": expert_bonus,
            "condition_bonus": cond_bonus,
            "risk_penalty": risk_penalty,
            "memory_bonus": mem_bonus,
            "raw_score": raw,
        })
        rows.append(d)

    out = pd.DataFrame(rows)
    vals = out["raw_score"].astype(float)
    expv = (vals - vals.max()).apply(lambda x: math.exp(x / 10))
    out["win_prob"] = expv / expv.sum()
    out["place_prob"] = (out["win_prob"] * 2.2 + 0.10).clip(0.05, 0.85)
    out["ev_win"] = out.apply(lambda x: safe_float(x["odds_win"], 0) * x["win_prob"] - 1 if safe_float(x["odds_win"], 0) > 0 else None, axis=1)
    out["ev_place"] = out.apply(lambda x: safe_float(x["odds_place"], 0) * x["place_prob"] - 1 if safe_float(x["odds_place"], 0) > 0 else None, axis=1)
    return out.sort_values("raw_score", ascending=False)

def build_bets(scored, bankroll, min_ev):
    unit = 1000
    spent = 0
    bets = []
    top = scored.iloc[0]
    axis = top["horse_no"]
    for p in scored.iloc[1:4]["horse_no"].tolist():
        if spent + unit <= bankroll:
            bets.append(f"쌍승 {axis}→{p} {unit}원")
            spent += unit
    for _, r in scored.sort_values("ev_win", ascending=False, na_position="last").head(2).iterrows():
        if spent + unit <= bankroll and r.get("ev_win") is not None and safe_float(r.get("ev_win"), -9) >= min_ev:
            bets.append(f"단승 {r['horse_no']} {unit}원")
            spent += unit
    for _, r in scored.sort_values("ev_place", ascending=False, na_position="last").head(2).iterrows():
        if spent + unit <= bankroll and r.get("ev_place") is not None and safe_float(r.get("ev_place"), -9) >= min_ev:
            bets.append(f"연승 {r['horse_no']} {unit}원")
            spent += unit
    return bets or ["기대값이 약합니다. 무리한 베팅보다 관망 권장"]

def update_memory(memory, scored, order, signals, note):
    top3 = set(order[:3])
    memory.setdefault("races", []).append({
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "finish_order": order,
        "note": note,
        "signals": signals,
        "scores": scored[["horse_no", "horse_name", "raw_score", "win_prob", "place_prob", "api_score"]].to_dict("records"),
    })
    memory["updated_at"] = datetime.now().isoformat(timespec="seconds")

    hs = memory.setdefault("horse_stats", {})
    for _, r in scored.iterrows():
        no = str(r["horse_no"])
        hs.setdefault(no, {"total": 0, "top1": 0, "top3": 0})
        hs[no]["total"] += 1
        if order and no == order[0]:
            hs[no]["top1"] += 1
        if no in top3:
            hs[no]["top3"] += 1

    ss = memory.setdefault("signal_stats", {})
    for no, sig in signals.items():
        hit = int(no in top3)
        if sig.get("expert_count", 0) > 0:
            ss.setdefault("expert_pick", {"total": 0, "top3": 0})
            ss["expert_pick"]["total"] += 1
            ss["expert_pick"]["top3"] += hit
        if safe_float(sig.get("condition_score", 0), 0) > 3:
            ss.setdefault("condition_good", {"total": 0, "top3": 0})
            ss["condition_good"]["total"] += 1
            ss["condition_good"]["top3"] += hit
        if sig.get("dark_horse"):
            ss.setdefault("dark_horse", {"total": 0, "top3": 0})
            ss["dark_horse"]["total"] += 1
            ss["dark_horse"]["top3"] += hit
        if sig.get("risk_flags"):
            ss.setdefault("risk_flag", {"total": 0, "top3": 0})
            ss["risk_flag"]["total"] += 1
            ss["risk_flag"]["top3"] += hit
    return memory

# ---------------- UI ----------------
st.title(APP_TITLE)
st.caption("출전마 기본명단도 사진에서 읽습니다. CSV는 보조일 뿐입니다.")

openai_key = sec("OPENAI_API_KEY", "")
data_key = sec("DATA_GO_KR_SERVICE_KEY", "")
vision_model = sec("OPENAI_VISION_MODEL", DEFAULT_MODEL)

with st.sidebar:
    st.header("상태")
    st.write("OpenAI 키:", "✅ 있음" if openai_key else "❌ 없음")
    st.write("공공데이터 키:", "✅ 있음" if data_key else "❌ 없음")
    bankroll = st.number_input("경주당 예산", 1000, 100000, 5000, step=1000)
    min_ev = st.slider("최소 기대값", -0.20, 0.50, 0.05, 0.01)

tabs = st.tabs(["1. 사진에서 출전마 추출", "2. 공공데이터 통계 보강", "3. 현장 사진 보정", "4. 최종 분석", "5. 결과 기록", "6. 저장/불러오기"])

with tabs[0]:
    st.header("사진에서 출전마 명단 만들기")
    st.info("출전표/예상지 첫 장을 올리면 마번·마명·게이트·거리·배당 등을 추출합니다.")
    entry_photos = st.file_uploader("출전표/예상지 사진 업로드", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True, key="entry_photos")
    if entry_photos:
        st.image(entry_photos, width=220)

    if st.button("사진에서 출전마 추출", type="primary"):
        if not entry_photos:
            st.error("사진을 먼저 올리세요.")
        else:
            result = extract_entries_from_photos(entry_photos, openai_key, vision_model)
            if "error" in result:
                st.error(result["error"])
            else:
                horses = result.get("horses", [])
                meta = result.get("race_meta", {})
                if not horses:
                    st.error("말 목록을 읽지 못했습니다. 사진을 더 크게/선명하게 찍어주세요.")
                else:
                    df = pd.DataFrame(horses)
                    rename_map = {}
                    # normalize expected columns
                    for col in ["horse_no","horse_name","gate_no","distance","odds_win","odds_place","recent_rank_avg","distance_score","jockey_score","trainer_score","weight_change"]:
                        if col not in df.columns:
                            df[col] = None
                    st.session_state.race_df = ensure_columns(df)
                    st.session_state.race_meta = meta
                    st.success("사진에서 출전마 명단을 만들었습니다.")
                    st.json(result)

    st.subheader("현재 출전마 표")
    if st.session_state.race_df is None:
        st.warning("아직 사진에서 출전마를 추출하지 않았습니다. 테스트용 샘플을 쓰려면 아래 버튼을 누르세요.")
        if st.button("샘플 출전마 사용"):
            st.session_state.race_df = load_sample()
    else:
        edited = st.data_editor(ensure_columns(st.session_state.race_df), use_container_width=True, num_rows="dynamic")
        st.session_state.race_df = ensure_columns(edited)

    uploaded_csv = st.file_uploader("CSV로 보조 업로드", type=["csv"])
    if uploaded_csv:
        try:
            st.session_state.race_df = ensure_columns(pd.read_csv(uploaded_csv))
            st.success("CSV를 불러왔습니다.")
        except Exception as e:
            st.error(f"CSV 읽기 실패: {e}")

with tabs[1]:
    st.header("공공데이터 통계 보강")
    st.caption("사진에서 추출한 마명으로 말별 1년 전적 API를 개별 조회해 api_score를 보강합니다.")
    if st.session_state.race_df is None:
        st.error("먼저 1번 탭에서 출전마 명단을 만들어야 합니다.")
    else:
        meta = st.session_state.get("race_meta", {})
        c1, c2, c3 = st.columns(3)
        meet = c1.text_input("경마장 코드/rccrs_cd/meet", value=str(meta.get("racecourse","")))
        race_date = c2.text_input("경주일자 예: 20260527", value=str(meta.get("race_date","")))
        race_no = c3.text_input("경주번호", value=str(meta.get("race_no","")))
        if st.button("공공데이터로 말별 통계 보강", type="primary"):
            df, logs = enrich_public_data(st.session_state.race_df, data_key, meet, race_date, race_no)
            st.session_state.race_df = df
            st.session_state.api_logs = logs
            st.success("공공데이터 보강 시도를 완료했습니다. 실패 항목이 있어도 최종 분석은 계속됩니다.")

        st.dataframe(ensure_columns(st.session_state.race_df), use_container_width=True)
        if "api_logs" in st.session_state:
            log_df = pd.DataFrame([{k:v for k,v in x.items() if k!="preview"} for x in st.session_state.api_logs])
            st.dataframe(log_df, use_container_width=True)
            with st.expander("공공데이터 원본 미리보기"):
                for x in st.session_state.api_logs:
                    st.write("###", x["target"])
                    st.write(x["preview"])

with tabs[2]:
    st.header("현장 사진 보정")
    st.caption("배당판, 마체중표, 전문가 추천표, 조교평 사진을 올려 오늘 정보를 보정합니다.")
    if st.session_state.race_df is None:
        st.error("먼저 1번 탭에서 출전마 명단을 만들어야 합니다.")
    else:
        field_photos = st.file_uploader("현장 사진 업로드", type=["jpg","jpeg","png","webp"], accept_multiple_files=True, key="field_photos")
        if field_photos:
            st.image(field_photos, width=220)
        if st.button("현장 사진 분석", type="primary"):
            if not field_photos:
                st.error("사진을 먼저 올리세요.")
            else:
                result = analyze_field_photos(field_photos, ensure_columns(st.session_state.race_df), openai_key, vision_model)
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.session_state.photo_signals = result.get("horses", {})
                    st.success("현장 사진 분석 완료")
                    st.json(result)

        st.subheader("수동 보정")
        # simple manual inputs
        manual = {}
        df = ensure_columns(st.session_state.race_df)
        for _, row in df.iterrows():
            no = str(row["horse_no"])
            with st.expander(f"{no}번 {row['horse_name']} 수동 보정"):
                a,b,c = st.columns(3)
                expert = a.number_input(f"{no} 전문가 지목 수", 0, 10, 0, key=f"ex_{no}")
                cond = b.slider(f"{no} 컨디션", -10, 10, 0, key=f"co_{no}")
                photo_score = c.slider(f"{no} 사진점수", -20, 20, 0, key=f"ph_{no}")
                ow = st.number_input(f"{no} 단승배당", 0.0, 999.0, 0.0, key=f"ow_{no}")
                op = st.number_input(f"{no} 연승배당", 0.0, 999.0, 0.0, key=f"op_{no}")
                risk = st.text_input(f"{no} 위험신호", "", key=f"ri_{no}")
                dark = st.checkbox(f"{no} 복병", key=f"da_{no}")
                if expert or cond or photo_score or ow or op or risk or dark:
                    manual[no] = {
                        "horse_name": row["horse_name"],
                        "expert_count": int(expert),
                        "condition_score": float(cond),
                        "distance_score_delta": 0,
                        "photo_odds_win": ow if ow else None,
                        "photo_odds_place": op if op else None,
                        "risk_flags": [x.strip() for x in risk.split(",") if x.strip()],
                        "dark_horse": bool(dark),
                        "photo_score": float(photo_score),
                        "evidence": "수동 입력",
                    }
        if st.button("수동 보정 적용"):
            merged = dict(st.session_state.photo_signals)
            merged.update(manual)
            st.session_state.photo_signals = merged
            st.success("수동 보정 적용 완료")

        if st.session_state.photo_signals:
            st.json(st.session_state.photo_signals)

with tabs[3]:
    st.header("최종 분석")
    if st.session_state.race_df is None:
        st.error("먼저 1번 탭에서 출전마 명단을 만들어야 합니다.")
    else:
        scored = compute_scores(st.session_state.race_df, st.session_state.photo_signals, st.session_state.memory)
        st.session_state.scored = scored
        cols = ["horse_no","horse_name","raw_score","win_prob","place_prob","odds_win","odds_place","ev_win","ev_place","api_score","photo_score","expert_bonus","risk_penalty","memory_bonus"]
        st.dataframe(scored[cols], use_container_width=True)
        st.subheader("추천 배팅안")
        for b in build_bets(scored, int(bankroll), float(min_ev)):
            st.write(f"- {b}")
        st.warning("경마 결과는 불확실합니다. 이 앱은 분석 보조용이며 자동 구매 기능은 없습니다.")

with tabs[4]:
    st.header("결과 기록")
    finish = st.text_input("도착순서 예: 4-2-3-7-11")
    note = st.text_area("메모")
    if st.button("결과 저장/통계 업데이트", type="primary"):
        order = [x.strip() for x in re.split(r"[-,/\s]+", finish) if x.strip()]
        if not order:
            st.error("도착순서를 입력하세요.")
        else:
            scored = st.session_state.get("scored", compute_scores(st.session_state.race_df, st.session_state.photo_signals, st.session_state.memory))
            st.session_state.memory = update_memory(st.session_state.memory, scored, order, st.session_state.photo_signals, note)
            st.success("통계가 업데이트되었습니다.")

    st.subheader("누적 통계")
    mem = st.session_state.memory
    st.write("저장 경주 수:", len(mem.get("races", [])))
    if mem.get("horse_stats"):
        rows = []
        for no, s in mem["horse_stats"].items():
            total = max(1, s.get("total", 0))
            rows.append({"horse_no": no, "total": s.get("total", 0), "top1": s.get("top1", 0), "top3": s.get("top3", 0), "top3_rate": s.get("top3", 0)/total})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

with tabs[5]:
    st.header("통계 저장/불러오기")
    up = st.file_uploader("이전 통계 JSON 업로드", type=["json"])
    if up:
        try:
            st.session_state.memory = json.loads(up.getvalue().decode("utf-8"))
            st.success("통계를 불러왔습니다.")
        except Exception as e:
            st.error(f"불러오기 실패: {e}")

    js = json.dumps(st.session_state.memory, ensure_ascii=False, indent=2)
    st.download_button(
        "통계 JSON 다운로드",
        data=js.encode("utf-8"),
        file_name=f"race_memory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
    )
    with st.expander("현재 통계 JSON"):
        st.code(js, language="json")
