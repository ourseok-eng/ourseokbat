import base64, json, math, re
from datetime import datetime
from io import BytesIO, StringIO
import pandas as pd
import requests
import streamlit as st
from PIL import Image

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

st.set_page_config(page_title="경마 PDF출전표 + 공공데이터", layout="wide")

DEFAULT_MODEL = "gpt-4.1-mini"
API_HORSE_RECORD = "https://apis.data.go.kr/B551015/API145"
API_RACE_INFO = "https://apis.data.go.kr/B551015/API187"
SAMPLE_CSV = """horse_no,horse_name,gate_no,distance,odds_win,odds_place,recent_rank_avg,distance_score,jockey_score,trainer_score,weight_change,api_score
1,샘플1,1,1200,4.5,1.7,3.2,77,70,68,-1,0
2,샘플2,3,1200,2.6,1.3,2.1,86,82,80,2,0
3,샘플3,7,1200,8.8,2.4,4.6,73,71,70,-3,0
4,샘플4,4,1200,5.5,1.9,3.1,81,77,75,1,0
"""

def sec(k, default=""):
    try:
        return str(st.secrets.get(k, default) or "")
    except Exception:
        return default

def sf(x, default=0.0):
    try:
        if pd.isna(x): return default
        s = str(x).replace(",", "").replace("%", "").strip()
        if s in ("", "None", "null"): return default
        return float(s)
    except Exception:
        return default

def si(x, default=0):
    try: return int(float(str(x).replace(",", "").strip()))
    except Exception: return default

def norm_no(x):
    try: return str(int(float(str(x).strip())))
    except Exception: return str(x).strip()

def sample_df():
    return pd.read_csv(StringIO(SAMPLE_CSV))

def ensure_df(df):
    if df is None or len(df) == 0:
        df = sample_df()
    df = df.copy()
    n = len(df)
    defaults = {
        "horse_no": list(range(1, n+1)),
        "horse_name": [f"{i}번마" for i in range(1, n+1)],
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
    df["horse_no"] = df["horse_no"].apply(norm_no)
    return df

if "race_df" not in st.session_state: st.session_state.race_df = None
if "race_meta" not in st.session_state: st.session_state.race_meta = {}
if "signals" not in st.session_state: st.session_state.signals = {}
if "memory" not in st.session_state: st.session_state.memory = {"races": [], "horse_stats": {}, "updated_at": ""}

# ---------- 선명 업로드 처리 ----------
def image_to_data_url(img, max_side=2600, quality=95):
    img = img.convert("RGB")
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1:
        img = img.resize((int(w*scale), int(h*scale)), Image.Resampling.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

def pdf_to_text_images(file, max_pages=5, zoom=3.2):
    if fitz is None:
        return "", []
    data = file.getvalue()
    doc = fitz.open(stream=data, filetype="pdf")
    texts, images = [], []
    for i in range(min(max_pages, len(doc))):
        page = doc[i]
        try:
            texts.append(page.get_text("text") or "")
        except Exception:
            pass
        try:
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            img = Image.open(BytesIO(pix.tobytes("png"))).convert("RGB")
            images.append(img)
        except Exception:
            pass
    return "\n".join(texts).strip(), images

def files_to_parts_and_previews(files):
    parts, previews, text_blocks = [], [], []
    for f in files:
        name = f.name.lower()
        if name.endswith(".pdf"):
            text, imgs = pdf_to_text_images(f)
            if text:
                text_blocks.append(f"[PDF 텍스트: {f.name}]\n{text[:12000]}")
            for img in imgs:
                previews.append(img)
                parts.append({"type": "input_image", "image_url": image_to_data_url(img)})
        else:
            img = Image.open(f).convert("RGB")
            previews.append(img)
            parts.append({"type": "input_image", "image_url": image_to_data_url(img)})
    if text_blocks:
        parts.insert(0, {"type": "input_text", "text": "\n\n".join(text_blocks)})
    return parts, previews

def vision_json(files, prompt):
    key = sec("OPENAI_API_KEY", "")
    model = sec("OPENAI_VISION_MODEL", DEFAULT_MODEL)
    if not key:
        return {"error": "OPENAI_API_KEY 없음"}
    if OpenAI is None:
        return {"error": "openai 패키지 오류"}
    parts, _ = files_to_parts_and_previews(files)
    try:
        client = OpenAI(api_key=key)
        res = client.responses.create(
            model=model,
            input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}] + parts}],
            temperature=0.1,
        )
        text = res.output_text.strip()
        text = re.sub(r"^```json\s*|\s*```$", "", text, flags=re.S).strip()
        return json.loads(text)
    except Exception as e:
        return {"error": f"분석 실패: {e}"}

# ---------- 공공데이터 ----------
def parse_items(text):
    text = (text or "").strip()
    if not text: return [], "응답 없음"
    try:
        data = json.loads(text)
        lists = []
        def walk(x):
            if isinstance(x, list): lists.append(x)
            elif isinstance(x, dict):
                for v in x.values(): walk(v)
        walk(data)
        if lists:
            longest = max(lists, key=len)
            return longest, longest[:3]
        return [data], data
    except Exception:
        pass
    items = re.findall(r"<item>(.*?)</item>", text, flags=re.S)
    out = []
    for block in items:
        d = {}
        for tag, val in re.findall(r"<([^/][^>]*)>(.*?)</\1>", block, flags=re.S):
            d[tag] = re.sub(r"\s+", " ", val).strip()
        out.append(d or {"raw": block[:300]})
    if out: return out, out[:3]
    return [], re.sub(r"<.*?>", " ", text)[:500]

def api_get(url, params):
    key = sec("DATA_GO_KR_SERVICE_KEY", "")
    if not key: return {"ok": False, "status": "서비스키 없음", "rows": 0, "items": [], "preview": "", "url": url}
    q = {"serviceKey": key, "pageNo": 1, "numOfRows": 30, "_type": "json"}
    q.update({k: v for k, v in params.items() if v not in (None, "")})
    try:
        r = requests.get(url, params=q, timeout=12)
        items, preview = parse_items(r.text)
        if r.status_code >= 500: status = f"HTTP {r.status_code}: 서버/요청변수 오류 가능"
        elif r.status_code >= 400: status = f"HTTP {r.status_code}: 요청 오류"
        elif not items: status = "응답 있음/데이터 없음"
        else: status = "성공"
        return {"ok": r.status_code < 400 and bool(items), "status": status, "rows": len(items), "items": items, "preview": preview, "url": r.url}
    except Exception as e:
        return {"ok": False, "status": f"연결 오류 {str(e)[:80]}", "rows": 0, "items": [], "preview": "", "url": url}

def rank_from_item(item):
    for k, v in item.items():
        lk = str(k).lower()
        if any(p in lk for p in ["rank", "ord", "plc", "chaksun", "착순", "순위"]):
            val = sf(v, None)
            if val and 0 < val < 30: return val
    return None

def score_record(items):
    ranks = [rank_from_item(x) for x in items[:20] if isinstance(x, dict)]
    ranks = [x for x in ranks if x]
    if ranks:
        avg = sum(ranks) / len(ranks)
        return max(0, min(20, 22 - avg*3))
    return min(8, len(items)*1.2) if items else 0

def enrich_public(df, meet="", race_date="", race_no=""):
    df = ensure_df(df)
    logs = []
    race_res = api_get(API_RACE_INFO, {"meet": meet, "rc_date": race_date, "rc_no": race_no, "rc_year": race_date[:4] if len(race_date)>=4 else "", "rc_month": race_date[4:6] if len(race_date)>=6 else ""})
    logs.append({"target": "경주정보", "status": race_res["status"], "rows": race_res["rows"], "url": race_res["url"], "preview": race_res["preview"]})
    race_bonus = 2 if race_res["ok"] else 0
    scores = []
    for _, row in df.iterrows():
        name = str(row["horse_name"]).strip()
        best = None
        for p in [{"hr_name": name, "rccrs_cd": meet}, {"hr_name": name}, {"hr_no": str(row["horse_no"]), "rccrs_cd": meet}]:
            res = api_get(API_HORSE_RECORD, p)
            if best is None or res["rows"] > best["rows"]: best = res
            if res["ok"]: break
        scores.append(score_record(best["items"]) + race_bonus)
        logs.append({"target": f"{row['horse_no']}번 {name}", "status": best["status"], "rows": best["rows"], "url": best["url"], "preview": best["preview"]})
    df["api_score"] = scores
    return df, logs

# ---------- 점수 ----------
def memory_bonus(no):
    hs = st.session_state.memory.get("horse_stats", {}).get(str(no), {})
    total = hs.get("total", 0)
    return 0 if total < 2 else (hs.get("top3", 0)/total - 0.33) * 10

def compute_scores(df):
    df = ensure_df(df)
    signals = st.session_state.signals
    rows = []
    for _, r in df.iterrows():
        no = str(r["horse_no"])
        sig = signals.get(no, {})
        odds_win = sf(sig.get("photo_odds_win"), sf(r["odds_win"], 0))
        odds_place = sf(sig.get("photo_odds_place"), sf(r["odds_place"], 0))
        weight = sf(sig.get("weight_change"), sf(r["weight_change"], 0))
        raw = max(0, 20 - sf(r["recent_rank_avg"], 5)*2.2)
        raw += sf(r["distance_score"],65)/5 + sf(r["jockey_score"],65)/7 + sf(r["trainer_score"],65)/8
        raw += max(0, 6 - abs(si(r["gate_no"],6)-3)*0.6) + sf(r["api_score"],0)
        raw += sf(sig.get("photo_score",0),0) + min(8, si(sig.get("expert_count",0))*2) + sf(sig.get("condition_score",0),0) + sf(sig.get("distance_score_delta",0),0)
        raw += 3 if sig.get("dark_horse") else 0
        raw -= 3 * len(sig.get("risk_flags", []) or [])
        raw += -4 if abs(weight) >= 14 else (-2 if abs(weight) >= 10 else 0)
        raw += memory_bonus(no)
        d = r.to_dict()
        d.update({"odds_win": odds_win, "odds_place": odds_place, "weight_change": weight, "raw_score": raw, "photo_score": sf(sig.get("photo_score",0),0), "expert_bonus": min(8, si(sig.get("expert_count",0))*2), "risk_penalty": -3*len(sig.get("risk_flags",[]) or []), "memory_bonus": memory_bonus(no)})
        rows.append(d)
    out = pd.DataFrame(rows)
    vals = out["raw_score"].astype(float)
    expv = (vals - vals.max()).apply(lambda x: math.exp(x/10))
    out["win_prob"] = expv / expv.sum()
    out["place_prob"] = (out["win_prob"]*2.2 + 0.1).clip(0.05, 0.85)
    out["ev_win"] = out.apply(lambda x: sf(x["odds_win"])*x["win_prob"]-1 if sf(x["odds_win"])>0 else None, axis=1)
    out["ev_place"] = out.apply(lambda x: sf(x["odds_place"])*x["place_prob"]-1 if sf(x["odds_place"])>0 else None, axis=1)
    return out.sort_values("raw_score", ascending=False)

def build_bets(scored, bankroll, min_ev):
    unit, spent, recs = 1000, 0, []
    axis = scored.iloc[0]["horse_no"]
    for p in scored.iloc[1:4]["horse_no"].tolist():
        if spent + unit <= bankroll:
            recs.append(f"쌍승 {axis}→{p} {unit}원"); spent += unit
    for _, r in scored.sort_values("ev_win", ascending=False, na_position="last").head(2).iterrows():
        if spent + unit <= bankroll and r.get("ev_win") is not None and sf(r.get("ev_win"), -9) >= min_ev:
            recs.append(f"단승 {r['horse_no']} {unit}원"); spent += unit
    for _, r in scored.sort_values("ev_place", ascending=False, na_position="last").head(2).iterrows():
        if spent + unit <= bankroll and r.get("ev_place") is not None and sf(r.get("ev_place"), -9) >= min_ev:
            recs.append(f"연승 {r['horse_no']} {unit}원"); spent += unit
    return recs or ["기대값이 약합니다. 관망 권장"]

# ---------- UI ----------
st.title("🏇 경마 PDF출전표 + 공공데이터 + 선명사진 분석")
st.caption("PDF 출전표를 올리면 출전마를 추출하고, 공공데이터와 현장 사진/PDF를 합쳐 최종 배팅안을 만듭니다.")
with st.sidebar:
    st.write("OpenAI 키:", "✅ 있음" if sec("OPENAI_API_KEY") else "❌ 없음")
    st.write("공공데이터 키:", "✅ 있음" if sec("DATA_GO_KR_SERVICE_KEY") else "❌ 없음")
    bankroll = st.number_input("경주당 예산", 1000, 100000, 5000, step=1000)
    min_ev = st.slider("최소 기대값", -0.2, 0.5, 0.05, 0.01)

tabs = st.tabs(["1. PDF 출전마 추출", "2. 공공데이터 보강", "3. 현장자료 보정", "4. 최종 분석", "5. 결과 기록", "6. 저장"])

with tabs[0]:
    st.header("PDF/사진에서 출전마 명단 만들기")
    files = st.file_uploader("출전표 PDF/사진 업로드", type=["pdf","jpg","jpeg","png","webp"], accept_multiple_files=True, key="entry")
    if files:
        _, previews = files_to_parts_and_previews(files)
        if previews:
            st.subheader("선명 미리보기")
            for im in previews[:3]:
                st.image(im, use_container_width=True)
    if st.button("PDF/사진에서 출전마 추출", type="primary"):
        if not files:
            st.error("파일을 먼저 올리세요.")
        else:
            prompt = '''한국 경마 출전표 PDF/사진에서 출전마를 추출해 JSON만 출력. 형식: {"race_meta":{"racecourse":"","race_no":"","race_date":"","distance":""},"horses":[{"horse_no":"1","horse_name":"","gate_no":"","distance":"","odds_win":null,"odds_place":null,"recent_rank_avg":null,"distance_score":null,"jockey_score":null,"trainer_score":null,"weight_change":null,"jockey":"","trainer":"","evidence":""}],"warnings":[]} 마번과 마명을 최우선. 불명확하면 warnings.'''
            res = vision_json(files, prompt)
            if "error" in res: st.error(res["error"])
            else:
                horses = res.get("horses", [])
                if not horses:
                    st.error("출전마를 읽지 못했습니다. PDF/사진을 더 선명하게 올려주세요.")
                else:
                    st.session_state.race_df = ensure_df(pd.DataFrame(horses))
                    st.session_state.race_meta = res.get("race_meta", {})
                    st.success("출전마 명단 생성 완료")
                    st.json(res)
    if st.session_state.race_df is None:
        if st.button("샘플 사용"):
            st.session_state.race_df = sample_df()
    else:
        st.session_state.race_df = ensure_df(st.data_editor(ensure_df(st.session_state.race_df), use_container_width=True, num_rows="dynamic"))

with tabs[1]:
    st.header("공공데이터 보강")
    if st.session_state.race_df is None:
        st.error("먼저 1번 탭에서 출전마를 추출하세요.")
    else:
        meta = st.session_state.race_meta
        c1,c2,c3 = st.columns(3)
        meet = c1.text_input("경마장 코드/meet/rccrs_cd", value=str(meta.get("racecourse", "")))
        race_date = c2.text_input("경주일자 예: 20260527", value=str(meta.get("race_date", "")))
        race_no = c3.text_input("경주번호", value=str(meta.get("race_no", "")))
        if st.button("공공데이터로 말별 통계 보강", type="primary"):
            df, logs = enrich_public(st.session_state.race_df, meet, race_date, race_no)
            st.session_state.race_df = df
            st.session_state.api_logs = logs
            st.success("보강 시도 완료")
        st.dataframe(ensure_df(st.session_state.race_df), use_container_width=True)
        if "api_logs" in st.session_state:
            st.dataframe(pd.DataFrame([{k:v for k,v in x.items() if k!="preview"} for x in st.session_state.api_logs]), use_container_width=True)
            with st.expander("원본 미리보기"):
                for x in st.session_state.api_logs:
                    st.write("### " + x["target"])
                    st.write(x["preview"])

with tabs[2]:
    st.header("현장자료 PDF/사진 보정")
    if st.session_state.race_df is None:
        st.error("먼저 출전마를 추출하세요.")
    else:
        f = st.file_uploader("배당판/마체중표/전문가표 PDF 또는 사진", type=["pdf","jpg","jpeg","png","webp"], accept_multiple_files=True, key="field")
        if f:
            _, previews = files_to_parts_and_previews(f)
            for im in previews[:3]: st.image(im, use_container_width=True)
        if st.button("현장자료 분석", type="primary"):
            if not f:
                st.error("파일을 먼저 올리세요.")
            else:
                horses = ensure_df(st.session_state.race_df)[["horse_no","horse_name"]].to_dict("records")
                prompt = f'''출전마 목록: {json.dumps(horses, ensure_ascii=False)}\n예상지/배당판/마체중표를 분석해 JSON만 출력. 형식: {{"summary":"","horses":{{"1":{{"horse_name":"","expert_count":0,"condition_score":0,"distance_score_delta":0,"photo_odds_win":null,"photo_odds_place":null,"weight_change":null,"pace_note":"","risk_flags":[],"dark_horse":false,"photo_score":0,"evidence":""}}}},"global_notes":[]}}'''
                res = vision_json(f, prompt)
                if "error" in res: st.error(res["error"])
                else:
                    st.session_state.signals = res.get("horses", {})
                    st.success("현장자료 분석 완료")
                    st.json(res)
        if st.session_state.signals: st.json(st.session_state.signals)

with tabs[3]:
    st.header("최종 분석")
    if st.session_state.race_df is None:
        st.error("먼저 출전마를 추출하세요.")
    else:
        scored = compute_scores(st.session_state.race_df)
        st.session_state.scored = scored
        cols = ["horse_no","horse_name","raw_score","win_prob","place_prob","odds_win","odds_place","ev_win","ev_place","api_score","photo_score","expert_bonus","risk_penalty","memory_bonus"]
        st.dataframe(scored[cols], use_container_width=True)
        st.subheader("추천 배팅안")
        for b in build_bets(scored, int(bankroll), float(min_ev)):
            st.write("- " + b)
        st.warning("분석 보조 도구입니다. 자동 구매 기능은 없습니다.")

with tabs[4]:
    st.header("결과 기록")
    finish = st.text_input("도착순서 예: 4-2-3-7-11")
    note = st.text_area("메모")
    if st.button("결과 저장", type="primary"):
        order = [x.strip() for x in re.split(r"[-,/\s]+", finish) if x.strip()]
        if not order:
            st.error("도착순서를 입력하세요.")
        else:
            scored = st.session_state.get("scored", compute_scores(st.session_state.race_df))
            top3 = set(order[:3])
            mem = st.session_state.memory
            mem.setdefault("races", []).append({"saved_at": datetime.now().isoformat(timespec="seconds"), "finish_order": order, "note": note, "signals": st.session_state.signals})
            hs = mem.setdefault("horse_stats", {})
            for _, r in scored.iterrows():
                no = str(r["horse_no"])
                hs.setdefault(no, {"total":0,"top1":0,"top3":0})
                hs[no]["total"] += 1
                hs[no]["top1"] += int(order and no == order[0])
                hs[no]["top3"] += int(no in top3)
            st.session_state.memory = mem
            st.success("저장 완료")
    st.json(st.session_state.memory.get("horse_stats", {}))

with tabs[5]:
    st.header("저장/불러오기")
    up = st.file_uploader("통계 JSON 업로드", type=["json"])
    if up:
        try:
            st.session_state.memory = json.loads(up.getvalue().decode("utf-8"))
            st.success("불러오기 완료")
        except Exception as e: st.error(e)
    js = json.dumps(st.session_state.memory, ensure_ascii=False, indent=2)
    st.download_button("통계 JSON 다운로드", data=js.encode("utf-8"), file_name=f"race_memory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", mime="application/json")
