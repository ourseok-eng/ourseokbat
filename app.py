import base64
import io
import json
import math
import os
import re
from copy import deepcopy
from datetime import datetime, date
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image

APP_TITLE = "경마 현장 분석 에이전트 FINAL"
MEMORY_VERSION = 1
DEFAULT_STATS_PATH = "race_memory.json"

SAMPLE_COLUMNS = [
    "horse_no", "horse_name", "gate_no", "distance", "odds_win", "odds_place",
    "recent_rank_avg", "distance_score", "jockey_score", "trainer_score",
    "weight_change", "style",
]

# -----------------------------
# Safe config helpers
# -----------------------------
def get_secret(name: str, default: str = "") -> str:
    try:
        val = st.secrets.get(name, default)
    except Exception:
        val = os.environ.get(name, default)
    if val is None:
        return default
    return str(val)


def mask_secret(text: str, keep: int = 4) -> str:
    if not text:
        return ""
    if len(text) <= keep * 2:
        return "***"
    return text[:keep] + "..." + text[-keep:]


def now_iso() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


# -----------------------------
# Memory / persistence
# -----------------------------
def empty_memory() -> Dict[str, Any]:
    return {
        "version": MEMORY_VERSION,
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "photo_observations": [],
        "race_results": [],
        "notes": [],
    }


def normalize_memory(obj: Any) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        return empty_memory()
    base = empty_memory()
    for k in base:
        if k in obj:
            base[k] = obj[k]
    if not isinstance(base.get("photo_observations"), list):
        base["photo_observations"] = []
    if not isinstance(base.get("race_results"), list):
        base["race_results"] = []
    if not isinstance(base.get("notes"), list):
        base["notes"] = []
    base["updated_at"] = obj.get("updated_at", now_iso())
    return base


def init_state() -> None:
    if "race_df" not in st.session_state:
        st.session_state.race_df = load_sample_df()
    if "photo_features" not in st.session_state:
        st.session_state.photo_features = pd.DataFrame()
    if "memory" not in st.session_state:
        st.session_state.memory = empty_memory()
    if "last_analysis_text" not in st.session_state:
        st.session_state.last_analysis_text = ""
    if "race_meta" not in st.session_state:
        st.session_state.race_meta = {
            "race_date": str(date.today()),
            "racecourse": "부산경남/서울/제주",
            "race_no": "",
            "distance": "",
            "memo": "",
        }


def memory_to_bytes(memory: Dict[str, Any]) -> bytes:
    memory = deepcopy(memory)
    memory["updated_at"] = now_iso()
    return json.dumps(memory, ensure_ascii=False, indent=2).encode("utf-8")


def github_config() -> Dict[str, str]:
    return {
        "token": get_secret("GITHUB_TOKEN", ""),
        "repo": get_secret("GITHUB_REPO", ""),
        "branch": get_secret("GITHUB_BRANCH", "main") or "main",
        "path": get_secret("STATS_FILE_PATH", DEFAULT_STATS_PATH) or DEFAULT_STATS_PATH,
    }


def github_headers(token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "race-agent-streamlit",
    }


def github_load_memory() -> Tuple[bool, str, Dict[str, Any]]:
    cfg = github_config()
    if not cfg["token"] or not cfg["repo"]:
        return False, "GITHUB_TOKEN 또는 GITHUB_REPO가 없습니다.", empty_memory()
    url = f"https://api.github.com/repos/{cfg['repo']}/contents/{cfg['path']}"
    try:
        r = requests.get(url, headers=github_headers(cfg["token"]), params={"ref": cfg["branch"]}, timeout=20)
        if r.status_code == 404:
            return True, "아직 저장된 통계 파일이 없습니다. 새로 시작합니다.", empty_memory()
        if not r.ok:
            return False, f"GitHub 읽기 실패: HTTP {r.status_code} {r.text[:200]}", empty_memory()
        data = r.json()
        raw = base64.b64decode(data.get("content", ""))
        memory = normalize_memory(json.loads(raw.decode("utf-8")))
        return True, f"GitHub에서 통계를 불러왔습니다: {cfg['path']}", memory
    except Exception as e:
        return False, f"GitHub 읽기 오류: {e}", empty_memory()


def github_save_memory(memory: Dict[str, Any]) -> Tuple[bool, str]:
    cfg = github_config()
    if not cfg["token"] or not cfg["repo"]:
        return False, "GITHUB_TOKEN 또는 GITHUB_REPO가 없습니다. 다운로드 백업을 사용하세요."
    url = f"https://api.github.com/repos/{cfg['repo']}/contents/{cfg['path']}"
    headers = github_headers(cfg["token"])
    try:
        sha = None
        get_r = requests.get(url, headers=headers, params={"ref": cfg["branch"]}, timeout=20)
        if get_r.ok:
            sha = get_r.json().get("sha")
        elif get_r.status_code != 404:
            return False, f"기존 파일 확인 실패: HTTP {get_r.status_code} {get_r.text[:200]}"
        content = base64.b64encode(memory_to_bytes(memory)).decode("utf-8")
        payload = {
            "message": f"Update race memory {now_iso()}",
            "content": content,
            "branch": cfg["branch"],
        }
        if sha:
            payload["sha"] = sha
        put_r = requests.put(url, headers=headers, json=payload, timeout=30)
        if not put_r.ok:
            return False, f"GitHub 저장 실패: HTTP {put_r.status_code} {put_r.text[:300]}"
        return True, f"GitHub에 통계를 저장했습니다: {cfg['path']}"
    except Exception as e:
        return False, f"GitHub 저장 오류: {e}"


# -----------------------------
# Data loading / cleaning
# -----------------------------
def load_sample_df() -> pd.DataFrame:
    try:
        return pd.read_csv("sample_race.csv")
    except Exception:
        return pd.DataFrame({
            "horse_no": [1, 2, 3, 4, 5, 6, 7, 8],
            "horse_name": ["인사이드킹", "스피드퀸", "막판추입", "가속흐름", "파워런", "선행강자", "꾸준한말", "복병후보"],
            "gate_no": [1, 3, 7, 4, 8, 2, 5, 6],
            "distance": [1200] * 8,
            "odds_win": [5.4, 3.1, 8.5, 4.0, 12.0, 6.8, 15.0, 18.0],
            "odds_place": [1.8, 1.4, 2.4, 1.6, 3.3, 2.1, 4.0, 4.8],
            "recent_rank_avg": [3.1, 2.0, 4.2, 2.8, 5.1, 3.5, 5.8, 6.0],
            "distance_score": [78, 86, 73, 82, 68, 77, 64, 66],
            "jockey_score": [74, 82, 69, 76, 65, 80, 62, 68],
            "trainer_score": [70, 79, 72, 75, 66, 73, 70, 67],
            "weight_change": [-1, 2, 0, 4, -3, 6, 1, -2],
            "style": ["선행", "선입", "추입", "선행", "선입", "선행", "자유", "추입"],
        })


def coerce_number(s: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(s):
            return default
        text = str(s).strip().replace(",", "")
        text = re.sub(r"[^0-9.\-]", "", text)
        if text in ("", "-", "."):
            return default
        return float(text)
    except Exception:
        return default


def standardize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Common Korean column mapping
    mapping = {
        "마번": "horse_no", "번호": "horse_no", "출주번호": "horse_no",
        "마명": "horse_name", "말이름": "horse_name",
        "게이트": "gate_no", "게이트번호": "gate_no", "출발번호": "gate_no",
        "거리": "distance", "경주거리": "distance",
        "단승": "odds_win", "단승배당": "odds_win", "단승식": "odds_win",
        "연승": "odds_place", "연승배당": "odds_place", "연승식": "odds_place",
        "최근평균순위": "recent_rank_avg", "최근순위평균": "recent_rank_avg",
        "거리점수": "distance_score", "기수점수": "jockey_score", "조교사점수": "trainer_score",
        "체중증감": "weight_change", "마체중증감": "weight_change",
        "각질": "style", "전개": "style",
    }
    df.rename(columns={c: mapping.get(c, c) for c in df.columns}, inplace=True)
    if "horse_no" not in df.columns:
        df["horse_no"] = np.arange(1, len(df) + 1)
    if "horse_name" not in df.columns:
        df["horse_name"] = df["horse_no"].apply(lambda x: f"{int(coerce_number(x, 0))}번마")
    for col in SAMPLE_COLUMNS:
        if col not in df.columns:
            if col == "style":
                df[col] = ""
            elif col == "horse_name":
                df[col] = ""
            elif col == "distance":
                df[col] = 1200
            elif col == "odds_win":
                df[col] = 9.9
            elif col == "odds_place":
                df[col] = 2.5
            elif col in ("distance_score", "jockey_score", "trainer_score"):
                df[col] = 60
            elif col == "recent_rank_avg":
                df[col] = 5
            else:
                df[col] = 0
    numeric_cols = ["horse_no", "gate_no", "distance", "odds_win", "odds_place", "recent_rank_avg", "distance_score", "jockey_score", "trainer_score", "weight_change"]
    for c in numeric_cols:
        df[c] = df[c].apply(coerce_number)
    df["horse_no"] = df["horse_no"].astype(int)
    df["gate_no"] = df["gate_no"].astype(int)
    df["horse_name"] = df["horse_name"].astype(str)
    return df[SAMPLE_COLUMNS].sort_values("horse_no").reset_index(drop=True)


# -----------------------------
# Vision/photo extraction
# -----------------------------
PHOTO_JSON_SCHEMA = {
    "race_context": {"racecourse": "", "race_no": "", "distance": "", "date": ""},
    "horses": [
        {
            "horse_no": 1,
            "horse_name": "",
            "expert_count": 0,
            "expert_labels": [],
            "condition_score": 50,
            "distance_score_photo": 50,
            "pace_style": "",
            "late_speed_score": 50,
            "weight_change": None,
            "odds_win": None,
            "odds_place": None,
            "risk_flags": [],
            "positive_notes": [],
            "raw_notes": "",
        }
    ],
    "summary": "",
}


def image_to_data_url(uploaded_file) -> str:
    raw = uploaded_file.getvalue()
    mime = uploaded_file.type or "image/jpeg"
    # Compress very large images to lower cost and reduce API failures.
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        max_side = 1600
        if max(img.size) > max_side:
            ratio = max_side / max(img.size)
            img = img.resize((int(img.width * ratio), int(img.height * ratio)))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=82)
        raw = buf.getvalue()
        mime = "image/jpeg"
    except Exception:
        pass
    return f"data:{mime};base64," + base64.b64encode(raw).decode("utf-8")


def parse_json_from_text(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.I).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    # Try to extract first JSON object
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}
    return {}


def analyze_images_with_openai(files: List[Any], race_df: pd.DataFrame, model: str) -> Tuple[bool, str, Dict[str, Any]]:
    api_key = get_secret("OPENAI_API_KEY", "")
    if not api_key:
        return False, "OPENAI_API_KEY가 없습니다. 사진 분석은 키가 있어야 작동합니다.", {}
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
    except Exception as e:
        return False, f"OpenAI 라이브러리 초기화 실패: {e}", {}

    horses_list = race_df[["horse_no", "horse_name"]].to_dict("records") if race_df is not None and not race_df.empty else []
    prompt = f"""
너는 한국 경마 예상지/출전표/배당판/마체중표를 읽어 베팅 보조용 구조화 데이터를 만드는 분석가다.
사진에서 보이는 내용만 사용하고, 모르는 값은 null 또는 빈 문자열로 둔다.
반드시 JSON만 출력한다. 설명 문장, 마크다운 금지.

현재 출전마 목록 참고:
{json.dumps(horses_list, ensure_ascii=False)}

목표:
- 말번호별 전문가 지목 수, 컨디션, 거리적성, 전개/각질, 추입/막판, 마체중 증감, 단승/연승 배당, 위험 신호를 추출한다.
- 사진이 여러 장이면 서로 합쳐서 같은 말번호를 통합한다.
- 전문가 표시가 ◎ ○ ▲ △ ☆ ★ 같은 기호이면 expert_labels에 기록하고 expert_count를 추정한다.
- condition_score, distance_score_photo, late_speed_score는 0~100 점수로 추정한다. 중립은 50.
- risk_flags에는 과체중/감량과다/외곽불리/거리불안/인기과열/기록부진/휴양후/승급부담 같은 위험을 넣는다.

출력 JSON 형식 예시:
{json.dumps(PHOTO_JSON_SCHEMA, ensure_ascii=False)}
"""
    content = [{"type": "input_text", "text": prompt}]
    for f in files:
        content.append({"type": "input_image", "image_url": image_to_data_url(f)})
    try:
        resp = client.responses.create(
            model=model,
            input=[{"role": "user", "content": content}],
            temperature=0.1,
            max_output_tokens=3000,
        )
        text = getattr(resp, "output_text", "") or str(resp)
        data = parse_json_from_text(text)
        if not data:
            return False, "사진 분석 결과를 JSON으로 파싱하지 못했습니다.", {"raw": text}
        return True, text, data
    except Exception as e:
        # Fallback to chat completions for accounts/models that don't support Responses image input.
        try:
            messages_content = [{"type": "text", "text": prompt}]
            for f in files:
                messages_content.append({"type": "image_url", "image_url": {"url": image_to_data_url(f)}})
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": messages_content}],
                temperature=0.1,
                max_tokens=3000,
            )
            text = resp.choices[0].message.content or ""
            data = parse_json_from_text(text)
            if not data:
                return False, f"사진 분석 JSON 파싱 실패. 원 오류: {e}", {"raw": text}
            return True, text, data
        except Exception as e2:
            return False, f"사진 분석 실패: {e2}", {}


def photo_json_to_df(data: Dict[str, Any]) -> pd.DataFrame:
    horses = data.get("horses", []) if isinstance(data, dict) else []
    rows = []
    for h in horses:
        if not isinstance(h, dict):
            continue
        no = h.get("horse_no")
        if no in (None, ""):
            continue
        rows.append({
            "horse_no": int(coerce_number(no, 0)),
            "horse_name_photo": str(h.get("horse_name") or ""),
            "expert_count": int(coerce_number(h.get("expert_count"), 0)),
            "expert_labels": ",".join(map(str, h.get("expert_labels", []) if isinstance(h.get("expert_labels"), list) else [])),
            "condition_score": coerce_number(h.get("condition_score"), 50),
            "distance_score_photo": coerce_number(h.get("distance_score_photo"), 50),
            "late_speed_score": coerce_number(h.get("late_speed_score"), 50),
            "pace_style": str(h.get("pace_style") or ""),
            "weight_change_photo": h.get("weight_change"),
            "odds_win_photo": h.get("odds_win"),
            "odds_place_photo": h.get("odds_place"),
            "risk_flags": ",".join(map(str, h.get("risk_flags", []) if isinstance(h.get("risk_flags"), list) else [])),
            "positive_notes": ",".join(map(str, h.get("positive_notes", []) if isinstance(h.get("positive_notes"), list) else [])),
            "raw_notes": str(h.get("raw_notes") or ""),
        })
    if not rows:
        return pd.DataFrame(columns=["horse_no", "expert_count", "condition_score", "distance_score_photo", "late_speed_score", "risk_flags"])
    df = pd.DataFrame(rows)
    for c in ["condition_score", "distance_score_photo", "late_speed_score", "odds_win_photo", "odds_place_photo", "weight_change_photo"]:
        if c in df.columns:
            df[c] = df[c].apply(lambda x: coerce_number(x, np.nan))
    return df.sort_values("horse_no").reset_index(drop=True)


# -----------------------------
# Scoring/statistics
# -----------------------------
def aggregate_memory_stats(memory: Dict[str, Any]) -> Dict[str, Any]:
    results = memory.get("race_results", []) if isinstance(memory, dict) else []
    total_races = len(results)
    horse_rows = []
    for r in results:
        finish = r.get("finish_order", [])
        if isinstance(finish, str):
            finish = parse_finish_order(finish)
        features = r.get("features", [])
        for h in features:
            if not isinstance(h, dict):
                continue
            no = int(coerce_number(h.get("horse_no"), 0))
            if no <= 0:
                continue
            rank = finish.index(no) + 1 if no in finish else None
            horse_rows.append({
                "horse_no": no,
                "horse_name": h.get("horse_name") or h.get("horse_name_photo") or "",
                "rank": rank,
                "win": rank == 1,
                "place": bool(rank and rank <= 3),
                "expert_count": coerce_number(h.get("expert_count"), 0),
                "condition_score": coerce_number(h.get("condition_score"), 50),
                "risk_count": len([x for x in str(h.get("risk_flags", "")).split(",") if x.strip()]),
            })
    if not horse_rows:
        return {
            "total_races": total_races,
            "total_horses": 0,
            "baseline_place_rate": 0.30,
            "expert_place_rate": None,
            "high_condition_place_rate": None,
            "risk_place_rate": None,
        }
    hdf = pd.DataFrame(horse_rows)
    known = hdf[hdf["rank"].notna()]
    if known.empty:
        return {"total_races": total_races, "total_horses": len(hdf), "baseline_place_rate": 0.30, "expert_place_rate": None, "high_condition_place_rate": None, "risk_place_rate": None}
    baseline = float(known["place"].mean())
    expert_df = known[known["expert_count"] >= 2]
    high_cond_df = known[known["condition_score"] >= 70]
    risk_df = known[known["risk_count"] >= 1]
    return {
        "total_races": total_races,
        "total_horses": len(known),
        "baseline_place_rate": baseline,
        "expert_place_rate": float(expert_df["place"].mean()) if not expert_df.empty else None,
        "high_condition_place_rate": float(high_cond_df["place"].mean()) if not high_cond_df.empty else None,
        "risk_place_rate": float(risk_df["place"].mean()) if not risk_df.empty else None,
    }


def reliability_multiplier(rate: Any, baseline: float) -> float:
    if rate is None or not isinstance(rate, (float, int)) or math.isnan(rate):
        return 1.0
    return float(np.clip(1.0 + (rate - baseline), 0.75, 1.35))


def softmax(x: np.ndarray, temp: float = 18.0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x
    z = (x - np.max(x)) / max(temp, 1e-6)
    e = np.exp(z)
    return e / e.sum()


def merge_photo_features(race_df: pd.DataFrame, photo_df: pd.DataFrame) -> pd.DataFrame:
    base = race_df.copy()
    if photo_df is None or photo_df.empty:
        return base
    merged = base.merge(photo_df, on="horse_no", how="left")
    # Let photo odds update current odds if present.
    for dest, src in [("odds_win", "odds_win_photo"), ("odds_place", "odds_place_photo"), ("weight_change", "weight_change_photo")]:
        if src in merged.columns:
            merged[dest] = merged.apply(lambda r: r[src] if not pd.isna(r[src]) else r[dest], axis=1)
    return merged


def score_race(race_df: pd.DataFrame, photo_df: pd.DataFrame, memory: Dict[str, Any]) -> pd.DataFrame:
    df = merge_photo_features(standardize_df(race_df), photo_df).copy()
    stats = aggregate_memory_stats(memory)
    baseline_place = stats.get("baseline_place_rate", 0.30) or 0.30
    expert_mult = reliability_multiplier(stats.get("expert_place_rate"), baseline_place)
    condition_mult = reliability_multiplier(stats.get("high_condition_place_rate"), baseline_place)
    risk_mult = reliability_multiplier(stats.get("risk_place_rate"), baseline_place)

    scores = []
    for _, r in df.iterrows():
        recent = max(1.0, coerce_number(r.get("recent_rank_avg"), 5))
        recent_score = max(0, 100 - (recent - 1) * 14)
        gate = coerce_number(r.get("gate_no"), 0)
        distance = coerce_number(r.get("distance"), 1200)
        gate_score = 55
        if distance <= 1300:
            gate_score = 70 if 1 <= gate <= 4 else 55 if gate <= 7 else 45
        elif distance >= 1600:
            gate_score = 60 if gate <= 6 else 52
        weight_chg = coerce_number(r.get("weight_change"), 0)
        weight_score = 60
        if -6 <= weight_chg <= 6:
            weight_score = 68
        elif abs(weight_chg) >= 12:
            weight_score = 42
        odds_win = max(1.01, coerce_number(r.get("odds_win"), 9.9))
        market_score = float(np.clip(100 / odds_win * 2.8, 10, 85))

        score = (
            coerce_number(r.get("distance_score"), 60) * 0.22 +
            coerce_number(r.get("jockey_score"), 60) * 0.13 +
            coerce_number(r.get("trainer_score"), 60) * 0.10 +
            recent_score * 0.17 +
            gate_score * 0.08 +
            weight_score * 0.07 +
            market_score * 0.08
        )
        # Photo adjustments
        expert_count = coerce_number(r.get("expert_count"), 0)
        cond = coerce_number(r.get("condition_score"), 50)
        dist_photo = coerce_number(r.get("distance_score_photo"), 50)
        late = coerce_number(r.get("late_speed_score"), 50)
        risks = [x.strip() for x in str(r.get("risk_flags", "")).split(",") if x.strip()]
        positives = [x.strip() for x in str(r.get("positive_notes", "")).split(",") if x.strip()]
        score += min(expert_count, 5) * 4.0 * expert_mult
        score += (cond - 50) * 0.22 * condition_mult
        score += (dist_photo - 50) * 0.13
        score += (late - 50) * 0.08
        score -= len(risks) * 4.5 * (1.0 if risk_mult >= baseline_place else 1.15)
        score += min(len(positives), 4) * 1.5
        scores.append(score)

    df["model_score"] = scores
    p = softmax(np.array(scores), temp=14.0)
    df["win_prob"] = p
    # place probability: calibrated-ish from win prob and score rank
    rank_factor = pd.Series(scores).rank(ascending=False, method="first").values
    df["place_prob"] = np.clip(df["win_prob"] * 2.15 + np.maximum(0, (len(df) + 1 - rank_factor)) / max(len(df), 1) * 0.10 + 0.12, 0.03, 0.88)
    df["ev_win"] = df["win_prob"] * df["odds_win"].apply(lambda x: max(1.01, coerce_number(x, 1.01))) - 1
    df["ev_place"] = df["place_prob"] * df["odds_place"].apply(lambda x: max(1.01, coerce_number(x, 1.01))) - 1
    df["risk_count"] = df.get("risk_flags", pd.Series([""] * len(df))).apply(lambda x: len([y for y in str(x).split(",") if y.strip()])) if "risk_flags" in df.columns else 0
    df["rank_model"] = df["model_score"].rank(ascending=False, method="first").astype(int)
    return df.sort_values("model_score", ascending=False).reset_index(drop=True)


def build_betting_plan(scored: pd.DataFrame, bankroll: int, min_ev: float) -> List[str]:
    if scored.empty:
        return []
    plan = []
    budget = int(bankroll)
    top = scored.iloc[0]
    second = scored.iloc[1] if len(scored) > 1 else None
    # Allocate carefully. Avoid overbetting one horse.
    unit = 1000 if budget >= 3000 else max(100, budget // 3)
    remaining = budget

    # Exacta axis-first if top win prob clear
    partners = scored.iloc[1:4]
    exacta_lines = []
    for _, r in partners.iterrows():
        exacta_lines.append((int(top.horse_no), int(r.horse_no), float(r.model_score)))
    if exacta_lines and remaining >= unit:
        main_partner = exacta_lines[0]
        amt = min(unit * 2 if budget >= 5000 else unit, remaining)
        plan.append(f"쌍승 {main_partner[0]}→{main_partner[1]} {amt}원")
        remaining -= amt
    for a, b, _ in exacta_lines[1:]:
        if remaining < unit:
            break
        plan.append(f"쌍승 {a}→{b} {unit}원")
        remaining -= unit

    # Positive EV place, usually safer
    place_candidates = scored[(scored["ev_place"] >= min_ev) | (scored["place_prob"] >= 0.48)].copy()
    place_candidates = place_candidates.sort_values(["ev_place", "place_prob"], ascending=False)
    for _, r in place_candidates.iterrows():
        if remaining < unit:
            break
        # Do not place-bet ultra low return unless probability high
        if coerce_number(r.odds_place, 1.0) < 1.2 and r.place_prob < 0.65:
            continue
        plan.append(f"연승 {int(r.horse_no)} {unit}원")
        remaining -= unit
        break

    # Win bet only if model finds value
    win_candidates = scored[scored["ev_win"] >= min_ev].sort_values("ev_win", ascending=False)
    if not win_candidates.empty and remaining >= unit:
        r = win_candidates.iloc[0]
        plan.append(f"단승 {int(r.horse_no)} {min(unit, remaining)}원")
        remaining -= min(unit, remaining)

    if remaining > 0 and plan:
        plan.append(f"잔액 {remaining}원은 보류 또는 배당 변동 확인")
    if not plan:
        plan = ["기대값 기준으로 강한 베팅 없음: 사진/배당 확인 후 소액 관망 추천"]
    return plan


def parse_finish_order(text: str) -> List[int]:
    nums = re.findall(r"\d+", text or "")
    return [int(n) for n in nums]


def current_features_for_memory(scored: pd.DataFrame) -> List[Dict[str, Any]]:
    if scored.empty:
        return []
    cols = [c for c in scored.columns if c not in []]
    rows = []
    for _, r in scored.iterrows():
        d = {}
        for c in cols:
            val = r[c]
            if isinstance(val, (np.integer, np.floating)):
                val = val.item()
            elif pd.isna(val):
                val = None
            d[c] = val
        rows.append(d)
    return rows


# -----------------------------
# Minimal API helper (safe, no crash)
# -----------------------------
def safe_api_test() -> pd.DataFrame:
    key = get_secret("DATA_GO_KR_SERVICE_KEY", "")
    api_map = {
        "경마경주정보": get_secret("KRA_RACE_INFO_API_URL", ""),
        "시행당일 경주결과종합": get_secret("KRA_RACE_RESULT_API_URL", ""),
        "경주마별 1년간 전적": get_secret("KRA_HORSE_RECORD_API_URL", ""),
        "매출액 및 확정배당율": get_secret("KRA_SALES_ODDS_API_URL", ""),
        "코너별 통과순위/주로빠르기": get_secret("KRA_SECTION_PACE_API_URL", ""),
        "시행당일 확정배당율종합": get_secret("KRA_FINAL_ODDS_API_URL", ""),
    }
    rows = []
    for name, url in api_map.items():
        if not key:
            rows.append({"API": name, "상태": "서비스키 없음", "행수": 0, "요청URL": url[:80]})
            continue
        if not url or not str(url).startswith("http"):
            rows.append({"API": name, "상태": "URL 없음", "행수": 0, "요청URL": str(url)[:80]})
            continue
        try:
            params = {
                "serviceKey": key,
                "pageNo": 1,
                "numOfRows": 5,
                "_type": "json",
            }
            # Do not force race-specific params; many APIs differ. This is only connection test.
            r = requests.get(url, params=params, timeout=12)
            if r.status_code >= 500:
                rows.append({"API": name, "상태": f"서버/요청 오류 HTTP {r.status_code}", "행수": 0, "요청URL": r.url.replace(key, "***")[:140]})
                continue
            if not r.ok:
                rows.append({"API": name, "상태": f"HTTP {r.status_code}", "행수": 0, "요청URL": r.url.replace(key, "***")[:140]})
                continue
            text = r.text[:2000]
            n = 0
            try:
                data = r.json()
                text_json = json.dumps(data, ensure_ascii=False)
                n = text_json.count("item") or text_json.count("items")
            except Exception:
                n = text.count("<item>") or text.count("<body>")
            rows.append({"API": name, "상태": "접속 성공/데이터는 조건에 따라 0행 가능", "행수": n, "요청URL": r.url.replace(key, "***")[:140]})
        except Exception as e:
            rows.append({"API": name, "상태": f"연결 오류: {str(e)[:80]}", "행수": 0, "요청URL": str(url)[:140]})
    return pd.DataFrame(rows)


# -----------------------------
# UI
# -----------------------------
def show_sidebar() -> Tuple[int, float, str]:
    with st.sidebar:
        st.header("설정")
        bankroll = st.number_input("경주당 예산", min_value=1000, max_value=100000, value=5000, step=1000)
        min_ev = st.slider("최소 기대값", min_value=-0.30, max_value=0.50, value=0.10, step=0.05)
        model = st.text_input("OpenAI Vision 모델", value=get_secret("OPENAI_VISION_MODEL", "gpt-4.1-mini") or "gpt-4.1-mini")
        st.divider()
        st.caption("키 상태")
        st.write("OpenAI:", "있음" if get_secret("OPENAI_API_KEY", "") else "없음")
        st.write("공공데이터:", "있음" if get_secret("DATA_GO_KR_SERVICE_KEY", "") else "없음")
        cfg = github_config()
        st.write("GitHub 저장:", "가능" if cfg["token"] and cfg["repo"] else "미설정")
        return int(bankroll), float(min_ev), model


def tab_data():
    st.subheader("1. 출전마 데이터")
    st.info("CSV를 올리지 않으면 샘플 데이터로 작동합니다. 현장에서는 사진분석만 해도 보정됩니다.")
    uploaded = st.file_uploader("출전마 CSV 업로드", type=["csv"], key="race_csv")
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.session_state.race_df = standardize_df(df)
            st.success("CSV를 불러왔습니다.")
        except Exception as e:
            st.error(f"CSV 읽기 실패: {e}")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("샘플 데이터로 초기화"):
            st.session_state.race_df = load_sample_df()
            st.success("샘플 데이터로 초기화했습니다.")
    with c2:
        if st.button("사진 보정 초기화"):
            st.session_state.photo_features = pd.DataFrame()
            st.success("사진 보정을 초기화했습니다.")
    with st.expander("경주 메모", expanded=False):
        meta = st.session_state.race_meta
        meta["race_date"] = st.text_input("경주일자", value=meta.get("race_date", str(date.today())))
        meta["racecourse"] = st.text_input("경마장", value=meta.get("racecourse", ""))
        meta["race_no"] = st.text_input("경주번호", value=meta.get("race_no", ""))
        meta["distance"] = st.text_input("거리", value=str(meta.get("distance", "")))
        meta["memo"] = st.text_area("메모", value=meta.get("memo", ""), height=80)
        st.session_state.race_meta = meta
    st.dataframe(st.session_state.race_df, width="stretch")


def tab_api():
    st.subheader("2. API 점검 / 과거 통계 보강")
    st.warning("API는 과거 통계 보강용입니다. 현장 당일 핵심 정보는 사진 업로드로 반영하는 구조가 더 안정적입니다.")
    with st.expander("설정값 확인", expanded=False):
        items = {
            "DATA_GO_KR_SERVICE_KEY": mask_secret(get_secret("DATA_GO_KR_SERVICE_KEY", "")),
            "KRA_RACE_INFO_API_URL": get_secret("KRA_RACE_INFO_API_URL", ""),
            "KRA_RACE_RESULT_API_URL": get_secret("KRA_RACE_RESULT_API_URL", ""),
            "KRA_HORSE_RECORD_API_URL": get_secret("KRA_HORSE_RECORD_API_URL", ""),
            "KRA_SALES_ODDS_API_URL": get_secret("KRA_SALES_ODDS_API_URL", ""),
            "KRA_SECTION_PACE_API_URL": get_secret("KRA_SECTION_PACE_API_URL", ""),
            "KRA_FINAL_ODDS_API_URL": get_secret("KRA_FINAL_ODDS_API_URL", ""),
        }
        st.json(items)
    if st.button("공공데이터 연결만 안전 점검", type="primary"):
        st.session_state.api_test = safe_api_test()
    if "api_test" in st.session_state:
        st.dataframe(st.session_state.api_test, width="stretch")
        st.caption("HTTP 500이 떠도 앱 분석은 계속 됩니다. 이 경우 해당 API의 요청변수 화면을 보고 별도 매핑해야 합니다.")


def tab_photo(model: str):
    st.subheader("3. 예상지/배당판/마체중표 사진")
    st.info("여기서 업로드한 사진 정보는 이번 경주 점수에 반영되고, 결과 저장 시 통계로 계속 업데이트됩니다.")
    files = st.file_uploader("사진 업로드", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True, key="photo_files")
    if files:
        cols = st.columns(min(3, len(files)))
        for i, f in enumerate(files[:3]):
            with cols[i % len(cols)]:
                st.image(f, caption=f.name, width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        analyze_btn = st.button("사진 분석 실행", type="primary", disabled=not bool(files))
    with c2:
        st.caption("OpenAI API 키가 없으면 아래 수동 JSON 입력을 사용하세요.")
    if analyze_btn:
        with st.spinner("사진을 분석 중입니다..."):
            ok, text, data = analyze_images_with_openai(files, st.session_state.race_df, model)
        st.session_state.last_analysis_text = text
        if ok:
            pdf = photo_json_to_df(data)
            st.session_state.photo_features = pdf
            obs = {
                "timestamp": now_iso(),
                "race_meta": deepcopy(st.session_state.race_meta),
                "summary": data.get("summary", ""),
                "features": pdf.to_dict("records"),
            }
            st.session_state.memory["photo_observations"].append(obs)
            st.success("사진 분석을 반영했습니다. 최종 분석 탭에서 확인하세요.")
        else:
            st.error(text)
    with st.expander("수동 JSON 입력 / 분석 결과 수정", expanded=False):
        default_json = json.dumps(PHOTO_JSON_SCHEMA, ensure_ascii=False, indent=2)
        manual = st.text_area("사진 분석 JSON을 직접 붙여넣기", value="", height=220, placeholder=default_json)
        if st.button("수동 JSON 반영"):
            data = parse_json_from_text(manual)
            pdf = photo_json_to_df(data)
            if pdf.empty:
                st.error("말번호별 데이터가 없습니다.")
            else:
                st.session_state.photo_features = pdf
                st.success("수동 입력을 반영했습니다.")
    if not st.session_state.photo_features.empty:
        st.write("사진에서 반영된 정보")
        st.dataframe(st.session_state.photo_features, width="stretch")


def tab_analysis(bankroll: int, min_ev: float):
    st.subheader("4. 최종 분석")
    scored = score_race(st.session_state.race_df, st.session_state.photo_features, st.session_state.memory)
    st.session_state.scored = scored
    show_cols = ["rank_model", "horse_no", "horse_name", "model_score", "win_prob", "place_prob", "odds_win", "odds_place", "ev_win", "ev_place", "risk_count"]
    existing = [c for c in show_cols if c in scored.columns]
    display = scored[existing].copy()
    for c in ["win_prob", "place_prob", "ev_win", "ev_place"]:
        if c in display.columns:
            display[c] = (display[c] * 100).round(1).astype(str) + "%"
    if "model_score" in display.columns:
        display["model_score"] = display["model_score"].round(1)
    st.dataframe(display, width="stretch")
    st.subheader("추천 배팅안")
    plan = build_betting_plan(scored, bankroll, min_ev)
    for p in plan:
        st.markdown(f"- {p}")
    st.warning("자동 구매 기능은 없습니다. 경마 결과는 불확실하며 손실 가능성을 항상 고려하세요.")
    with st.expander("판단 메모", expanded=False):
        top3 = scored.head(3)
        for _, r in top3.iterrows():
            notes = []
            if "expert_count" in r and coerce_number(r.get("expert_count"), 0) >= 2:
                notes.append("전문가 지목")
            if "condition_score" in r and coerce_number(r.get("condition_score"), 50) >= 70:
                notes.append("사진상 컨디션 우수")
            if "risk_flags" in r and str(r.get("risk_flags", "")).strip():
                notes.append(f"위험: {r.get('risk_flags')}")
            st.write(f"{int(r.horse_no)}번 {r.horse_name}: {', '.join(notes) if notes else '기본 통계 우위'}")


def tab_result_update():
    st.subheader("5. 결과 저장 / 사진 통계 업데이트")
    st.info("경주가 끝나면 실제 도착순서를 입력하세요. 이 데이터가 누적되어 사진 지표의 신뢰도가 업데이트됩니다.")
    finish_text = st.text_input("실제 도착순서", placeholder="예: 4-5-3-8-9")
    memo = st.text_area("결과 메모/반성", height=90, placeholder="예: 3번 복병을 놓쳤다, 4번 축은 맞았다 등")
    if st.button("이번 경주 결과 저장", type="primary"):
        finish = parse_finish_order(finish_text)
        if not finish:
            st.error("도착순서를 숫자로 입력하세요. 예: 4-5-3-8-9")
        else:
            scored = st.session_state.get("scored")
            if scored is None or scored.empty:
                scored = score_race(st.session_state.race_df, st.session_state.photo_features, st.session_state.memory)
            record = {
                "timestamp": now_iso(),
                "race_meta": deepcopy(st.session_state.race_meta),
                "finish_order": finish,
                "memo": memo,
                "features": current_features_for_memory(scored),
            }
            st.session_state.memory["race_results"].append(record)
            st.session_state.memory["updated_at"] = now_iso()
            st.success("결과를 메모리에 저장했습니다. 저장소 탭에서 GitHub 저장 또는 다운로드 백업을 하세요.")
    stats = aggregate_memory_stats(st.session_state.memory)
    st.write("누적 통계")
    st.json(stats)
    if st.session_state.memory.get("race_results"):
        summary_rows = []
        for i, r in enumerate(st.session_state.memory["race_results"][-10:], start=1):
            summary_rows.append({
                "순번": i,
                "저장시각": r.get("timestamp"),
                "경주": f"{r.get('race_meta', {}).get('race_date','')} {r.get('race_meta', {}).get('race_no','')}R",
                "도착순서": "-".join(map(str, r.get("finish_order", []))),
                "메모": r.get("memo", "")[:60],
            })
        st.dataframe(pd.DataFrame(summary_rows), width="stretch")


def tab_storage():
    st.subheader("6. 저장소 / 통계 유지")
    st.warning("Streamlit Cloud의 임시 파일은 영구 저장이 불안정합니다. 자동 저장은 GitHub Token, 아니면 JSON 다운로드/업로드를 사용하세요.")
    cfg = github_config()
    st.write("GitHub 설정")
    st.json({
        "GITHUB_TOKEN": "있음" if cfg["token"] else "없음",
        "GITHUB_REPO": cfg["repo"],
        "GITHUB_BRANCH": cfg["branch"],
        "STATS_FILE_PATH": cfg["path"],
    })
    c1, c2 = st.columns(2)
    with c1:
        if st.button("GitHub에서 통계 불러오기"):
            ok, msg, memory = github_load_memory()
            if ok:
                st.session_state.memory = memory
                st.success(msg)
            else:
                st.error(msg)
    with c2:
        if st.button("GitHub에 통계 저장"):
            ok, msg = github_save_memory(st.session_state.memory)
            if ok:
                st.success(msg)
            else:
                st.error(msg)
    st.download_button(
        "통계 JSON 다운로드 백업",
        data=memory_to_bytes(st.session_state.memory),
        file_name="race_memory.json",
        mime="application/json",
    )
    uploaded_memory = st.file_uploader("기존 통계 JSON 업로드", type=["json"], key="memory_upload")
    if uploaded_memory is not None:
        try:
            memory = normalize_memory(json.loads(uploaded_memory.getvalue().decode("utf-8")))
            if st.button("업로드한 통계 적용"):
                st.session_state.memory = memory
                st.success("통계를 불러왔습니다.")
        except Exception as e:
            st.error(f"통계 파일 읽기 실패: {e}")
    with st.expander("현재 메모리 원본 보기", expanded=False):
        st.json(st.session_state.memory)


def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🏇", layout="wide")
    init_state()
    bankroll, min_ev, model = show_sidebar()
    st.title("🏇 경마 현장 분석 에이전트 FINAL")
    st.caption("과거 통계는 API/저장 통계로 보강하고, 현장 정보는 사진 업로드로 반영합니다. 사진 분석 결과와 실제 결과는 계속 누적됩니다.")

    tabs = st.tabs(["1. 데이터", "2. API 점검", "3. 예상지 사진", "4. 최종 분석", "5. 결과/통계", "6. 저장소"])
    with tabs[0]:
        tab_data()
    with tabs[1]:
        tab_api()
    with tabs[2]:
        tab_photo(model)
    with tabs[3]:
        tab_analysis(bankroll, min_ev)
    with tabs[4]:
        tab_result_update()
    with tabs[5]:
        tab_storage()


if __name__ == "__main__":
    main()
