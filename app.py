import base64, json, math, re
from datetime import datetime, date
from io import StringIO, BytesIO
import pandas as pd
import requests
import streamlit as st
from PIL import Image
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

st.set_page_config(page_title='🏇 경마 공공데이터 제대로 버전', layout='wide')

DEFAULT_ENDPOINTS = {
    # 경주 전/당일 분석용: race_dt + race_no 중심
    'seoul_entries': 'https://apis.data.go.kr/B551015/API314',
    'jeju_entries': 'https://apis.data.go.kr/B551015/API315',
    'busan_entries': 'https://apis.data.go.kr/B551015/API316',
    'seoul_weight': 'https://apis.data.go.kr/B551015/API317',
    'jeju_weight': 'https://apis.data.go.kr/B551015/API318',
    'busan_weight': 'https://apis.data.go.kr/B551015/API319',
    # 출전등록현황: race_dt + gb 중심
    'seoul_registered': 'https://apis.data.go.kr/B551015/API323',
    'jeju_registered': 'https://apis.data.go.kr/B551015/API324',
    'busan_registered': 'https://apis.data.go.kr/B551015/API325',
    # 사후 통계/검증용
    'race_info_old': 'https://apis.data.go.kr/B551015/API187',
    'race_result': 'https://apis.data.go.kr/B551015/API299',
    'horse_record': 'https://apis.data.go.kr/B551015/API145',
    'sales_odds': 'https://apis.data.go.kr/B551015/API179_1',
    'section_pace': 'https://apis.data.go.kr/B551015/API303',
    'final_odds': 'https://apis.data.go.kr/B551015/API301',
}

MEET_KEY = {'서울': 'seoul', '제주': 'jeju', '부경': 'busan', '부산경남': 'busan'}
MEET_CODE = {'서울': '1', '제주': '2', '부경': '3', '부산경남': '3'}

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

def secret(name, default=''):
    try:
        return str(st.secrets.get(name, default) or '')
    except Exception:
        return default

def sf(x, default=0.0):
    try:
        if pd.isna(x): return default
        return float(str(x).replace(',', '').strip())
    except Exception:
        return default

def si(x, default=0):
    try:
        if pd.isna(x): return default
        return int(float(str(x).replace(',', '').strip()))
    except Exception:
        return default

def norm_date(d) -> str:
    if isinstance(d, date): return d.strftime('%Y%m%d')
    s = str(d).replace('-', '').replace('/', '').strip()
    return s[:8]

def get_endpoint(key):
    return secret(key.upper() + '_URL', DEFAULT_ENDPOINTS.get(key, '')) or DEFAULT_ENDPOINTS.get(key, '')

def make_sample():
    return pd.read_csv(StringIO(SAMPLE_CSV))

def ensure_df(df):
    df = df.copy()
    defaults = {
        'horse_no': list(range(1, len(df)+1)),
        'horse_name': [f'{i}번마' for i in range(1, len(df)+1)],
        'gate_no': 0, 'distance': 1200, 'odds_win': 0.0, 'odds_place': 0.0,
        'recent_rank_avg': 5.0, 'distance_score': 65.0, 'jockey_score': 65.0,
        'trainer_score': 65.0, 'weight_change': 0.0, 'api_score': 0.0
    }
    for k, v in defaults.items():
        if k not in df.columns:
            df[k] = v if not isinstance(v, list) else v
    df['horse_no'] = df['horse_no'].apply(lambda x: str(si(x, x)).strip())
    df['horse_name'] = df['horse_name'].astype(str)
    return df

def find_lists(obj):
    found = []
    def walk(x):
        if isinstance(x, list):
            found.append(x)
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
    walk(obj)
    return found

def parse_response(text):
    text = (text or '').strip()
    if not text:
        return [], '빈 응답'
    try:
        data = json.loads(text)
        lists = find_lists(data)
        if lists:
            longest = max(lists, key=len)
            if longest and isinstance(longest[0], dict):
                return longest, data
        return [], data
    except Exception:
        pass
    items = []
    for item in re.findall(r'<item>(.*?)</item>', text, flags=re.S):
        d = {}
        for m in re.finditer(r'<([^/>]+)>(.*?)</\\1>', item, flags=re.S):
            d[m.group(1)] = re.sub(r'\s+', ' ', m.group(2)).strip()
        if d:
            items.append(d)
    if items:
        return items, items[:5]
    return [], text[:800]

def call_api(url, params, service_key, label=''):
    if not url:
        return {'ok': False, 'status': 'URL 없음', 'rows': 0, 'data': [], 'raw': '', 'url': ''}
    if not service_key:
        return {'ok': False, 'status': '서비스키 없음', 'rows': 0, 'data': [], 'raw': '', 'url': url}
    base_params = {'serviceKey': service_key, 'pageNo': 1, 'numOfRows': 200, '_type': 'json'}
    base_params.update({k: v for k, v in params.items() if v not in (None, '')})
    attempts = []
    attempts.append(base_params)
    p2 = base_params.copy(); p2['ServiceKey'] = p2.pop('serviceKey')
    attempts.append(p2)
    p3 = base_params.copy(); p3.pop('_type', None)
    attempts.append(p3)
    last = None
    for p in attempts:
        try:
            r = requests.get(url, params=p, timeout=15)
            rows, raw = parse_response(r.text)
            if r.status_code < 400 and rows:
                return {'ok': True, 'status': '성공', 'rows': len(rows), 'data': rows, 'raw': raw, 'url': r.url}
            if r.status_code < 400:
                last = {'ok': False, 'status': '데이터 없음/파싱 실패', 'rows': 0, 'data': [], 'raw': raw, 'url': r.url}
            else:
                last = {'ok': False, 'status': f'서버/요청 오류 HTTP {r.status_code}', 'rows': 0, 'data': [], 'raw': r.text[:800], 'url': r.url}
        except Exception as e:
            last = {'ok': False, 'status': f'연결 오류 {str(e)[:100]}', 'rows': 0, 'data': [], 'raw': '', 'url': url}
    return last

def normalize_entries(rows, distance=None):
    out = []
    for i, r in enumerate(rows, start=1):
        no = r.get('pthrNo') or r.get('chulNo') or r.get('ord') or r.get('entryNo') or r.get('horse_no') or i
        name = r.get('hrnm') or r.get('hrName') or r.get('hr_name') or r.get('horse_name') or r.get('마명') or f'{no}번마'
        out.append({
            'horse_no': str(si(no, i)), 'horse_name': str(name), 'gate_no': si(no, i),
            'distance': sf(distance, 1200), 'odds_win': 0.0, 'odds_place': 0.0,
            'recent_rank_avg': 5.0,
            'distance_score': 60 + min(20, sf(r.get('ratg') or r.get('rating') or 0, 0)/5),
            'jockey_score': 65.0, 'trainer_score': 65.0, 'weight_change': 0.0,
            'api_score': 8.0, 'rating': r.get('ratg') or r.get('rating') or '',
            'jockey': r.get('jckyNm') or r.get('jockey') or '',
            'trainer': r.get('trarNm') or r.get('trainer') or '',
            'owner': r.get('ownerNm') or '', 'raw_api': json.dumps(r, ensure_ascii=False)
        })
    return ensure_df(pd.DataFrame(out)) if out else pd.DataFrame()

def merge_weight(df, weight_rows):
    if df is None or df.empty or not weight_rows:
        return df
    df = df.copy(); wmap = {}
    for r in weight_rows:
        no = str(si(r.get('pthrNo') or r.get('chulNo') or r.get('raceNo') or 0, 0))
        name = str(r.get('hrnm') or r.get('hrName') or '')
        item = {'weight': r.get('hrWeg') or r.get('weight') or '', 'change': r.get('indec') or r.get('change') or ''}
        if no and no != '0': wmap[('no', no)] = item
        if name: wmap[('name', name)] = item
    weights = []; changes = []
    for _, row in df.iterrows():
        item = wmap.get(('no', str(row['horse_no']))) or wmap.get(('name', str(row['horse_name']))) or {}
        weights.append(item.get('weight', ''))
        changes.append(sf(item.get('change', row.get('weight_change', 0)), 0))
    df['horse_weight'] = weights
    df['weight_change'] = changes
    return ensure_df(df)

def img_to_data_url(img):
    img = img.convert('RGB')
    max_side = 1600
    scale = min(1.0, max_side/max(img.size))
    if scale < 1:
        img = img.resize((int(img.size[0]*scale), int(img.size[1]*scale)))
    buf = BytesIO(); img.save(buf, format='JPEG', quality=85)
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()

def analyze_photos(files, df, api_key, model):
    if not api_key:
        return {'error': 'OPENAI_API_KEY 없음'}
    if OpenAI is None:
        return {'error': 'openai 패키지 로드 실패'}
    horses = df[['horse_no','horse_name']].to_dict('records')
    prompt = f'''
한국 경마 예상지/출전표/배당판/마체중표 사진을 분석한다.
출전마 목록: {json.dumps(horses, ensure_ascii=False)}
JSON만 출력. 형식:
{{"summary":"요약", "horses":{{"1":{{"expert_count":0,"condition_score":0,"distance_score_delta":0,"risk_flags":[],"dark_horse":false,"photo_score":0,"evidence":"근거"}}}}}}
condition_score -10~10, distance_score_delta -5~5, photo_score -20~20. 흐리면 추측하지 말고 불명확이라고 써라.
'''
    content = [{'type':'input_text', 'text': prompt}]
    for f in files:
        content.append({'type':'input_image', 'image_url': img_to_data_url(Image.open(f))})
    try:
        client = OpenAI(api_key=api_key)
        resp = client.responses.create(model=model, input=[{'role':'user','content':content}], temperature=0.1)
        text = re.sub(r'^```json\s*|\s*```$', '', resp.output_text.strip(), flags=re.S)
        return json.loads(text)
    except Exception as e:
        return {'error': str(e)}

def compute(df, signals, memory):
    df = ensure_df(df); rows = []
    for _, r in df.iterrows():
        no = str(r['horse_no']); sig = signals.get(no, {})
        raw = 0
        raw += max(0, 20 - sf(r['recent_rank_avg'], 5)*2)
        raw += sf(r['distance_score'], 65)/5
        raw += sf(r['jockey_score'], 65)/7
        raw += sf(r['trainer_score'], 65)/8
        raw += sf(r.get('api_score', 0), 0)
        raw += sf(sig.get('photo_score', 0), 0)
        raw += min(8, si(sig.get('expert_count', 0), 0)*2)
        raw += sf(sig.get('condition_score', 0), 0)
        raw += sf(sig.get('distance_score_delta', 0), 0)
        raw -= 3*len(sig.get('risk_flags', []) or [])
        raw += 3 if sig.get('dark_horse') else 0
        if abs(sf(r.get('weight_change', 0), 0)) >= 12: raw -= 3
        rows.append({**r.to_dict(), 'raw_score': raw, 'photo_score': sf(sig.get('photo_score', 0), 0), 'risk_count': len(sig.get('risk_flags', []) or [])})
    out = pd.DataFrame(rows)
    exp = (out.raw_score - out.raw_score.max()).apply(lambda x: math.exp(x/10))
    out['win_prob'] = exp/exp.sum()
    out['place_prob'] = (out.win_prob*2.2 + 0.10).clip(0.05, 0.85)
    out['ev_win'] = out.apply(lambda x: sf(x.odds_win, 0)*x.win_prob-1 if sf(x.odds_win, 0)>0 else None, axis=1)
    out['ev_place'] = out.apply(lambda x: sf(x.odds_place, 0)*x.place_prob-1 if sf(x.odds_place, 0)>0 else None, axis=1)
    return out.sort_values('raw_score', ascending=False)

def bets(scored, bankroll=5000):
    unit = 1000; spent = 0; out = []
    top = scored.iloc[0]
    partners = scored.iloc[1:4].horse_no.tolist()
    for p in partners:
        if spent + unit <= bankroll:
            out.append(f"쌍승 {top.horse_no}→{p} {unit}원"); spent += unit
    for _, r in scored.sort_values('ev_place', ascending=False, na_position='last').head(2).iterrows():
        if spent + unit <= bankroll and r.ev_place is not None and r.ev_place > 0.05:
            out.append(f"연승 {r.horse_no} {unit}원"); spent += unit
    for _, r in scored.sort_values('ev_win', ascending=False, na_position='last').head(1).iterrows():
        if spent + unit <= bankroll and r.ev_win is not None and r.ev_win > 0.08:
            out.append(f"단승 {r.horse_no} {unit}원"); spent += unit
    return out or ['기대값이 약합니다. 관망 또는 소액만 권장']

if 'race_df' not in st.session_state: st.session_state.race_df = make_sample()
if 'signals' not in st.session_state: st.session_state.signals = {}
if 'memory' not in st.session_state: st.session_state.memory = {'races': [], 'updated_at': ''}

st.title('🏇 경마 공공데이터 제대로 버전')
st.caption('경주 전 API와 경주 후 API를 분리했습니다. 경주 전 자동 로딩은 출전마현황/출전마체중만 사용하고, 결과/배당/코너 API는 사후 통계용으로만 둡니다.')

with st.sidebar:
    st.header('기본 설정')
    key = secret('DATA_GO_KR_SERVICE_KEY', '')
    openai_key = secret('OPENAI_API_KEY', '')
    model = secret('OPENAI_VISION_MODEL', 'gpt-4.1-mini')
    st.write('공공데이터 키:', '✅ 있음' if key else '❌ 없음')
    st.write('OpenAI 키:', '✅ 있음' if openai_key else '❌ 없음')
    bankroll = st.number_input('예산', 1000, 100000, 5000, 1000)

tabs = st.tabs(['1. 공공데이터로 출전마 불러오기','2. CSV/데이터 확인','3. 사진 분석','4. 최종 분석','5. 결과 저장','6. 사후 통계 API'])

with tabs[0]:
    st.header('경주 전 공공데이터 전용 호출')
    st.info('여기서는 경주 전/당일에 실제 필요한 출전마현황 + 출전마체중만 호출합니다. 결과/배당/코너 API는 여기서 부르지 않습니다.')
    c1, c2, c3, c4 = st.columns(4)
    meet_label = c1.selectbox('경마장', ['서울','부경','제주'])
    race_dt = c2.date_input('경주일자')
    race_no = c3.number_input('경주번호', 1, 20, 1)
    distance = c4.number_input('거리', 800, 3200, 1200, 100)
    race_dt_s = norm_date(race_dt)
    meet_key = MEET_KEY[meet_label]
    entry_key = f'{meet_key}_entries'; weight_key = f'{meet_key}_weight'; reg_key = f'{meet_key}_registered'
    with st.expander('사용할 엔드포인트 확인/수정'):
        entry_url = st.text_input('출전마현황 URL', get_endpoint(entry_key))
        weight_url = st.text_input('출전마체중 URL', get_endpoint(weight_key))
        reg_url = st.text_input('출전등록현황 URL', get_endpoint(reg_key))
        st.caption('공공데이터 포털의 상세기능 화면에서 요청주소가 별도로 보이면 이 칸에 그대로 붙여넣으세요.')
    if st.button('출전마/체중 API로 불러오기', type='primary'):
        params = {'race_dt': race_dt_s, 'race_no': str(int(race_no)), 'pageNo': 1, 'numOfRows': 200}
        entry = call_api(entry_url, params, key, 'entries')
        weight = call_api(weight_url, params, key, 'weight')
        reg = call_api(reg_url, {'race_dt': race_dt_s, 'gb': '', 'pageNo': 1, 'numOfRows': 200}, key, 'registered')
        st.session_state['last_api'] = {'entry': entry, 'weight': weight, 'registered': reg}
        status = pd.DataFrame([
            {'구분':'출전마현황','상태':entry['status'],'행수':entry['rows'],'요청URL':entry['url']},
            {'구분':'출전마체중','상태':weight['status'],'행수':weight['rows'],'요청URL':weight['url']},
            {'구분':'출전등록현황','상태':reg['status'],'행수':reg['rows'],'요청URL':reg['url']},
        ])
        st.dataframe(status, use_container_width=True)
        source_rows = entry['data'] if entry['data'] else reg['data']
        if source_rows:
            df = normalize_entries(source_rows, distance)
            if weight['data']: df = merge_weight(df, weight['data'])
            st.session_state.race_df = df
            st.success(f'{len(df)}두를 앱 데이터로 불러왔습니다.')
            st.dataframe(df, use_container_width=True)
        else:
            st.error('출전마 데이터가 없습니다. 아래 원본 미리보기를 보고 요청주소/날짜/경주번호를 확인하세요.')
        for name, res in st.session_state['last_api'].items():
            with st.expander(f'{name} 원본 미리보기'):
                st.write(res['raw'])

with tabs[1]:
    st.header('CSV/데이터 확인')
    file = st.file_uploader('CSV 업로드', type=['csv'])
    if file:
        st.session_state.race_df = ensure_df(pd.read_csv(file)); st.success('CSV 적용 완료')
    if st.button('샘플로 초기화'):
        st.session_state.race_df = make_sample()
    st.dataframe(ensure_df(st.session_state.race_df), use_container_width=True)

with tabs[2]:
    st.header('사진 분석')
    files = st.file_uploader('예상지/배당판/마체중표 사진', type=['jpg','jpeg','png','webp'], accept_multiple_files=True)
    if files: st.image(files, width=180)
    if st.button('OpenAI로 사진 분석'):
        res = analyze_photos(files or [], ensure_df(st.session_state.race_df), openai_key, model)
        if 'error' in res: st.error(res['error'])
        else:
            st.session_state.signals = res.get('horses', {})
            st.success('사진 분석 완료')
            st.json(res)
    st.subheader('수동 보정')
    manual = {}
    for _, r in ensure_df(st.session_state.race_df).iterrows():
        no = str(r.horse_no)
        with st.expander(f'{no}번 {r.horse_name}'):
            exp = st.number_input(f'{no} 전문가 지목', 0, 10, 0, key='e'+no)
            cond = st.slider(f'{no} 컨디션', -10, 10, 0, key='c'+no)
            ps = st.slider(f'{no} 사진점수', -20, 20, 0, key='p'+no)
            risk = st.text_input(f'{no} 위험신호', key='r'+no)
            dark = st.checkbox(f'{no} 복병', key='d'+no)
            if exp or cond or ps or risk or dark:
                manual[no] = {'expert_count': exp, 'condition_score': cond, 'photo_score': ps, 'risk_flags': [x.strip() for x in risk.split(',') if x.strip()], 'dark_horse': dark, 'evidence': '수동'}
    if st.button('수동 보정 적용'):
        s = dict(st.session_state.signals); s.update(manual); st.session_state.signals = s; st.success('적용됨')
    if st.session_state.signals: st.json(st.session_state.signals)

with tabs[3]:
    st.header('최종 분석')
    scored = compute(st.session_state.race_df, st.session_state.signals, st.session_state.memory)
    st.session_state.scored = scored
    cols = ['horse_no','horse_name','raw_score','win_prob','place_prob','odds_win','odds_place','ev_win','ev_place','api_score','photo_score','risk_count','weight_change']
    st.dataframe(scored[cols], use_container_width=True)
    st.subheader('추천 배팅안')
    for b in bets(scored, int(bankroll)): st.write('- ' + b)
    st.warning('분석 보조 도구입니다. 자동 구매 기능은 없고 손실 가능성을 고려하세요.')

with tabs[4]:
    st.header('결과 저장/통계 누적')
    finish = st.text_input('도착순서 예: 4-2-3-7-11')
    note = st.text_area('메모')
    if st.button('결과 저장'):
        order = [x for x in re.split(r'[-, /]+', finish.strip()) if x]
        st.session_state.memory['races'].append({'at': datetime.now().isoformat(timespec='seconds'), 'finish': order, 'signals': st.session_state.signals, 'note': note})
        st.session_state.memory['updated_at'] = datetime.now().isoformat(timespec='seconds')
        st.success('저장됨')
    data = json.dumps(st.session_state.memory, ensure_ascii=False, indent=2)
    up = st.file_uploader('이전 통계 JSON 불러오기', type=['json'])
    if up:
        st.session_state.memory = json.loads(up.getvalue().decode('utf-8')); st.success('불러옴')
    st.download_button('통계 JSON 다운로드', data=data.encode('utf-8'), file_name='race_memory.json', mime='application/json')
    st.code(data, language='json')

with tabs[5]:
    st.header('사후 통계 API')
    st.caption('경주결과/확정배당/코너순위 등은 경주 후 학습용입니다. 경주 전 자동 추천 데이터로 직접 쓰지 않습니다.')
    c1, c2, c3 = st.columns(3)
    meet = c1.selectbox('meet', ['1','2','3'], index=0)
    rc_date = c2.text_input('rc_date', value=datetime.now().strftime('%Y%m%d'))
    rc_no = c3.text_input('rc_no', value='1')
    post_apis = [('race_result','경주결과',get_endpoint('race_result')),('final_odds','확정배당',get_endpoint('final_odds')),('section_pace','코너/주로',get_endpoint('section_pace')),('horse_record','말별1년전적',get_endpoint('horse_record'))]
    if st.button('사후 API 개별 점검'):
        rows = []
        for keyname, label, url in post_apis:
            res = call_api(url, {'meet':meet,'rc_date':rc_date,'rc_no':rc_no,'rc_year':rc_date[:4],'rc_month':rc_date[:6]}, key, label)
            rows.append({'API':label,'상태':res['status'],'행수':res['rows'],'요청URL':res['url'],'원본':res['raw']})
        st.dataframe(pd.DataFrame([{k:v for k,v in r.items() if k!='원본'} for r in rows]), use_container_width=True)
        for r in rows:
            with st.expander(r['API']+' 원본'):
                st.write(r['원본'])
