"""
AI 기반 도로 위험 신고 지원 시스템 — Streamlit 웹 프로토타입 (4NOVA)

화면 흐름 (4단계)
  ① 사진 업로드
  ② 위험요소 인식 결과 - [민서] HyperCLOVA X(HCX-005) 유형/심각도 + [민서] YOLO 파손 위치 검출
  ③ 우선순위(등급) 판정 - 파손 면적 비율 기반
  ④ 민원문 확인

연동 현황
  [민서] analyze_road_damage_clova()  : HCX-005 연동 완료 (심각도 경미=1, 보통=2, 심각=3, 긴급=4)
  [민서] detect_damage()              : YOLO 연동 완료 (best.pt 를 app.py 옆에 두면 자동 사용)
  [주혜] calculate_single_image_damage_ratio() : 10/29 수정본 (균열=선을 폭 12px 띠로 계산)
         + 11/9 수정: 균열·포트홀 전체를 합집합으로 계산 (겹친 부분 한 번만)
         -> 좌표가 있는 라벨 사진용. YOLO 모델이 없을 때의 임시 시연에도 사용.
  [윤수] 채점 함수 : 자리만 표시 (서비스 화면에는 직접 쓰이지 않음)

확정된 규칙
  - 면적 비율 분모는 "사진 전체 면적(픽셀 기준)".
  - 등급 3단계: 일반 <2% / 보통 2~5% / 긴급 >=5%
    (300장 실데이터 기준 등급 분포 246/37/17장, 파손 비율 최댓값 10.54%)
  - AI 심각도(4단계)와 서비스 등급(3단계)은 매핑하지 않고 독립적으로 사용
    (심각도는 신뢰성 검증용 참고 지표).
  - 어린이 보호구역 인근 체크 시 등급은 그대로 두고 민원문에 '해당 위치는 어린이 보호구역 인근입니다.' 한 문장만 추가 (등급 상향은 이후 과제).

실행: streamlit run app.py   (필요 패키지: requirements.txt)
"""

import base64
import os
import random
import re
import tempfile
import unicodedata

from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

import streamlit as st

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import numpy as np
    from ultralytics import YOLO
except ImportError:  # ultralytics 미설치 시 임시 좌표 방식으로 동작
    YOLO = None
    np = None

st.set_page_config(page_title="도로 위험 신고", page_icon="🛣️", layout="centered",
                   initial_sidebar_state="expanded")

# ---------------- 노션 스타일 디자인 (CSS) ----------------
LIGHT = dict(bg="#ffffff", fg="#37352f", muted="#787774", side="#f7f7f5", line="#e9e9e7",
             hover="#efefed", card="#ffffff", accent="#2383e2", code="#f7f6f3",
             red_bg="#fdebec", red_fg="#9f2f2d", org_bg="#fbecdd", org_fg="#b35c1e",
             grn_bg="#edf3ec", grn_fg="#346538", gry_bg="#f1f1ef", gry_fg="#5f5e5b",
             blu_bg="#e7f3f8", blu_fg="#28456c", call_bg="#f1f1ef")
DARK = dict(bg="#191919", fg="#ebebea", muted="#9b9a97", side="#202020", line="#373737",
            hover="#2f2f2f", card="#252525", accent="#529cca", code="#252525",
            red_bg="#522e2a", red_fg="#ff9d97", org_bg="#5c3b23", org_fg="#ffb36b",
            grn_bg="#243d30", grn_fg="#7ec699", gry_bg="#373737", gry_fg="#c3c2bf",
            blu_bg="#1f3a4d", blu_fg="#79b8e0", call_bg="#252525")


def inject_css(dark=False):
    pal = DARK if dark else LIGHT
    root = ":root{" + ";".join(f"--{k.replace('_','-')}:{v}" for k, v in pal.items()) + "}"
    st.markdown("<style>" + root + """
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css');
html, body, [class*="css"], .stApp, button, input, textarea {
  font-family: "Pretendard", ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", "Apple SD Gothic Neo", sans-serif !important;
  -webkit-font-smoothing: antialiased; }
.stApp, [data-testid="stAppViewContainer"] { background: var(--bg); color: var(--fg); }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container { max-width: 760px; padding-top: 2.2rem; padding-bottom: 5rem; }
p, span, label, li, div { color: inherit; }
h1, h2, h3, h4 { color: var(--fg); letter-spacing: -0.02em; font-weight: 700; }
[data-testid="stCaptionContainer"], .stCaption { color: var(--muted) !important; }

/* 사이드바 */
[data-testid="stSidebar"] { background: var(--side); border-right: 1px solid var(--line); }
[data-testid="stSidebar"] * { color: var(--fg); }
.brand { display:flex; align-items:center; gap:8px; font-weight:700; font-size:1.02rem; margin:.2rem 0 .9rem; }
.brand .b-ic { font-size:1.25rem; }
.side-h { font-size:.72rem; font-weight:600; color: var(--muted) !important; margin:1.1rem 0 .3rem; letter-spacing:.04em; }
[data-testid="stSidebar"] [role="radiogroup"] { gap: 2px; }
[data-testid="stSidebar"] [role="radiogroup"] label { padding: 6px 8px; border-radius: 6px; width:100%; }
[data-testid="stSidebar"] [role="radiogroup"] label:hover { background: var(--hover); }
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child { display:none; }
[data-testid="stSidebar"] [data-testid="stExpander"] { border:1px solid var(--line); border-radius:8px; background: transparent; }

/* 페이지 헤더 */
.page-icon { font-size: 3.4rem; line-height:1; margin: .2rem 0 .5rem; }
.page-title { font-size: 2.35rem; font-weight: 800; letter-spacing:-0.03em; margin:0 0 .25rem; color: var(--fg); }
.crumb { font-size:.82rem; color: var(--muted); margin-bottom: 1rem; }
.props { border-top:1px solid var(--line); border-bottom:1px solid var(--line); padding:.35rem 0; margin:.6rem 0 1.2rem; }
.prop { display:flex; align-items:center; gap:.6rem; padding:.28rem .2rem; border-radius:6px; font-size:.92rem; }
.prop:hover { background: var(--hover); }
.pk { width: 120px; color: var(--muted); flex-shrink:0; }
.pv { flex:1; }

/* 스텝 */
.steps { display:flex; gap:6px; flex-wrap:wrap; margin: .2rem 0 1.2rem; }
.step { font-size:.8rem; padding:3px 10px; border-radius:999px; background: var(--gry-bg); color: var(--gry-fg); }
.step.on { background: var(--blu-bg); color: var(--blu-fg); font-weight:600; }
.step.done { background: var(--grn-bg); color: var(--grn-fg); }

/* 태그 */
.tag { display:inline-block; padding:1px 8px; border-radius:4px; font-size:.82rem; font-weight:500; margin-right:4px; }
.tag.red { background: var(--red-bg); color: var(--red-fg); }
.tag.orange { background: var(--org-bg); color: var(--org-fg); }
.tag.green { background: var(--grn-bg); color: var(--grn-fg); }
.tag.gray { background: var(--gry-bg); color: var(--gry-fg); }
.tag.blue { background: var(--blu-bg); color: var(--blu-fg); }

/* 콜아웃 */
.callout { display:flex; gap:.7rem; background: var(--call-bg); border:1px solid var(--line); border-radius:6px;
           padding: .8rem 1rem; margin: .6rem 0 1rem; font-size:.93rem; line-height:1.55; }
.callout .ci { font-size:1.15rem; }
.callout.warn { background: var(--org-bg); border-color: transparent; }

/* 게이지 */
.gauge { margin:.4rem 0 1rem; }
.gtrack { position:relative; display:flex; height:14px; border-radius:7px; overflow:hidden; background: var(--gry-bg); }
.gz { height:100%; opacity:.9; }
.gz.l { width:20%; background: var(--grn-bg); } .gz.m { width:30%; background: var(--org-bg); } .gz.h { width:50%; background: var(--red-bg); }
.gneedle { position:absolute; top:-3px; width:4px; height:20px; border-radius:2px; background: var(--fg); transition:left .4s; }
.glabels { display:flex; font-size:.72rem; color: var(--muted); margin-top:4px; }
.glabels span:nth-child(1){ width:20%; } .glabels span:nth-child(2){ width:30%; } .glabels span:nth-child(3){ width:50%; }

/* 입력 */
.stTextInput input, .stTextArea textarea, [data-baseweb="select"] > div {
  background: var(--bg) !important; color: var(--fg) !important; border:1px solid var(--line) !important; border-radius:6px !important; }
.stTextInput input:focus, .stTextArea textarea:focus { border-color: var(--accent) !important; box-shadow: 0 0 0 3px rgba(35,131,226,.18) !important; }
[data-testid="stFileUploaderDropzone"] { background: var(--side); border:1px dashed var(--line); border-radius:8px; }
[data-testid="stImage"] img { border-radius: 8px; border:1px solid var(--line); }

/* 버튼 */
.stButton > button, .stDownloadButton > button { border-radius:6px; border:1px solid var(--line); background: var(--bg); color: var(--fg);
  font-weight:500; padding:.4rem 1rem; transition: background .12s; }
.stButton > button:hover, .stDownloadButton > button:hover { background: var(--hover); border-color: var(--line); color: var(--fg); }
.stButton > button[kind="primary"] { background: var(--accent); border-color: var(--accent); color:#fff; }
.stButton > button[kind="primary"]:hover { filter: brightness(1.08); background: var(--accent); color:#fff; }
.stButton > button:disabled { opacity:.45; }

/* 지표 / 탭 / 코드 / 표 */
[data-testid="stMetric"] { background: var(--card); border:1px solid var(--line); border-radius:8px; padding:.8rem 1rem; }
[data-testid="stMetricLabel"] { color: var(--muted); }
[data-testid="stMetricValue"] { font-weight:700; }
[data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--line); }
[data-baseweb="tab"] { background: transparent !important; color: var(--muted) !important; }
[aria-selected="true"][data-baseweb="tab"] { color: var(--fg) !important; font-weight:600; }
[data-testid="stCode"], pre { background: var(--code) !important; border:1px solid var(--line); border-radius:6px; }
[data-testid="stAlert"] { border-radius:6px; border:none; }
[data-testid="stExpander"] { border:1px solid var(--line); border-radius:8px; }
hr { border-color: var(--line); }
</style>""", unsafe_allow_html=True)


# =========================================================================
# [민서 전달 함수 원본] — analyze_road_damage_clova (HyperCLOVA X / HCX-005)
# =========================================================================

def analyze_road_damage_clova(image_path, api_key):
    """
    도로 사진을 넣으면 파손 유형과 심각도를 반환하는 메인 함수
    (최종 선정 메인 모델: HyperCLOVA X / HCX-005)

    출력: {"유형": "균열" 또는 "포트홀", "심각도": "경미"/"보통"/"심각"/"긴급"}
    """
    client = OpenAI(
        api_key=api_key,
        base_url="https://clovastudio.stream.ntruss.com/v1/openai"
    )

    PROMPT = """이 사진 속 도로 파손을 분석해줘. 아래 형식으로만 정확히 답해줘 (다른 말 붙이지 말고):

유형: 균열 또는 포트홀 (둘 중 하나만)

심각도: 아래 기준을 참고해서 판단해줘
- 경미: 표면에 실금 수준이거나 아주 작은 파임, 즉각적인 위험 없음
- 보통: 육안으로 뚜렷이 보이는 파손, 주의가 필요한 수준
- 심각: 파손 범위가 넓거나 깊어 통행에 지장을 줄 수 있는 수준
- 긴급: 깊은 구멍, 넓은 함몰, 도로 일부 붕괴 등 사고 위험이 임박한 수준

반드시 아래 형식 그대로만 답하고, 기준 설명은 다시 쓰지 마:
유형: [답]
심각도: [답]"""

    with open(image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode("utf-8")

    response = client.chat.completions.create(
        model="HCX-005",
        max_tokens=100,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        }]
    )

    answer = response.choices[0].message.content
    type_match = re.search(r"유형[:\s]*(균열|포트홀|해당 없음)", answer)
    severity_match = re.search(r"심각도[:\s]*(경미|보통|심각|긴급|해당 없음)", answer)

    return {
        "유형": type_match.group(1) if type_match else "인식 실패",
        "심각도": severity_match.group(1) if severity_match else "인식 실패"
    }


SEVERITY_TO_SCORE = {"경미": 1, "보통": 2, "심각": 3, "긴급": 4}


# =========================================================================
# [주혜 전달 함수] — 사진 1장 실시간 파손비율 계산 (10/29 수정본)
# COCO 스타일 annotation(dict)을 넣으면 균열/포트홀 면적, 파손비율(%)을 반환.
# 수정 이유: 균열 라벨 [x1,y1,x2,y2]는 '선'인데 기존 코드가 사각형 [x,y,w,h]로 읽어
#            균열 면적이 부풀려졌음 -> 균열은 선을 폭 12px 띠로 보고 계산하도록 교체.
# =========================================================================

CRACK_WIDTH_PX = 12  # 균열 선 폭(픽셀). 8~16으로 바꿔도 결과가 거의 같음 (주혜 확인)


def calculate_crack_area(crack_segs, width=CRACK_WIDTH_PX):
    shapes = []
    for s in crack_segs:
        if not isinstance(s, list) or len(s) < 4:
            continue
        pts = list(zip(s[0::2], s[1::2]))
        line = LineString(pts)
        if line.length > 0:
            shapes.append(line.buffer(width / 2))
    return unary_union(shapes).area if shapes else 0.0


def calculate_polygon_area(segmentation):
    area = 0.0
    for poly in segmentation:
        if len(poly) < 6:
            continue
        x = poly[0::2]
        y = poly[1::2]
        area += 0.5 * abs(
            sum(
                x[i] * y[(i + 1) % len(y)] - x[(i + 1) % len(x)] * y[i]
                for i in range(len(x))
            )
        )
    return area


def get_area_from_seg_like_list(val):
    total = 0.0
    for sub in val:
        if not isinstance(sub, list):
            continue
        if len(sub) == 4:
            total += abs(sub[2] * sub[3])
        elif len(sub) >= 6:
            total += calculate_polygon_area([sub])
    return total


def get_robust_area(ann):
    if ann.get("area"):
        return float(ann["area"])

    max_area = 0
    for key, val in ann.items():
        if isinstance(val, list):
            if (
                "bbox" in key.lower()
                and len(val) == 4
                and all(isinstance(v, (int, float)) for v in val)
            ):
                area = val[2] * val[3]
                if area > max_area:
                    max_area = area
            elif len(val) > 0 and isinstance(val[0], list):
                seg_area = get_area_from_seg_like_list(val)
                if seg_area > max_area:
                    max_area = seg_area
            elif len(val) > 4 and all(isinstance(v, (int, float)) for v in val):
                poly_area = calculate_polygon_area([val])
                if poly_area > max_area:
                    max_area = poly_area
    return max_area


def crack_shapes(crack_segs, width=CRACK_WIDTH_PX):
    """균열 선 라벨 -> 폭 width 띠 도형 목록 (합집합 계산용)."""
    shapes = []
    for s in crack_segs:
        if not isinstance(s, list) or len(s) < 4:
            continue
        line = LineString(list(zip(s[0::2], s[1::2])))
        if line.length > 0:
            shapes.append(line.buffer(width / 2))
    return shapes


def pothole_shapes(ann):
    """
    포트홀 라벨 1개 -> 도형 목록 (합집합 계산용).
    segmentation 안의 [x,y,w,h] 사각형 / 다각형, 또는 bbox [x,y,w,h]를 도형으로 바꾼다.
    좌표가 하나도 없으면 빈 목록 (이때는 area 숫자만 따로 더함).
    """
    shapes = []
    for sub in ann.get("segmentation") or []:
        if not isinstance(sub, list):
            continue
        if len(sub) == 4:
            x, y, w, h = sub
            shapes.append(Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)]))
        elif len(sub) >= 6:
            shapes.append(Polygon(list(zip(sub[0::2], sub[1::2]))).buffer(0))
    if not shapes:
        bbox = ann.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox):
            x, y, w, h = bbox
            shapes.append(Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)]))
    return [s for s in shapes if not s.is_empty and s.area > 0]


def calculate_single_image_damage_ratio(data):
    """
    사진 1장의 JSON 형태 분석 결과 -> 균열/포트홀 면적, 총 파손면적, 파손비율(%), 면적 기준 유형.
    [11/9 수정] 균열·포트홀 라벨을 모두 도형으로 바꿔 '합집합(union)'으로 면적을 잰다.
      - 포트홀끼리 겹친 부분, 균열과 포트홀이 겹친 부분을 한 번만 계산 (YOLO 쪽과 같은 방식).
      - 균열면적/포트홀면적은 각 종류 안에서의 합집합, 총파손면적은 전체 합집합.
      - 좌표 없이 area 숫자만 있는 라벨은 겹침을 알 수 없어 그 값만 더함.
    """
    cat_map = {c["id"]: c["name"] for c in data["categories"]}

    crack_segs = []
    pothole_geoms = []
    pothole_area_only = 0.0  # 좌표 없는 포트홀 라벨의 area 값

    for ann in data["annotations"]:
        category_id = ann.get("category_id")
        cat_name = unicodedata.normalize("NFC", str(cat_map.get(category_id, "")))
        if "균열" in cat_name:
            crack_segs += ann.get("segmentation") or []
        elif "홀" in cat_name:
            geoms = pothole_shapes(ann)
            if geoms:
                pothole_geoms += geoms
            else:
                pothole_area_only += get_robust_area(ann)

    crack_geoms = crack_shapes(crack_segs)
    crack_area = unary_union(crack_geoms).area if crack_geoms else 0.0
    pothole_area = (unary_union(pothole_geoms).area if pothole_geoms else 0.0) + pothole_area_only

    img_width = float(data["images"][0]["width"])
    img_height = float(data["images"][0]["height"])
    image_total_area = img_width * img_height
    if image_total_area <= 0:
        raise ValueError("이미지 가로/세로 크기가 잘못되었습니다.")

    # 균열 + 포트홀 전체를 하나로 합친 뒤 넓이 (겹친 부분 한 번만)
    all_geoms = crack_geoms + pothole_geoms
    total_damage_area = (unary_union(all_geoms).area if all_geoms else 0.0) + pothole_area_only
    damage_ratio = total_damage_area / image_total_area * 100

    if crack_area == 0 and pothole_area == 0:
        final_type = "확인필요"
    elif crack_area >= pothole_area:
        final_type = "균열"
    else:
        final_type = "포트홀"

    return {
        "정답유형(면적기준)": final_type,
        "이미지전체면적": image_total_area,
        "균열면적": crack_area,
        "포트홀면적": pothole_area,
        "총파손면적": total_damage_area,
        "파손비율(%)": damage_ratio,
    }


# =========================================================================
# [민서 YOLO 연동] damage_detecotor.ipynb 의 detect_damage()
# 사진에서 균열/포트홀 위치를 찾아 파손 면적 비율을 계산한다.
# best.pt 를 app.py 와 같은 폴더에 두면 자동 사용 (최종 모델이 나오면 파일만 교체).
# =========================================================================

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.pt")
YOLO_NAMES = {0: "균열", 1: "포트홀"}
_yolo_model = None


def yolo_available():
    return YOLO is not None and os.path.exists(MODEL_PATH)


def _get_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO(MODEL_PATH)
    return _yolo_model


def _poly_area(xy):
    x, y = xy[:, 0], xy[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def detect_damage(image_path, conf=0.25):
    """
    (검출 목록, 검출 결과가 그려진 RGB 이미지, 전체 파손 비율) 반환.
    검출 목록의 면적비율 = 검출 1개의 마스크 면적 / 사진 전체 면적(%).
    전체 파손 비율 = 모든 마스크를 먼저 하나로 합친(union) 뒤 넓이를 잰 값.
    -> 마스크끼리 겹친 부분을 한 번만 계산한다 (민서 피드백 반영).
    """
    r = _get_yolo_model().predict(image_path, imgsz=1280, conf=conf, verbose=False)[0]
    h, w = r.orig_shape
    out = []
    polys = []
    for i, box in enumerate(r.boxes):
        area = 0.0
        if r.masks is not None:
            area = _poly_area(r.masks.xy[i]) / (w * h) * 100
            pts = r.masks.xy[i]
            if len(pts) >= 3:
                polys.append(Polygon(pts).buffer(0))  # buffer(0): 꼬인 다각형 보정
        out.append({
            "종류": YOLO_NAMES[int(box.cls[0])],
            "신뢰도": round(float(box.conf[0]), 3),
            "박스": [round(float(v), 1) for v in box.xyxy[0]],
            "면적비율(%)": round(float(area), 2),
        })
    annotated = r.plot()[..., ::-1]  # BGR -> RGB
    union_ratio = unary_union(polys).area / (w * h) * 100 if polys else 0.0
    return out, annotated, union_ratio


def run_yolo(image_file):
    """업로드 파일에 YOLO 적용. 실패하면 {'error': ...} (임시 좌표 방식으로 대체)."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(image_file.getbuffer())
            tmp_path = tmp.name
        dets, annotated, union_ratio = detect_damage(tmp_path)
        ratio = round(min(union_ratio, 100.0), 2)
        return {"detections": dets, "annotated": annotated, "ratio": ratio}
    except Exception as e:
        return {"error": str(e)}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# =========================================================================
# [연동 지점]
# =========================================================================

def call_vlm(image_file, api_key):
    """
    [민서] 연동 완료 (analyze_road_damage_clova 사용).
    API 키가 없거나 라이브러리가 없거나 호출이 실패하면 가짜 데이터로 대체해서
    나머지 화면 흐름은 계속 확인할 수 있게 함.
    """
    if not image_file or not api_key or OpenAI is None:
        return _fake_vlm_result(reason="API 키 미입력 (또는 openai 패키지 없음) - 가짜 데이터")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(image_file.getbuffer())
            tmp_path = tmp.name
        result = analyze_road_damage_clova(tmp_path, api_key)
        if result["유형"] not in ("균열", "포트홀"):
            return _fake_vlm_result(reason=f"인식 실패 응답({result}) - 가짜 데이터로 대체")
        return {
            "위험유형": result["유형"],
            "ai_심각도": SEVERITY_TO_SCORE.get(result["심각도"], 0),
            "is_mock": False,
        }
    except Exception as e:
        return _fake_vlm_result(reason=f"호출 오류({e}) - 가짜 데이터로 대체")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def _fake_vlm_result(reason=""):
    fake_type = random.choice(["포트홀", "균열"])
    return {
        "위험유형": fake_type,
        "ai_심각도": random.randint(1, 4),
        "is_mock": True,
        "mock_reason": reason,
    }


def _generate_fake_annotation(risk_type, photo_size=(1920, 1080)):
    """
    [임시] YOLO 모델(best.pt)이 없을 때만 사용. 가짜 좌표를 만들어 주혜의 실제 계산 로직에 넣어 시연.
    균열은 '선'(점 2개), 포트홀은 [x, y, w, h] 사각형.
    """
    width, height = photo_size
    if risk_type == "균열":
        x1, y1 = random.uniform(0, width), random.uniform(0, height)
        x2 = min(max(x1 + random.uniform(-0.4, 0.4) * width, 0), width)
        y2 = min(max(y1 + random.uniform(-0.4, 0.4) * height, 0), height)
        seg, cat_id = [[x1, y1, x2, y2]], 1
    else:
        box_w = width * random.uniform(0.03, 0.25)
        box_h = height * random.uniform(0.03, 0.25)
        x = random.uniform(0, max(width - box_w, 1))
        y = random.uniform(0, max(height - box_h, 1))
        seg, cat_id = [[x, y, box_w, box_h]], 2
    return {
        "categories": [{"id": 1, "name": "도로균열"}, {"id": 2, "name": "도로(홀)"}],
        "images": [{"width": width, "height": height}],
        "annotations": [{
            "segmentation": seg,
            "area": None,
            "bbox": None,
            "category_id": cat_id,
        }],
    }


def calc_area_ratio(risk_type, photo_size=(1920, 1080)):
    """YOLO 모델이 없을 때의 임시 계산 (가짜 좌표 + 주혜 실제 로직)."""
    fake_data = _generate_fake_annotation(risk_type, photo_size)
    result = calculate_single_image_damage_ratio(fake_data)
    return round(result["파손비율(%)"], 2)


def score_against_answer(pred_type, answer_type, pred_area_ratio, answer_ai_severity):
    """
    TODO(윤수): 채점 함수로 교체 (파손유형 O/X, 면적비율-AI심각도 상관계수)
    서비스 메인 화면에는 직접 쓰이지 않지만 팀 평가 리포트용으로 자리만 표시해둠.
    """
    return None


# =========================================================================
# 등급 판정 & 민원문
# =========================================================================

def determine_grade(area_ratio):
    """
    면적비율(사진 전체 면적 기준 %) -> 최종 등급(일반/보통/긴급) 판정
    [민서 반영] 3단계 체계: 일반 <2% / 보통 2~5% / 긴급 >=5%
    300장 실데이터 기준 등급 분포 246/37/17장 (파손 비율 최댓값 10.54%).
    """
    if area_ratio >= 5.0:
        return "긴급"
    elif area_ratio >= 2.0:
        return "보통"
    else:
        return "일반"


GRADE_ORDER = ["일반", "보통", "긴급"]


def apply_zone_bump(  # (현재 미사용: 등급 상향은 이후 과제)
        base_grade, near_sensitive_zone):
    """
    [제안 — 팀 결정 대기] 사용자가 '어린이 보호구역 등 보행 취약 구역 인근'을 직접 선택한 경우에만
    등급을 한 단계 올린다 (최대 긴급). 자동 판정이 아니라 사용자 선택 기반.
    """
    if not near_sensitive_zone:
        return base_grade
    idx = GRADE_ORDER.index(base_grade)
    return GRADE_ORDER[min(idx + 1, len(GRADE_ORDER) - 1)]


REPORT_SYSTEM_PROMPT = """너는 도로 파손을 안전신문고에 신고하려는 시민을 돕는 민원문 작성 도우미야.
아래 [입력 정보]만 사용해서 한국어 민원문을 요구, 현황, 위험, 조치 요청 순서로 정확히 4문장으로 써.
한 문장을 한 줄에 쓰고(총 4줄), 위험 문장과 조치 요청 문장을 한 문장으로 합치지 마. 감사 인사는 붙이지 마.

문장 구성:
1. 요구: 위치와 위험 유형을 넣어 보수를 요청드린다고 먼저 밝혀.
2. 현황: '첨부한 사진 기준으로' 파손 면적이 사진 전체의 약 몇 %인지만 써.
3. 위험: 위험 정도를 판단하지 말고 가능성만 써. 포트홀은 '차량 파손이나 사고로 이어질 우려가 있습니다.', 균열은 '방치할 경우 파손이 확대될 우려가 있습니다.'로 써.
4. 조치 요청: 등급에 맞춰 시급성을 정해. 긴급 '빠른 시일 내에', 보통 '가능한 시일 내에', 일반 '정기 점검 시'.

규칙:
- 입력 정보에 없는 사실(사고 이력, 파손 원인, 모양·길이·깊이, 발생 시기)은 절대 쓰지 마.
- 어린이 보호구역 인근이 '예'이면 현황 다음 줄에 '해당 위치는 어린이 보호구역 인근입니다.' 한 문장만 추가해. (이때만 총 5줄)
- '크다', '심각하다', '매우', '빠르게', '큰 지장' 같은 위험 정도를 판단하는 표현은 쓰지 마. 시급성은 조치 요청 문장의 '빠른 시일 내에' 같은 표현으로만 나타내.
- 존댓말을 쓰고, 민원문 본문만 출력해. 제목, 번호, 설명, 따옴표는 붙이지 마.

예시 1 (포트홀, 긴급, 약 10%):
OO동 123 앞 도로의 포트홀 보수를 요청드립니다.
첨부한 사진 기준으로 포트홀 파손 면적이 사진 전체의 약 10%를 차지합니다.
차량 파손이나 사고로 이어질 우려가 있습니다.
빠른 시일 내에 현장을 확인하시고 보수 조치를 해주시기 바랍니다.

예시 2 (균열, 보통, 약 3%):
OO동 45 앞 도로의 균열 점검과 보수를 요청드립니다.
첨부한 사진 기준으로 균열이 사진 전체의 약 3%를 차지합니다.
방치할 경우 파손이 확대될 우려가 있습니다.
가능한 시일 내에 현장을 확인하시고 보수해 주시기 바랍니다."""


def _fmt_ratio(x):
    """10.54 -> '10.5', 3.0 -> '3' (민원문에 쓰는 비율 표기)"""
    t = f"{round(float(x), 1):.1f}"
    return t[:-2] if t.endswith(".0") else t


def report_sentences(location, risk_type, area_ratio, grade, near_zone=False):
    """
    민원문 문장 리스트: 요구 -> 현황 -> (어린이 보호구역 인근이면 한 문장 추가) -> 위험 -> 조치 요청.
    어린이 보호구역은 등급을 바꾸지 않고 문장만 한 줄 더한다.
    """
    pct = _fmt_ratio(area_ratio)
    # 위험 정도를 판단하는 표현(큽니다·심각·빠르게 등)은 쓰지 않고, 가능성만 서술한다 (민서 피드백).
    RISK = {"포트홀": "차량 파손이나 사고로 이어질 우려가 있습니다.",
            "균열": "방치할 경우 파손이 확대될 우려가 있습니다."}
    s3 = RISK.get(risk_type, "방치할 경우 파손이 확대될 우려가 있습니다.")
    if risk_type == "포트홀":
        s1 = f"{location}의 포트홀 보수를 요청드립니다."
        s2 = f"첨부한 사진 기준으로 포트홀 파손 면적이 사진 전체의 약 {pct}%를 차지합니다."
    else:
        s1 = f"{location}의 균열 점검과 보수를 요청드립니다."
        s2 = f"첨부한 사진 기준으로 균열이 사진 전체의 약 {pct}%를 차지합니다."
    zone = ["해당 위치는 어린이 보호구역 인근입니다."] if near_zone else []
    s4 = {"긴급": "빠른 시일 내에 현장을 확인하시고 보수 조치를 해주시기 바랍니다.",
          "보통": "가능한 시일 내에 현장을 확인하시고 보수해 주시기 바랍니다.",
          "일반": "정기 점검 시 현장을 확인하시고 보수해 주시기 바랍니다."}.get(
        grade, "현장을 확인하시고 보수해 주시기 바랍니다.")
    return [s1, s2, *zone, s3, s4]


def generate_report_text(location, risk_type, area_ratio, grade, near_sensitive_zone=False, detections=None):
    """민원문 템플릿 버전 (AI 호출이 실패하거나 키가 없을 때 대신 사용 -> 앱이 멈추지 않음)."""
    return " ".join(report_sentences(location, risk_type, area_ratio, grade, near_sensitive_zone))


_BAD_WORDS = ("큽니다", "심각", "매우", "큰 지장", "빠르게", "위험합니다")


def _split_sentences(text):
    """AI 답변을 문장 리스트로 나눈다. 줄 단위 우선, 한 줄에 여러 문장이면 마침표 기준으로 나눈다."""
    out = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[-•*]|\d+[.)])\s*", "", line.strip())
        if not line:
            continue
        out += [p.strip() for p in re.split(r"(?<=[.!?])\s+", line) if p.strip()]
    return out


def generate_report_llm(api_key, location, risk_type, area_ratio, grade, near_sensitive_zone=False, detections=None):
    """
    HyperCLOVA X(HCX-005)로 민원문 생성. 입력 5가지: 위치, 위험 유형, 파손 면적 비율, 등급, 어린이 보호구역 인근 여부.
    답변이 정해진 문장 수(4문장, 보호구역이면 5문장)가 아니거나 금지 표현이 들어 있으면 한 번 더 요청하고,
    그래도 안 맞으면 None을 반환한다 (-> 호출한 쪽이 템플릿으로 대체해 앱이 멈추지 않음).
    """
    if not api_key or OpenAI is None:
        return None
    want = 4 + (1 if near_sensitive_zone else 0)
    user_msg = (
        "[입력 정보]\n"
        f"- 위치: {location}\n"
        f"- 위험 유형: {risk_type}\n"
        f"- 파손 면적 비율: 사진 전체의 약 {_fmt_ratio(area_ratio)}%\n"
        f"- 등급: {grade}\n"
        f"- 어린이 보호구역 인근: {'예' if near_sensitive_zone else '아니오'}\n\n"
        f"총 {want}줄로, 한 줄에 한 문장씩 써 주세요."
    )
    try:
        client = OpenAI(api_key=api_key, base_url="https://clovastudio.stream.ntruss.com/v1/openai")
        for _ in range(2):
            resp = client.chat.completions.create(
                model="HCX-005",
                max_tokens=400,
                messages=[
                    {"role": "system", "content": REPORT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
            )
            text = (resp.choices[0].message.content or "").strip()
            sents = _split_sentences(text)
            if len(sents) == want and not any(w in text for w in _BAD_WORDS):
                return " ".join(sents)
    except Exception:
        return None
    return None


# =========================================================================
# 민원 양식 (윤수 제안 구조: 제목 / 요청 사항 / 현황 / 위험성 / 조치 요청)
# =========================================================================

def build_title(location, risk_type, grade):
    return f"[도로 보수 요청] {location} {risk_type} 파손 건"


def build_structured_report(location, risk_type, area_ratio, grade, near_zone, detections):
    """안전신문고 상황 설명 칸용 구조화 양식 (민원문과 같은 문장을 항목별로 정리)."""
    sents = report_sentences(location, risk_type, area_ratio, grade, near_zone)
    s1, s2, s4 = sents[0], sents[1], sents[-1]
    s3 = " ".join(sents[2:-1])
    return "\n".join([
        build_title(location, risk_type, grade), "",
        "1. 요청 사항", f"- {s1}", "",
        "2. 현황", f"- {s2}", "",
        "3. 위험성", f"- {s3}", "",
        "4. 조치 요청", f"- {s4}",
        "- 보수 완료 예정일 또는 향후 조치 계획을 회신해 주시기 바랍니다.", "",
        "(첨부: 현장 사진)",
    ])


# =========================================================================
# 화면 도우미 (노션 스타일 블록)
# =========================================================================

import html as _html
import datetime as _dt

GRADE_TAG = {"긴급": "red", "보통": "orange", "일반": "green"}


def tag(text, color="gray"):
    return f'<span class="tag {color}">{_html.escape(str(text))}</span>'


def props_block(rows):
    items = "".join(
        f'<div class="prop"><span class="pk">{k}</span><span class="pv">{v}</span></div>' for k, v in rows
    )
    st.markdown(f'<div class="props">{items}</div>', unsafe_allow_html=True)


def callout(icon, text, warn=False):
    st.markdown(
        f'<div class="callout{" warn" if warn else ""}"><span class="ci">{icon}</span><div>{text}</div></div>',
        unsafe_allow_html=True,
    )


def gauge(ratio):
    """0~10% 눈금 위에 현재 파손 비율을 표시. 2%/5% 경계로 일반·보통·긴급 구간을 색으로 나눈다."""
    pos = max(0.0, min(ratio / 10.0, 1.0)) * 100
    st.markdown(
        f'<div class="gauge"><div class="gtrack"><div class="gz l"></div><div class="gz m"></div>'
        f'<div class="gz h"></div><div class="gneedle" style="left:calc({pos}% - 2px)"></div></div>'
        f'<div class="glabels"><span>일반 (2% 미만)</span><span>보통 (2~5%)</span><span>긴급 (5% 이상)</span></div></div>',
        unsafe_allow_html=True,
    )


def page_header(icon, title, crumb):
    st.markdown(
        f'<div class="page-icon">{icon}</div><div class="page-title">{title}</div>'
        f'<div class="crumb">{crumb}</div>',
        unsafe_allow_html=True,
    )


STEP_NAMES = {1: "사진 업로드", 2: "위험요소 인식", 3: "등급 판정", 4: "민원문"}


def steps_block(cur):
    chips = "".join(
        f'<span class="step {"on" if i == cur else "done" if i < cur else ""}">{"✓ " if i < cur else ""}{i}. {n}</span>'
        for i, n in STEP_NAMES.items()
    )
    st.markdown(f'<div class="steps">{chips}</div>', unsafe_allow_html=True)


# =========================================================================
# 세션 상태 & 사이드바
# =========================================================================

if "step" not in st.session_state:
    st.session_state.step = 1
if "result" not in st.session_state:
    st.session_state.result = {}
if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.markdown('<div class="brand"><span class="b-ic">🛣️</span>도로 위험 신고</div>', unsafe_allow_html=True)
    page = st.radio("메뉴", ["🆕  새 신고", "🗂  신고 기록", "📖  사용 방법"], label_visibility="collapsed")
    st.markdown('<div class="side-h">상태</div>', unsafe_allow_html=True)
    _api_now = st.session_state.get("api_key_input", "")
    st.markdown(
        tag("AI 연결됨" if _api_now else "AI 가짜 데이터", "green" if _api_now else "gray")
        + tag("YOLO 사용 중" if yolo_available() else "YOLO 없음", "blue" if yolo_available() else "gray"),
        unsafe_allow_html=True,
    )
    st.markdown('<div class="side-h">설정</div>', unsafe_allow_html=True)
    with st.expander("API · 화면", expanded=False):
        st.text_input("CLOVA Studio API 키", type="password", key="api_key_input",
                      help="비워두면 실제 AI 호출 대신 가짜 인식 결과로 화면 흐름만 확인합니다.")
        dark = st.toggle("다크 모드", key="dark_mode")
    st.markdown('<div class="side-h">등급 기준</div>', unsafe_allow_html=True)
    st.caption("일반 < 2% · 보통 2~5% · 긴급 ≥ 5%\n사진 전체 면적 대비 파손 면적")

inject_css(st.session_state.get("dark_mode", False))


# =========================================================================
# 페이지: 새 신고 (4단계)
# =========================================================================

def run_new_report():
    step = st.session_state.step
    page_header("📸" if step == 1 else "🔍" if step == 2 else "📊" if step == 3 else "📝",
                "새 도로 위험 신고", f"신고 / 새 신고 / {STEP_NAMES[step]}")
    steps_block(step)

    # ---------------- STEP 1: 업로드 ----------------
    if step == 1:
        location = st.text_input("📍 위치", value="OO동 123 앞 도로",
                                 help="예: OO동 123 앞 도로. 민원문 첫 문장에 들어갑니다.")
        uploaded = st.file_uploader("도로/보도 사진을 업로드하세요", type=["jpg", "jpeg", "png"])
        if uploaded:
            st.image(uploaded, caption="업로드된 사진", width="stretch")
        c1, c2 = st.columns([1, 1])
        if c1.button("분석하기 →", disabled=uploaded is None, type="primary"):
            st.session_state.result = {"location": location, "image": uploaded,
                                       "api_key": st.session_state.get("api_key_input", "")}
            st.session_state.step = 2
            st.rerun()
        if c2.button("사진 없이 체험하기"):
            st.session_state.result = {"location": location, "image": None,
                                       "api_key": st.session_state.get("api_key_input", "")}
            st.session_state.step = 2
            st.rerun()
        callout("💡", "사진이 없어도 <b>체험하기</b>로 화면 흐름을 확인할 수 있어요. "
                      "이때 인식 결과와 파손 비율은 임시(가짜) 값입니다.")

    # ---------------- STEP 2: 위험요소 인식 ----------------
    elif step == 2:
        if "vlm" not in st.session_state.result:
            with st.spinner("AI가 사진을 분석하는 중..."):
                st.session_state.result["vlm"] = call_vlm(
                    st.session_state.result.get("image"), st.session_state.result.get("api_key"))
        vlm = st.session_state.result["vlm"]

        if "yolo" not in st.session_state.result:
            img = st.session_state.result.get("image")
            if img is not None and yolo_available():
                with st.spinner("파손 위치를 찾는 중..."):
                    st.session_state.result["yolo"] = run_yolo(img)
            else:
                st.session_state.result["yolo"] = None
        yres = st.session_state.result["yolo"]

        if vlm.get("is_mock"):
            callout("⚠️", f"실제 AI 응답 대신 가짜 데이터를 표시하고 있습니다. ({_html.escape(vlm.get('mock_reason', ''))})", warn=True)

        props_block([
            ("📍 위치", _html.escape(st.session_state.result["location"])),
            ("⚠️ 위험 유형", tag(vlm["위험유형"], "blue")),
            ("🤖 AI 심각도", tag(f"{vlm['ai_심각도']}단계", "gray") + ' <span style="color:var(--muted);font-size:.8rem">참고용 · 등급 산정에는 사용하지 않음</span>'),
        ])

        if yres and "annotated" in yres:
            st.image(yres["annotated"], caption="파손 위치 검출 결과", width="stretch")
            if yres["detections"]:
                st.dataframe(
                    [{"종류": d["종류"], "신뢰도": d["신뢰도"], "개별 면적(%)": d["면적비율(%)"]} for d in yres["detections"]],
                    width="stretch", hide_index=True,
                )
            else:
                callout("ℹ️", "검출된 파손이 없습니다. 파손 비율 0%로 처리됩니다.")
        elif yres and "error" in yres:
            callout("⚠️", f"위치 검출에 실패해 임시 좌표 방식으로 진행합니다. ({_html.escape(yres['error'])})", warn=True)

        nc1, nc2 = st.columns(2)
        if nc1.button("← 다시 업로드"):
            st.session_state.step = 1
            st.rerun()
        if nc2.button("다음 →", type="primary"):
            st.session_state.step = 3
            st.rerun()

    # ---------------- STEP 3: 등급 판정 ----------------
    elif step == 3:
        vlm = st.session_state.result["vlm"]
        if "area_ratio" not in st.session_state.result:
            _y = st.session_state.result.get("yolo")
            if _y and "ratio" in _y:
                st.session_state.result["area_ratio"] = _y["ratio"]
                st.session_state.result["area_is_real"] = True
            else:
                st.session_state.result["area_ratio"] = calc_area_ratio(vlm["위험유형"])
                st.session_state.result["area_is_real"] = False
        area_ratio = st.session_state.result["area_ratio"]
        base_grade = determine_grade(area_ratio)

        near_zone = st.toggle("🚸 어린이 보호구역 인근입니다 (등급은 그대로, 민원문에 한 문장만 추가됩니다)",
                              value=st.session_state.result.get("near_zone", False))
        st.session_state.result["near_zone"] = near_zone
        grade = base_grade
        st.session_state.result["base_grade"] = base_grade
        st.session_state.result["grade"] = grade

        props_block([
            ("📐 파손 면적 비율", f"<b>{area_ratio}%</b> <span style='color:var(--muted);font-size:.8rem'>사진 전체 면적 기준</span>"),
            ("🚦 우선순위 등급", tag(grade, GRADE_TAG[grade])),
        ])
        gauge(area_ratio)

        if st.session_state.result.get("area_is_real"):
            callout("✅", "YOLO가 찾은 파손 영역을 <b>하나로 합친 뒤</b>(겹친 부분은 한 번만) 사진 전체 면적으로 나눈 값입니다.")
        else:
            callout("🧪", "YOLO 모델(best.pt)이 없거나 사진이 없어 <b>임시 좌표</b>로 계산한 값입니다. "
                          "best.pt를 app.py 옆에 두고 사진을 올리면 실제 검출 결과를 사용합니다.", warn=True)

        nc1, nc2 = st.columns(2)
        if nc1.button("← 인식 결과로"):
            st.session_state.step = 2
            st.rerun()
        if nc2.button("다음 →", type="primary"):
            st.session_state.step = 4
            st.rerun()

    # ---------------- STEP 4: 민원문 ----------------
    elif step == 4:
        r = st.session_state.result
        _dets = (r.get("yolo") or {}).get("detections")
        _key = (r["location"], r["vlm"]["위험유형"], r["area_ratio"], r["grade"], r.get("near_zone", False))
        if r.get("report_key") != _key:
            with st.spinner("민원문을 작성하는 중..."):
                _llm = generate_report_llm(r.get("api_key"), *_key, detections=_dets)
            r["report_text"] = _llm or generate_report_text(*_key, detections=_dets)
            r["report_by_llm"] = bool(_llm)
            r["report_key"] = _key
            r["saved"] = False
        title = build_title(r["location"], r["vlm"]["위험유형"], r["grade"])
        structured = build_structured_report(*_key, _dets)

        props_block([
            ("📝 제목", _html.escape(title)),
            ("🚦 등급", tag(r["grade"], GRADE_TAG[r["grade"]])),
            ("✍️ 작성 방식", tag("HyperCLOVA X", "blue") if r.get("report_by_llm") else tag("템플릿", "gray")),
        ])

        t1, t2, t3, t4 = st.tabs(["📝 민원문", "📋 안전신문고 양식", "🔍 분석 근거", "🧠 프롬프트"])
        with t1:
            st.caption("수정해서 쓰세요. 아래 상자 오른쪽 위 복사 버튼으로 한 번에 복사할 수 있어요.")
            edited = st.text_area("민원문 편집", r["report_text"], height=150, label_visibility="collapsed")
            r["report_text"] = edited
            st.code(edited, language=None, wrap_lines=True)
            st.download_button("⬇️ 텍스트로 저장", edited, file_name="민원문.txt")
        with t2:
            st.caption("제목·요청 사항·현황·위험성·조치 요청 순서의 양식입니다. 발생 시기·치수·사고 이력 등은 직접 확인해 덧붙이세요.")
            st.code(structured, language=None, wrap_lines=True)
            st.download_button("⬇️ 양식 저장", structured, file_name="민원양식.txt")
        with t3:
            _src = "YOLO 검출 결과" if r.get("area_is_real") else "임시 좌표(가짜) 계산"
            st.markdown(
                f"- 위험 유형: **{r['vlm']['위험유형']}** (HyperCLOVA X{' · 가짜 데이터' if r['vlm'].get('is_mock') else ''})\n"
                f"- 파손 면적 비율: **{r['area_ratio']}%** ({_src})\n"
                f"- 등급: **{r['grade']}** (면적 비율로만 결정)\n"
                f"- 보행 취약 구역: {'예 (민원문에 한 문장 추가, 등급 그대로)' if r.get('near_zone') else '아니오'}"
            )
            gauge(r["area_ratio"])
        with t4:
            st.caption("API 키가 있으면 HyperCLOVA X에 아래 지침을 보냅니다. (프롬프트 문서와 동일)")
            st.code(REPORT_SYSTEM_PROMPT, language=None, wrap_lines=True)

        nc1, nc2, nc3 = st.columns(3)
        if nc1.button("← 등급으로"):
            st.session_state.step = 3
            st.rerun()
        if nc2.button("💾 기록에 저장", disabled=r.get("saved", False), type="primary"):
            st.session_state.history.insert(0, {
                "시각": _dt.datetime.now().strftime("%m/%d %H:%M"),
                "위치": r["location"], "유형": r["vlm"]["위험유형"],
                "면적(%)": r["area_ratio"], "등급": r["grade"],
                "제목": title, "민원문": r["report_text"],
            })
            r["saved"] = True
            st.toast("기록에 저장했어요 ✅")
            st.rerun()
        if nc3.button("＋ 새 신고"):
            st.session_state.step = 1
            st.session_state.result = {}
            st.rerun()
        if r.get("saved"):
            st.caption("✅ 이 신고는 '신고 기록'에 저장되었습니다.")


# =========================================================================
# 페이지: 신고 기록
# =========================================================================

def run_history():
    page_header("🗂️", "신고 기록", "신고 / 기록")
    hist = st.session_state.history
    if not hist:
        callout("📭", "아직 저장한 신고가 없어요. 새 신고를 마치고 <b>기록에 저장</b>을 눌러 보세요.")
        return
    c1, c2 = st.columns([2, 3])
    pick = c1.multiselect("등급 필터", ["긴급", "보통", "일반"], default=["긴급", "보통", "일반"])
    q = c2.text_input("검색", placeholder="위치·제목으로 검색")
    rows = [h for h in hist if h["등급"] in pick and (q in h["위치"] or q in h["제목"])]
    n = {g: sum(1 for h in hist if h["등급"] == g) for g in ["긴급", "보통", "일반"]}
    st.markdown(tag(f"전체 {len(hist)}건", "gray") + tag(f"긴급 {n['긴급']}", "red")
                + tag(f"보통 {n['보통']}", "orange") + tag(f"일반 {n['일반']}", "green"), unsafe_allow_html=True)
    st.dataframe([{k: h[k] for k in ["시각", "위치", "유형", "면적(%)", "등급"]} for h in rows],
                 width="stretch", hide_index=True)
    for h in rows:
        with st.expander(f"{h['시각']} · {h['제목']}"):
            st.code(h["민원문"], language=None, wrap_lines=True)
    import csv, io
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(hist[0].keys()))
    w.writeheader()
    w.writerows(hist)
    d1, d2 = st.columns(2)
    d1.download_button("⬇️ CSV로 내보내기", buf.getvalue().encode("utf-8-sig"), file_name="신고기록.csv")
    if d2.button("기록 모두 지우기"):
        st.session_state.history = []
        st.rerun()


# =========================================================================
# 페이지: 사용 방법
# =========================================================================

def run_guide():
    page_header("📖", "사용 방법", "신고 / 사용 방법")
    st.markdown("""
**이 서비스는 이렇게 동작해요**

1. **사진 업로드** — 도로 사진과 위치를 입력해요.
2. **위험요소 인식** — AI(HyperCLOVA X)가 균열/포트홀을 구분하고, YOLO가 파손 위치를 찾아요.
3. **등급 판정** — 파손이 사진에서 차지하는 비율로 일반·보통·긴급을 정해요. 겹친 부분은 한 번만 계산해요.
4. **민원문** — 안전신문고에 붙여 넣을 문장과 양식을 만들어 줘요.
""")
    callout("🚦", "등급 기준: <b>일반</b> 2% 미만 · <b>보통</b> 2~5% · <b>긴급</b> 5% 이상 (사진 전체 면적 대비)")
    callout("🚸", "어린이 보호구역 인근을 선택해도 <b>등급은 그대로</b>예요. 민원문에 \"해당 위치는 어린이 보호구역 인근입니다.\" 한 문장만 들어가요. (등급 상향은 이후 과제)")
    st.markdown("""
**알아 두세요**
- API 키가 없으면 인식 결과와 민원문은 가짜 데이터/템플릿으로 만들어져요.
- `best.pt` 모델이 없으면 파손 비율이 임시 값이에요.
- 만든 민원문은 사실과 다른 내용이 없는지 꼭 확인한 뒤 제출하세요.
""")


if page.endswith("새 신고"):
    run_new_report()
elif page.endswith("신고 기록"):
    run_history()
else:
    run_guide()
