import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from PIL import Image, ImageTk
import os
import fitz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import easyocr
import random
import base64
import json
import urllib.request
import urllib.error
from collections import Counter

# LM Studio(로컬 REST 서버, OpenAI 호환)에 로드해둔 비전 모델(Gemma). 점선/
# 빗금으로 끊긴 도면 폰트를 실측 100% 정확도로 읽어낸 방식. TrOCR/Tesseract/
# EasyOCR 단일 크롭 방식은 실측 정확도가 낮아 제거하고 Gemma 판독만 남긴다.
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
LM_STUDIO_MODELS_URL = "http://localhost:1234/api/v0/models"
LM_STUDIO_MODEL = "qwen/qwen2.5-vl-7b"
# 처음엔 "API로 조회 불가"로 보고 GUI 팝업으로 물어보게 했었는데,
# LM Studio의 REST API v0(/api/v0/models)이 각 모델의 현재
# loaded_context_length를 그대로 내려준다는 걸 실측으로 확인했다
# (실측: 262144). get_lm_studio_context_length()로 자동 조회하고,
# 서버가 꺼져있거나 API가 실패하면 이 값을 안전한 기본값으로 쓴다.
LM_STUDIO_CONTEXT_LENGTH_FALLBACK = 8192


def get_lm_studio_context_length(model=LM_STUDIO_MODEL, timeout=5):
    """LM Studio REST API v0에서 현재 로드된 모델의 실제 context length를
    조회한다. /api/v0/models는 로드된 모델들의 loaded_context_length를
    포함한 목록을 반환하므로(실측 확인), 거기서 model과 이름이 일치하고
    state가 "loaded"인 항목을 찾는다. 실패하면(서버 꺼짐, API 형식
    변경 등) None을 반환하고 호출부가 LM_STUDIO_CONTEXT_LENGTH_FALLBACK
    으로 대체한다."""
    try:
        req = urllib.request.Request(LM_STUDIO_MODELS_URL)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        for m in data.get("data", []):
            if m.get("id") == model and m.get("state") == "loaded":
                ctx = m.get("loaded_context_length") or m.get("max_context_length")
                if ctx:
                    return int(ctx)
        return None
    except Exception:
        return None


def ask_local_vlm_whole(image_bgr, n_expected, layout, max_tokens=3000, timeout=180, temperature=0.0):
    """크롭 전체를 한 번에 Gemma에 보여주고, 박스 안 글자를 순서대로 모두
    읽어달라고 요청한다. 셀 단위 개별 질의보다 정확도가 살짝 낮지만
    (실측 90% vs 100%), 픽셀 기반 경계선 검출(morphology kernel/ratio
    튜닝)이 아예 필요 없어서 도면마다 노이즈/해상도가 달라져도 깨지지
    않고, VLM 호출이 1회라 훨씬 빠르다.
    Gemma는 답을 낸 뒤에도 "Wait, let me check again"을 반복하며 자기
    검증을 계속하는 습성이 있어 content가 끝내 비고 finish_reason=length로
    끊기는 경우가 잦다 - 그래도 reasoning 안에는 이미 정답 목록이 여러 번
    나타나므로, "알파벳,알파벳,..." 형태의 n_expected개짜리 패턴 중
    마지막 것을 정답으로 채택한다."""
    dir_word = "위에서 아래로" if layout == "vertical" else "왼쪽에서 오른쪽으로"
    prompt = (
        f"이 이미지는 케이블 배선도의 커넥터 핀 이름을 나열한 박스 목록입니다. "
        f"총 {n_expected}개의 박스가 {dir_word} 있고, 각 박스 안에는 알파벳 한 "
        f"글자가 점선처럼 끊긴 획으로 그려져 있습니다. "
        f"{dir_word} 순서대로 각 박스의 글자를 읽어서, 쉼표로 구분한 한 줄로만 "
        f"답하세요. 예: S,Q,P,N,... 다른 설명은 하지 마세요."
    )
    ok, buf = cv2.imencode(".png", image_bgr)
    if not ok:
        return [], "이미지 인코딩 실패"
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    payload = {
        "model": LM_STUDIO_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        }],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        LM_STUDIO_URL, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        msg = result["choices"][0]["message"]
        content = (msg.get("content", "") or "").strip()
        text = content
        err = None
        if not text:
            reasoning = (msg.get("reasoning_content", "") or "")
            if reasoning:
                import re
                pattern = r'(?:[A-Z]\s*,\s*){' + str(n_expected - 1) + r'}[A-Z]'
                candidates = re.findall(pattern, reasoning)
                if candidates:
                    text = candidates[-1]
                    err = "reasoning에서 회수"
        if not text:
            return [], "빈 응답(경계선 검출 실패 또는 모델이 답을 내지 못함)"
        letters = [c.strip().upper()[:1] for c in text.replace("\n", ",").split(",") if c.strip()]
        return letters, err
    except urllib.error.URLError as e:
        return [], f"연결 실패: {e}"
    except Exception as e:
        return [], f"예외: {e}"


def ask_local_vlm_extract_all_text(image_bgr, max_tokens=6000, timeout=180, temperature=0.0):
    """배선도 전체를 Gemma에 보여주고, 보이는 모든 텍스트(핀 문자,
    커넥터명 P1/J14 등, 와이어 번호 K-16 등, 신호명 SIG_LIMIT_ELE 등,
    각주 번호)를 위치와 함께 JSON 리스트로 뽑아낸다.
    ask_local_vlm_whole은 "고정 개수 박스가 정해진 순서로 나열"이라는
    좁은 레이아웃 가정이 있어서, 박스 안/밖/원 안에 제각각 흩어진
    실제 배선도 텍스트에는 안 맞는다(사용자 스크린샷으로 확인). 이
    함수는 레이아웃 가정 없이 "보이는 모든 텍스트를 위치와 함께
    나열"만 요청해서 어떤 배선도 형식에도 동일하게 적용할 수 있다.
    연결관계(넷리스트) 추론은 이 함수의 책임이 아니다 - 사용자 지시로
    텍스트 추출만 담당하고, 좌표 근접도로 텍스트를 연결 짓는 건 이후
    별도 단계에서 처리한다.
    반환: (items, err) - items는 [{"text":str, "x":int, "y":int}, ...]
    (x,y는 원본 image_bgr 픽셀 좌표, 텍스트 중심 대략 위치)."""
    # LM Studio/llama.cpp 비전 백엔드는 이미지 장변에 상한이 있어서, 배선도
    # 원본(3000~4000px대)을 그대로 보내면 400 Bad Request로 거부되거나
    # (실측: 60309822 도면) 아예 모델 프로세스가 크래시한다(실측: 1600px로
    # 줄여도 "model has crashed" - VRAM 부족으로 추정). 1024px까지 더
    # 공격적으로 줄여서 보내고, 모델이 준 좌표는 이 축소 비율의 역수를
    # 곱해 원본 좌표계로 되돌린다.
    orig_h, orig_w = image_bgr.shape[:2]
    max_side = 1024
    scale = min(1.0, max_side / max(orig_h, orig_w))
    if scale < 1.0:
        send_img = cv2.resize(image_bgr, (int(orig_w * scale), int(orig_h * scale)),
                               interpolation=cv2.INTER_AREA)
    else:
        send_img = image_bgr

    # 이 모델(Gemma QAT)은 LM Studio UI에 reasoning을 끄는 토글이 없어서
    # (실측: Info/Load/Inference 탭 어디에도 없음) 항상 "생각 과정"부터
    # 뱉는다. 그 생각 과정이 같은 문장을 반복하며 max_tokens를 전부
    # 소진해버려(실측: 28핀을 세려다 "Wait, let me re-check"를 수십 번
    # 반복) 정작 JSON 답변까지 못 감. "/nothink" 접두사는 Qwen3/Gemma
    # 계열 일부 모델이 reasoning을 끄는 관례적 트리거라 프롬프트 맨
    # 앞에 붙여서 시도한다(모델이 무시해도 안전 - 그냥 텍스트로 처리됨).
    prompt = (
        "/nothink\n"
        "이 이미지는 케이블 배선도입니다. 이미지 안에 보이는 모든 텍스트를 "
        "빠짐없이 읽어주세요. 커넥터 이름(P1, J14 등), 핀 문자/번호(A, B, S, "
        "16 등), 와이어 번호(K-1, K-16 등), 신호명(SIG_LIMIT_ELE 등), 원 안의 "
        "각주 번호, 그 외 모든 문자/숫자를 포함합니다. 박스 안에 있든, 박스 "
        "밖에 있든, 원(써클) 안에 있든 상관없이 보이는 대로 전부 읽으세요. "
        "핀 개수를 세거나 검증하려 하지 말고, 보이는 텍스트를 그대로 한 번만 "
        "나열하세요. 결과는 반드시 JSON 배열로만 답하세요. 각 항목은 "
        '{"text": "읽은 텍스트", "x": 대략적인 가로 픽셀 위치, "y": 대략적인 '
        "세로 픽셀 위치} 형식입니다. 예: "
        '[{"text":"P1","x":50,"y":80},{"text":"K-16","x":300,"y":150}]. '
        "다른 설명 없이 JSON 배열만 출력하세요."
    )
    ok, buf = cv2.imencode(".png", send_img)
    if not ok:
        return [], "이미지 인코딩 실패"
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    payload = {
        "model": LM_STUDIO_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        }],
        "temperature": temperature,
        "max_tokens": max_tokens,
        # LM Studio(llama.cpp 서버)가 지원하면 이 필드로 chat template의
        # thinking 블록 자체를 끈다(Qwen3류 관례 - 지원 안 하는 모델/서버
        # 버전이면 그냥 무시됨. 실측 필요, 안전한 시도).
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        LM_STUDIO_URL, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        msg = result["choices"][0]["message"]
        content = (msg.get("content", "") or "").strip()
        text = content
        if not text:
            reasoning = (msg.get("reasoning_content", "") or "")
            text = reasoning
        if not text:
            return [], "빈 응답"

        import re
        # 모델이 코드펜스(```json ... ```)나 앞뒤 설명을 붙이는 경우가
        # 많아서, 가장 바깥 대괄호 [ ... ] 블록만 정규식으로 추출한다.
        match = re.search(r'\[.*\]', text, re.DOTALL)
        items = None
        if match:
            try:
                items = json.loads(match.group(0))
            except json.JSONDecodeError:
                items = None
        if items is None:
            # max_tokens에 걸려 배열이 닫히지 않은 채 끊긴 경우(실측:
            # A20016149에서 finish_reason=length로 "[{...},{...},{"처럼
            # 마지막 객체 중간에 잘림 - 닫는 ]가 없어 위 정규식이
            # 아예 매치를 못 해 타일 전체가 통째로 날아갔었음). 완결된
            # {"text":...,"x":...,"y":...} 객체만이라도 하나씩 건져낸다 -
            # 사용자 지시("커넥터 옆 텍스트는 박스 유무 상관없이 뽑아야
            # 한다")대로 잘린 마지막 항목 하나를 잃더라도 앞쪽 항목들은
            # 살려야 한다.
            obj_matches = re.findall(
                r'\{\s*"text"\s*:\s*"([^"]*)"\s*,\s*"x"\s*:\s*(-?\d+)\s*,\s*"y"\s*:\s*(-?\d+)\s*\}',
                text)
            if obj_matches:
                items = [{"text": t, "x": int(x), "y": int(y)} for t, x, y in obj_matches]
        if items is None:
            # max_tokens 안에 reasoning만으로 다 써버려 JSON 답변까지
            # 못 간 경우(실측: finish_reason=length, content 빈 문자열),
            # reasoning이 "1. **텍스트** (위치 설명)" 같은 번호 매긴 목록
            # 형태로 나온다 - 이 패턴에서라도 텍스트만 회수한다. 좌표는
            # 모델이 안 줬으므로 0,0으로 채우고 이후 단계(좌표 근접
            # 그룹화)에서는 이 항목들을 위치 정보 없이 다뤄야 한다.
            fallback = re.findall(r'\*\*([^*]+)\*\*', text)
            if fallback:
                items = [{"text": t, "x": 0, "y": 0} for t in fallback]
        if items is None:
            return [], f"JSON 배열을 찾지 못함: {text[:200]}"

        # 모델에게는 축소된 send_img 기준 좌표를 요청했으므로, scale의
        # 역수를 곱해 원본 image_bgr 픽셀 좌표계로 되돌린다.
        inv_scale = 1.0 / scale if scale > 0 else 1.0
        cleaned = []
        for it in items:
            if not isinstance(it, dict) or "text" not in it:
                continue
            cleaned.append({
                "text": str(it.get("text", "")).strip(),
                "x": int(int(it.get("x", 0)) * inv_scale),
                "y": int(int(it.get("y", 0)) * inv_scale),
            })
        return cleaned, None
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:500]
        return [], f"HTTP {e.code} 오류: {body}"
    except urllib.error.URLError as e:
        return [], f"연결 실패: {e}"
    except Exception as e:
        return [], f"예외: {e}"


def detect_connector_box_columns(image_bgr, min_box_height_ratio=0.12, min_width=25, max_width_ratio=0.35):
    """커넥터 핀 박스(세로로 긴 직사각형, 예: 50073731의 J20/J5)의 좌/우
    테두리를 이미지 처리로 찾는다. VLM은 "대략적인" 좌표를 추정할 뿐이라
    실측으로 이미지 범위를 넘는 좌표까지 준 사례가 있었다(사용자 지적 -
    A20016147의 B핀 y=820이 타일 높이 766을 초과). 반면 격자선은
    모폴로지 연산으로 100% 정확한 픽셀 위치를 계산할 수 있다(실측:
    50073731에서 박스 테두리 x좌표(462,657)과 가로 구분선(650,845,1040,
    1235,1430)을 정확히 검출, 그 사이를 잘라보니 실제 B 셀 문자와 일치).
    min_box_height는 이미지 세로 크기에 비례해서 계산한다(고정값이면
    도면마다 원본 크기가 제각각이라 작은 도면(예: 50073721, 964x951)에서
    박스 테두리를 통째로 놓치는 문제가 있었다 - 실측).
    반환: [(x0, x1), ...] (박스 좌/우 테두리 x좌표 쌍, 왼쪽부터)."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    _, binimg = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    h, w = binimg.shape
    min_box_height = max(60, int(h * min_box_height_ratio))

    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_box_height))
    vert_lines = cv2.morphologyEx(binimg, cv2.MORPH_OPEN, vert_kernel)
    col_ink = (vert_lines > 0).sum(axis=0)
    candidates = np.where(col_ink > min_box_height * 0.5)[0]
    if len(candidates) == 0:
        return []

    groups = []
    cur = [int(candidates[0])]
    for x in candidates[1:]:
        x = int(x)
        if x - cur[-1] <= 10:
            cur.append(x)
        else:
            groups.append((cur[0] + cur[-1]) // 2)
            cur = [x]
    groups.append((cur[0] + cur[-1]) // 2)

    # 연속한 두 수직선을 "박스의 좌/우 테두리" 쌍으로 짝짓는다. 테두리
    # 사이 폭이 너무 좁으면(글자가 들어갈 공간이 안 됨) 실제 박스가
    # 아니라 배선 중간의 우연한 겹침선일 수 있고, 반대로 너무 넓으면
    # (예: 배선 다발 점선까지 우연히 잡힌 경우) 박스가 아닐 가능성이
    # 높다.
    max_width = max(200, int(w * max_width_ratio))
    pairs = []
    for i in range(len(groups) - 1):
        x0, x1 = groups[i], groups[i + 1]
        if min_width <= (x1 - x0) <= max_width:
            pairs.append((x0, x1))
    if not pairs:
        return []

    # 배선 다발 표시(점선 타원)의 곡선이 세로선처럼 검출되면, 진짜
    # 커넥터 박스 바로 옆에 비슷한 위치에서 더 넓은 폭의 가짜 박스 쌍이
    # 잡힌다(실측: A60024605에서 진짜 박스 폭 49px 옆에 가짜 139/141px
    # 쌍이 붙어있음 - 셀 높이/간격 균일성 검사만으로는 못 거름, 가짜의
    # 양끝 캡 셀을 걸러내고 남은 중간 셀들이 우연히 균일해서 통과함).
    # 실제 핀 박스는 보통 폭이 비슷하므로, 가장 좁은 폭을 기준으로 그
    # 절반~2배 범위 밖의 후보는 다발 표시로 보고 제외한다.
    widths = sorted(x1 - x0 for x0, x1 in pairs)
    narrowest = widths[0]
    return [(x0, x1) for x0, x1 in pairs if narrowest * 0.5 <= (x1 - x0) <= narrowest * 2.2]


def detect_grid_cells(image_bgr, box_x0, box_x1, min_line_width_ratio=0.5):
    """detect_connector_box_columns로 찾은 박스 하나(x0~x1) 안에서 가로
    구분선을 찾아, 핀 하나하나에 해당하는 셀 경계 y좌표 목록을 계산한다.
    반환: [(y0,y1), ...] (셀 상/하 경계 쌍, 위에서 아래 순서)."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    _, binimg = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    col_slice = binimg[:, box_x0:box_x1]
    row_ink = (col_slice > 0).sum(axis=1)
    threshold = (box_x1 - box_x0) * min_line_width_ratio
    line_rows = np.where(row_ink > threshold)[0]
    if len(line_rows) < 2:
        return []

    groups = []
    cur = [int(line_rows[0])]
    for y in line_rows[1:]:
        y = int(y)
        if y - cur[-1] <= 5:
            cur.append(y)
        else:
            groups.append((cur[0] + cur[-1]) // 2)
            cur = [y]
    groups.append((cur[0] + cur[-1]) // 2)

    raw_cells = [(groups[i], groups[i + 1]) for i in range(len(groups) - 1)]
    # 커넥터 위/아래 끝의 둥근 캡(반원 테두리)이 가로줄처럼 검출되면 두
    # 가지 형태로 오탐이 생긴다: (1) 캡과 첫/마지막 실제 구분선 사이의
    # "셀"이 다른 셀들보다 훨씬 큼(실측: 50073731에서 303px vs 195px),
    # (2) 캡의 곡선 시작점 자체가 별도의 짧은 가짜 구분선으로 잡혀 바로
    # 다음 구분선과의 사이가 다른 셀보다 훨씬 작음(실측: A60026426-1에서
    # 16px vs 정상 51px). 중앙값 높이 기준 0.5~1.4배를 벗어나는 셀은
    # 진짜 핀 칸이 아니라고 보고 제외한다.
    if len(raw_cells) >= 3:
        heights = sorted(y1 - y0 for y0, y1 in raw_cells)
        median_h = heights[len(heights) // 2]
        raw_cells = [
            (y0, y1) for y0, y1 in raw_cells
            if median_h * 0.5 <= (y1 - y0) <= median_h * 1.4
        ]
    return raw_cells


def ask_local_vlm_pin_sequence(image_bgr, n_expected, timeout=90, temperature=0.0, horizontal=False):
    """커넥터 박스 전체(모든 핀 칸이 다 보이는 이미지)를 Gemma에게 보여주고,
    위에서 아래로 핀 이름을 순서대로 읽어달라고 요청한다.
    셀 하나씩 잘라서 개별로 물어보면(이전 방식) 주변 문맥이 사라져서
    VLM이 전통 OCR과 똑같이 "고립된 이미지 조각"만 보고 판단하게 되고,
    그 결과 A→I, S→1처럼 체계적인 오독이 생겼다(실측: 50073721에서 18핀
    중 다수 오류). 반면 박스 전체를 한 번에 보여주면 VLM이 "핀은 보통
    A,B,C,D,E,F,G,H,J,K,L,M,N,P,R,S,T,U 순서(I,O,Q 생략)로 나열된다"는
    문맥/추론을 글자 판독에 활용할 수 있다 - 이게 비전 모델이 전통 OCR과
    다른 강점이다(사용자와의 논의로 확인). 실측: 이 방식으로 50073721
    양쪽 박스(18핀×2)와 50073731(4핀) 전부 100% 정확했음(이전 셀별
    개별 질의는 36개 중 34개, 그마저도 판독 자체는 부정확했음).
    좌표는 이 함수가 관여하지 않는다 - 호출부(extract_pins_via_grid)가
    detect_grid_cells로 계산한 셀 경계와, 이 함수가 반환한 순서를
    그대로 y좌표 순으로 매칭한다.
    반환: (letters, err) - letters는 [str, ...] 길이가 n_expected와
    다를 수 있음(호출부에서 개수 불일치를 처리해야 함)."""
    direction = "왼쪽에서 오른쪽으로" if horizontal else "위에서 아래로"
    prompt = (
        "/nothink\n"
        f"이 이미지는 케이블 커넥터의 핀 이름을 {direction} 나열한 "
        f"목록입니다. 총 {n_expected}개의 칸이 있고, 각 칸에는 알파벳 "
        "한 글자 또는 숫자 1~2자리가 있습니다. 커넥터 핀은 흔히 "
        "A,B,C,D,E,F,G,H,J,K,L,M,N,P,R,S,T,U 순서로 나열됩니다(I,O,Q는 "
        "혼동을 피하려고 건너뜁니다). 다만 도면에 따라 이 순서를 따르지 "
        "않거나 숫자로 된 핀 번호일 수도 있습니다 - 실제로 보이는 글자를"
        " 우선하고, 획이 점선처럼 끊기거나 흐릿해서 진짜 애매할 때만 "
        "표준 순서를 참고해 보정하세요. "
        f"{direction} 정확히 {n_expected}개의 글자를 쉼표로 구분해서 "
        "한 줄로만 답하세요. 다른 설명은 하지 마세요."
    )
    ok, buf = cv2.imencode(".png", image_bgr)
    if not ok:
        return [], "이미지 인코딩 실패"
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    payload = {
        "model": LM_STUDIO_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        }],
        "temperature": temperature,
        "max_tokens": 3000,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        LM_STUDIO_URL, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        msg = result["choices"][0]["message"]
        text = (msg.get("content", "") or "").strip()
        err = None
        if not text:
            # max_tokens 안에 reasoning만으로 다 써버려 답을 못 낸 경우,
            # reasoning 안에 "Row 1: A" 같은 패턴이 남아있을 수 있어 그걸
            # 회수한다(ask_local_vlm_whole과 동일한 패턴, 실측 검증됨).
            reasoning = (msg.get("reasoning_content", "") or "")
            import re
            pattern = r'(?:[A-Za-z0-9]{1,2}\s*,\s*){' + str(max(0, n_expected - 1)) + r'}[A-Za-z0-9]{1,2}'
            candidates = re.findall(pattern, reasoning)
            if candidates:
                text = candidates[-1]
                err = "reasoning에서 회수"
        if not text:
            return [], "빈 응답"
        letters = [c.strip().upper() for c in text.replace("\n", ",").split(",") if c.strip()]
        return letters, err
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:300]
        return [], f"HTTP {e.code} 오류: {body}"
    except urllib.error.URLError as e:
        return [], f"연결 실패: {e}"
    except Exception as e:
        return [], f"예외: {e}"


def detect_grid_cells_horizontal(image_bgr, box_y0, box_y1, box_x0=None, box_x1=None,
                                  min_line_width_ratio=0.3):
    """detect_grid_cells의 가로 버전 - 박스(y0~y1, 선택적으로 x0~x1) 안에서
    세로 구분선을 찾아 핀 하나하나에 해당하는 셀 경계 x좌표 목록을
    계산한다. 마찬가지로 transpose해서 기존 detect_grid_cells를 재사용한다.
    box_x0/box_x1을 주지 않으면 이미지 가로 전체를 스캔하는데, 이는 같은
    y범위 안에 여러 커넥터가 나란히 있을 때(실측: A60023103에서 P9,P8,
    P7,P6,P5가 한 줄에 붙어있음) 전부 하나로 뭉뚱그려 잘못 검출되는
    사고로 이어진다 - 사람이 3-B2에서 커넥터 하나만 감싸는 사각형을
    드래그하면 box_x0/box_x1도 함께 넘겨 이 문제를 막는다.
    min_line_width_ratio 기본값이 세로 버전(0.5)보다 낮은 이유: 사람이
    3-B2에서 손으로 드래그한 박스는 자동검출된 세로 격자 박스처럼
    타이트하지 않고 커넥터 라벨(P9 텍스트)이나 테두리 캡, 여백까지
    넉넉하게 포함되는 경우가 많다(실측: A60023103의 P9 박스). 그러면
    "박스 전체 높이" 대비 진짜 구분선 비율이 낮아져 0.5 기준으로는
    구분선을 하나도 못 찾는 사고가 났다 - 0.3으로 낮추니 A,B 사이
    구분선과 양끝 테두리 3개가 정확히 검출됨(실측 확인).
    반환: [(x0,x1), ...] (셀 좌/우 경계 쌍, 왼쪽부터, 원본 이미지 좌표계)."""
    h, w = image_bgr.shape[:2]
    x0 = 0 if box_x0 is None else max(0, box_x0)
    x1 = w if box_x1 is None else min(w, box_x1)
    cropped = image_bgr[:, x0:x1]
    transposed = cv2.transpose(cropped)
    cells = detect_grid_cells(transposed, box_y0, box_y1, min_line_width_ratio=min_line_width_ratio)
    return [(cx0 + x0, cx1 + x0) for cx0, cx1 in cells]


def _is_regular_grid(row_ranges, max_cv=0.08):
    """row_ranges(셀 경계 목록)의 셀 간격이 얼마나 균일한지 검사한다.
    진짜 커넥터 핀 박스는 칸 간격이 기계적으로 일정하지만(변동계수
    실측 0.015), 배선 다발을 표시하는 점선 타원의 곡선 구간이 우연히
    가로줄처럼 잡히면 양 끝은 넓고 가운데는 좁은 식으로 간격이 들쭉날쭉
    해진다(실측: A60024605에서 다발 타원 구간 변동계수 0.163). 변동계수
    (표준편차/평균)가 max_cv를 넘으면 진짜 핀 박스가 아니라고 판단한다."""
    if len(row_ranges) < 3:
        return True  # 셀이 2개 이하면 간격이 하나뿐이라 균일성 판단 불가, 통과시킴
    heights = [y1 - y0 for y0, y1 in row_ranges]
    mean_h = sum(heights) / len(heights)
    if mean_h == 0:
        return False
    variance = sum((h - mean_h) ** 2 for h in heights) / len(heights)
    cv = (variance ** 0.5) / mean_h
    return cv <= max_cv


def extract_pins_via_grid(image_bgr, max_workers=4, progress_cb=None, manual_horizontal_boxes=None):
    """이미지에서 커넥터 박스(격자)를 자동 검출해, 각 셀을 크롭한 뒤
    VLM에게 글자만 물어보는 방식으로 핀 텍스트를 뽑는다. 좌표는 셀
    경계에서 기하학적으로 계산하므로 100% 정확하다(사용자 지시 - "좌표가
    정확하게 들어가야 되기 때문에 틀린 좌표는 필요가 없다"). 격자(박스)가
    없는 도면(예: A20016147처럼 테두리 없이 문자만 나열된 스타일)에서는
    박스 자체가 검출되지 않으므로 빈 리스트를 반환한다 - 호출부에서 이
    경우 기존 extract_all_text_tiled로 폴백해야 한다.
    핀 문자는 박스(커넥터) 단위로 ask_local_vlm_pin_sequence에 한 번씩
    질의한다 - 셀을 하나씩 잘라서 개별 질의하면 문맥이 사라져 VLM이
    체계적으로 오독했다(실측: A→I, S→1). 박스가 여러 개면(보통 2개,
    좌우 커넥터) 박스별로 병렬 질의해 전체 처리 시간은 비슷하게 유지한다.
    VLM이 준 글자 개수가 셀 개수와 다르면(누락/과다 응답) 그 박스는
    신뢰할 수 없으므로 좌표만 남기고 글자는 "?"로 표시해 사람이 3-E에서
    확인하게 한다(틀린 글자를 조용히 채우는 것보다 안전).
    manual_horizontal_boxes는 사람이 GUI(3-B2)로 직접 드래그해서 지정한
    가로 격자 박스 [(x0,y0,x1,y1), ...] 목록이다(커넥터 하나를 감싸는
    사각형 전체 - y범위만 주면 같은 y줄에 나란히 붙은 여러 커넥터가
    하나로 뭉뚱그려 잘못 검출된다, 실측: A60023103에서 P9,P8,P7,P6,P5가
    한 줄에 붙어있어 y범위만으로는 못 나눔). 가로 격자(핀이 왼쪽→오른쪽
    나열)는 자동 검출을 시도해봤으나, 세로 격자와 달리 배선 자체가
    가로선이 많아(신호선이 수평으로 지나감) 커넥터 진짜 테두리와 배선을
    구분하지 못해 오탐이 심했다(실측: A60023103에서 12개, A60025766에서
    23개의 가짜 후보 - 사용자와 함께 확인 후 자동검출 대신 수동 지정으로
    전환하기로 함). 사람이 박스 영역만 지정하면, 그 안에서 셀 경계
    계산(detect_grid_cells_horizontal)은 여전히 기하학적으로 100%
    정확하다 - 사람이 하는 일은 "어디가 커넥터 박스인지" 뿐이고, 정확한
    셀 좌표 계산은 여전히 자동이다.
    반환: (items, box_regions) - items는 [{"text","x","y","source":"grid_cell"}, ...],
    box_regions는 [(x0,y0,x1,y1), ...] - 검출된 각 커넥터 박스의 전체 영역
    (핀 칸들을 감싸는 사각형). 호출부가 이 영역 안에 들어오는 tile_items를
    걸러내는 데 쓴다 - 격자 칸 옆의 작은 배선 연결 심볼을 VLM이 핀 문자로
    잘못 읽어 같은 글자가 중복 등장하는 문제가 있었다(실측: 50073721에서
    격자 x=127의 A~U와 거의 같은 y에 x=80으로 A~U가 또 검출됨 - 좌표가
    30px dedup 기준을 넘어서 걸러지지 않았음). 좌표 근접 비교 대신 박스
    영역 자체로 걸러내는 게 더 근본적이다."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # valid_boxes 항목 형태: (x0, x1, y0, y1, cell_ranges, horizontal)
    # 세로 격자는 cell_ranges가 y구간 목록, 가로 격자는 x구간 목록이다.
    # horizontal 플래그로 run_box/좌표 계산 시 축을 구분한다.
    valid_boxes = []

    box_columns = detect_connector_box_columns(image_bgr)
    for x0, x1 in box_columns:
        row_ranges = detect_grid_cells(image_bgr, x0, x1)
        # 셀이 3개 미만이면 진짜 핀 격자로 보지 않는다 - 테두리 없이
        # 핀 문자만 나열된 도면(A20016147 등)에서 P1 박스 테두리나
        # 신호명 텍스트가 있는 가로줄 구간이 우연히 규칙적인 간격으로
        # 잡혀 CV 기준(_is_regular_grid)을 통과하는 사고가 있었다(실측:
        # A20016147에서 (36,141) 1셀, (698,750)/(750,803) 2~4셀짜리
        # 가짜 격자가 검출되어 "SOL_LIMIT_ELE_RTN" 같은 신호명 텍스트가
        # 통째로 핀 문자인 것처럼 잘못 저장됨). 실제 커넥터 격자는 실측
        # 사례 전부 핀이 여러 개(최소 4개 이상)였다.
        if len(row_ranges) < 4:
            continue
        if not _is_regular_grid(row_ranges):
            # 배선 다발 표시(점선 타원)가 우연히 커넥터 박스처럼 검출된
            # 가짜 박스는 통째로 건너뛴다(실측: A60024605에서 다발 타원
            # 구간이 진짜 커넥터 박스 바로 옆에 비슷한 폭으로 붙어있어
            # 폭 필터만으로는 못 거름).
            continue
        y0, y1 = row_ranges[0][0], row_ranges[-1][1]
        valid_boxes.append((x0, x1, y0, y1, row_ranges, False))

    # 가로 격자는 자동 검출하지 않고, 사람이 지정한 사각형(x0,y0,x1,y1)
    # 안에서만 셀 경계를 계산한다(위 docstring 참고 - 자동검출은 오탐이
    # 심했고, y범위만으로는 같은 줄에 붙은 여러 커넥터가 안 나뉘었음).
    # 사람이 직접 지정했지만 셀 경계를 못 찾은 박스(격자가 아니라 단일
    # 심볼인 경우)도 box_regions에는 반드시 포함시킨다 - 그래야 그 영역
    # 안의 화살표 지시선 숫자, 원 안 숫자 같은 VLM 잡음이 걸러진다(실측:
    # A60023104의 P1/P9는 격자가 아니라 valid_boxes에서 continue로
    # 빠지는데, box_regions에도 안 들어가면 필터링이 전혀 안 돼서 "47",
    # "19", "10" 같은 잡음이 애매 항목으로 그대로 남았음). 사람이 그린
    # 박스는 "여기가 커넥터 영역이다"라는 선언 자체가 신뢰할 수 있는
    # 정보이므로, 셀 인식 성공 여부와 무관하게 필터 영역으로 써야 한다.
    manual_box_regions = []
    for bx0, by0, bx1, by1 in (manual_horizontal_boxes or []):
        col_ranges = detect_grid_cells_horizontal(image_bgr, by0, by1, box_x0=bx0, box_x1=bx1)
        y0, y1 = by0, by1
        if len(col_ranges) < 2:
            # 사람이 직접 지정한 영역이므로 세로 격자처럼 4개 이상을
            # 강제하지 않는다 - 핀이 2~3개뿐인 작은 커넥터도 있을 수
            # 있고, 오탐 방지용 최소 개수 제약은 자동검출에서만 필요했다.
            manual_box_regions.append((bx0, by0, bx1, by1))
            continue
        x0, x1 = col_ranges[0][0], col_ranges[-1][1]
        valid_boxes.append((x0, x1, y0, y1, col_ranges, True))

    if not valid_boxes and not manual_box_regions:
        return [], []

    # 박스 영역을 살짝 넉넉하게(margin) 잡아둔다 - 핀 문자 칸 바로 옆의
    # 연결 심볼/여백까지 포함해서 tile_items를 걸러내기 위함(실측:
    # 50073721에서 심볼이 격자 칸 x=127 대비 약 47px 왼쪽인 x=80에 있었음).
    box_regions = []
    for x0, x1, y0, y1, cell_ranges, horizontal in valid_boxes:
        margin = max(60, (x1 - x0) if not horizontal else (y1 - y0))
        box_regions.append((x0 - margin, y0 - 20, x1 + margin, y1 + 20))
    box_regions.extend(manual_box_regions)

    def run_box(x0, x1, y0, y1, cell_ranges, horizontal):
        whole = image_bgr[y0:y1, x0:x1]
        # 너무 좁은 박스(41px대)는 VLM이 보기 어려우므로 2배 확대해서
        # 보낸다(실측: 50073721에서 확대 없이는 판독 실패, 확대 후 100%).
        short_side = whole.shape[0] if horizontal else whole.shape[1]
        if short_side < 100:
            whole = cv2.resize(whole, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        letters, err = ask_local_vlm_pin_sequence(
            whole, len(cell_ranges), horizontal=horizontal)
        return x0, x1, y0, y1, cell_ranges, horizontal, letters, err

    items = []
    done_count = [0]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(run_box, *b) for b in valid_boxes]
        for fut in as_completed(futures):
            x0, x1, y0, y1, cell_ranges, horizontal, letters, err = fut.result()
            done_count[0] += 1
            if progress_cb:
                progress_cb(done_count[0], len(valid_boxes))
            n = len(cell_ranges)
            if len(letters) != n:
                letters = ["?"] * n  # 개수 불일치 - 틀린 값을 채우지 않고 사람 확인 유도
            # VLM이 읽어온 값의 절반 이상이 핀 문자/핀 번호 패턴(단일
            # 알파벳 또는 숫자 1~2자리)이 아니면, 격자 검출 자체가
            # 신호명 텍스트 같은 엉뚱한 영역을 핀 칸으로 착각한 것으로
            # 보고 이 박스를 통째로 버린다(실측: A20016147에서 진짜
            # 핀 격자가 아닌 영역이 CV 기준을 우연히 통과해 "TN","LE"
            # 같은 신호명 조각이 핀 문자처럼 저장됨 - 셀 개수 필터만으론
            # 못 걸러서 내용 자체로 이중 검증).
            valid_pin_pattern = sum(
                1 for l in letters if PIN_LETTER_RE.match(l) or PIN_DIGIT_RE.match(l))
            if valid_pin_pattern < max(1, n // 2):
                continue
            if horizontal:
                cy = (y0 + y1) // 2
                for (cx0, cx1), letter in zip(cell_ranges, letters):
                    cx = (cx0 + cx1) // 2
                    items.append({"text": letter, "x": cx, "y": cy, "source": "grid_cell"})
            else:
                cx = (x0 + x1) // 2
                for (cy0, cy1), letter in zip(cell_ranges, letters):
                    cy = (cy0 + cy1) // 2
                    items.append({"text": letter, "x": cx, "y": cy, "source": "grid_cell"})

    return items, box_regions


def compute_auto_tile_grid(image_bgr, target_tile_px=900, min_grid=1, max_grid=6):
    """이미지 크기에 맞춰 cols x rows 타일 그리드를 자동으로 계산한다.
    실측으로 확인된 것: 커넥터 1개 크기(가로/세로 약 900px대) 타일은
    reasoning이 3000토큰대로 끝나 안정적으로 성공하지만, 전체 도면을
    통째로(3000~4000px) 보내면 reasoning이 폭주해 실패한다. 그래서
    "타일 한 변이 대략 target_tile_px가 되도록" 각 축을 독립적으로
    나눈다(가로가 세로보다 훨씬 긴 배선도가 많아서 cols/rows를 항상
    같게 묶으면 한쪽이 과도하게 커지거나 작아짐). 사용자가 매번 가로/
    세로 조각 수를 직접 입력하던 것을 대체한다(사용자 지시 - PNG마다
    자동으로 맞춰서 진행)."""
    h, w = image_bgr.shape[:2]
    cols = min(max_grid, max(min_grid, round(w / target_tile_px)))
    rows = min(max_grid, max(min_grid, round(h / target_tile_px)))
    return cols, rows


def compute_safe_parallelism(context_length, per_request_tokens=6500, server_max_parallel=4):
    """LM Studio는 로드된 모델의 Context Length를 Parallel 슬롯 수만큼
    나눠 쓴다(실측: Context=2048, Parallel=4로 동시 4개 질의 시 슬롯당
    컨텍스트가 부족해져 전부 "Context size has been exceeded"로 실패).
    반대로 슬롯 하나가 다 쓰는 게 아니라 "동시에 뜬 요청 수만큼" 나눠
    쓰이므로, 실제로 동시에 보내는 요청 수를 context_length //
    per_request_tokens로 제한하면 이 실패를 피할 수 있다.
    per_request_tokens는 prompt_tokens(이미지+지시문, 실측 500대) +
    max_tokens(reasoning 포함, 기본 6000)의 여유있는 합. 최소 1, 그리고
    LM Studio 자체에 설정된 Parallel 슬롯 수(server_max_parallel)를
    넘지 않게 클램프한다(그 이상은 서버가 어차피 대기열에 넣으므로
    의미 없음)."""
    if not context_length or context_length <= 0:
        return 1
    n = max(1, context_length // per_request_tokens)
    return min(n, server_max_parallel)


def extract_all_text_tiled(image_bgr, cols=3, rows=3, overlap_ratio=0.1, max_workers=4,
                            progress_cb=None):
    """배선도 전체를 한 번에 Gemma에 보내면 reasoning이 항목 수에 비례해
    폭주한다(실측: 전체 도면 요청 시 reasoning 6000 토큰을 다 쓰고도
    JSON 답변을 못 냄, finish_reason=length - 28개 핀을 세다가 같은
    문장을 반복). 반면 사용자가 직접 커넥터 하나 크기로 잘라 질의하면
    reasoning이 3355 토큰으로 줄고 성공한다(실측). 이 함수는 그 수동
    분할을 자동화한다 - 이미지를 cols x rows 타일로 나누고(타일 경계에서
    텍스트가 잘리는 걸 막기 위해 overlap_ratio만큼 타일끼리 겹치게 함),
    LM Studio의 Parallel 슬롯을 활용해 여러 타일을 동시에 질의한다(순차
    대비 최대 max_workers배 단축). max_workers는 호출부에서
    compute_safe_parallelism으로 미리 계산해 넘겨야 한다 - 무작정 4로
    고정하면 Context Length가 작을 때 전부 실패한다(실측).
    겹친 영역 때문에 같은 텍스트가 여러 타일에서 중복 추출될 수 있으므로,
    (text, 반올림한 x, y) 기준으로 근접 중복을 제거한다.
    반환: (items, tile_errors) - items는 원본 image_bgr 좌표계의
    [{"text","x","y"}, ...], tile_errors는 실패한 타일의 에러 메시지 리스트."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    h, w = image_bgr.shape[:2]
    tile_w, tile_h = w / cols, h / rows
    ov_x, ov_y = int(tile_w * overlap_ratio), int(tile_h * overlap_ratio)

    tiles = []  # (tile_img, offset_x, offset_y)
    for r in range(rows):
        for c in range(cols):
            x0 = max(0, int(c * tile_w) - ov_x)
            y0 = max(0, int(r * tile_h) - ov_y)
            x1 = min(w, int((c + 1) * tile_w) + ov_x)
            y1 = min(h, int((r + 1) * tile_h) + ov_y)
            if x1 <= x0 or y1 <= y0:
                continue
            tiles.append((image_bgr[y0:y1, x0:x1], x0, y0))

    all_items = []
    tile_errors = []
    done_count = [0]

    def run_tile(tile_img, off_x, off_y):
        items, err = ask_local_vlm_extract_all_text(tile_img)
        return items, err, off_x, off_y

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(run_tile, t_img, ox, oy) for t_img, ox, oy in tiles]
        for fut in as_completed(futures):
            items, err, off_x, off_y = fut.result()
            done_count[0] += 1
            if progress_cb:
                progress_cb(done_count[0], len(tiles))
            if err and not items:
                tile_errors.append(f"타일@({off_x},{off_y}): {err}")
                continue
            for it in items:
                all_items.append({
                    "text": it["text"],
                    "x": it["x"] + off_x,
                    "y": it["y"] + off_y,
                })

    # 겹친 영역에서 같은 텍스트가 두 타일 모두에서 뽑혀 중복되는 걸
    # 제거한다. 좌표가 정확히 같지는 않으므로(각 타일이 독립적으로 추정)
    # 텍스트가 같고 좌표가 타일 겹침 폭(ov_x, ov_y) 이내로 가까우면
    # 같은 항목으로 보고 하나만 남긴다.
    dedup = []
    for it in all_items:
        is_dup = False
        for kept in dedup:
            if kept["text"] == it["text"] and \
               abs(kept["x"] - it["x"]) <= max(ov_x, 5) and \
               abs(kept["y"] - it["y"]) <= max(ov_y, 5):
                is_dup = True
                break
        if not is_dup:
            dedup.append(it)

    return dedup, tile_errors


def _contains_hangul(text):
    """text에 한글 완성형 음절(가-힣) 또는 자모(ㄱ-ㅎ, ㅏ-ㅣ)가 하나라도
    있으면 True. 배선도 텍스트 추출 결과에서 한글 각주/설명 문구를
    걸러낼 때 쓴다(사용자 지시 - JSON에는 한글을 포함시키지 않는다)."""
    return any(
        '가' <= ch <= '힣' or 'ㄱ' <= ch <= 'ㅣ'
        for ch in text
    )


def filter_out_hangul(items):
    """items([{"text","x","y"}, ...])에서 text에 한글이 하나라도 섞인
    항목을 통째로 제외한다(부분적으로 한글만 지우면 "K-1(계속)"같은
    항목이 "K-1()"처럼 망가지므로, 항목 단위로 버린다)."""
    return [it for it in items if not _contains_hangul(it.get("text", ""))]


import re as _re

# 커넥터명 패턴: 알파벳 1~2글자 + 숫자 1~3자리 (P1, P11, J14, J26 등).
# 사용자 지시 - "지금은 핀번호/핀문자 외에는 다 필요없다"는 방침에 따라,
# 후처리 3단계 분류의 1차 기준으로 쓴다.
CONNECTOR_NAME_RE = _re.compile(r'^[A-Za-z]{1,2}\d{1,3}$')
# 핀 문자: 알파벳 단일 문자(대소문자 무관, 케이블 핀 명명 관례상 I/O는
# 안 쓰는 경우가 많지만 실측 데이터에 A~Z 전부 나왔으므로 전체 허용).
PIN_LETTER_RE = _re.compile(r'^[A-Za-z]$')
# 핀 번호: 순수 숫자 1~2자리. 단, 이 패턴은 원 안 지시선 참조번호(12,13,
# 17...)와 좌표만으로는 구별 불가하므로 "애매(ambiguous)"로 분류해
# 자동 확정하지 않고 사람이 PNG로 확인하게 한다.
PIN_DIGIT_RE = _re.compile(r'^\d{1,2}$')

# 핀 번호는 반드시 커넥터 근처(상하좌우)에 붙어 있다는 도면 관례(사용자
# 확인) - 커넥터명(connectors)에서 이 거리보다 먼 순수 숫자는 핀 번호가
# 아니라 원 안 지시선 참조번호/주석이므로 3-C에서 바로 제외한다. 격자
# 박스 폭(50073721 기준 약 330px)보다 넉넉하게 잡아야 커넥터 반대편
# 끝의 진짜 핀 번호까지 포함된다.
REFERENCE_NUMBER_MAX_DIST = 400


def filter_out_stray_reference_numbers(items):
    """items에서, 순수 숫자(PIN_DIGIT_RE)인데 어떤 커넥터명(CONNECTOR_NAME_RE)
    과도 REFERENCE_NUMBER_MAX_DIST 이내에 있지 않은 항목을 제외한다.
    핀 번호는 반드시 커넥터가 상하좌우로 존재한다는 도면 관례(사용자
    지적)를 이용해, 지시선 안의 참조번호(예: 50073721의 원 안 "12","13")
    처럼 커넥터와 무관하게 멀리 떨어진 숫자를 걸러낸다. 알파벳 핀 문자나
    커넥터명, 그 외 텍스트는 건드리지 않는다."""
    connectors = [it for it in items if CONNECTOR_NAME_RE.match(it.get("text", "").strip())]
    if not connectors:
        return items
    result = []
    for it in items:
        text = it.get("text", "").strip()
        if PIN_DIGIT_RE.match(text):
            near = any(
                abs(c["x"] - it["x"]) <= REFERENCE_NUMBER_MAX_DIST and
                abs(c["y"] - it["y"]) <= REFERENCE_NUMBER_MAX_DIST
                for c in connectors)
            if not near:
                continue
        result.append(it)
    return result


def classify_pin_items(items):
    """items를 세 그룹으로 나눈다:
    - connectors: 커넥터명(P1, J14 등) - 항상 유지
    - pins: 핀 문자(A~Z 단일 문자) - 항상 유지
    - ambiguous: 1~2자리 순수 숫자 - 핀 번호일 수도, 원 안 지시선
      참조번호(예: "12","13")일 수도 있어 좌표만으로 확정 불가.
      GUI에서 PNG 크롭을 보여주고 사람이 최종 판단하게 한다.
    - excluded: 나머지 전부(신호명, AWG, 색상, 전압 등) - 항상 제외
    반환: {"connectors":[...], "pins":[...], "ambiguous":[...], "excluded":[...]}
    (원본 딕셔너리를 그대로 담되 분류 리스트만 나눈다)"""
    connectors, pins, ambiguous, excluded = [], [], [], []
    for it in items:
        text = it.get("text", "").strip()
        if CONNECTOR_NAME_RE.match(text):
            connectors.append(it)
        elif PIN_LETTER_RE.match(text):
            pins.append(it)
        elif PIN_DIGIT_RE.match(text):
            ambiguous.append(it)
        else:
            excluded.append(it)
    return {
        "connectors": connectors,
        "pins": pins,
        "ambiguous": ambiguous,
        "excluded": excluded,
    }


def crop_around_point(image_bgr, x, y, half_size=110):
    """image_bgr에서 (x,y)를 중심으로 한 변 (half_size*2)px 정사각형을
    잘라낸다. 좌표가 이미지 경계에 가까우면 잘리는 범위를 이미지 안으로
    당겨서 보정한다(중심이 정확히 안 맞더라도 잘림 없이 항상 정사각형을
    반환). 사람이 "이 숫자가 원(지시선) 안에 있는지" 실제 그림으로
    판단할 수 있게 후처리 필터 GUI에서 쓴다."""
    h, w = image_bgr.shape[:2]
    size = half_size * 2
    x0 = int(x) - half_size
    y0 = int(y) - half_size
    x0 = max(0, min(x0, w - size)) if w >= size else 0
    y0 = max(0, min(y0, h - size)) if h >= size else 0
    x1 = min(w, x0 + size)
    y1 = min(h, y0 + size)
    return image_bgr[y0:y1, x0:x1]


def _cluster_pins_into_columns(pins, x_tol=40, gap_ratio=1.8, gap_abs_min=60):
    """pins를 x좌표가 비슷한(±x_tol) 것끼리 1차로 "열(column)"로 묶고,
    같은 열 안에서 y간격이 그 열의 "평소 간격"보다 훨씬 크게 벌어지는
    지점에서 별개 그룹(같은 x열에 다른 커넥터의 핀이 이어지는 경우)으로
    2차 분리한다. 커넥터 핀 문자(A,B,C...)는 실측 도면 전부에서 같은
    커넥터면 거의 같은 x좌표에 세로로 촘촘히 정렬되어 나타난다(예:
    A20016147의 P4는 x=1391에 A~E가 간격 70씩 y=330~610로 이어짐).
    개별 핀을 하나씩 최근접 커넥터에 배정하면 열 중간의 핀 하나가 옆
    커넥터한테 잘못 가로채이는 문제가 생기므로(실측: A20016147에서 P4열의
    첫 핀 A가 근처 P2에 잘못 배정됨), 열 전체를 하나의 단위로 취급해
    통째로 같은 커넥터에 배정한다. 다만 P3의 A,B(y=750,820, 간격70)와
    P4의 A~E(y=330~610, 간격70)가 같은 x=1391에 있어 고정 문턱값으로는
    분리가 안 되므로(실측: 두 그룹 사이 간격 140 vs 내부 간격 70 - 절대
    문턱값 150을 쓰면 못 갈랐음), "직전까지의 평균 간격 대비 gap_ratio
    배" 초과를 기준으로 상대적으로 판단한다(핀 1개짜리 열은 비교 기준이
    없으므로 gap_abs_min을 최소 문턱값으로 같이 적용)."""
    remaining = sorted(pins, key=lambda p: p.get("x", 0))
    x_columns = []
    for p in remaining:
        placed = False
        for col in x_columns:
            if abs(col[-1].get("x", 0) - p.get("x", 0)) <= x_tol:
                col.append(p)
                placed = True
                break
        if not placed:
            x_columns.append([p])

    columns = []
    for col in x_columns:
        col.sort(key=lambda p: p.get("y", 0))
        sub = [col[0]]
        gaps = []
        for p in col[1:]:
            gap = p.get("y", 0) - sub[-1].get("y", 0)
            avg_gap = (sum(gaps) / len(gaps)) if gaps else gap
            is_break = gap > gap_abs_min and gap > avg_gap * gap_ratio
            if is_break:
                columns.append(sub)
                sub = [p]
                gaps = []
            else:
                sub.append(p)
                gaps.append(gap)
        columns.append(sub)
    return columns


def assemble_connector_pins(connectors, pins, x_tol=40, gap_ratio=1.8, gap_abs_min=60):
    """3차 필터: 2차 필터(classify_pin_items)로 이미 걸러낸 connectors/pins
    리스트를 받아 커넥터에 배정한다. 핀 하나씩 최근접 커넥터를 찾는 대신,
    먼저 x좌표가 비슷한 핀들을 "열"로 묶고(_cluster_pins_into_columns),
    그 열의 무게중심과 가장 가까운 커넥터에 열 전체를 통째로 배정한다.
    도면마다 레이아웃이 달라(커넥터명이 핀 위/옆/커넥터 사이 등) 좌표
    근접성만으로는 100% 정확한 배정이 보장되지 않으므로, 이 함수의 결과는
    항상 GUI에서 사람이 검증/재배치하는 것을 전제로 한다.
    반환: {커넥터텍스트: [{"text","x","y"}, ...]} (같은 커넥터명이
    두 번 나오면 좌표가 다른 별개 인스턴스이므로 "이름#순번" 키로 구분)"""
    if not connectors:
        return {}
    conn_entries = []
    for i, c in enumerate(connectors):
        key = c["text"] if sum(1 for cc in connectors if cc["text"] == c["text"]) == 1 else f"{c['text']}#{i}"
        conn_entries.append({"key": key, "text": c["text"], "x": c.get("x", 0), "y": c.get("y", 0), "pins": []})

    for col in _cluster_pins_into_columns(pins, x_tol=x_tol, gap_ratio=gap_ratio, gap_abs_min=gap_abs_min):
        cx = sum(p.get("x", 0) for p in col) / len(col)
        cy = sum(p.get("y", 0) for p in col) / len(col)
        nearest = min(
            conn_entries,
            key=lambda c: (c["x"] - cx) ** 2 + (c["y"] - cy) ** 2)
        nearest["pins"].extend(col)

    for c in conn_entries:
        c["pins"].sort(key=lambda p: (p.get("y", 0), p.get("x", 0)))

    return {c["key"]: c["pins"] for c in conn_entries}


def process_image_folder_extract_text(folder, max_workers, progress_cb=None):
    """folder 안의 모든 PNG(자유곡선으로 이미 잘라낸 배선도 크롭 이미지들,
    사용자가 GUI로 55개를 미리 만들어둠)에서 텍스트+좌표를 뽑아 파일마다
    {도면번호}.json으로 저장한다(한글 항목은 filter_out_hangul로 제외).
    핀 문자(A,B,C... 커넥터 격자 칸)는 먼저 extract_pins_via_grid로
    시도한다 - 격자선을 이미지 처리로 검출해 셀 좌표를 기하학적으로
    계산하므로 100% 정확하다(VLM이 좌표를 추정만 해서 이미지 범위를
    벗어난 값을 준 사례가 있었음 - 사용자 지적, A20016147의 B핀 y=820이
    타일 높이 766 초과). 격자(박스)가 없는 도면(예: A20016147처럼
    테두리 없이 문자만 나열된 스타일)에서는 격자 자체가 검출되지 않으니
    이 부분은 폴백 없이 비워둔다.
    그와 별개로, 커넥터명/신호명 등 격자 방식이 다루지 못하는 텍스트는
    항상 기존 extract_all_text_tiled(타일 분할 + VLM)로 뽑는다. 두
    결과를 합치되, 격자 결과와 좌표가 아주 가까운(±30px) 타일 결과는
    같은 핀을 가리키는 중복으로 보고 격자 쪽(좌표 확정)을 우선한다.
    도면마다 크기가 제각각이므로 타일 그리드는 고정값을 받지 않고
    파일별로 compute_auto_tile_grid로 자동 계산한다(사용자 지시 -
    PNG마다 자동으로 맞춰서 진행). 파일들은 GUI 세션과 무관하게 이미
    확정된 크롭 이미지이므로, GUI를 계속 붙들지 않고 파일 목록만
    훑으면 된다.
    반환: [{"file":str, "count":int, "tile_errors":[...]} 또는
           {"file":str, "error":str}, ...] (파일별 처리 요약)."""
    png_files = sorted(
        f for f in os.listdir(folder) if f.lower().endswith(".png"))
    # 1차 추출 결과는 폴더 바로 아래가 아니라 "1_Raw_JSON" 서브폴더에
    # 모은다(사용자 지시 - 1차/2차/3차 결과를 번호가 붙은 폴더로 구분).
    raw_dir = os.path.join(folder, "1_Raw_JSON")
    os.makedirs(raw_dir, exist_ok=True)

    # hboxes.json(3-B2에서 사람이 지정한 가로 격자 박스)을 PNG와 같은
    # 폴더뿐 아니라 그 아래 모든 하위폴더까지 재귀적으로 찾는다(사용자가
    # "Horizental-Box" 같은 별도 하위폴더로 정리해서 보관하길 원함).
    # 도면번호가 같은 파일이 여러 곳에 있으면 먼저 찾은(os.walk 순서상
    # 얕은 폴더 우선) 것을 쓴다.
    hbox_map = {}
    for dirpath, _, filenames in os.walk(folder):
        for fn in filenames:
            if fn.endswith(".hboxes.json"):
                drawing_key = fn[:-len(".hboxes.json")]
                hbox_map.setdefault(drawing_key, os.path.join(dirpath, fn))

    results = []
    for i, fname in enumerate(png_files):
        path = os.path.join(folder, fname)
        drawing_no = os.path.splitext(fname)[0]
        try:
            img = cv2.imread(path)
            if img is None:
                results.append({"file": fname, "error": "이미지를 읽지 못함"})
                continue

            # 3-B2(가로 격자 박스 지정)에서 사람이 저장해둔 수동 박스가
            # 있으면 읽어서 넘긴다. 파일이 없으면 이 도면엔 가로 격자가
            # 없다는 뜻이므로 그냥 빈 목록으로 진행(세로 격자 자동검출만
            # 적용됨, 기존 동작과 동일).
            hbox_path = hbox_map.get(drawing_no)
            manual_horizontal_boxes = []
            if hbox_path and os.path.exists(hbox_path):
                try:
                    with open(hbox_path, encoding="utf-8") as f:
                        manual_horizontal_boxes = [tuple(b) for b in json.load(f)]
                except (json.JSONDecodeError, OSError):
                    manual_horizontal_boxes = []

            grid_items, box_regions = extract_pins_via_grid(
                img, max_workers=max_workers,
                manual_horizontal_boxes=manual_horizontal_boxes)

            cols, rows = compute_auto_tile_grid(img)
            tile_items, tile_errors = extract_all_text_tiled(
                img, cols=cols, rows=rows, max_workers=max_workers)

            # 격자 박스 영역(box_regions) 안에 들어오는 tile_items는 버린다.
            # 격자 칸 옆의 작은 배선 연결 심볼을 VLM이 핀 문자로 잘못
            # 읽어 같은 글자가 격자 결과와 거의 같은 y, 다른 x로 중복
            # 등장하는 문제가 있었다(실측: 50073721에서 격자 x=127의
            # A~U와 거의 같은 y에 x=80으로 A~U가 또 검출됨 - x 차이가
            # 47~128px라 예전의 ±30px 좌표 근접 dedup으로는 못 걸렀음).
            merged = list(grid_items)
            for it in tile_items:
                text = it.get("text", "").strip()
                # 커넥터명(J14, J26 등)은 박스 영역 필터에서 항상 예외로
                # 둔다 - box_region의 y0 margin(-20)이 타이트해서, tile
                # VLM이 매 실행마다 살짝 다른 좌표를 주면(비결정적) 커넥터
                # 바로 위에 있는 커넥터명이 근소한 차이로 걸러지는 사고가
                # 있었다(실측: 50073721 재실행에서 J26 y=135가 region
                # y0=136 경계에 걸려 통째로 사라짐). 커넥터명은 애초에 A~U
                # 심볼 오인식 문제와 무관하므로 항상 살려둔다.
                if CONNECTOR_NAME_RE.match(text):
                    merged.append(it)
                    continue
                in_box = any(
                    bx0 <= it["x"] <= bx1 and by0 <= it["y"] <= by1
                    for bx0, by0, bx1, by1 in box_regions)
                if not in_box:
                    merged.append(it)

            items = filter_out_hangul(merged)
            items = filter_out_stray_reference_numbers(items)
            out_path = os.path.join(raw_dir, f"{drawing_no}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(items, f, ensure_ascii=False, indent=2)
            results.append({
                "file": fname, "count": len(items),
                "grid_count": len(grid_items), "tile_errors": tile_errors})
        except Exception as e:
            results.append({"file": fname, "error": str(e)})
        if progress_cb:
            progress_cb(i + 1, len(png_files), fname)
    return results


# Gemma가 판독한 (셀 이미지, 정답 글자) 쌍을 쌓아두는 폴더. TrOCR 파인튜닝용
# 데이터셋으로 그대로 사용한다.
TRAIN_DATA_DIR = os.path.join(os.path.dirname(__file__), "trocr_finetune_data")

# 배선도/회로도/결선도 영역 자동 검출용 제목 키워드. 도면 관례상 이 제목은
# 보통 다이어그램 바로 아래에 캡션처럼 붙고, 4분할 기준 좌측 하단 ~ 중단
# 근처에 위치한다(사용자 관찰 + 27개 실패 사례 실측 확인, 일부 도면은
# 제목이 세로 중앙(y=0.45h)까지 올라와 있었음).
DIAGRAM_TITLE_KEYWORDS = ["배선도", "회로도", "결선도", "배선", "회로", "결선"]
# "배선공차", "표2. 배선공차" 처럼 공차표 제목에도 "배선"이 들어있어서
# 키워드만으로는 진짜 배선도 제목과 구분이 안 된다(실측: A60026426-1에서
# "배선공차"가 prob=1.00으로 오검출됨). 공차표는 제목에 항상 "공차"가
# 붙으므로 이 단어가 있으면 후보에서 제외한다.
DIAGRAM_TITLE_EXCLUDE = ["공차", "도면번호", "도면크기", "관련도면"]
# "배선/회로/결선"(2글자) 단독 키워드는 "회로번호 표시", "배선방법" 처럼
# 무관한 문구에도 흔하게 걸린다(실측: A60023186 "회로번호 표시"
# prob=1.00 오검출). 이 2글자 다음에 "도"가 아닌 다른 글자가 곧바로
# 이어지면(=제목이 "배선도"류가 아니라는 뜻) 후보에서 제외한다.
DIAGRAM_TITLE_2CHAR_SUFFIX_BLOCK = ["번호", "방법", "설명", "공차", "표시", "기호"]


def _edit_distance(a, b):
    """두 문자열 사이의 레벤슈타인 편집거리(삽입/삭제/치환 1회당 1)."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def _fuzzy_contains(text, keyword, max_dist=1):
    """text 안에 keyword와 같은 길이의 부분 문자열 중 편집거리(치환만,
    삽입/삭제 없음) max_dist 이내인 것이 있는지 검사한다. 스캔/인쇄
    품질이 낮은 도면에서 "배선도"가 "버선도", "비선도" 처럼 한 글자가
    깨져서 완전 일치(in 연산)로는 못 잡는 경우를 구제하기 위함(사용자
    스크린샷으로 실측 확인된 오인식 패턴).
    길이를 keyword와 고정으로 맞춰서만 비교한다 - 길이가 다른 부분
    문자열까지 허용하면(이전 구현) "조립로도" 같은 무관한 4글자 조각이
    "회로도"(3글자)와 편집거리 1로 오매칭되는 문제가 있었다(실측).
    매칭된 위치 바로 뒤에 DIAGRAM_TITLE_2CHAR_SUFFIX_BLOCK 단어가
    이어지면 그 매칭은 인정하지 않는다("회로번호"의 "회로번"이 "회로도"와
    편집거리 1로 오매칭되던 문제, 실측: A60023186 "회로번호 표시")."""
    klen = len(keyword)
    for i in range(len(text) - klen + 1):
        sub = text[i:i + klen]
        if _edit_distance(sub, keyword) <= max_dist:
            # 매칭 윈도우 끝(i+klen)부터가 아니라, 치환으로 흡수됐을 수 있는
            # 앞쪽 1글자(max_dist)까지 포함해 넓게 뒤 문맥을 본다("회로번호"의
            # "회로번"이 "회로도"와 편집거리 1로 걸릴 때, 그 뒤가 "호표시"라서
            # 매칭 끝 지점만 보면 "번호"라는 단어를 놓쳤던 문제).
            rest = text[max(0, i + klen - max_dist):]
            if any(suf in rest[:len(suf) + max_dist] for suf in DIAGRAM_TITLE_2CHAR_SUFFIX_BLOCK):
                continue
            return True
    return False


def _has_blocked_2char_match(text, keyword):
    """text 안에서 2글자 키워드(예: '회로')가 나타난 위치 바로 뒤에
    DIAGRAM_TITLE_2CHAR_SUFFIX_BLOCK에 속한 단어가 곧바로 이어지는지
    검사한다("회로번호", "배선방법" 등). 이런 경우는 "배선도"류 제목이
    아니라고 보고 이 매칭을 무효화한다."""
    start = 0
    while True:
        idx = text.find(keyword, start)
        if idx == -1:
            return False
        rest = text[idx + len(keyword):]
        if any(rest.startswith(suf) for suf in DIAGRAM_TITLE_2CHAR_SUFFIX_BLOCK):
            return True
        start = idx + 1


def _search_title_in_roi(reader, img, roi_x0, roi_y0, roi_x1, roi_y1):
    """지정된 영역(원본 이미지 좌표계) 안에서 배선도류 제목 텍스트를
    찾는다. 공차표 제목("배선공차" 등)은 "공차"라는 단어로 걸러서
    제외한다. 완전 일치뿐 아니라 편집거리 1 이내의 유사 텍스트("버선도",
    "비선도" 등 OCR 오인식)도 신뢰도 0.3 이상일 때 후보로 잡는다.
    반환: [(text, prob, (x0,y0,x1,y1)), ...] (원본 이미지 좌표계)"""
    roi = img[roi_y0:roi_y1, roi_x0:roi_x1]
    results = reader.readtext(roi)
    candidates = []
    for bbox, text, prob in results:
        clean = text.strip()
        clean_no_space = clean.replace(" ", "")  # 띄어쓰기 무시 ("배 선 도" -> "배선도")
        if any(kw in clean_no_space for kw in DIAGRAM_TITLE_EXCLUDE):
            continue
        # "회로번호", "배선방법" 처럼 2글자 키워드 뒤에 무관한 단어가 곧장
        # 붙는 경우("도"가 아닌 경우)는 그 2글자 키워드로 인한 매칭을
        # 인정하지 않는다(실측: A60023186 "회로번호 표시" prob=1.00 오검출).
        exact_hit = any(
            kw in clean_no_space for kw in DIAGRAM_TITLE_KEYWORDS
            if len(kw) >= 3 or not _has_blocked_2char_match(clean_no_space, kw))
        # fuzzy(편집거리 허용) 매칭은 완전 일치보다 오검출 위험이 크므로,
        # 신뢰도가 너무 낮은 검출(도형/노이즈 오인식)은 애초에 대상에서
        # 제외한다(실측: "재선도" prob=0.06 오검출 사례).
        fuzzy_hit = prob >= 0.3 and any(
            _fuzzy_contains(clean_no_space, kw) for kw in DIAGRAM_TITLE_KEYWORDS if len(kw) >= 3)
        if exact_hit or fuzzy_hit:
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            x0, x1 = min(xs) + roi_x0, max(xs) + roi_x0
            y0, y1 = min(ys) + roi_y0, max(ys) + roi_y0
            candidates.append((clean, prob, (x0, y0, x1, y1)))
    return candidates


def find_diagram_title(reader, img):
    """원본 도면 전체 이미지에서 배선도류 제목 텍스트를 찾는다. 기본은
    좌측 하단(y: 0.2h~h, x: 0~w/2)을 먼저 탐색하고 - 대부분의 도면이
    여기 해당(사용자 관찰) - 거기서 후보를 하나도 못 찾으면 좌측 상단
    (y: 0~0.5h, x: 0~w/2)을 추가로 탐색한다. 하단 탐색을 우선하고
    상단은 못 찾았을 때만 시도하는 이유는, 두 영역을 항상 같이 열면
    상단의 타이틀블록/부품표 텍스트가 후보로 섞여 최고 신뢰도 선택이
    흔들리기 때문(실측: 성공하던 도면이 실패로 바뀜)."""
    h, w = img.shape[:2]
    candidates = _search_title_in_roi(reader, img, 0, int(h * 0.2), w // 2, h)
    if candidates:
        return candidates
    return _search_title_in_roi(reader, img, 0, 0, w // 2, int(h * 0.5))


def _find_right_divider(img, y0, y1, cx, w, search_ratio=0.30, min_gap_ratio=0.01):
    """배선도와 부품목록표 사이는 항상 빈 여백(세로 방향으로 잉크가 아예
    없는 좁은 흰 띠)으로 떨어져 있다. 예전에는 "높이의 70% 이상 이어지는
    수직선"으로 구분선을 찾았는데, 표의 촘촘한 셀 경계선도 같은 조건을
    만족해서 표 내부나 표를 넘어간 지점을 구분선으로 오검출했다(실측:
    "키 배열 결선도" 스크린샷에서 표1/표2가 통째로 배선도에 딸려 나옴).
    빈 여백 기준으로 찾으면 표 앞에서 확실히 멈춘다. 제목 중심(cx) 기준
    오른쪽 search_ratio*w 범위 안에서 첫 번째 빈 여백 띠(폭 min_gap_ratio*w
    이상)의 시작 x좌표를 반환한다. 못 찾으면 None(호출부 고정 비율 폴백)."""
    h_img, w_img = img.shape[:2]
    search_x1 = min(w_img, int(cx + w * search_ratio))
    band = img[y0:y1, int(cx):search_x1]
    if band.size == 0:
        return None
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY) if band.ndim == 3 else band
    _, bin_img = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    col_has_ink = bin_img.sum(axis=0) > 0
    min_gap = max(3, int(w * min_gap_ratio))

    n = len(col_has_ink)
    i = 0
    # 배선도 내용(잉크)이 시작되기 전까지는 건너뛴다 - anchor 바로 오른쪽은
    # 아직 배선도 안쪽일 수 있어서, 첫 잉크 구간을 지난 다음부터 "그 뒤에
    # 오는 첫 공백 띠"를 진짜 구분선 후보로 본다.
    while i < n and not col_has_ink[i]:
        i += 1
    while i < n:
        if not col_has_ink[i]:
            j = i
            while j < n and not col_has_ink[j]:
                j += 1
            if j - i >= min_gap:
                return int(cx) + i
            i = j
        else:
            i += 1
    return None


def _expand_from_anchor(mask, anchor, gap):
    """mask(bool 배열)에서 anchor 위치를 포함하는 "내용이 있는 덩어리"를
    양쪽으로 넓혀간다. gap 길이 이상 연속으로 내용이 없는 공백을 만나면
    그 직전에서 멈춘다. anchor 자체가 공백일 수 있으므로(실측: 제목 바로
    위는 캡션과 다이어그램 사이 여백이라 anchor가 공백에 놓임 -
    A60023104에서 이 여백만 보고 곧장 gap 판정이 나 버려서 배선도 전체가
    "배선도" 글자 한 줄만 남는 문제가 있었다), 먼저 anchor에서부터
    gap 범위 안의 첫 잉크 지점을 찾아 그곳으로 anchor를 옮긴 뒤 확장한다."""
    n = len(mask)
    anchor = min(max(anchor, 0), n - 1)

    if not mask[anchor]:
        found = None
        for d in range(1, gap * 4 + 1):
            if anchor + d < n and mask[anchor + d]:
                found = anchor + d
                break
            if anchor - d >= 0 and mask[anchor - d]:
                found = anchor - d
                break
        if found is not None:
            anchor = found
        else:
            return anchor, anchor + 1

    lo = anchor
    empty_run = 0
    while lo > 0:
        if mask[lo - 1]:
            empty_run = 0
        else:
            empty_run += 1
            if empty_run > gap:
                break
        lo -= 1

    hi = anchor
    empty_run = 0
    while hi < n - 1:
        if mask[hi + 1]:
            empty_run = 0
        else:
            empty_run += 1
            if empty_run > gap:
                break
        hi += 1

    return lo, hi + 1


def _tight_content_bbox(img, x0, y0, x1, y1, anchor_x, anchor_y, row_gap_ratio=0.02, min_row_ratio=0.003):
    """지정된 넉넉한 탐색 영역(x0,y0,x1,y1) 안에서, (anchor_x, anchor_y)
    (보통 제목 텍스트 위치)를 포함하는 "내용이 이어진 덩어리"만의 타이트한
    바운딩 박스를 구한다. 배선도는 도면마다 실제 높이/폭이 크게 다른데
    (실측: 어떤 도면은 세로로 좁고, 어떤 도면은 커넥터가 많아 가로로
    길다) 고정 비율로 자르면 내용이 잘리거나, 반대로 탐색 범위를 넓히면
    공백 띠 건너 케이블 형상도 일부까지 딸려 들어온다(실측: A60023104에서
    vertical_ratio만 키웠더니 위쪽 상세도 조각이 같이 포함됨). anchor에서
    출발해 공백 띠(row_gap_ratio 이상)를 만나면 거기서 멈추는 방식으로
    막는다."""
    h_img, w_img = img.shape[:2]
    x0 = max(0, x0); y0 = max(0, y0)
    x1 = min(w_img, x1); y1 = min(h_img, y1)
    band = img[y0:y1, x0:x1]
    if band.size == 0:
        return x0, y0, x1, y1
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY) if band.ndim == 3 else band
    _, bin_img = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    bh, bw = bin_img.shape

    row_has_ink = bin_img.sum(axis=1) / 255 > max(1, bw * min_row_ratio)
    col_has_ink = bin_img.sum(axis=0) / 255 > max(1, bh * min_row_ratio)

    row_gap = max(2, int(bh * row_gap_ratio))
    col_gap = max(2, int(bw * row_gap_ratio))

    local_ay = min(max(anchor_y - y0, 0), bh - 1)
    local_ax = min(max(anchor_x - x0, 0), bw - 1)

    ry0, ry1 = _expand_from_anchor(row_has_ink, local_ay, row_gap)
    rx0, rx1 = _expand_from_anchor(col_has_ink, local_ax, col_gap)

    return x0 + rx0, y0 + ry0, x0 + rx1, y0 + ry1


def crop_diagram_region(img, title_bbox, vertical_ratio=0.35, side_ratio=0.30, margin_ratio=0.01):
    """제목의 위치(Y좌표)를 파악해 배선도가 있을 법한 넉넉한 탐색 영역을
    먼저 잡고(제목이 아래에 있으면 위쪽, 위에 있으면 아래쪽), 그 안에서
    _tight_content_bbox로 실제 선/글자가 그려진 부분만의 타이트한 경계를
    구한다. 이전에는 고정 비율(vertical_ratio=0.16)만큼만 잘랐는데, 배선도
    실제 높이가 도면마다 달라 더 큰 도면은 위쪽이 잘리는 문제가 있었다
    (실측: "성공"으로 표시된 것도 일부만 나온 경우 다수). 넉넉한 탐색
    영역(vertical_ratio=0.35로 확대)을 먼저 주고 내용 경계로 다시 타이트하게
    맞추는 2단계 방식으로 바꿔서, 도면 크기에 관계없이 실제 내용 전체를
    포함하게 한다."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = title_bbox
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2

    if cy > h * 0.5:
        # 타이틀이 이미지 아래쪽에 있음 -> 도면은 타이틀 '위'에 있음.
        # 배선도는 제목 바로 위에서부터 시작하므로, anchor를 그 지점
        # (search_y1 바로 안쪽)으로 잡아야 내용 확장이 올바른 덩어리에서
        # 출발한다 - anchor를 탐색 범위 중앙에 두면 그 위치가 우연히
        # 공백이라 확장이 아예 안 되는 경우가 있었다.
        search_y1 = int(y0)
        search_y0 = max(0, int(y0 - h * vertical_ratio))
        anchor_y = search_y1 - 1
    else:
        # 타이틀이 이미지 위쪽에 있음 -> 도면은 타이틀 '아래'에 있음
        search_y0 = int(y1)
        search_y1 = min(h, int(y1 + h * vertical_ratio))
        anchor_y = search_y0

    search_x0 = max(0, int(cx - w * side_ratio))
    divider_x = _find_right_divider(img, search_y0, search_y1, cx, w, search_ratio=side_ratio)
    search_x1 = divider_x if divider_x is not None else min(w, int(cx + w * side_ratio))

    tx0, ty0, tx1, ty1 = _tight_content_bbox(img, search_x0, search_y0, search_x1, search_y1,
                                              anchor_x=int(cx), anchor_y=anchor_y)

    # 타이트하게 잡은 경계 바로 위/아래/옆에서 선/글자 끝이 살짝 잘리지
    # 않도록 페이지 크기 대비 작은 여백을 둔다.
    mx = int(w * margin_ratio)
    my = int(h * margin_ratio)
    crop_x0 = max(0, tx0 - mx)
    crop_y0 = max(0, ty0 - my)
    crop_x1 = min(w, tx1 + mx)
    crop_y1 = min(h, ty1 + my)
    return crop_x0, crop_y0, crop_x1, crop_y1


def _find_outer_frame(img, border_search_ratio=0.15):
    """도면 용지의 굵은 외곽 테두리(도면번호/격자 눈금이 붙은 프레임)
    위치를 찾는다. 페이지 가장자리 border_search_ratio 범위 안에서
    폭/높이의 50% 이상 이어지는 가장 굵은 가로/세로선을 프레임으로
    본다. 못 찾으면 페이지 전체를 프레임으로 간주한다."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    _, bin_img = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    top_band = int(h * border_search_ratio)
    bottom_band = h - top_band
    left_band = int(w * border_search_ratio)
    right_band = w - left_band

    row_sum = bin_img.sum(axis=1) / 255
    col_sum = bin_img.sum(axis=0) / 255

    top_candidates = np.where(row_sum[:top_band] > w * 0.5)[0]
    bottom_candidates = np.where(row_sum[bottom_band:] > w * 0.5)[0]
    left_candidates = np.where(col_sum[:left_band] > h * 0.5)[0]
    right_candidates = np.where(col_sum[right_band:] > h * 0.5)[0]

    frame_y0 = int(top_candidates.max()) + 1 if len(top_candidates) else 0
    frame_y1 = bottom_band + int(bottom_candidates.min()) if len(bottom_candidates) else h
    frame_x0 = int(left_candidates.max()) + 1 if len(left_candidates) else 0
    frame_x1 = right_band + int(right_candidates.min()) if len(right_candidates) else w

    return frame_x0, frame_y0, frame_x1, frame_y1


def _find_center_gap(arr, lo_ratio=0.3, hi_ratio=0.7, thresh=None, thresh_margin=1.5):
    """1차원 잉크량 배열(arr)의 중앙 lo_ratio~hi_ratio 구간에서 값이
    thresh 이하로 거의 비어있는 가장 넓은 연속 구간을 찾는다. 도면
    한가운데는 격자 눈금/방향 화살표 등으로 완전히 0은 아니지만 거의
    비어있는 띠가 있다. thresh를 고정 픽셀값으로 주면 렌더링 zoom
    배율에 따라 같은 선도 두께(픽셀 수)가 달라져서 깨진다(실측: zoom=2.0
    에서는 최저 잉크량이 1픽셀이었는데 zoom=3.0에서는 같은 선이 21픽셀로
    나와 고정 thresh=5를 넘어서 gap을 하나도 못 찾음). 그래서 thresh를
    주지 않으면 이 구간의 최저 잉크값 * thresh_margin으로 동적으로
    잡는다 - 해상도가 달라져도 "이 구간에서 가장 옅은 선 근방"을 항상
    공백 후보로 인정한다.
    반환: (start, end, width) - 못 찾으면 (0,0,0)."""
    n = len(arr)
    lo, hi = int(n * lo_ratio), int(n * hi_ratio)
    seg = arr[lo:hi]
    if len(seg) == 0:
        return (0, 0, 0)
    if thresh is None:
        thresh = max(1.0, float(seg.min()) * thresh_margin)
    near_empty = seg <= thresh
    best = (0, 0, 0)
    i = 0
    while i < len(near_empty):
        if near_empty[i]:
            j = i
            while j < len(near_empty) and near_empty[j]:
                j += 1
            if j - i > best[2]:
                best = (lo + i, lo + j, j - i)
            i = j
        else:
            i += 1
    return best


def _find_all_gaps(arr, thresh_margin=1.5, min_gap_len_ratio=0.01, min_segment_ratio=0.05):
    """1차원 잉크량 배열(arr) 전체에서 "내용이 거의 없는 공백 띠"를
    모두 찾아 그 중앙 좌표 리스트를 반환한다(경계 0, len(arr) 제외).
    임계값은 전체 배열의 최저 잉크량 * thresh_margin으로 동적으로 잡아서
    렌더링 해상도가 달라져도 일관되게 동작한다(_find_center_gap과 동일
    이유). 폭이 min_gap_len_ratio*len(arr) 이상인 공백만 진짜 구획
    경계로 인정하고, 양쪽 세그먼트가 너무 얇으면(min_segment_ratio 미만)
    버려서 도면 테두리 바로 안쪽의 얇은 여백을 분할선으로 잘못 잡지
    않게 한다."""
    n = len(arr)
    if n == 0:
        return []
    thresh = max(1.0, float(arr.min()) * thresh_margin)
    near_empty = arr <= thresh
    min_gap_len = max(2, int(n * min_gap_len_ratio))
    min_segment = max(1, int(n * min_segment_ratio))

    gaps = []
    i = 0
    while i < n:
        if near_empty[i]:
            j = i
            while j < n and near_empty[j]:
                j += 1
            if j - i >= min_gap_len:
                gaps.append((i, j))
            i = j
        else:
            i += 1

    # 세그먼트(구획)가 너무 얇아지는 분할선은 버린다.
    boundaries = [0] + [ (g[0]+g[1])//2 for g in gaps ] + [n]
    kept = [0]
    for b in boundaries[1:-1]:
        if b - kept[-1] >= min_segment:
            kept.append(b)
    if kept[-1] != n:
        if n - kept[-1] < min_segment and len(kept) > 1:
            kept.pop()
        kept.append(n)
    return kept[1:-1]  # 양끝(0, n)은 제외하고 내부 분할선만


def split_into_sections(img):
    """도면 용지를 실제 그어진 선이 아니라 내용물의 밀도 패턴으로 N개
    섹션(그리드)으로 나눈다. 분면 개수를 4개로 못박지 않고, 가로/세로
    각각에서 발견되는 모든 공백 분할선을 기준으로 그리드를 만든다 -
    도면마다 섹션 배치가 2단, 3단 등으로 다를 수 있어서(사용자 지시:
    "4분면이던 3분면이던 N분면이던 섹션별로 보면 된다"), 분할 개수를
    고정하지 않는 것이 더 일반적이다.
    반환: [(x0,y0,x1,y1), ...] (원본 이미지 좌표계, 섹션별 사각형 리스트)"""
    frame_x0, frame_y0, frame_x1, frame_y1 = _find_outer_frame(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    _, bin_img = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    inner = bin_img[frame_y0:frame_y1, frame_x0:frame_x1]
    ih, iw = inner.shape
    if ih == 0 or iw == 0:
        return []

    col_ink = inner.sum(axis=0) / 255
    row_ink = inner.sum(axis=1) / 255

    col_splits = [frame_x0] + [frame_x0 + x for x in _find_all_gaps(col_ink)] + [frame_x1]
    row_splits = [frame_y0] + [frame_y0 + y for y in _find_all_gaps(row_ink)] + [frame_y1]

    sections = []
    for ry0, ry1 in zip(row_splits[:-1], row_splits[1:]):
        for rx0, rx1 in zip(col_splits[:-1], col_splits[1:]):
            sections.append((rx0, ry0, rx1, ry1))
    return sections


def find_diagram_quadrant(reader, img):
    """도면을 내용물 밀도 기반 N개 섹션으로 나누고(split_into_sections),
    배선도류 제목("배선도/회로도/결선도")을 EasyOCR로 찾아 그 텍스트가
    속한 섹션 전체를 배선도 영역으로 채택한다. find_diagram_title +
    crop_diagram_region(제목 위치에서 고정 비율/gap으로 확장)보다
    안정적인 이유: 실제 도면이 갖고 있는 섹션 레이아웃 자체를 먼저
    확정하고 나서 "이 중 어디가 배선도인가"만 판정하므로, 형상도나
    부품목록표 조각이 섞여 들어올 여지가 훨씬 적다(실측: 50073721에서
    상세도까지 포함한 정확한 섹션 크롭 확인됨). 섹션이 정확히 4개로
    나뉘지 않는 도면(2단, 3단 등)에도 동일하게 동작한다.
    반환: (section_index, region(x0,y0,x1,y1), title_text, prob) 또는
    후보를 못 찾으면 None."""
    sections = split_into_sections(img)
    if not sections:
        return None
    candidates = find_diagram_title(reader, img)
    if not candidates:
        return None
    best = max(candidates, key=lambda c: c[1])
    tx0, ty0, tx1, ty1 = best[2]
    tcx, tcy = (tx0 + tx1) / 2, (ty0 + ty1) / 2

    for idx, (qx0, qy0, qx1, qy1) in enumerate(sections):
        if qx0 <= tcx <= qx1 and qy0 <= tcy <= qy1:
            return idx, (qx0, qy0, qx1, qy1), best[0], best[1]
    return None


def mask_outside_region(img, region, fill=(255, 255, 255)):
    """원본 페이지 크기/좌표계는 그대로 유지한 채, region(x0,y0,x1,y1)
    바깥 전체를 fill 색으로 덮어써서 케이블 형상도/부품목록표/타이틀블록을
    지운다(사용자 지시: 잘라내는 게 아니라 지우는 것 - 도면번호 등 원본
    좌표 참조가 필요할 수 있어 페이지 크기를 바꾸지 않는다)."""
    x0, y0, x1, y1 = region
    out = np.full_like(img, fill, dtype=img.dtype)
    out[y0:y1, x0:x1] = img[y0:y1, x0:x1]
    return out


def mask_outside_polygon(img, points, fill=(255, 255, 255)):
    """원본 페이지 크기/좌표계는 그대로 유지한 채, points(원본 이미지
    좌표계의 (x,y) 리스트로 이루어진 자유곡선/다각형) 바깥을 fill 색으로
    덮어쓴다. 실제 도면 경계는 직선 격자가 아니라 삐뚤빼뚤하고 계단식으로
    꺾인 손그림 경계인 경우가 많아서(사용자 예시), 직선 구획선 그리드
    (mask_outside_region)만으로는 못 따라가는 경계를 이 함수로 처리한다.
    폴리곤은 항상 닫힌 도형으로 취급한다(마지막 점과 첫 점을 자동으로
    잇는다)."""
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    pts = np.array([points], dtype=np.int32)
    cv2.fillPoly(mask, pts, 255)
    out = np.full_like(img, fill, dtype=img.dtype)
    out[mask == 255] = img[mask == 255]
    return out


def extract_vector_region(pdf_path, region_px, zoom, page_index=0, margin_pt=2.0):
    """원본 PDF에서 region_px(x0,y0,x1,y1, 래스터화에 쓴 zoom 배율 기준
    픽셀 좌표) 안에 놓인 벡터 도형(선/사각형/곡선)과 텍스트만 골라
    새 벡터 PDF로 저장한다. 지금까지의 크롭/마스킹은 래스터 이미지
    (PNG)만 다뤄서 확대하면 깨지고 선이 다시 벡터 좌표로 돌아갈 수
    없었다(사용자 지시: "벡터로 저장해야 한다") - PyMuPDF의
    get_drawings()/get_text("dict")는 원본 PDF가 갖고 있던 벡터 좌표를
    그대로 반환하므로, 그중 region 안에 있는 것만 새 페이지에 원본
    좌표 그대로 다시 그리면 벡터 상태가 유지된다.
    반환: (out_doc, saved_bool). out_doc은 호출부에서 save()/close() 처리."""
    doc = fitz.open(pdf_path)
    page = doc[page_index]
    x0_px, y0_px, x1_px, y1_px = region_px
    # 래스터 픽셀 좌표 -> PDF 포인트 좌표로 환산(래스터화에 쓴 zoom의 역수).
    x0 = x0_px / zoom - margin_pt
    y0 = y0_px / zoom - margin_pt
    x1 = x1_px / zoom + margin_pt
    y1 = y1_px / zoom + margin_pt
    clip = fitz.Rect(x0, y0, x1, y1)

    out_doc = fitz.open()
    out_page = out_doc.new_page(width=clip.width, height=clip.height)
    shape = out_page.new_shape()
    # 원본 좌표계를 그대로 쓰되, 출력 페이지 원점(0,0)이 clip.x0,y0에
    # 오도록 이동시키는 행렬. new_shape().draw_*는 좌표를 그대로 받고
    # commit(matrix=...)에서 변환을 적용한다.
    shift = fitz.Matrix(1, 0, 0, 1, -clip.x0, -clip.y0)

    drew_any = False
    for path in page.get_drawings():
        rect = path["rect"]
        if not clip.intersects(rect):
            continue
        for item in path["items"]:
            kind = item[0]
            if kind == "l":  # line
                shape.draw_line(item[1], item[2])
                drew_any = True
            elif kind == "re":  # rectangle
                shape.draw_rect(item[1])
                drew_any = True
            elif kind == "c":  # bezier curve
                shape.draw_bezier(item[1], item[2], item[3], item[4])
                drew_any = True
            elif kind == "qu":  # quad
                shape.draw_quad(item[1])
                drew_any = True
        color = path.get("color")
        fill = path.get("fill")
        width = path.get("width") or 0.5
        shape.finish(color=color, fill=fill, width=width,
                     closePath=path.get("closePath", False))

    doc.close()
    if drew_any:
        shape.commit(matrix=shift)
    else:
        shape.commit()

    # 텍스트는 도형과 별도 경로(insert_text)로 옮긴다 - get_drawings()는
    # 글자 획을 포함하지 않는다.
    src_doc = fitz.open(pdf_path)
    src_page = src_doc[page_index]
    text_dict = src_page.get_text("dict", clip=clip)
    for block in text_dict.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                sx0, sy0, sx1, sy1 = span["bbox"]
                if not clip.intersects(fitz.Rect(sx0, sy0, sx1, sy1)):
                    continue
                origin = span.get("origin", (sx0, sy1))
                out_page.insert_text(
                    (origin[0] - clip.x0, origin[1] - clip.y0),
                    span["text"],
                    fontsize=span.get("size", 8),
                    color=(0, 0, 0),
                )
    src_doc.close()

    return out_doc, True


class TemplateMatcherGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("배선도 자동 검출 + Gemma 앙상블 판독 툴")
        self.root.geometry("1500x900")

        self.filepath = None
        self.original_img = None
        self.cropped_img = None
        self.results = []
        self.extracted_text_items = []
        self.clean_img = None
        self.reader = None
        self.pdf_zoom = None
        self.crop_coords = None

        self.rect_id = None
        self.disp_scale = 1.0
        # 마우스 휠 확대/축소 배율. fit_scale(캔버스에 꽉 채우는 기본
        # 배율) * zoom_factor = 실제 disp_scale. 1.0이 원본 fit 크기.
        self.fit_scale = 1.0
        self.zoom_factor = 1.0

        # 자유곡선(폴리라인) 경계 그리기 상태(항상 활성 - 클릭할 때마다
        # 점 추가). 삐뚤빼뚤/계단식 경계(사용자 예시)를 따라 그릴 수
        # 있다. points는 원본 이미지 좌표계 (x,y) 리스트, 닫히면 폴리곤으로
        # 채택.
        self.polyline_points = []
        self.polyline_canvas_ids = []

        self.setup_ui()

    def setup_ui(self):
        control_frame = tk.Frame(self.root, width=350, padx=10, pady=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y)

        tk.Button(control_frame, text="📁 1. 파일 불러오기 (이미지/PDF)", command=self.load_file, height=2).pack(fill=tk.X, pady=(0, 10))

        tk.Button(control_frame, text="🧬 2-A. 그린 영역을 벡터 PDF로 저장\n(원본 PDF의 선/텍스트를 그대로 유지, 확대해도 깨지지 않음)", command=self.save_vector_region, bg="plum", height=2).pack(fill=tk.X, pady=(0, 10))

        tk.Label(control_frame, text="--- 자유곡선으로 경계 그리기 ---", font=("", 9, "bold")).pack(anchor='w', pady=(0, 5))
        tk.Label(control_frame, text="원본 이미지 위를 클릭해서 점을 찍어\n삐뚤빼뚤/계단식 경계를 따라 그리세요.\n마지막에 '경계 완성'을 누르면 자동으로\n첫 점과 이어져 폐곡선이 됩니다.",
                 fg="blue", justify=tk.LEFT).pack(anchor='w', pady=(0, 5))

        polyline_btn_row = tk.Frame(control_frame)
        polyline_btn_row.pack(fill=tk.X, pady=(0, 5))
        tk.Button(polyline_btn_row, text="↩ 마지막 점 취소", command=self.undo_polyline_point).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 3))
        tk.Button(polyline_btn_row, text="🗑 전체 초기화", command=self.clear_polyline).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(3, 0))

        tk.Button(control_frame, text="✅ 경계 완성 (안쪽만 남기고 마스킹)", command=self.finish_polyline, bg="khaki", height=2).pack(fill=tk.X, pady=(0, 5))

        tk.Button(control_frame, text="💾 잘라낸 배선도 PNG로 저장", command=self.save_cropped_png, bg="lightgreen", height=2).pack(fill=tk.X, pady=(0, 15))

        tk.Label(control_frame, text="--- 자동 검출(참고용, 정확도 낮음) ---", font=("", 8), fg="gray").pack(anchor='w')

        tk.Button(control_frame, text="🔍 1-B. 배선도 영역 자동 검출\n(\"배선도/회로도/결선도\" 제목 찾아서 자동 크롭)", command=self.process_auto_region, bg="lightcyan", height=2).pack(fill=tk.X, pady=(0, 5))

        tk.Button(control_frame, text="📦 1-C. 폴더 전체 배치 자동 검출\n(폴더 안 모든 PDF에서 자동 크롭해서 저장)", command=self.process_batch_auto_region, bg="lightsteelblue", height=2).pack(fill=tk.X, pady=(0, 15))

        tk.Label(control_frame, text="정답 시퀀스 강제 덮어쓰기").pack(anchor='w')
        self.entry_sequence = tk.Entry(control_frame)
        self.entry_sequence.pack(fill=tk.X, pady=(0, 20))

        tk.Label(control_frame, text="--- 추출 방식 ---", font=("", 9, "bold")).pack(anchor='w', pady=(0, 5))

        tk.Button(control_frame, text="🎯 3. Gemma(LM Studio) 앙상블 판독\n(같은 크롭을 N회 미세 지터링해서 다수결 확정)", command=self.process_local_vlm, bg="lightgreen", font=("", 9, "bold"), height=3).pack(fill=tk.X, pady=(0, 10))

        tk.Button(control_frame, text="🔤 3-B. 전체 텍스트 추출 (타일 분할 병렬)\n(격자로 나눠 동시 질의, 위치 포함 JSON으로)", command=self.process_extract_all_text, bg="lightblue", height=2).pack(fill=tk.X, pady=(0, 5))

        tk.Button(control_frame, text="📐 3-B2. 가로 격자 박스 지정\n(핀이 좌→우로 나열된 커넥터를 드래그로 직접 표시)", command=self.open_horizontal_box_window, bg="lightyellow", height=2).pack(fill=tk.X, pady=(0, 5))

        tk.Button(control_frame, text="📂 3-C. 폴더 전체 텍스트 추출 (PNG → JSON)\n(폴더 안 모든 PNG를 타일 분할 병렬 처리, 한글 제외)", command=self.process_folder_extract_text, bg="lightblue", height=2).pack(fill=tk.X, pady=(0, 15))

        tk.Button(control_frame, text="🧹 3-D. 후처리 필터 (커넥터명 + 핀만 남기기)\n(신호명/전압/AWG 등 제외, 애매한 숫자는 PNG로 확인)", command=self.open_pin_filter_window, bg="lightyellow", height=2).pack(fill=tk.X, pady=(0, 15))

        tk.Button(control_frame, text="🔗 3-E. 커넥터-핀 조립\n(핀을 가장 가까운 커넥터에 자동 배정 후 사람이 검증)", command=self.open_pin_assembly_window, bg="lightyellow", height=2).pack(fill=tk.X, pady=(0, 15))

        tk.Button(control_frame, text="✨ 4. 텍스트와 박스 클린 렌더링", command=self.render_clean, bg="lightgreen", height=2).pack(fill=tk.X, pady=(0, 10))

        tk.Button(control_frame, text="💾 클린 벡터 PDF / PNG 저장", command=self.save_image, height=2).pack(fill=tk.X)

        main_frame = tk.Frame(self.root)
        main_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        image_frame = tk.Frame(main_frame)
        image_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        orig_frame = tk.Frame(image_frame)
        orig_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        tk.Label(orig_frame, text="[ 원본 이미지 (클릭으로 경계점 추가, Ctrl+휠 확대/축소, 휠 스크롤) ]").pack()

        canvas_container = tk.Frame(orig_frame)
        canvas_container.pack(fill=tk.BOTH, expand=True)

        h_scroll = tk.Scrollbar(canvas_container, orient=tk.HORIZONTAL)
        v_scroll = tk.Scrollbar(canvas_container, orient=tk.VERTICAL)
        self.canvas_orig = tk.Canvas(canvas_container, bg="gray", cursor="cross",
                                      xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)
        h_scroll.config(command=self.canvas_orig.xview)
        v_scroll.config(command=self.canvas_orig.yview)

        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas_orig.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas_orig.bind("<ButtonPress-1>", self.on_button_press)
        # 윈도우/맥은 <MouseWheel>(delta), 리눅스는 <Button-4>/<Button-5>.
        self.canvas_orig.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas_orig.bind("<Button-4>", self.on_mouse_wheel)
        self.canvas_orig.bind("<Button-5>", self.on_mouse_wheel)

        proc_frame = tk.Frame(image_frame)
        proc_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        tk.Label(proc_frame, text="[ AI 추출 및 렌더링 결과 ]").pack()
        self.lbl_proc = tk.Label(proc_frame, bg="white")
        self.lbl_proc.pack(fill=tk.BOTH, expand=True)

        text_frame = tk.Frame(main_frame, height=250)
        text_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        text_frame.pack_propagate(False)

        tk.Label(text_frame, text="[ AI 비전 처리 로그 ]").pack(anchor='w')
        self.txt_log = scrolledtext.ScrolledText(text_frame, wrap=tk.WORD, height=10)
        self.txt_log.pack(fill=tk.BOTH, expand=True)

    def load_file(self):
        self.filepath = filedialog.askopenfilename(
            filetypes=[("All Supported", "*.pdf;*.png;*.jpg;*.jpeg;*.bmp"),
                       ("PDF Files", "*.pdf"),
                       ("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")]
        )
        if not self.filepath:
            return

        self.clean_img = None
        self.cropped_img = None
        self.results = []
        self.extracted_text_items = []
        self.pdf_zoom = None
        self.crop_coords = None
        self.polyline_points = []
        self.polyline_canvas_ids = []
        self.zoom_factor = 1.0
        self.txt_log.delete('1.0', tk.END)
        self.lbl_proc.configure(image='')
        self.canvas_orig.delete("all")

        if self.filepath.lower().endswith('.pdf'):
            self.load_pdf(self.filepath)
        else:
            self.original_img = cv2.imread(self.filepath)
            self.txt_log.insert(tk.END, f"이미지 로드 완료: {os.path.basename(self.filepath)}\n크롭할 영역을 마우스로 드래그하세요.\n")

        self.display_canvas_image()

    def load_pdf(self, pdf_path):
        try:
            self.txt_log.insert(tk.END, "PDF 고해상도 변환 중...\n")
            self.root.update()
            doc = fitz.open(pdf_path)
            page = doc[0]
            zoom = 3.0
            self.pdf_zoom = zoom
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

            if pix.n == 4:
                self.original_img = cv2.cvtColor(img_data, cv2.COLOR_RGBA2BGR)
            elif pix.n == 3:
                self.original_img = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
            else:
                self.original_img = cv2.cvtColor(img_data, cv2.COLOR_GRAY2BGR)

            self.txt_log.insert(tk.END, f"PDF 로드 완료. 드래그하세요.\n")
            doc.close()
        except Exception as e:
            messagebox.showerror("오류", f"PDF 오류:\n{e}")

    def on_button_press(self, event):
        """캔버스 클릭은 항상 자유곡선 경계에 점을 하나 추가한다(사용자
        지시: 직선 구획선/드래그 크롭은 없애고 자유곡선 하나만 남긴다).
        스크롤(줌 후 스크롤바 이동)된 상태에서는 event.x/y(위젯 좌표)가
        실제 캔버스 콘텐츠 좌표와 어긋나므로 canvasx/canvasy로 보정한다."""
        if self.original_img is None: return
        x = self.canvas_orig.canvasx(event.x)
        y = self.canvas_orig.canvasy(event.y)
        self._add_polyline_point(x, y)


    def _add_polyline_point(self, sx, sy):
        """캔버스 표시 좌표(sx,sy)를 원본 이미지 좌표계로 변환해 점
        목록에 추가하고, 점들을 잇는 선을 다시 그린다."""
        orig_x = int(sx / self.disp_scale)
        orig_y = int(sy / self.disp_scale)
        self.polyline_points.append((orig_x, orig_y))
        self.txt_log.insert(tk.END, f"경계점 추가: ({orig_x},{orig_y}) [총 {len(self.polyline_points)}개]\n")
        self.txt_log.see(tk.END)
        self._redraw_polyline()

    def _redraw_polyline(self):
        for cid in self.polyline_canvas_ids:
            self.canvas_orig.delete(cid)
        self.polyline_canvas_ids = []
        pts = self.polyline_points
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            dx0, dy0 = x0 * self.disp_scale, y0 * self.disp_scale
            dx1, dy1 = x1 * self.disp_scale, y1 * self.disp_scale
            cid = self.canvas_orig.create_line(dx0, dy0, dx1, dy1, fill='magenta', width=2)
            self.polyline_canvas_ids.append(cid)
        for (x, y) in pts:
            dx, dy = x * self.disp_scale, y * self.disp_scale
            cid = self.canvas_orig.create_oval(dx - 3, dy - 3, dx + 3, dy + 3, fill='magenta', outline='')
            self.polyline_canvas_ids.append(cid)

    def undo_polyline_point(self):
        if not self.polyline_points:
            return
        removed = self.polyline_points.pop()
        self.txt_log.insert(tk.END, f"경계점 취소: {removed}\n")
        self.txt_log.see(tk.END)
        self._redraw_polyline()

    def clear_polyline(self):
        self.polyline_points = []
        self._redraw_polyline()
        self.txt_log.insert(tk.END, "자유곡선 경계점을 모두 초기화했습니다.\n")
        self.txt_log.see(tk.END)

    def finish_polyline(self):
        """지금까지 찍은 점들을 닫힌 폴리곤으로 채택해 그 바깥을
        마스킹한다(mask_outside_polygon - 마지막 점과 첫 점이 자동으로
        이어진다). 미리보기(오른쪽 결과 패널)는 폴리곤의 바운딩박스로
        실제 크롭해서 케이블 도면 크기에 맞춰 보여준다(사용자 지시 -
        원본 페이지 크기 그대로 보여주면 도면이 작게 찌그러져 보임).
        self.crop_coords/self.cropped_img도 이 크롭 결과로 채택해서
        이후 벡터 저장/Gemma 판독에 그대로 쓸 수 있게 한다."""
        if self.original_img is None:
            messagebox.showwarning("경고", "먼저 파일을 불러와주세요.")
            return
        if len(self.polyline_points) < 3:
            messagebox.showwarning("경고", "점을 3개 이상 찍어 경계를 만들어주세요.")
            return

        masked_full_img = mask_outside_polygon(self.original_img, self.polyline_points)

        xs = [p[0] for p in self.polyline_points]
        ys = [p[1] for p in self.polyline_points]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        self.crop_coords = (x0, y0, x1, y1)
        self.cropped_img = masked_full_img[y0:y1, x0:x1]
        self.display_result_image(self.cropped_img, self.lbl_proc)

        # 닫힌 도형임을 보여주기 위해 첫 점과 마지막 점을 잇는 선도 그린다.
        x0p, y0p = self.polyline_points[0]
        x1p, y1p = self.polyline_points[-1]
        dx0, dy0 = x0p * self.disp_scale, y0p * self.disp_scale
        dx1, dy1 = x1p * self.disp_scale, y1p * self.disp_scale
        cid = self.canvas_orig.create_line(dx0, dy0, dx1, dy1, fill='magenta', width=2, dash=(4, 2))
        self.polyline_canvas_ids.append(cid)

        self.txt_log.insert(tk.END, f"\n자유곡선 경계 완성 ({len(self.polyline_points)}개 점). 바깥을 마스킹했습니다.\n")
        self.txt_log.see(tk.END)

    def save_cropped_png(self):
        """경계 완성(finish_polyline) 또는 자동 검출로 채택된
        self.cropped_img(배선도 크기에 맞춰 이미 크롭된 이미지)를 PNG로
        저장한다. 기존 저장 버튼들(save_image/save_vector_region)은
        각각 Gemma 판독 결과(clean_img+results)나 PDF 벡터가 있어야만
        동작해서, 경계만 그리고 바로 저장하고 싶을 때 쓸 버튼이 없었다
        (사용자 지시로 추가)."""
        if self.cropped_img is None:
            messagebox.showwarning("경고", "먼저 자유곡선으로 경계를 완성해주세요.")
            return

        # PDF_File 폴더의 원본 파일명은 "도면번호_스펙번호_Cable.pdf" 형식
        # (예: A60023104_6150-37-520-3487_Cable.pdf)이라, 첫 "_" 앞부분이
        # 곧 도면번호다. 저장 대화상자의 기본 파일명으로 그대로 제안해서
        # 매번 새로 타이핑하지 않게 한다(사용자 지시).
        default_name = "cropped.png"
        if self.filepath:
            base = os.path.splitext(os.path.basename(self.filepath))[0]
            drawing_no = base.split("_")[0] if "_" in base else base
            default_name = f"{drawing_no}.png"

        save_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image files", "*.png")],
            initialfile=default_name,
        )
        if not save_path:
            return
        cv2.imwrite(save_path, self.cropped_img)
        self.txt_log.insert(tk.END, f"\n배선도 PNG 저장 완료: {save_path}\n")
        self.txt_log.see(tk.END)
        messagebox.showinfo("완료", "PNG 저장 완료.")

    def on_mouse_wheel(self, event):
        """Ctrl+휠은 확대/축소, 일반 휠은 세로 스크롤로 동작한다(사용자
        지시 - Ctrl 없이 휠만 굴리면 그냥 위아래로 움직이게). event.state의
        Control 비트(0x4, 윈도우/리눅스 공통)로 Ctrl 눌림 여부를 판정한다.
        확대/축소는 zoom_factor를 고정 배율(1.1)씩 곱/나눠서
        disp_scale = fit_scale * zoom_factor로 갱신한다. 구획선/폴리라인
        좌표는 원본 이미지 좌표계로 저장돼 있으므로(disp_scale 곱해서
        화면에 그림) 다시 그리기만 하면 된다."""
        if self.original_img is None:
            return
        ctrl_held = bool(event.state & 0x4)
        scroll_up = getattr(event, 'delta', 0) > 0 or getattr(event, 'num', None) == 4
        if ctrl_held:
            if scroll_up:
                self.zoom_factor = min(self.zoom_factor * 1.1, 20.0)
            else:
                self.zoom_factor = max(self.zoom_factor / 1.1, 0.1)
            self.display_canvas_image()
        else:
            self.canvas_orig.yview_scroll(-1 if scroll_up else 1, "units")

    def display_canvas_image(self):
        if self.original_img is None: return
        h, w = self.original_img.shape[:2]
        self.root.update_idletasks()
        cw = self.canvas_orig.winfo_width()
        ch = self.canvas_orig.winfo_height()
        if cw < 10: cw = 400
        if ch < 10: ch = 500

        self.fit_scale = min(cw/w, ch/h)
        self.disp_scale = self.fit_scale * self.zoom_factor
        new_w, new_h = max(1, int(w * self.disp_scale)), max(1, int(h * self.disp_scale))

        interp = cv2.INTER_AREA if self.disp_scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(self.original_img, (new_w, new_h), interpolation=interp)
        if len(resized.shape) == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        img_pil = Image.fromarray(resized)
        self.tk_img = ImageTk.PhotoImage(img_pil)
        self.canvas_orig.delete("all")
        self.canvas_orig.create_image(0, 0, anchor=tk.NW, image=self.tk_img)
        self.canvas_orig.config(scrollregion=(0, 0, new_w, new_h))
        self._redraw_polyline()

    def _draw_results(self, target_img=None, show_img=True):
        seq_input = self.entry_sequence.get().strip()
        expected_order = list(seq_input) if seq_input else None

        matched_img = target_img.copy() if target_img is not None else self.cropped_img.copy()

        for idx, r in enumerate(self.results):
            p0, p1 = r['p0'], r['p1']
            layout = r.get('layout', 'vertical')
            letter = r['letter']

            if expected_order and idx < len(expected_order):
                letter = expected_order[idx]
                r['letter'] = letter
                r['score'] = 1.0

            if layout == 'vertical':
                cv2.rectangle(matched_img, (0, p0), (matched_img.shape[1], p1), (0, 255, 0), 2)
                cv2.putText(matched_img, letter, (10, p1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.rectangle(matched_img, (p0, 0), (p1, matched_img.shape[0]), (0, 255, 0), 2)
                cv2.putText(matched_img, letter, (p0 + 5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if show_img:
            self.display_result_image(matched_img, self.lbl_proc)

    def save_vector_region(self):
        """1-B/1-C로 검출된 배선도 영역(self.crop_coords)을 원본 PDF의
        벡터 좌표계로 다시 잘라서 새 벡터 PDF로 저장한다. 지금까지의
        1-B/1-C 결과(마스킹된 PNG)는 래스터 이미지라 확대하면 깨지고
        선을 다시 벡터로 되돌릴 수 없었다(사용자 지시) - extract_vector_region
        은 원본 PDF가 가진 get_drawings()/get_text() 벡터 좌표를 그대로
        읽어 필터링하므로 결과물도 벡터로 남는다. PDF를 불러온 경우에만
        가능하고(self.pdf_zoom), 이미지 파일을 불러왔거나 아직 영역을
        검출하지 않았으면 동작하지 않는다."""
        if not self.filepath or not self.filepath.lower().endswith('.pdf') or self.pdf_zoom is None:
            messagebox.showwarning("경고", "벡터 저장은 PDF를 불러온 경우에만 가능합니다.")
            return
        if self.crop_coords is None:
            messagebox.showwarning("경고", "먼저 1-B로 배선도 영역을 검출해주세요.")
            return

        save_path = filedialog.asksaveasfilename(
            defaultextension=".pdf", filetypes=[("Vector PDF files", "*.pdf")])
        if not save_path:
            return

        try:
            out_doc, _ = extract_vector_region(self.filepath, self.crop_coords, self.pdf_zoom)
            out_doc.save(save_path)
            out_doc.close()
            self.txt_log.insert(tk.END, f"\n벡터 PDF 저장 완료: {save_path}\n")
            self.txt_log.see(tk.END)
            messagebox.showinfo("완료", f"벡터 PDF로 저장했습니다:\n{save_path}")
        except Exception as e:
            messagebox.showerror("오류", f"벡터 PDF 저장 실패:\n{e}")

    def process_auto_region(self):
        """도면을 4분면(좌상/우상/좌하/우하)으로 나누고, '배선도/회로도/
        결선도' 제목이 속한 사분면 전체를 배선도 영역으로 채택한다
        (find_diagram_quadrant). 제목 위치에서 고정 비율/gap으로
        확장하던 이전 방식(crop_diagram_region)은 형상도나 부품목록표
        조각이 자주 섞여 들어왔는데(실측: "키 배열 결선도" 도면에서
        표1/표2가 통째로 포함됨), 도면 내용물의 밀도로 먼저 4분면을
        확정하고 그중 하나를 통째로 채택하는 방식이 훨씬 안정적이었다
        (실측: 50073721 도면에서 배선도+상세도만 정확히 분리됨)."""
        if self.original_img is None:
            messagebox.showwarning("경고", "먼저 파일을 불러와주세요.")
            return

        self.txt_log.insert(tk.END, "\n========== 배선도 영역 자동 검출 시작(4분면 방식) ==========\n")
        self.root.update()

        original_img = self.original_img
        import threading

        def worker():
            reader = self.reader
            if reader is None:
                reader = easyocr.Reader(['ko', 'en'], gpu=False)

            result = find_diagram_quadrant(reader, original_img)

            def report_result():
                if result is None:
                    self.txt_log.insert(tk.END, "배선도 사분면을 찾지 못했습니다. 수동으로 드래그해주세요.\n")
                    self.txt_log.see(tk.END)
            self.root.after(0, report_result)

            if result is None:
                return

            quad_name, (x0, y0, x1, y1), title_text, prob = result

            def apply_crop():
                self.reader = reader
                self.crop_coords = (x0, y0, x1, y1)
                self.cropped_img = original_img[y0:y1, x0:x1]
                self.txt_log.insert(
                    tk.END,
                    f"\n선택: '{title_text}'(신뢰도 {prob:.2f})가 속한 섹션 #{quad_name} 채택: "
                    f"({x0}, {y0}) -> ({x1}, {y1})\n")
                self.txt_log.see(tk.END)

                # 원본 캔버스에 검출된 영역을 표시
                if self.rect_id:
                    self.canvas_orig.delete(self.rect_id)
                dx0, dy0 = x0 * self.disp_scale, y0 * self.disp_scale
                dx1, dy1 = x1 * self.disp_scale, y1 * self.disp_scale
                self.rect_id = self.canvas_orig.create_rectangle(dx0, dy0, dx1, dy1, outline='red', width=2)

                # 벡터/좌표 참조용으로는 원본 페이지 크기를 유지한 채 배선도
                # 영역 바깥만 지운 버전도 보관하지만, 미리보기는 그 영역
                # 크기에 맞춰 실제로 크롭한 이미지를 보여준다(사용자 지시
                # - 원본 페이지 크기 그대로 보여주면 도면이 작게 보임).
                self.masked_full_img = mask_outside_region(original_img, (x0, y0, x1, y1))
                self.display_result_image(self.cropped_img, self.lbl_proc)
            self.root.after(0, apply_crop)

        threading.Thread(target=worker, daemon=True).start()

    def process_batch_auto_region(self):
        """폴더를 하나 선택받아서 그 안의 모든 PDF 첫 페이지에 대해
        배선도/회로도/결선도 영역 자동 검출을 순서대로 실행하고, 크롭
        결과를 <선택폴더>/auto_region_out/ 에 저장한다. batch_auto_region.py
        스크립트를 GUI 버튼으로 그대로 옮긴 것 - 백그라운드 스레드에서
        돌려서 처리 중에도 창이 멈추지 않는다."""
        folder = filedialog.askdirectory(title="PDF들이 들어있는 폴더 선택")
        if not folder:
            return

        import glob
        pdf_files = sorted(glob.glob(os.path.join(folder, "*.pdf")))
        if not pdf_files:
            messagebox.showwarning("경고", "선택한 폴더에 PDF 파일이 없습니다.")
            return

        out_dir = os.path.join(folder, "auto_region_out")
        os.makedirs(out_dir, exist_ok=True)

        self.txt_log.insert(tk.END, f"\n========== 폴더 전체 배치 자동 검출 시작 ==========\n")
        self.txt_log.insert(tk.END, f"대상: {len(pdf_files)}개 PDF, 저장 위치: {out_dir}\n")
        self.txt_log.see(tk.END)
        self.root.update()

        import threading

        def worker():
            reader = self.reader
            if reader is None:
                reader = easyocr.Reader(['ko', 'en'], gpu=False)

            ok_count = 0
            fail_list = []

            for i, pdf_path in enumerate(pdf_files):
                base = os.path.splitext(os.path.basename(pdf_path))[0]

                def report_progress(idx=i, b=base):
                    self.txt_log.insert(tk.END, f"[{idx+1}/{len(pdf_files)}] {b} ... ")
                    self.txt_log.see(tk.END)
                self.root.after(0, report_progress)

                try:
                    doc = fitz.open(pdf_path)
                    page = doc[0]
                    mat = fitz.Matrix(3.0, 3.0)
                    pix = page.get_pixmap(matrix=mat)
                    img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                    if pix.n == 4:
                        img = cv2.cvtColor(img_data, cv2.COLOR_RGBA2BGR)
                    elif pix.n == 3:
                        img = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
                    else:
                        img = cv2.cvtColor(img_data, cv2.COLOR_GRAY2BGR)
                    doc.close()

                    result = find_diagram_quadrant(reader, img)
                    if result is None:
                        fail_list.append((base, "사분면 판정 실패"))

                        def report_fail():
                            self.txt_log.insert(tk.END, "실패 (사분면 판정 실패)\n")
                            self.txt_log.see(tk.END)
                        self.root.after(0, report_fail)
                        continue

                    quad_name, (x0, y0, x1, y1), title_text, prob = result
                    region = img[y0:y1, x0:x1]
                    if region.size == 0:
                        fail_list.append((base, "빈 영역"))

                        def report_empty():
                            self.txt_log.insert(tk.END, "실패 (빈 영역)\n")
                            self.txt_log.see(tk.END)
                        self.root.after(0, report_empty)
                        continue

                    # 잘라내지 않고, 원본 페이지에서 배선도 영역 바깥(케이블
                    # 형상도/부품목록표/타이틀블록)만 흰색으로 지운다(사용자
                    # 지시: 페이지 크기/좌표계 유지).
                    masked = mask_outside_region(img, (x0, y0, x1, y1))
                    out_path = os.path.join(out_dir, f"{base}.png")
                    cv2.imwrite(out_path, masked)
                    ok_count += 1

                    def report_ok(txt=title_text, prob=prob, quad=quad_name, cw=region.shape[1], ch=region.shape[0]):
                        self.txt_log.insert(tk.END, f"OK ('{txt}' 신뢰도 {prob:.2f}, 섹션=#{quad}, {cw}x{ch})\n")
                        self.txt_log.see(tk.END)
                    self.root.after(0, report_ok)

                except Exception as e:
                    fail_list.append((base, f"예외: {e}"))

                    def report_exc(err=e):
                        self.txt_log.insert(tk.END, f"예외: {err}\n")
                        self.txt_log.see(tk.END)
                    self.root.after(0, report_exc)

            def finish():
                self.reader = reader
                self.txt_log.insert(tk.END, f"\n=== 완료: {ok_count}/{len(pdf_files)} 성공 ===\n")
                self.txt_log.insert(tk.END, f"결과 저장 위치: {out_dir}\n")
                if fail_list:
                    self.txt_log.insert(tk.END, "실패 목록:\n")
                    for b, reason in fail_list:
                        self.txt_log.insert(tk.END, f"  - {b}: {reason}\n")
                self.txt_log.see(tk.END)
                messagebox.showinfo("완료", f"배치 자동 검출 완료: {ok_count}/{len(pdf_files)} 성공\n저장 위치: {out_dir}")
            self.root.after(0, finish)

        threading.Thread(target=worker, daemon=True).start()

    def process_local_vlm(self):
        """크롭 영역을 Gemma에게 N회(기본 20회) 미세하게 좌표를 흔들어가며
        (EasyOCR 앙상블 3-C와 동일한 ±15px 지터링) 반복 질의하고, 각
        박스 위치별로 다수결(최빈값)을 취해 최종 판독을 확정한다.
        temperature=0.0(결정적)으로는 같은 크롭을 몇 번을 물어도 매번
        같은 답이 나와 앙상블의 의미가 없으므로, 지터링으로 실제 입력
        이미지 자체를 매번 바꿔서 판독 다양성을 만든다(사용자 지시)."""
        if self.cropped_img is None:
            messagebox.showwarning("경고", "먼저 영역을 드래그해주세요.")
            return

        crop_h, crop_w = self.cropped_img.shape[:2]
        layout = "vertical" if crop_h >= crop_w else "horizontal"

        from tkinter import simpledialog
        n_cells = simpledialog.askinteger(
            "셀 개수", f"크롭 영역 안의 박스(글자) 개수를 입력하세요 (배치 감지: {layout}):",
            minvalue=1, maxvalue=100)
        if not n_cells:
            return

        n_iter = simpledialog.askinteger(
            "앙상블 반복 횟수", "Gemma에게 몇 번 반복 질의할까요? (지터링으로 매번 다른 크롭 전달)",
            minvalue=1, maxvalue=50, initialvalue=20)
        if not n_iter:
            return

        self.txt_log.insert(tk.END, f"\n========== Gemma 앙상블 판독 시작 ({n_iter}회) ==========\n")
        self.txt_log.insert(tk.END, f"배치 감지: {layout}, 박스 {n_cells}개\n")
        self.root.update()

        original_img = self.original_img
        orig_x0, orig_y0, orig_x1, orig_y1 = self.crop_coords
        img_h, img_w = original_img.shape[:2]
        os.makedirs(TRAIN_DATA_DIR, exist_ok=True)
        import time
        session_tag = time.strftime("%Y%m%d_%H%M%S")

        import threading

        def worker():
            # 각 iteration마다 원본 크롭 좌표를 ±15px 흔들어서 실제로 다른
            # 이미지를 Gemma에 보낸다(EasyOCR 앙상블 3-C와 동일한 지터 폭).
            per_slot_votes = [[] for _ in range(n_cells)]
            n_success = 0

            for it in range(n_iter):
                shift_x0 = random.randint(-15, 15)
                shift_y0 = random.randint(-15, 15)
                shift_x1 = random.randint(-15, 15)
                shift_y1 = random.randint(-15, 15)
                jx0 = max(0, orig_x0 + shift_x0)
                jy0 = max(0, orig_y0 + shift_y0)
                jx1 = min(img_w, orig_x1 + shift_x1)
                jy1 = min(img_h, orig_y1 + shift_y1)
                if jx1 <= jx0 or jy1 <= jy0:
                    continue
                jittered = original_img[jy0:jy1, jx0:jx1]
                big = cv2.resize(jittered, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

                letters, err = ask_local_vlm_whole(big, n_cells, layout)
                if letters:
                    n_success += 1
                    for i in range(min(n_cells, len(letters))):
                        per_slot_votes[i].append(letters[i])

                def report_progress(idx=it, e=err, got=len(letters)):
                    msg = f"[{idx+1}/{n_iter}] 판독 {got}/{n_cells}자"
                    if e:
                        msg += f" ({e})"
                    self.txt_log.insert(tk.END, msg + "\n")
                    self.txt_log.see(tk.END)
                self.root.after(0, report_progress)

            def finish():
                if n_success == 0:
                    self.txt_log.insert(tk.END, "\n모든 시도가 실패했습니다. LM Studio 연결을 확인하세요.\n")
                    return

                total = crop_h if layout == "vertical" else crop_w
                step = total / n_cells
                cropped_img = self.cropped_img
                local_results = []
                self.txt_log.insert(tk.END, f"\n[ 앙상블 결과 (성공 {n_success}/{n_iter}회) ]\n")

                for i in range(n_cells):
                    votes = per_slot_votes[i]
                    p0, p1 = int(i * step), int((i + 1) * step)
                    if votes:
                        counter = Counter(votes)
                        letter, count = counter.most_common(1)[0]
                        consistency = count / len(votes) * 100
                    else:
                        letter, count, consistency = "?", 0, 0.0

                    local_results.append({"layout": layout, "p0": p0, "p1": p1,
                                           "letter": letter, "score": consistency})

                    self.txt_log.insert(
                        tk.END,
                        f"박스 {i+1}/{n_cells}: '{letter}' (일치율 {consistency:.0f}%, {count}/{len(votes)}표)\n")

                    if letter != "?":
                        if layout == "vertical":
                            cell = cropped_img[p0:p1, :]
                        else:
                            cell = cropped_img[:, p0:p1]
                        fname = f"{session_tag}_{i:03d}_{letter}.png"
                        cv2.imwrite(os.path.join(TRAIN_DATA_DIR, fname), cell)

                self.results = local_results
                self.txt_log.insert(tk.END, f"\n=> 앙상블 판독 완료. {TRAIN_DATA_DIR} 에 저장됨.\n")
                self.txt_log.see(tk.END)
                self._draw_results()
            self.root.after(0, finish)

        threading.Thread(target=worker, daemon=True).start()

    def process_extract_all_text(self):
        """crop된 배선도를 자동으로 격자 타일로 나눠(extract_all_text_tiled)
        Gemma에 병렬로 질의하고, 박스/원/자유텍스트 등 형태에 상관없이
        보이는 모든 텍스트를 위치와 함께 JSON으로 뽑아 로그와 JSON 파일로
        남긴다. 전체 이미지를 한 번에 통째로 보내면 reasoning이 항목
        수에 비례해 폭주해서 max_tokens를 다 쓰고도 답을 못 내는 경우가
        실측됐다(사용자가 직접 커넥터 하나 크기로 잘라 질의했을 때는
        성공) - 그래서 타일 자동 분할 + 병렬 요청(LM Studio의 Parallel
        슬롯 활용)으로 바꿨다(사용자 지시)."""
        if self.cropped_img is None:
            messagebox.showwarning("경고", "먼저 자유곡선으로 배선도 경계를 완성해주세요.")
            return

        # 이미지 크기에 맞춰 타일 그리드를 자동 계산한다(사용자 지시 -
        # 매번 가로/세로 조각 수를 직접 입력하지 않고 PNG마다 자동으로
        # 맞춘다). 계산 근거는 compute_auto_tile_grid 주석 참고.
        cols, rows = compute_auto_tile_grid(self.cropped_img)

        # LM Studio는 Context Length를 동시 요청 수만큼 나눠 쓰므로
        # (실측: Context=2048에서 4개 동시 요청 시 전부 "Context size
        # exceeded"로 실패), 안전한 병렬 수를 계산하려면 실제 로드된
        # Context Length 값이 필요하다. /api/v0/models로 자동 조회한다
        # (실측 확인됨) - 실패하면 보수적인 기본값으로 대체.
        ctx_length = get_lm_studio_context_length() or LM_STUDIO_CONTEXT_LENGTH_FALLBACK
        max_workers = compute_safe_parallelism(ctx_length)
        self.txt_log.insert(
            tk.END,
            f"\n========== 전체 텍스트 추출 시작 (Gemma, {cols}x{rows} 타일, "
            f"Context={ctx_length}로 계산한 병렬 {max_workers}개) ==========\n")
        self.root.update()

        crop = self.cropped_img
        import threading

        def progress_cb(done, total):
            self.root.after(0, lambda: (
                self.txt_log.insert(tk.END, f"타일 진행: {done}/{total}\n"),
                self.txt_log.see(tk.END)))

        def worker():
            items, tile_errors = extract_all_text_tiled(
                crop, cols=cols, rows=rows, max_workers=max_workers, progress_cb=progress_cb)
            items = filter_out_hangul(items)

            def finish():
                for e in tile_errors:
                    self.txt_log.insert(tk.END, f"경고: {e}\n")
                if not items:
                    self.txt_log.insert(tk.END, "실패: 어떤 타일에서도 텍스트를 추출하지 못했습니다.\n")
                    self.txt_log.see(tk.END)
                    return

                self.extracted_text_items = items
                self.txt_log.insert(tk.END, f"\n총 {len(items)}개 텍스트 항목 추출됨(중복 제거/한글 제외 후):\n")
                for it in items:
                    self.txt_log.insert(tk.END, f"  '{it['text']}' @ ({it['x']}, {it['y']})\n")

                os.makedirs(TRAIN_DATA_DIR, exist_ok=True)
                import time
                out_path = os.path.join(
                    TRAIN_DATA_DIR, f"extracted_text_{time.strftime('%Y%m%d_%H%M%S')}.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(items, f, ensure_ascii=False, indent=2)
                self.txt_log.insert(tk.END, f"\n=> JSON 저장 완료: {out_path}\n")
                self.txt_log.see(tk.END)
            self.root.after(0, finish)

        threading.Thread(target=worker, daemon=True).start()

    def open_horizontal_box_window(self):
        """핀이 왼쪽→오른쪽으로 나열된 가로 격자 커넥터(예: A60023103의
        P9/P8/P5, A60025766/776의 P2~P5)는 자동 검출을 시도했으나 배선
        자체가 가로선이 많아 커넥터 진짜 테두리와 배선을 구분하지 못해
        오탐이 심했다(사용자와 함께 실측 확인 후 수동 지정으로 전환).
        이 창은 폴더 안 PNG들을 하나씩 보여주고, 사람이 드래그로 가로
        커넥터 박스 영역만 표시하게 한다 - 정확한 셀 좌표 계산
        (detect_grid_cells_horizontal)은 여전히 자동이므로, 사람이 할
        일은 "어디가 박스인지" 뿐이다. 박스는 커넥터 하나를 딱 감싸는
        사각형(x0,y0,x1,y1)이어야 한다 - y범위만 지정하면 같은 줄에
        나란히 붙은 여러 커넥터가 한 번에 잡혀 잘못 병합된다(실측:
        A60023103에서 P9,P8,P7,P6,P5가 한 줄에 붙어있음). 각 도면의
        박스 목록은 {도면번호}.hboxes.json으로 저장되고,
        3-C(process_image_folder_extract_text)가 이 파일이 있으면 읽어서
        반영한다."""
        folder = filedialog.askdirectory(title="가로 격자 박스를 지정할 PNG들이 있는 폴더 선택")
        if not folder:
            return
        png_files = sorted(f for f in os.listdir(folder) if f.lower().endswith(".png"))
        if not png_files:
            messagebox.showwarning("경고", "선택한 폴더에 PNG 파일이 없습니다.")
            return

        # 기존 hboxes.json을 PNG 폴더뿐 아니라 그 아래 모든 하위폴더까지
        # 재귀적으로 찾는다(사용자가 "Horizental-Box" 같은 별도 하위폴더로
        # 정리해서 보관 - process_image_folder_extract_text의 hbox_map과
        # 동일한 탐색 방식). 새로 저장할 때는 항상 이 고정 하위폴더에
        # 쓴다 - PNG 폴더를 어질러놓지 않고 한 곳에 모아두기 위함.
        hbox_map = {}
        for dirpath, _, filenames in os.walk(folder):
            for fn in filenames:
                if fn.endswith(".hboxes.json"):
                    drawing_key = fn[:-len(".hboxes.json")]
                    hbox_map.setdefault(drawing_key, os.path.join(dirpath, fn))
        save_dir = os.path.join(folder, "Horizental-Box")

        win = tk.Toplevel(self.root)
        win.title("3-B2. 가로 격자 박스 지정")
        win.geometry("1100x800")

        state = {"idx": 0, "img": None, "disp_scale": 1.0, "boxes": [], "drag_start": None, "drag_rect_id": None}

        top = tk.Frame(win)
        top.pack(fill=tk.X, padx=8, pady=6)
        lbl_file = tk.Label(top, text="", font=("", 10, "bold"))
        lbl_file.pack(side=tk.LEFT)
        lbl_hint = tk.Label(
            top, text="드래그로 커넥터 하나만 딱 감싸는 사각형을 그리세요(핀이 나열된 가로 박스). 여러 개 지정 가능.")
        lbl_hint.pack(side=tk.LEFT, padx=20)

        canvas_frame = tk.Frame(win)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        canvas = tk.Canvas(canvas_frame, bg="gray")
        canvas.pack(fill=tk.BOTH, expand=True)

        btn_frame = tk.Frame(win)
        btn_frame.pack(fill=tk.X, padx=8, pady=6)

        def load_current():
            fname = png_files[state["idx"]]
            path = os.path.join(folder, fname)
            img = cv2.imread(path)
            state["img"] = img
            drawing_no = os.path.splitext(fname)[0]
            hbox_path = hbox_map.get(drawing_no)
            if hbox_path and os.path.exists(hbox_path):
                try:
                    with open(hbox_path, encoding="utf-8") as f:
                        state["boxes"] = [tuple(b) for b in json.load(f)]
                except (json.JSONDecodeError, OSError):
                    state["boxes"] = []
            else:
                state["boxes"] = []
            lbl_file.config(text=f"[{state['idx']+1}/{len(png_files)}] {fname} (저장된 박스 {len(state['boxes'])}개)")
            redraw()

        def redraw():
            img = state["img"]
            if img is None:
                return
            h, w = img.shape[:2]
            cw = max(canvas.winfo_width(), 400)
            ch = max(canvas.winfo_height(), 300)
            scale = min(cw / w, ch / h, 1.0)
            state["disp_scale"] = scale
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            state["tk_img"] = ImageTk.PhotoImage(Image.fromarray(rgb))
            canvas.delete("all")
            canvas.create_image(0, 0, anchor=tk.NW, image=state["tk_img"])
            for x0, y0, x1, y1 in state["boxes"]:
                canvas.create_rectangle(
                    x0 * scale, y0 * scale, x1 * scale, y1 * scale, outline="red", width=2)

        def on_press(event):
            state["drag_start"] = (event.x, event.y)

        def on_drag(event):
            if state["drag_start"] is None:
                return
            if state["drag_rect_id"] is not None:
                canvas.delete(state["drag_rect_id"])
            sx0, sy0 = state["drag_start"]
            state["drag_rect_id"] = canvas.create_rectangle(
                sx0, sy0, event.x, event.y, outline="lime", width=2)

        def on_release(event):
            if state["drag_start"] is None or state["img"] is None:
                return
            sx0, sy0 = state["drag_start"]
            x0_disp, x1_disp = sorted([sx0, event.x])
            y0_disp, y1_disp = sorted([sy0, event.y])
            state["drag_start"] = None
            if state["drag_rect_id"] is not None:
                canvas.delete(state["drag_rect_id"])
                state["drag_rect_id"] = None
            scale = state["disp_scale"]
            if scale <= 0 or (y1_disp - y0_disp) < 5 or (x1_disp - x0_disp) < 5:
                return  # 너무 작게 드래그하면(클릭 실수) 무시
            x0 = int(x0_disp / scale)
            y0 = int(y0_disp / scale)
            x1 = int(x1_disp / scale)
            y1 = int(y1_disp / scale)
            state["boxes"].append((x0, y0, x1, y1))
            redraw()

        canvas.bind("<ButtonPress-1>", on_press)
        canvas.bind("<B1-Motion>", on_drag)
        canvas.bind("<ButtonRelease-1>", on_release)

        def undo_box():
            if state["boxes"]:
                state["boxes"].pop()
                redraw()

        def save_current():
            fname = png_files[state["idx"]]
            drawing_no = os.path.splitext(fname)[0]
            # 기존 파일이 어디(예: Horizental-Box 하위폴더)에 있었다면
            # 그 자리에 덮어쓰고, 새로 만드는 것이면 save_dir(고정
            # 하위폴더)에 쓴다 - 사용자가 PNG 폴더와 분리해서 정리하는
            # 걸 선호하므로, 매번 PNG 폴더 바로 아래에 새로 만들지 않는다.
            hbox_path = hbox_map.get(drawing_no)
            if hbox_path is None:
                os.makedirs(save_dir, exist_ok=True)
                hbox_path = os.path.join(save_dir, f"{drawing_no}.hboxes.json")
                hbox_map[drawing_no] = hbox_path
            if state["boxes"]:
                with open(hbox_path, "w", encoding="utf-8") as f:
                    json.dump(state["boxes"], f)
            elif os.path.exists(hbox_path):
                os.remove(hbox_path)  # 박스를 다 지웠으면 파일도 정리

        def go_prev():
            save_current()
            if state["idx"] > 0:
                state["idx"] -= 1
                load_current()

        def go_next():
            save_current()
            if state["idx"] < len(png_files) - 1:
                state["idx"] += 1
                load_current()

        tk.Button(btn_frame, text="◀ 이전", command=go_prev).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="다음 ▶", command=go_next).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="↩ 마지막 박스 취소", command=undo_box).pack(side=tk.LEFT, padx=20)
        tk.Button(btn_frame, text="💾 저장하고 닫기", command=lambda: (save_current(), win.destroy())).pack(side=tk.RIGHT)

        win.after(100, load_current)

    def process_folder_extract_text(self):
        """폴더 선택 다이얼로그로 고른 폴더 안의 모든 PNG(자유곡선으로
        미리 잘라둔 배선도 크롭 이미지 55개 등)에 대해, 하나씩
        extract_all_text_tiled(타일 분할 + 병렬 질의)를 돌려서 파일마다
        {도면번호}.json으로 저장한다(process_image_folder_extract_text).
        한글이 섞인 텍스트 항목은 filter_out_hangul로 제외한다(사용자
        지시). GUI로 폴더를 매번 다시 여는 3-B와 달리, 폴더 하나를
        고르면 안의 파일 전체를 순서대로 처리한다."""
        folder = filedialog.askdirectory(title="텍스트를 추출할 PNG들이 있는 폴더 선택")
        if not folder:
            return

        png_count = len([f for f in os.listdir(folder) if f.lower().endswith(".png")])
        if png_count == 0:
            messagebox.showwarning("경고", "선택한 폴더에 PNG 파일이 없습니다.")
            return

        ctx_length = get_lm_studio_context_length() or LM_STUDIO_CONTEXT_LENGTH_FALLBACK
        max_workers = compute_safe_parallelism(ctx_length)

        self.txt_log.insert(
            tk.END,
            f"\n========== 폴더 전체 텍스트 추출 시작: {folder}\n"
            f"({png_count}개 PNG, 파일별 자동 타일 분할, 병렬 {max_workers}개) ==========\n")
        self.root.update()

        import threading

        def show_processing_image(fname):
            # 지금 3-C가 처리 중인 PNG를 메인 화면 오른쪽(lbl_proc, 평소
            # "AI 추출 및 렌더링 결과"를 보여주는 자리)에 미리보기로
            # 띄운다(사용자 요청 - 추출 중 어떤 이미지가 처리되는지 화면
            # 으로 보고 싶다는 지시). process_image_folder_extract_text가
            # 백그라운드 스레드에서 돌므로 Tk 위젯 갱신은 반드시
            # root.after로 메인 스레드에 넘겨야 한다.
            path = os.path.join(folder, fname)
            img = cv2.imread(path)
            if img is None:
                return
            h, w = img.shape[:2]
            lbl_w = max(self.lbl_proc.winfo_width(), 200)
            lbl_h = max(self.lbl_proc.winfo_height(), 200)
            scale = min(lbl_w / w, lbl_h / h, 1.0)
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            self.proc_preview_img = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.lbl_proc.configure(image=self.proc_preview_img)

        def progress_cb(done, total, fname):
            self.root.after(0, lambda: (
                self.txt_log.insert(tk.END, f"[{done}/{total}] {fname} 처리 완료\n"),
                self.txt_log.see(tk.END)))
            self.root.after(0, lambda: show_processing_image(fname))

        def worker():
            results = process_image_folder_extract_text(
                folder, max_workers=max_workers, progress_cb=progress_cb)

            def finish():
                ok = sum(1 for r in results if "error" not in r)
                fail = [r for r in results if "error" in r]
                # 격자 방식(100% 정확한 좌표)으로 뽑힌 핀 개수를 파일별로
                # 표시해서, 어떤 도면이 격자 검출 혜택을 받았는지(=좌표가
                # 확정적인지) 바로 알 수 있게 한다(사용자 지시 - 좌표
                # 정확도가 중요).
                total_grid = sum(r.get("grid_count", 0) for r in results if "error" not in r)
                self.txt_log.insert(
                    tk.END,
                    f"\n=> 폴더 처리 완료: {ok}/{len(results)}개 성공 "
                    f"(격자 검출로 좌표 확정된 핀 총 {total_grid}개)\n")
                for r in fail:
                    self.txt_log.insert(tk.END, f"  실패: {r['file']} - {r['error']}\n")
                self.txt_log.see(tk.END)
                messagebox.showinfo("완료", f"{ok}/{len(results)}개 PNG 처리 완료.\nPNG 폴더 아래 1_Raw_JSON에 저장됨.\n격자 검출로 좌표 확정된 핀: {total_grid}개")
            self.root.after(0, finish)

        threading.Thread(target=worker, daemon=True).start()

    def open_pin_filter_window(self):
        """JSON(+같은 폴더의 PNG) 폴더를 골라서, 커넥터명/핀 문자는
        자동으로 유지하고 신호명/AWG/전압 등은 자동으로 제외한 뒤,
        "애매(순수 숫자 1~2자리)" 항목만 PNG 크롭 썸네일로 보여줘서
        사람이 직접 포함/제외를 클릭으로 확정하게 한다(사용자 지시 -
        "핀번호/핀문자 외에는 다 필요없다", 원 안 지시선 참조번호는
        좌표만으로 구별 불가하므로 실제 이미지를 보고 사람이 판단).
        확정 후 파일마다 {도면번호}_filtered.json으로 저장한다."""
        folder = filedialog.askdirectory(title="필터링할 JSON들이 있는 폴더 선택")
        if not folder:
            return

        json_files = sorted(f for f in os.listdir(folder) if f.lower().endswith(".json")
                             and not f.lower().endswith("_filtered.json"))
        if not json_files:
            messagebox.showwarning("경고", "선택한 폴더에 JSON 파일이 없습니다.")
            return

        # PNG는 보통 1_Raw_JSON 폴더가 아니라 그 상위(원본 크롭 이미지)
        # 폴더에 있으므로(사용자 폴더 구조: Image/*.png, Image/1_Raw_JSON/
        # *.json) 둘 다에서 찾는다. 저장은 형제 폴더 "2_Filtered_JSON"에
        # 모은다(사용자 지시 - 1차/2차/3차 결과를 번호 붙은 폴더로 구분).
        png_search_dirs = [folder, os.path.dirname(folder)]
        filtered_dir = os.path.join(os.path.dirname(folder), "2_Filtered_JSON")
        os.makedirs(filtered_dir, exist_ok=True)

        win = tk.Toplevel(self.root)
        win.title("3-D. 후처리 필터 - 커넥터명 + 핀만 남기기")
        win.geometry("1000x750")

        state = {
            "files": json_files,
            "idx": 0,
            "folder": folder,
            "classified": None,
            "amb_vars": [],      # ambiguous 항목과 1:1 대응하는 BooleanVar(체크=포함)
            "img": None,         # 현재 파일과 같은 이름의 PNG(cv2), 없으면 None
        }

        top_bar = tk.Frame(win)
        top_bar.pack(fill=tk.X, padx=10, pady=(10, 5))
        lbl_file = tk.Label(top_bar, text="", font=("", 11, "bold"))
        lbl_file.pack(side=tk.LEFT)
        lbl_progress = tk.Label(top_bar, text="")
        lbl_progress.pack(side=tk.RIGHT)

        summary_frame = tk.Frame(win)
        summary_frame.pack(fill=tk.X, padx=10)
        lbl_summary = tk.Label(summary_frame, text="", justify=tk.LEFT, fg="darkgreen")
        lbl_summary.pack(anchor=tk.W)

        tk.Label(win, text="애매한 항목(순수 숫자 1~2자리) - PNG를 보고 핀번호면 체크 유지, 지시선 참조번호면 체크 해제",
                 fg="darkred").pack(anchor=tk.W, padx=10, pady=(10, 0))

        amb_canvas_frame = tk.Frame(win)
        amb_canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        amb_canvas = tk.Canvas(amb_canvas_frame, bg="white")
        amb_scroll = tk.Scrollbar(amb_canvas_frame, orient=tk.VERTICAL, command=amb_canvas.yview)
        amb_inner = tk.Frame(amb_canvas, bg="white")
        amb_inner.bind("<Configure>", lambda e: amb_canvas.configure(scrollregion=amb_canvas.bbox("all")))
        amb_canvas.create_window((0, 0), window=amb_inner, anchor="nw")
        amb_canvas.configure(yscrollcommand=amb_scroll.set)
        amb_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        amb_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        def _on_mousewheel(event):
            delta = getattr(event, 'delta', 0)
            if delta:
                amb_canvas.yview_scroll(-1 if delta > 0 else 1, "units")
            elif getattr(event, 'num', None) == 4:
                amb_canvas.yview_scroll(-1, "units")
            elif getattr(event, 'num', None) == 5:
                amb_canvas.yview_scroll(1, "units")

        def _bind_wheel(_e):
            amb_canvas.bind_all("<MouseWheel>", _on_mousewheel)
            amb_canvas.bind_all("<Button-4>", _on_mousewheel)
            amb_canvas.bind_all("<Button-5>", _on_mousewheel)

        def _unbind_wheel(_e):
            amb_canvas.unbind_all("<MouseWheel>")
            amb_canvas.unbind_all("<Button-4>")
            amb_canvas.unbind_all("<Button-5>")

        amb_canvas.bind("<Enter>", _bind_wheel)
        amb_canvas.bind("<Leave>", _unbind_wheel)

        btn_bar = tk.Frame(win)
        btn_bar.pack(fill=tk.X, padx=10, pady=10)

        def load_current():
            for w in amb_inner.winfo_children():
                w.destroy()
            state["amb_vars"] = []

            fname = state["files"][state["idx"]]
            drawing_no = os.path.splitext(fname)[0]
            lbl_file.config(text=f"{fname}")
            lbl_progress.config(text=f"{state['idx']+1} / {len(state['files'])}")

            with open(os.path.join(folder, fname), "r", encoding="utf-8") as f:
                items = json.load(f)
            classified = classify_pin_items(items)
            state["classified"] = classified

            state["img"] = None
            for d in png_search_dirs:
                png_path = os.path.join(d, f"{drawing_no}.png")
                if os.path.exists(png_path):
                    state["img"] = cv2.imread(png_path)
                    break

            conn_names = ", ".join(sorted({it["text"] for it in classified["connectors"]})) or "(없음)"
            pin_names = ", ".join(sorted({it["text"] for it in classified["pins"]})) or "(없음)"
            lbl_summary.config(text=(
                f"커넥터명({len(classified['connectors'])}개, 자동 유지): {conn_names}\n"
                f"핀 문자({len(classified['pins'])}개, 자동 유지): {pin_names}\n"
                f"제외 예정({len(classified['excluded'])}개, 자동 제외)\n"
                f"애매({len(classified['ambiguous'])}개, 아래에서 직접 확인)"))

            if not classified["ambiguous"]:
                tk.Label(amb_inner, text="(애매한 숫자 항목 없음)", bg="white").pack(pady=20)
                return

            for it in classified["ambiguous"]:
                row = tk.Frame(amb_inner, bg="white", bd=1, relief=tk.SOLID)
                row.pack(fill=tk.X, padx=5, pady=4)

                thumb_label = tk.Label(row, bg="white")
                thumb_label.pack(side=tk.LEFT, padx=5, pady=5)
                if state["img"] is not None:
                    crop = crop_around_point(state["img"], it.get("x", 0), it.get("y", 0))
                    if crop.size > 0:
                        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(crop_rgb).resize((140, 140))
                        tk_img = ImageTk.PhotoImage(pil_img)
                        thumb_label.configure(image=tk_img)
                        thumb_label.image = tk_img
                else:
                    thumb_label.configure(text="(PNG 없음)", width=18, height=8)

                info = tk.Frame(row, bg="white")
                info.pack(side=tk.LEFT, padx=10, fill=tk.Y)
                tk.Label(info, text=f"값: {it.get('text','')}  (x={it.get('x')}, y={it.get('y')})",
                         bg="white", font=("", 10, "bold")).pack(anchor=tk.W)
                var = tk.BooleanVar(value=True)
                tk.Checkbutton(info, text="핀 번호로 포함", variable=var, bg="white").pack(anchor=tk.W)
                state["amb_vars"].append((it, var))

        def save_current():
            classified = state["classified"]
            if classified is None:
                return
            kept = list(classified["connectors"]) + list(classified["pins"])
            kept += [it for it, var in state["amb_vars"] if var.get()]
            fname = state["files"][state["idx"]]
            drawing_no = os.path.splitext(fname)[0]
            out_path = os.path.join(filtered_dir, f"{drawing_no}_filtered.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(kept, f, ensure_ascii=False, indent=2)
            return out_path

        def go_next():
            out_path = save_current()
            if state["idx"] < len(state["files"]) - 1:
                state["idx"] += 1
                load_current()
            else:
                messagebox.showinfo("완료", f"마지막 파일까지 저장했습니다.\n({out_path})")

        def go_prev():
            save_current()
            if state["idx"] > 0:
                state["idx"] -= 1
                load_current()

        def save_only():
            out_path = save_current()
            messagebox.showinfo("저장됨", f"저장 완료:\n{out_path}")

        tk.Button(btn_bar, text="◀ 이전 (저장 후 이동)", command=go_prev).pack(side=tk.LEFT)
        tk.Button(btn_bar, text="저장만", command=save_only, bg="lightgray").pack(side=tk.LEFT, padx=10)
        tk.Button(btn_bar, text="다음 (저장 후 이동) ▶", command=go_next, bg="lightgreen").pack(side=tk.RIGHT)

        load_current()

    def open_pin_assembly_window(self):
        """3-E: Filtered_JSON(커넥터명+핀만 남은 2차 결과) 폴더를 골라서,
        각 핀을 "가장 가까운 커넥터"에 자동 배정(assemble_connector_pins)한
        뒤, {커넥터명: [핀 목록]} 형태로 사람이 검증하는 화면을 띄운다.
        좌표 근접성만으로는 완전 자동화가 불가능하므로(도면마다 레이아웃이
        다르고, 같은 핀 문자가 여러 커넥터에 중복 등장 - 사용자 질문에
        답한 설명), 커넥터별 핀 목록을 보여주고 각 핀 옆의 드롭다운으로
        사람이 소속 커넥터를 바로 잡을 수 있게 한다."""
        folder = filedialog.askdirectory(title="조립할 Filtered JSON들이 있는 폴더 선택")
        if not folder:
            return

        json_files = sorted(f for f in os.listdir(folder) if f.lower().endswith(".json"))
        if not json_files:
            messagebox.showwarning("경고", "선택한 폴더에 JSON 파일이 없습니다.")
            return

        # 원본 PNG는 2_Filtered_JSON 폴더가 아니라 그 상위(원본 크롭 이미지)
        # 폴더에 있는 경우가 많으므로(사용자 작업 폴더 구조: Image/*.png,
        # Image/2_Filtered_JSON/*.json), 선택한 폴더와 그 상위 폴더 둘
        # 다에서 {도면번호}.png를 찾는다. 조립 결과는 형제 폴더
        # "3_Assembled_JSON"에 모은다(사용자 지시 - 1차/2차/3차 결과를
        # 번호 붙은 폴더로 구분).
        image_search_dirs = [folder, os.path.dirname(folder)]
        assembled_dir = os.path.join(os.path.dirname(folder), "3_Assembled_JSON")
        os.makedirs(assembled_dir, exist_ok=True)

        win = tk.Toplevel(self.root)
        win.title("3-E. 커넥터-핀 조립 - 자동 배정 후 검증")
        win.geometry("1400x750")

        state = {"files": json_files, "idx": 0, "assembled": None, "conn_keys": []}

        top_bar = tk.Frame(win)
        top_bar.pack(fill=tk.X, padx=10, pady=(10, 5))
        lbl_file = tk.Label(top_bar, text="", font=("", 11, "bold"))
        lbl_file.pack(side=tk.LEFT)
        lbl_progress = tk.Label(top_bar, text="")
        lbl_progress.pack(side=tk.RIGHT)

        content_frame = tk.Frame(win)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # 좌: 커넥터-핀 목록(폭 40%, 흰 여백이 많이 남으므로 좁게),
        # 우: 원본 PNG 미리보기(폭 60%, 크게) - 사용자 지시.
        body_frame = tk.Frame(content_frame, width=560)
        body_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)
        body_frame.pack_propagate(False)
        body_canvas = tk.Canvas(body_frame, bg="white")
        body_scroll = tk.Scrollbar(body_frame, orient=tk.VERTICAL, command=body_canvas.yview)
        body_inner = tk.Frame(body_canvas, bg="white")
        body_inner.bind("<Configure>", lambda e: body_canvas.configure(scrollregion=body_canvas.bbox("all")))
        body_canvas.create_window((0, 0), window=body_inner, anchor="nw")
        body_canvas.configure(yscrollcommand=body_scroll.set)
        body_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        body_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # 우측: 원본 PNG 미리보기(스크롤 가능, 커넥터-핀 목록과 대조하며
        # 검증할 수 있게 나란히 표시 - 사용자 지시).
        img_frame = tk.Frame(content_frame, bg="gray90")
        img_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        img_canvas = tk.Canvas(img_frame, bg="gray90")
        img_vscroll = tk.Scrollbar(img_frame, orient=tk.VERTICAL, command=img_canvas.yview)
        img_hscroll = tk.Scrollbar(img_frame, orient=tk.HORIZONTAL, command=img_canvas.xview)
        img_canvas.configure(yscrollcommand=img_vscroll.set, xscrollcommand=img_hscroll.set)
        img_vscroll.pack(side=tk.RIGHT, fill=tk.Y)
        img_hscroll.pack(side=tk.BOTTOM, fill=tk.X)
        img_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        # 이미지를 Label 위젯(create_window)으로 얹으면 캔버스에 그리는
        # 다른 도형(마커)이 항상 그 뒤에 가려진다(Tkinter Canvas의 window
        # 아이템은 z-order상 always-on-top이라 create_oval을 나중에 그려도
        # 안 보임 - 실측으로 확인된 원인). 그래서 이미지 자체를
        # create_image로 캔버스에 직접 그리고, 마커는 그 이미지 아이템
        # 다음에 그려서 위에 겹치게 한다.
        img_item_id = img_canvas.create_image(0, 0, anchor="nw")
        img_no_text_id = img_canvas.create_text(20, 20, anchor="nw", fill="black", text="")

        def _on_img_mousewheel(event):
            delta = getattr(event, 'delta', 0)
            if delta:
                img_canvas.yview_scroll(-1 if delta > 0 else 1, "units")
            elif getattr(event, 'num', None) == 4:
                img_canvas.yview_scroll(-1, "units")
            elif getattr(event, 'num', None) == 5:
                img_canvas.yview_scroll(1, "units")

        def _bind_img_wheel(_e):
            img_canvas.bind_all("<MouseWheel>", _on_img_mousewheel)
            img_canvas.bind_all("<Button-4>", _on_img_mousewheel)
            img_canvas.bind_all("<Button-5>", _on_img_mousewheel)

        def _unbind_img_wheel(_e):
            img_canvas.unbind_all("<MouseWheel>")
            img_canvas.unbind_all("<Button-4>")
            img_canvas.unbind_all("<Button-5>")

        img_canvas.bind("<Enter>", _bind_img_wheel)
        img_canvas.bind("<Leave>", _unbind_img_wheel)

        def load_preview_image(drawing_no):
            """PNG를 찾아 img_canvas에 화면 폭에 맞춰 축소해서 표시한다.
            JSON 좌표는 원본 픽셀 좌표계이므로, 마커를 찍을 때는
            state["img_scale"](이 축소 비율)을 곱해서 화면 좌표로 정확히
            변환한다(원본을 그대로 띄우면 도면이 너무 커서 스크롤이
            번거로우므로 사용자 지시로 다시 축소 표시 - 좌표 자체는 JSON에
            원본 그대로 저장되니 화면 표시만 줄이는 것은 문제 없음)."""
            # 새 파일을 불러올 때는 이전 파일에서 찍었던 마커 도형을 캔버스
            # 에서 실제로 지워야 한다. state["marker_id"]만 None으로
            # 리셋하면 상태값만 초기화될 뿐 캔버스에 그려진 oval은 그대로
            # 남아 "이전 파일 이동할 때마다 원이 하나씩 늘어나는" 잔상
            # 버그가 생긴다(사용자 지적).
            if state.get("marker_id") is not None:
                img_canvas.delete(state["marker_id"])
                state["marker_id"] = None

            for d in image_search_dirs:
                png_path = os.path.join(d, f"{drawing_no}.png")
                if os.path.exists(png_path):
                    pil_img = Image.open(png_path)
                    max_w = 900
                    scale = min(max_w / pil_img.width, 1.0)
                    new_size = (int(pil_img.width * scale), int(pil_img.height * scale))
                    pil_img = pil_img.resize(new_size)
                    tk_img = ImageTk.PhotoImage(pil_img)
                    img_canvas.itemconfigure(img_item_id, image=tk_img)
                    img_canvas.itemconfigure(img_no_text_id, text="")
                    state["img_tk_ref"] = tk_img  # PhotoImage GC 방지용 강한 참조
                    img_canvas.configure(scrollregion=(0, 0, new_size[0], new_size[1]))
                    state["img_scale"] = scale
                    return
            img_canvas.itemconfigure(img_item_id, image="")
            img_canvas.itemconfigure(img_no_text_id, text=f"({drawing_no}.png 없음)")
            state["img_tk_ref"] = None
            img_canvas.configure(scrollregion=(0, 0, 300, 60))
            state["img_scale"] = None

        def mark_pin_on_image(pin):
            """pin(원본 픽셀 좌표를 가진 JSON 항목)의 위치에 빨간 원 마커를
            찍는다. 이전 마커는 지우고 하나만 유지. 화면 좌표 = 원본좌표 *
            state["img_scale"](축소 비율 보정, 사용자 지적사항 반영).
            찍은 뒤 그 위치가 보이도록 스크롤도 이동시킨다."""
            if state.get("marker_id") is not None:
                img_canvas.delete(state["marker_id"])
                state["marker_id"] = None
            scale = state.get("img_scale")
            if not scale:
                return
            sx = pin.get("x", 0) * scale
            sy = pin.get("y", 0) * scale
            r = 14
            state["marker_id"] = img_canvas.create_oval(
                sx - r, sy - r, sx + r, sy + r, outline="red", width=4)
            region = img_canvas.cget("scrollregion")
            if region:
                _, _, w, h = (float(v) for v in region.split())
                w, h = max(w, 1), max(h, 1)
                # xview_moveto/yview_moveto는 "스크롤 전체 영역 대비 뷰포트
                # 좌상단 비율"을 받는데, 뷰포트(실제 보이는 캔버스 크기)를
                # 빼지 않고 (sx-200)/w 식으로만 계산하면 뷰포트가 이미지보다
                # 크거나 비슷할 때 스크롤이 과도하게 밀려서 마커가 이미지
                # 바깥 회색 여백에 있는 것처럼 보이는 버그가 있었다(사용자
                # 스크린샷으로 실측 확인 - C 핀 마커가 이미지 밖에 찍힘).
                # 스크롤 가능한 실제 폭/높이(전체 - 뷰포트)로 나눠야 정확하다.
                img_canvas.update_idletasks()
                view_w = max(img_canvas.winfo_width(), 1)
                view_h = max(img_canvas.winfo_height(), 1)
                scrollable_w = max(w - view_w, 1)
                scrollable_h = max(h - view_h, 1)
                target_x = sx - view_w / 2
                target_y = sy - view_h / 2
                img_canvas.xview_moveto(min(1, max(0, target_x / scrollable_w)))
                img_canvas.yview_moveto(min(1, max(0, target_y / scrollable_h)))

        def _on_mousewheel(event):
            delta = getattr(event, 'delta', 0)
            if delta:
                body_canvas.yview_scroll(-1 if delta > 0 else 1, "units")
            elif getattr(event, 'num', None) == 4:
                body_canvas.yview_scroll(-1, "units")
            elif getattr(event, 'num', None) == 5:
                body_canvas.yview_scroll(1, "units")

        def _bind_wheel(_e):
            body_canvas.bind_all("<MouseWheel>", _on_mousewheel)
            body_canvas.bind_all("<Button-4>", _on_mousewheel)
            body_canvas.bind_all("<Button-5>", _on_mousewheel)

        def _unbind_wheel(_e):
            body_canvas.unbind_all("<MouseWheel>")
            body_canvas.unbind_all("<Button-4>")
            body_canvas.unbind_all("<Button-5>")

        body_canvas.bind("<Enter>", _bind_wheel)
        body_canvas.bind("<Leave>", _unbind_wheel)

        def select_pin(idx):
            """idx번째 핀 행을 "현재 선택"으로 표시하고(파란 배경), 오른쪽
            이미지에 마커를 찍고, 그 행이 보이도록 목록도 스크롤한다.
            방향키(위/아래)로 이 함수를 호출해 순서대로 훑을 수 있게 한다
            (사용자 지시 - 체크는 잘 되니 방향키로 순회하며 확인하고 싶다)."""
            rows = state.get("pin_rows") or []
            widgets = state.get("pin_widgets") or []
            if not rows or not (0 <= idx < len(rows)):
                return
            deleted = state.get("deleted_pins", set())
            prev = state.get("sel_idx", -1)
            if 0 <= prev < len(rows):
                prev_bg = "#eee" if id(widgets[prev][0]) in deleted else "white"
                rows[prev].configure(bg=prev_bg)
                for w in rows[prev].winfo_children():
                    try:
                        w.configure(bg=prev_bg)
                    except tk.TclError:
                        pass
            state["sel_idx"] = idx
            sel_bg = "#eee" if id(widgets[idx][0]) in deleted else "#cfe8ff"
            rows[idx].configure(bg=sel_bg)
            for w in rows[idx].winfo_children():
                try:
                    w.configure(bg=sel_bg)
                except tk.TclError:
                    pass

            pin = widgets[idx][0]
            mark_pin_on_image(pin)

            # body_canvas 스크롤을 선택된 행이 보이는 위치로 이동.
            body_inner.update_idletasks()
            row_y = rows[idx].winfo_y()
            total_h = max(body_inner.winfo_height(), 1)
            body_canvas.yview_moveto(max(0, (row_y - 40) / total_h))

        def move_selection(delta):
            widgets = state.get("pin_widgets") or []
            if not widgets:
                return
            cur = state.get("sel_idx", -1)
            new_idx = min(max(cur + delta, 0), len(widgets) - 1)
            select_pin(new_idx)

        win.bind("<Down>", lambda e: move_selection(1))
        win.bind("<Up>", lambda e: move_selection(-1))

        btn_bar = tk.Frame(win)
        btn_bar.pack(fill=tk.X, padx=10, pady=10)

        def load_current():
            for w in body_inner.winfo_children():
                w.destroy()

            fname = state["files"][state["idx"]]
            drawing_no = os.path.splitext(fname)[0].replace("_filtered", "")
            lbl_file.config(text=fname)
            lbl_progress.config(text=f"{state['idx']+1} / {len(state['files'])}")
            load_preview_image(drawing_no)

            with open(os.path.join(folder, fname), "r", encoding="utf-8") as f:
                items = json.load(f)
            connectors = [it for it in items if CONNECTOR_NAME_RE.match(it.get("text", "").strip())]
            pins = [it for it in items if not CONNECTOR_NAME_RE.match(it.get("text", "").strip())]
            assembled = assemble_connector_pins(connectors, pins)
            state["assembled"] = assembled
            state["conn_keys"] = list(assembled.keys())

            if not assembled:
                tk.Label(body_inner, text="(커넥터명을 찾지 못했습니다)", bg="white").pack(pady=20)
                return

            state["pin_widgets"] = []   # (pin_item, combobox_var) - rows와 같은 순서
            state["pin_rows"] = []      # row Frame들, pin_widgets와 같은 인덱스로 대응
            state["deleted_pins"] = set()  # id(pin) 집합 - 삭제 표시된 항목
            state["sel_idx"] = -1       # 방향키로 이동 중인 현재 선택 인덱스
            for conn_key in state["conn_keys"]:
                group = tk.LabelFrame(body_inner, text=f"{conn_key}  ({len(assembled[conn_key])}핀)",
                                       bg="white", padx=8, pady=6)
                group.pack(fill=tk.X, padx=5, pady=6, anchor=tk.W)
                if not assembled[conn_key]:
                    tk.Label(group, text="(배정된 핀 없음)", bg="white", fg="gray").pack(anchor=tk.W)
                    continue
                for pin in assembled[conn_key]:
                    row = tk.Frame(group, bg="white")
                    row.pack(fill=tk.X, pady=1)
                    row_idx = len(state["pin_widgets"])

                    # 핀 문자 자체를 VLM이 잘못 읽은 경우(실측: E여야 할
                    # 핀이 B로 중복 인식됨)가 있어, 라벨이 아니라 직접
                    # 타이핑해 고칠 수 있는 입력칸으로 만든다(사용자 지시 -
                    # "직접 수정할 수 있게 기능을 추가하자").
                    text_var = tk.StringVar(value=pin.get("text", ""))

                    def on_text_change(*_a, p=pin, v=text_var):
                        p["text"] = v.get()
                    text_var.trace_add("write", on_text_change)
                    entry_text = tk.Entry(row, textvariable=text_var, width=6)
                    entry_text.pack(side=tk.LEFT)
                    lbl_coord = tk.Label(row, text=f"(x={pin.get('x')}, y={pin.get('y')})", bg="white", fg="gray", width=18, anchor=tk.W, cursor="hand2")
                    lbl_coord.pack(side=tk.LEFT)

                    # 사용자 지시: 핀 클릭 시 오른쪽 원본 이미지의 해당
                    # 좌표에 표시(리사이즈 비율 보정은 mark_pin_on_image에서).
                    # 클릭한 행을 "현재 선택"으로 맞춰서, 이후 방향키로 그
                    # 지점부터 이어서 위/아래로 훑을 수 있게 한다.
                    def on_click(_e, i=row_idx):
                        select_pin(i)
                    entry_text.bind("<FocusIn>", on_click)
                    lbl_coord.bind("<Button-1>", on_click)

                    var = tk.StringVar(value=conn_key)
                    combo = ttk.Combobox(row, textvariable=var, values=state["conn_keys"], width=10, state="readonly")
                    combo.pack(side=tk.LEFT, padx=5)

                    def on_delete(p=pin, r=row):
                        # 삭제된 핀은 저장 시 제외된다(rebuild_from_widgets가
                        # deleted_pins에 있는 id(pin)은 건너뜀). 행은 시각적
                        # 피드백으로 취소선처럼 회색 처리하고 버튼을 잠근다.
                        state["deleted_pins"].add(id(p))
                        r.configure(bg="#eee")
                        for w in r.winfo_children():
                            try:
                                w.configure(state="disabled")
                            except tk.TclError:
                                pass
                    tk.Button(row, text="삭제", fg="red", command=on_delete).pack(side=tk.LEFT, padx=5)

                    state["pin_widgets"].append((pin, var))
                    state["pin_rows"].append(row)

            if state["pin_widgets"]:
                select_pin(0)

        def rebuild_from_widgets():
            """드롭다운에서 사람이 바꾼 소속을 반영해 assembled를 다시 만든다.
            "삭제" 버튼을 누른 핀(state["deleted_pins"]에 id가 있는 것)은
            결과에서 완전히 제외한다."""
            new_assembled = {k: [] for k in state["conn_keys"]}
            deleted = state.get("deleted_pins", set())
            for pin, var in state["pin_widgets"]:
                if id(pin) in deleted:
                    continue
                new_assembled.setdefault(var.get(), []).append(pin)
            return new_assembled

        def save_current():
            assembled = rebuild_from_widgets() if state.get("pin_widgets") else state["assembled"]
            if assembled is None:
                return None
            out = {k: [p.get("text", "") for p in v] for k, v in assembled.items()}
            fname = state["files"][state["idx"]]
            drawing_no = os.path.splitext(fname)[0].replace("_filtered", "")
            out_path = os.path.join(assembled_dir, f"{drawing_no}_assembled.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            return out_path

        def go_next():
            out_path = save_current()
            if state["idx"] < len(state["files"]) - 1:
                state["idx"] += 1
                load_current()
            else:
                messagebox.showinfo("완료", f"마지막 파일까지 저장했습니다.\n({out_path})")

        def go_prev():
            save_current()
            if state["idx"] > 0:
                state["idx"] -= 1
                load_current()

        def save_only():
            out_path = save_current()
            messagebox.showinfo("저장됨", f"저장 완료:\n{out_path}")

        tk.Button(btn_bar, text="◀ 이전 (저장 후 이동)", command=go_prev).pack(side=tk.LEFT)
        tk.Button(btn_bar, text="저장만", command=save_only, bg="lightgray").pack(side=tk.LEFT, padx=10)
        tk.Button(btn_bar, text="다음 (저장 후 이동) ▶", command=go_next, bg="lightgreen").pack(side=tk.RIGHT)

        load_current()

    def render_clean(self):
        if not self.results or self.cropped_img is None:
            messagebox.showwarning("경고", "먼저 추출을 진행해주세요.")
            return

        self.txt_log.insert(tk.END, "\n--- 클린 텍스트 & 박스 렌더링 ---\n")
        h, w = self.cropped_img.shape[:2]
        self.clean_img = np.full((h, w, 3), 255, dtype=np.uint8)

        for r in self.results:
            layout = r.get('layout', 'vertical')
            if 'p0' in r:
                p0, p1 = r['p0'], r['p1']
            else:
                p0, p1 = r['y0'], r['y1']
            letter = r['letter']
            font = cv2.FONT_HERSHEY_SIMPLEX
            thickness = 2
            if layout == 'vertical':
                cv2.rectangle(self.clean_img, (0, p0), (w, p1), (0, 0, 0), 2)
                font_scale = max(0.8, (w / 50.0))
                text_size = cv2.getTextSize(letter, font, font_scale, thickness)[0]
                text_x = (w - text_size[0]) // 2
                text_y = p0 + (p1 - p0 + text_size[1]) // 2
                cv2.putText(self.clean_img, letter, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)
            else:
                cv2.rectangle(self.clean_img, (p0, 0), (p1, h), (0, 0, 0), 2)
                font_scale = max(0.8, (h / 50.0))
                text_size = cv2.getTextSize(letter, font, font_scale, thickness)[0]
                text_x = p0 + (p1 - p0 - text_size[0]) // 2
                text_y = (h + text_size[1]) // 2
                cv2.putText(self.clean_img, letter, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)

        self.display_result_image(self.clean_img, self.lbl_proc)
        self.txt_log.insert(tk.END, "클린 렌더링 완료.\n")
        self.txt_log.see(tk.END)

    def display_result_image(self, cv_img, label_widget):
        h, w = cv_img.shape[:2]
        max_h = 500
        scale = min(max_h / h, 1.0)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(cv_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        if len(resized.shape) == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(resized)
        img_tk = ImageTk.PhotoImage(img_pil)
        label_widget.configure(image=img_tk)
        label_widget.image = img_tk

    def save_image(self):
        if self.clean_img is None or not self.results:
            messagebox.showwarning("경고", "저장할 결과가 없습니다.")
            return

        save_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("Vector PDF files", "*.pdf"), ("PNG Image files", "*.png")]
        )
        if not save_path: return

        if save_path.lower().endswith(".pdf"):
            h, w = self.cropped_img.shape[:2]
            fig, ax = plt.subplots(figsize=(w/100, h/100), dpi=100)
            ax.set_xlim(0, w)
            ax.set_ylim(-h, 0)
            ax.axis("off")
            for r in self.results:
                layout = r.get('layout', 'vertical')
                p0, p1 = (r['p0'], r['p1']) if 'p0' in r else (r['y0'], r['y1'])
                letter = r['letter']
                if layout == 'vertical':
                    box_h = p1 - p0
                    rect = Rectangle((0, -p1), w, box_h, fill=False, edgecolor='black', linewidth=1.0)
                    ax.add_patch(rect)
                    ax.text(w / 2, -(p0 + box_h / 2), letter, ha="center", va="center", fontsize=max(10, w * 0.3))
                else:
                    box_w = p1 - p0
                    rect = Rectangle((p0, -h), box_w, h, fill=False, edgecolor='black', linewidth=1.0)
                    ax.add_patch(rect)
                    ax.text(p0 + box_w / 2, -h / 2, letter, ha="center", va="center", fontsize=max(10, h * 0.3))
            fig.savefig(save_path, format="pdf", bbox_inches="tight")
            plt.close(fig)
            messagebox.showinfo("완료", "벡터 PDF 저장 완료.")
        else:
            cv2.imwrite(save_path, self.clean_img)
            messagebox.showinfo("완료", "PNG 저장 완료.")

if __name__ == "__main__":
    root = tk.Tk()
    app = TemplateMatcherGUI(root)
    root.mainloop()
