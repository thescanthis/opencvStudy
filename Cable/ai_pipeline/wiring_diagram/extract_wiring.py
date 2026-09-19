"""
케이블 도면 하단 "배선도(결선도)" 영역에서 핀 결선 관계를 자동 추출한다.

방법:
  1. 배선도 영역을 크게 확대 + 이진화해서 OCR(Tesseract) 정확도를 높인다.
  2. "K-숫자" 형태의 배선번호 텍스트를 전부 찾는다(왼쪽 열 + 오른쪽 열에
     각각 한 번씩, 같은 배선이 중복 표기됨).
  3. 같은 y좌표(줄) 에 있는 왼쪽 K-n / 오른쪽 K-n 을 한 배선으로 짝짓는다.
  4. OpenCV HoughLinesP 로 그 줄의 가로선(실제 배선 인출선)이 존재하는지
     확인해서 신뢰도를 보강한다(텍스트만으로 매칭 오류가 나는 걸 줄임).

출력: {"wires": [{"code": "K-16", "y": 41, "line_detected": true}, ...]}
아직은 "어느 핀 이름(S,Q,P...)과 어느 목적지 핀(H,F,E...)에 연결되는지"까지는
포함하지 않는다 - 그건 2단계(핀 열 OCR + x좌표 매칭)에서 추가한다.
"""
import os
import re
import json
import cv2
import numpy as np
import pytesseract

TESSERACT_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE

HERE = os.path.dirname(__file__)
OUT_DIR = os.path.join(HERE, "out")

WIRE_RE = re.compile(r"^K-?\s?(\d{1,3})[;:.]?$", re.IGNORECASE)


def _normalize_wire_code(text):
    """OCR 결과에서 'K-16', 'K-16;', 'K16' 등을 'K-16' 표준형으로 정규화.
    아니면 None."""
    t = text.strip().upper().replace(" ", "")
    m = WIRE_RE.match(t)
    if not m:
        return None
    return f"K-{int(m.group(1))}"


def ocr_words(gray_img, scale=3, psm=11, lang="eng"):
    """gray_img(원본 crop, 확대 전)를 scale배 확대+이진화한 뒤 OCR,
    단어 단위 바운딩박스 리스트를 원본 좌표계로 변환해 반환.
    반환: [{"text":..., "x":..., "y":..., "w":..., "h":..., "conf":...}, ...]
    """
    big = cv2.resize(gray_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    _, bin_img = cv2.threshold(big, 180, 255, cv2.THRESH_BINARY)
    data = pytesseract.image_to_data(bin_img, lang=lang, config=f"--psm {psm}",
                                      output_type=pytesseract.Output.DICT)
    words = []
    n = len(data["text"])
    for i in range(n):
        t = data["text"][i].strip()
        conf = int(data["conf"][i]) if data["conf"][i] != "-1" else -1
        if not t or conf < 0:
            continue
        words.append({
            "text": t,
            "x": data["left"][i] / scale,
            "y": data["top"][i] / scale,
            "w": data["width"][i] / scale,
            "h": data["height"][i] / scale,
            "conf": conf,
        })
    return words


def detect_horizontal_lines(gray_img, min_len=60):
    edges = cv2.Canny(gray_img, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40,
                             minLineLength=min_len, maxLineGap=5)
    if lines is None:
        return []
    out = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if angle < 3 or angle > 177:
            out.append((min(x1, x2), max(x1, x2), (y1 + y2) / 2))
    return out


def extract_wire_codes(gray_img):
    """배선도 crop 이미지에서 K-n 코드들을 전부 찾아 y좌표 순으로 묶는다.
    같은 배선은 왼쪽/오른쪽에 한 번씩 두 번 나타나므로, y좌표가 가까운
    (±6px) 것끼리 좌/우로 나눠 짝짓는다."""
    words = ocr_words(gray_img)
    codes = []
    for w in words:
        code = _normalize_wire_code(w["text"])
        if code:
            codes.append({**w, "code": code})

    codes.sort(key=lambda c: c["y"])

    h_lines = detect_horizontal_lines(gray_img)

    # y좌표 클러스터링(같은 줄) - 6px 이내면 같은 줄로 취급
    rows = []
    for c in codes:
        placed = False
        for row in rows:
            if abs(row["y"] - c["y"]) < 8:
                row["items"].append(c)
                row["y"] = sum(i["y"] for i in row["items"]) / len(row["items"])
                placed = True
                break
        if not placed:
            rows.append({"y": c["y"], "items": [c]})

    wires = []
    for row in rows:
        items = sorted(row["items"], key=lambda c: c["x"])
        left = items[0]
        right = items[-1] if len(items) > 1 else None
        # 같은 줄에 가로선이 있는지 확인(신뢰도 보강)
        has_line = any(abs(ly - row["y"]) < 10 for (lx1, lx2, ly) in h_lines)
        wires.append({
            "code": left["code"],
            "y": round(row["y"], 1),
            "left_x": round(left["x"], 1),
            "right_x": round(right["x"], 1) if right else None,
            "duplicated": right is not None,
            "line_detected": has_line,
        })
    return wires


def fix_wire_order(wires, expected_order=None):
    """OCR 숫자 1자리 오독(K-16->K-18, K-6->K-68 등)을 순서 규칙으로 보정한다.

    expected_order 를 안 주면, wires 를 y좌표 순으로 나열했을 때 "값이 앞뒤
    이웃과 비교해 비정상적으로 튀는" 항목만 교정 후보로 표시한다(자동으로
    확신 있게 고치지 않고, 의심스러운 것만 표시 - 오탐 방지). expected_order
    를 리스트로 주면(예: ['K-16','K-15',...,'K-1','K-17','K-18','K-19','K-20'])
    그 순서에 맞춰 그대로 덮어쓴다(가장 안전, 도면 구조를 사람이 확인했을 때)."""
    if expected_order:
        if len(expected_order) != len(wires):
            print(f"  [warn] expected_order 길이({len(expected_order)}) != "
                  f"검출된 wires 개수({len(wires)}) - 보정 건너뜀")
            return wires
        for w, exp in zip(wires, expected_order):
            if w["code"] != exp:
                print(f"  [fix] y={w['y']}: '{w['code']}' -> '{exp}'")
                w["code"] = exp
                w["ocr_corrected"] = True
        return wires

    # expected_order 없을 때: 숫자만 뽑아 이웃과 비교, 튀는 값만 flag
    nums = []
    for w in wires:
        m = re.match(r"K-(\d+)", w["code"])
        nums.append(int(m.group(1)) if m else None)
    for i, w in enumerate(wires):
        if nums[i] is None:
            continue
        neighbors = [nums[j] for j in (i - 1, i + 1)
                     if 0 <= j < len(nums) and nums[j] is not None]
        if neighbors and all(abs(nums[i] - nb) > 5 for nb in neighbors):
            w["suspect_ocr_error"] = True
            print(f"  [suspect] y={w['y']}: '{w['code']}' 이웃과 값 차이가 큼 - 확인 필요")
    return wires


def main(image_path, crop=None, expected_order=None):
    """crop: (x0,y0,x1,y1) 원본 이미지에서 배선도 영역. None이면 전체."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(image_path)
    if crop:
        x0, y0, x1, y1 = crop
        sub = img[y0:y1, x0:x1]
    else:
        sub = img

    wires = extract_wire_codes(sub)
    wires = fix_wire_order(wires, expected_order=expected_order)

    name = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{name}_wires.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"image": image_path, "crop": crop, "wires": wires}, f,
                   indent=2, ensure_ascii=False)
    print(f"-> {out_path} ({len(wires)} wires)")
    return wires


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python extract_wiring.py <image_path> [x0 y0 x1 y1]")
        sys.exit(1)
    image_path = sys.argv[1]
    crop = tuple(int(v) for v in sys.argv[2:6]) if len(sys.argv) >= 6 else None
    main(image_path, crop)
