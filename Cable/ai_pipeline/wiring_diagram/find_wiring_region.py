"""
도면 원본(PDF에서 추출한 고해상도 PNG)에서 "배선도(결선도)" 영역을 텍스트
OCR 없이, 표 구조(균등 간격으로 촘촘히 반복되는 가로선) 특징만으로 자동
탐지한다.

이유: 이 도면들의 스캔 폰트/노이즈 특성상 Tesseract/EasyOCR 모두 "배선도"
제목 텍스트 인식이 거의 실패한다(별도 조사로 확인됨). 반면 배선도 표
자체는 핀 개수만큼(보통 15~30개) 균등한 간격(약 80~95px, 2000px 렌더
기준 약 16~19px)으로 반복되는 가로선이라는 뚜렷한 기하학적 특징이 있어,
이 패턴을 찾는 게 훨씬 안정적이다.
"""
import os
import cv2
import numpy as np

HERE = os.path.dirname(__file__)


def _detect_horizontal_lines(bin_img, min_len_ratio=0.04):
    h, w = bin_img.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, int(w * min_len_ratio)), 1))
    horiz = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)
    row_sum = horiz.sum(axis=1)
    rows = np.where(row_sum > 0)[0]
    if len(rows) == 0:
        return []
    groups = []
    cur = [rows[0]]
    for r in rows[1:]:
        if r - cur[-1] <= 3:
            cur.append(r)
        else:
            groups.append(int(np.mean(cur)))
            cur = [r]
    groups.append(int(np.mean(cur)))
    return groups


def _find_dense_even_run(lines, gap_range=(15, 100), gap_tolerance=25, min_run=8):
    """lines(y좌표 오름차순) 중, 인접 간격이 gap_range 안에 들고 서로
    비슷한(오차 gap_tolerance 이내) 구간이 min_run개 이상 연속되는 가장
    긴 구간을 찾는다. 배선도 표의 "균등 간격 반복 가로선" 특징을 탐지."""
    if len(lines) < min_run:
        return None
    gaps = [lines[i + 1] - lines[i] for i in range(len(lines) - 1)]

    best = None  # (start_idx, end_idx, count)
    i = 0
    n = len(gaps)
    while i < n:
        if not (gap_range[0] <= gaps[i] <= gap_range[1]):
            i += 1
            continue
        j = i
        ref_gap = gaps[i]
        while j + 1 < n and gap_range[0] <= gaps[j + 1] <= gap_range[1] \
                and abs(gaps[j + 1] - ref_gap) <= gap_tolerance:
            j += 1
        count = j - i + 2  # 선 개수 = 구간 개수+1
        if best is None or count > best[2]:
            best = (i, j + 1, count)
        i = j + 1
    if best is None or best[2] < min_run:
        return None
    start_idx, end_idx, count = best
    return lines[start_idx], lines[end_idx], count


def find_wiring_table_y_range(gray_img, scan_region=None, bin_thresh=150):
    """gray_img(전체 도면, 그레이스케일)에서 배선도 표의 y범위(y0,y1)와
    검출된 선 개수를 반환. scan_region=(x0,y0,x1,y1) 을 주면 그 안에서만
    탐색(속도/정확도 향상, 대략적인 하단 영역 등)."""
    h, w = gray_img.shape
    if scan_region:
        sx0, sy0, sx1, sy1 = scan_region
    else:
        sx0, sy0, sx1, sy1 = 0, int(h * 0.5), w, h  # 기본: 하단 절반에서 탐색

    sub = gray_img[sy0:sy1, sx0:sx1]
    _, bin_img = cv2.threshold(sub, bin_thresh, 255, cv2.THRESH_BINARY_INV)
    lines = _detect_horizontal_lines(bin_img)
    if not lines:
        return None

    run = _find_dense_even_run(lines)
    if run is None:
        return None
    y0_local, y1_local, count = run
    # 표 위아래로 여유(핀 이름 헤더, 목적지 커넥터 라벨 등) 약간 추가
    pad = 40
    return sy0 + y0_local - pad, sy0 + y1_local + pad, count


def find_wiring_table_bbox(gray_img, scan_region=None):
    """y범위를 찾은 뒤, 그 y범위 안에서 텍스트/선이 있는 x범위(좌우 끝)도
    추정해서 (x0,y0,x1,y1) 전체 bbox를 반환."""
    h, w = gray_img.shape
    yr = find_wiring_table_y_range(gray_img, scan_region=scan_region)
    if yr is None:
        return None
    y0, y1, count = yr
    y0 = max(0, y0)
    y1 = min(h, y1)

    sub = gray_img[y0:y1, :]
    _, bin_img = cv2.threshold(sub, 150, 255, cv2.THRESH_BINARY_INV)
    col_sum = bin_img.sum(axis=0)
    cols = np.where(col_sum > 0)[0]
    if len(cols) == 0:
        x0, x1 = 0, w
    else:
        x0, x1 = int(cols.min()), int(cols.max())
        pad_x = 30
        x0 = max(0, x0 - pad_x)
        x1 = min(w, x1 + pad_x)

    return x0, y0, x1, y1, count


def main(image_path, scan_region=None):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(image_path)
    result = find_wiring_table_bbox(img, scan_region=scan_region)
    if result is None:
        print("wiring table not found")
        return None
    x0, y0, x1, y1, count = result
    print(f"table region: ({x0},{y0})-({x1},{y1})  ~{count} lines detected")

    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 255), 4)
    name = os.path.splitext(os.path.basename(image_path))[0]
    out_dir = os.path.join(HERE, "out")
    os.makedirs(out_dir, exist_ok=True)
    small = cv2.resize(vis, None, fx=0.25, fy=0.25)
    out_path = os.path.join(out_dir, f"{name}_region_vis.png")
    cv2.imwrite(out_path, small)
    print(f"-> {out_path}")
    return (x0, y0, x1, y1)


if __name__ == "__main__":
    import sys
    main(sys.argv[1])
