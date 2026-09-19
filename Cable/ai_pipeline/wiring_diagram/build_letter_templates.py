"""
배선도 좌측 핀 이름 열(S,Q,P,N,M,L,K,J,H,G,F,E,C,D,B,A,T,U,R,V ...)처럼
좁은 테두리 셀 안의 알파벳 한 글자는 일반 OCR(Tesseract/EasyOCR)로 인식이
거의 안 된다(별도 검증 완료). 대신 이 폰트 자체가 도면마다 똑같으므로,
알려진 정답이 있는 한 장(60309822)에서 글자별 템플릿 이미지를 잘라
템플릿 매칭(cv2.matchTemplate)용 라이브러리로 저장해둔다.

사용법:
    python build_letter_templates.py
      out/_left_col_hires_clean.png (20행, 이미 셀 경계로 크롭된 좌측 열
      이미지) 를 20등분해서 templates/<letter>.png 로 저장한다.
"""
import os
import cv2
import numpy as np

HERE = os.path.dirname(__file__)
OUT_DIR = os.path.join(HERE, "out")
TPL_DIR = os.path.join(HERE, "templates")

# out/_left_col_hires_clean.png 의 위에서 아래 순서(60309822 P1 좌측 열 기준)
KNOWN_ORDER = list("SQPNMLKJHGFECDBATURV")


def _find_row_dividers(bin_img):
    """각 행을 나누는 가로 테두리선의 y좌표들을 찾는다. 글자 내부의 가로 획
    (F/E/H 등)과 구분하기 위해, 셀 폭의 70% 이상을 가로지르는 선만 인정한다."""
    h, w = bin_img.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(w * 0.7), 1))
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


def _find_letter_region_x(bin_img):
    """열 이미지 안에서 세로로 쭉 이어진 굵은 선(셀 테두리/리드선 경계)들의
    x좌표를 모두 찾고, 그 중 폭이 가장 넓은 "빈 구간"을 글자 영역으로
    추정해 (x0, x1)을 반환한다. 테두리가 글자 왼쪽에 있든 오른쪽에 있든
    (도면마다 crop 방식에 따라 다름) 안정적으로 글자만 골라내기 위함."""
    h, w = bin_img.shape
    col_sum = bin_img.sum(axis=0)
    thresh = h * 255 * 0.5
    border_cols = col_sum > thresh
    # border_cols가 True인 지점들을 경계로 구간을 나누고, 가장 넓은 구간 선택
    bounds = [0] + [i for i in range(w) if border_cols[i]] + [w - 1]
    best = (0, w)
    best_len = 0
    for i in range(len(bounds) - 1):
        x0, x1 = bounds[i], bounds[i + 1]
        if x1 - x0 > best_len:
            best_len = x1 - x0
            best = (x0, x1)
    return best


def _tight_crop_main_component(cell_bin, cell_gray, margin=2):
    """cell_bin(이진, 글자=255) 안에서, 살짝 팽창(dilate)시켜 글자의 끊긴
    획들을 하나의 성분으로 합친 뒤 가장 큰 연결 성분(=글자 본체)의 bbox를
    구하고, 그 bbox로 (팽창 전) cell_gray를 tight crop한다. 점선 테두리
    잔여물이나 리드선 파편처럼 글자와 떨어진 작은 성분은 제외된다."""
    dilated = cv2.dilate(cell_bin, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    if n <= 1:
        return cell_gray
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = 1 + int(np.argmax(areas))
    x, y, w, h = stats[best, cv2.CC_STAT_LEFT], stats[best, cv2.CC_STAT_TOP], \
        stats[best, cv2.CC_STAT_WIDTH], stats[best, cv2.CC_STAT_HEIGHT]
    x0 = max(0, x - margin)
    y0 = max(0, y - margin)
    x1 = min(cell_gray.shape[1], x + w + margin)
    y1 = min(cell_gray.shape[0], y + h + margin)
    return cell_gray[y0:y1, x0:x1]


def build(src_path, order=KNOWN_ORDER):
    img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(src_path)
    _, bin_img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
    dividers = _find_row_dividers(bin_img)
    lx0, lx1 = _find_letter_region_x(bin_img)
    print(f"dividers found: {len(dividers)} -> {dividers}, letter_x=({lx0},{lx1})")

    n_cells = len(dividers) - 1
    if n_cells != len(order):
        print(f"[warn] cell count({n_cells}) != order length({len(order)}) - "
              f"균등분할로 대체")
        h = img.shape[0]
        step = h / len(order)
        dividers = [int(i * step) for i in range(len(order) + 1)]

    os.makedirs(TPL_DIR, exist_ok=True)
    pad = 3
    for i, letter in enumerate(order):
        y0, y1 = dividers[i] + pad, dividers[i + 1] - pad
        if y1 <= y0:
            y0, y1 = dividers[i], dividers[i + 1]
        cell = img[y0:y1, lx0:lx1]
        # 연결 성분 중 가장 큰 것(글자 본체)만 남기고 tight crop
        _, cell_bin = cv2.threshold(cell, 150, 255, cv2.THRESH_BINARY_INV)
        cell = _tight_crop_main_component(cell_bin, cell)
        out_path = os.path.join(TPL_DIR, f"{letter}.png")
        cv2.imwrite(out_path, cell)
        print(f"  {letter}: {cell.shape} -> {out_path}")

    print(f"\n완료: {len(order)}개 템플릿 -> {TPL_DIR}")


if __name__ == "__main__":
    build(os.path.join(OUT_DIR, "_left_col_hires_clean.png"))
