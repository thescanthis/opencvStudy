"""
templates/<letter>.png 라이브러리를 이용해, 배선도 좌측(또는 우측) 핀
이름 열의 각 셀 이미지에 어떤 글자가 있는지 템플릿 매칭으로 추정한다.

일반 OCR이 실패하는 좁은 테두리 셀 + 특수 CAD 폰트 조합에서, 이 도면
세트가 어차피 고정된 폰트/스타일을 쓴다는 점을 이용해 우회한다.
"""
import os
import cv2
import numpy as np

HERE = os.path.dirname(__file__)
TPL_DIR = os.path.join(HERE, "templates")

LETTERS = list("SQPNMLKJHGFECDBATURV")


def load_templates():
    tpls = {}
    for l in LETTERS:
        p = os.path.join(TPL_DIR, f"{l}.png")
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if im is not None:
            tpls[l] = im
    return tpls


def _find_row_dividers(bin_img):
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
    추정해 (x0, x1)을 반환한다."""
    h, w = bin_img.shape
    col_sum = bin_img.sum(axis=0)
    thresh = h * 255 * 0.5
    border_cols = col_sum > thresh
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
    """cell_bin(이진, 글자=255) 안에서 살짝 팽창시켜 글자의 끊긴 획들을
    하나의 성분으로 합친 뒤 가장 큰 연결 성분(=글자 본체)만 tight crop."""
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


def split_cells(col_img_path, n_expected=None):
    """열 이미지를 셀 경계선 기준으로 분할해 (y0,y1,cell_img) 리스트 반환.
    테두리선/리드선은 제외하고 글자 영역만 남긴다."""
    img = cv2.imread(col_img_path, cv2.IMREAD_GRAYSCALE)
    _, bin_img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
    dividers = _find_row_dividers(bin_img)
    lx0, lx1 = _find_letter_region_x(bin_img)
    cells = []
    pad = 3
    for i in range(len(dividers) - 1):
        y0, y1 = dividers[i] + pad, dividers[i + 1] - pad
        if y1 <= y0:
            y0, y1 = dividers[i], dividers[i + 1]
        cell = img[y0:y1, lx0:lx1]
        _, cell_bin = cv2.threshold(cell, 150, 255, cv2.THRESH_BINARY_INV)
        cell = _tight_crop_main_component(cell_bin, cell)
        cells.append((dividers[i], dividers[i + 1], cell))
    return cells


def match_cell(cell_img, templates):
    """cell_img 하나를 모든 템플릿과 매칭해 (best_letter, best_score, all_scores) 반환.

    그레이스케일 그대로 스케일을 스윕하며 매칭하면 D/L처럼 내부 흰 공간
    구조가 다른 글자끼리 혼동이 잦았다(검증됨). 대신 둘 다 이진화한 뒤
    템플릿을 셀 크기에 맞춰 정확히 리사이즈해서 한 번만 매칭한다 - 글자
    획의 유무(흑백 패턴) 자체를 비교하는 것이라 더 안정적이다."""
    if cell_img.size == 0 or cell_img.shape[0] < 3 or cell_img.shape[1] < 3:
        return None, 0.0, {}

    _, cell_bin = cv2.threshold(cell_img, 150, 255, cv2.THRESH_BINARY)
    ch, cw = cell_bin.shape

    scores = {}
    for letter, tpl in templates.items():
        _, tpl_bin = cv2.threshold(tpl, 150, 255, cv2.THRESH_BINARY)
        tpl_r = cv2.resize(tpl_bin, (cw, ch), interpolation=cv2.INTER_AREA)
        _, tpl_r = cv2.threshold(tpl_r, 127, 255, cv2.THRESH_BINARY)
        res = cv2.matchTemplate(cell_bin, tpl_r, cv2.TM_CCOEFF_NORMED)
        scores[letter] = float(res.max())

    best_letter = max(scores, key=scores.get)
    return best_letter, scores[best_letter], scores


def match_column(col_img_path, expected_order=None):
    """expected_order 를 주면(예: 도면에서 사람이 한 번 확인한 핀 이름
    시퀀스), 템플릿 매칭은 "셀 개수가 맞는지"만 검증하는 용도로 쓰고,
    실제 글자는 그 순서를 그대로 배정한다 - 글자 하나하나를 매칭하는
    것보다(자기 열 기준 100%지만 다른 열에서는 63% 수준으로 불안정,
    검증됨) 훨씬 신뢰도가 높다. 커넥터 핀 배열은 도면마다 고정이므로
    한 번 사람이 확인해두면 재사용 가능."""
    templates = load_templates()
    cells = split_cells(col_img_path)
    results = []
    for y0, y1, cell in cells:
        letter, score, _ = match_cell(cell, templates)
        results.append({"y0": y0, "y1": y1, "letter": letter, "score": round(float(score), 3)})

    if expected_order is not None:
        if len(expected_order) != len(results):
            print(f"  [warn] expected_order 길이({len(expected_order)}) != "
                  f"검출된 셀 개수({len(results)}) - 셀 분할부터 다시 확인 필요")
        else:
            for r, exp in zip(results, expected_order):
                if r["letter"] != exp:
                    print(f"  [fix] y={r['y0']}: matcher='{r['letter']}' -> '{exp}' (expected_order)")
                r["ocr_letter"] = r["letter"]
                r["letter"] = exp
                r["order_corrected"] = True
    return results


if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    expected = list(sys.argv[2]) if len(sys.argv) > 2 else None
    results = match_column(path, expected_order=expected)
    for r in results:
        print(f"  y={r['y0']:>5}-{r['y1']:<5} -> {r['letter']}  (score={r['score']})")
    seq = "".join(r["letter"] for r in results)
    print(f"\n인식 순서: {seq}")
