import cv2
import numpy as np


class OwnerLookup:
    """원본 이진 이미지의 연결요소 정보. '이 선분이 속한 덩어리가 글자만 한가?'를 판단할 때 쓴다.
    진짜 선은 박스/결선과 이어져 큰 덩어리를 이루고, 글자 획('1', 'J', '_')은 글자 크기 덩어리에 속한다."""

    def __init__(self, binary, char_h):
        _, self.labels, self.stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        self.limit = char_h * 2.5

    def is_small(self, ys, xs):
        """ys, xs: 선분 픽셀 좌표 배열"""
        ids = self.labels[ys, xs]
        ids = ids[ids > 0]
        if len(ids) == 0:
            return True
        owner = np.bincount(ids).argmax()
        _, _, w, h, _ = self.stats[owner]
        return max(w, h) < self.limit


def _segments_from_mask(mask, axis, owners):
    """형태학 열기 결과 마스크의 연결요소 하나 = 선분 하나.
    글자 크기 덩어리에 속한 선분은 마스크에서 지우고 버린다.
    axis='H' → (x0, y, x1, y), axis='V' → (x, y0, x, y1)"""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    segs = []
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        crop = labels[y:y + h, x:x + w] == i
        ys, xs = np.nonzero(crop)
        if owners.is_small(ys + y, xs + x):
            mask[y:y + h, x:x + w][crop] = 0
            continue
        if axis == 'H':
            yc = y + h // 2
            segs.append((int(x), int(yc), int(x + w - 1), int(yc)))
        else:
            xc = x + w // 2
            segs.append((int(xc), int(y), int(xc), int(y + h - 1)))
    return segs


def extract_hv_lines(binary, char_h, owners):
    """글자 높이의 0.8배보다 긴 가로/세로 획을 후보로 뽑고, 글자 크기 덩어리에 속한 것은 버린다.
    열기(open) 연산은 긴 획의 픽셀만 남기므로, 선에 글자가 붙어 있어도 글자는 살아남는다."""
    min_len = max(int(char_h * 0.8), 8)
    h_k = cv2.getStructuringElement(cv2.MORPH_RECT, (min_len, 1))
    v_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_len))
    h_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_k)
    v_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_k)
    h_segs = _segments_from_mask(h_mask, 'H', owners)
    v_segs = _segments_from_mask(v_mask, 'V', owners)
    return h_mask, v_mask, h_segs, v_segs


def extract_dashed_lines(residual, char_h, stroke_w, min_run=3):
    """점선 = 속이 꽉 찬 짧고 가는 막대(대시)가 같은 축 위에 일정 간격으로 3개 이상 늘어선 것.
    글자 '1', '_' 등도 막대 모양이지만, 대시 길이를 글자 높이의 0.8배 미만으로 제한하고
    간격이 글자 높이 이내로 연속되어야 하므로 대부분 걸러진다."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(residual, connectivity=8)
    max_thick = stroke_w * 2 + 1
    max_dash = char_h * 0.8
    pos_tol = max(2, stroke_w)
    max_gap = char_h

    h_dash, v_dash = [], []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        fill = area / max(w * h, 1)
        if fill < 0.7:
            continue
        if w >= 2 * h and h <= max_thick and w < max_dash:
            h_dash.append((x, x + w - 1, y + h / 2, i))
        elif h >= 2 * w and w <= max_thick and h < max_dash:
            v_dash.append((y, y + h - 1, x + w / 2, i))

    def chain(dashes):
        # 같은 위치(pos) 그룹 → 진행 방향 정렬 → 간격이 max_gap 이내면 한 줄
        dashes = sorted(dashes, key=lambda d: (d[2], d[0]))
        groups, cur = [], []
        for d in dashes:
            if cur and abs(d[2] - cur[-1][2]) > pos_tol:
                groups.append(cur)
                cur = []
            cur.append(d)
        if cur:
            groups.append(cur)

        runs = []
        for g in groups:
            g.sort(key=lambda d: d[0])
            run = [g[0]]
            for d in g[1:]:
                if 0 <= d[0] - run[-1][1] <= max_gap:
                    run.append(d)
                else:
                    runs.append(run)
                    run = [d]
            runs.append(run)
        return [r for r in runs if len(r) >= min_run]

    mask = np.zeros_like(residual)
    segs = []
    for axis, dashes in (('H', h_dash), ('V', v_dash)):
        for run in chain(dashes):
            a0 = int(min(d[0] for d in run))
            a1 = int(max(d[1] for d in run))
            p = int(round(sum(d[2] for d in run) / len(run)))
            segs.append((a0, p, a1, p) if axis == 'H' else (p, a0, p, a1))
            for d in run:
                mask[labels == d[3]] = 255
    return mask, segs


def _line_pixels(x1, y1, x2, y2):
    n = max(abs(x2 - x1), abs(y2 - y1)) + 1
    xs = np.round(np.linspace(x1, x2, n)).astype(int)
    ys = np.round(np.linspace(y1, y2, n)).astype(int)
    return ys, xs


def extract_diagonal_lines(residual, char_h, stroke_w, owners):
    """가로/세로가 아닌 직선(지시선 등).
    Hough 후보 중 ① 중심선 잉크 비율 90% 이상(원호 제외) ② 글자 크기 덩어리가 아닌 것만 인정."""
    min_len = max(int(char_h * 1.5), 20)
    raw = cv2.HoughLinesP(residual, 1, np.pi / 180, threshold=int(min_len * 0.8),
                          minLineLength=min_len, maxLineGap=stroke_w * 2)
    segs = []
    if raw is not None:
        for x1, y1, x2, y2 in raw[:, 0]:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))) % 180
            if not (5 < ang < 85 or 95 < ang < 175):
                continue
            # 중심선을 따라 잉크가 차 있는 비율. 원호는 중간이 휘어서 낮게 나온다.
            ys, xs = _line_pixels(x1, y1, x2, y2)
            on = residual[ys, xs] > 0
            if on.mean() < 0.9:
                continue
            if owners.is_small(ys[on], xs[on]):
                continue
            segs.append((x1, y1, x2, y2))

    # 굵은 선은 Hough가 여러 번 잡으므로, 그린 뒤 잔여 잉크와 AND 해서 마스크만 사용
    drawn = np.zeros_like(residual)
    for x1, y1, x2, y2 in segs:
        cv2.line(drawn, (x1, y1), (x2, y2), 255, stroke_w + 2)
    mask = cv2.bitwise_and(drawn, residual)
    return mask, segs


def extract_lines(binary, char_h, stroke_w, owner_source=None):
    """owner_source: 글자 크기 판단용 원본 이진 이미지(원 제거 전). 없으면 binary 사용."""
    owners = OwnerLookup(binary if owner_source is None else owner_source, char_h)
    h_mask, v_mask, h_segs, v_segs = extract_hv_lines(binary, char_h, owners)
    hv_mask = cv2.bitwise_or(h_mask, v_mask)

    # 선을 뺀 나머지에서 점선과 사선을 찾는다
    residual = cv2.bitwise_and(binary, cv2.bitwise_not(hv_mask))
    dash_mask, dash_segs = extract_dashed_lines(residual, char_h, stroke_w)
    residual = cv2.bitwise_and(residual, cv2.bitwise_not(dash_mask))
    diag_mask, diag_segs = extract_diagonal_lines(residual, char_h, stroke_w, owners)
    residual = cv2.bitwise_and(residual, cv2.bitwise_not(diag_mask))

    return {
        'h_mask': h_mask, 'v_mask': v_mask, 'dash_mask': dash_mask, 'diag_mask': diag_mask,
        'line_mask': cv2.bitwise_or(cv2.bitwise_or(hv_mask, dash_mask), diag_mask),
        'h_segs': h_segs, 'v_segs': v_segs, 'dash_segs': dash_segs, 'diag_segs': diag_segs,
        'residual': residual,
    }
