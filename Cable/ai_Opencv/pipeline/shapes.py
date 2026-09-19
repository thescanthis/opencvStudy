import cv2
import numpy as np


def _holes(binary):
    """잉크로 둘러싸인 안쪽 구멍(hole) 윤곽들. RETR_CCOMP에서 부모가 있는 윤곽 = 구멍."""
    contours, hier = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hier is None:
        return []
    return [c for c, h in zip(contours, hier[0]) if h[3] != -1]


def _erase_outline(out_mask, contour, labels, stroke_w):
    """구멍 둘레의 테두리 잉크를 out_mask에 표시한다.
    - 구멍 안쪽(도형 속 글자)은 건드리지 않는다.
    - 테두리 띠 안에서도 '구멍을 둘러싼 덩어리'와 같은 연결요소의 잉크만 지운다.
      (테두리 바깥에 붙어 있는 다른 글자를 같이 지우지 않도록)
    바운딩 박스 영역만 잘라서 처리."""
    # 테두리 두께 + 안티앨리어싱 번짐까지 덮도록 여유 있게
    pad = stroke_w * 4 + 3
    H, W = labels.shape
    x, y, w, h = cv2.boundingRect(contour)
    x0, y0 = max(x - pad, 0), max(y - pad, 0)
    x1, y1 = min(x + w + pad, W), min(y + h + pad, H)
    lab = labels[y0:y1, x0:x1]
    hole = np.zeros(lab.shape, np.uint8)
    cv2.drawContours(hole, [contour - [x0, y0]], -1, 255, cv2.FILLED)

    # 구멍 바로 바깥 1~2px에 있는 잉크 라벨 중 최빈값 = 테두리 덩어리
    inner = cv2.dilate(hole, np.ones((5, 5), np.uint8)) & ~hole
    ids = lab[(inner > 0) & (lab > 0)]
    if len(ids) == 0:
        return
    owner = np.bincount(ids).argmax()

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad, pad))
    ring = (cv2.dilate(hole, k) > 0) & (hole == 0) & (lab == owner)
    out_mask[y0:y1, x0:x1][ring] = 255


def detect_circles(binary, char_h, stroke_w):
    """원형 도형(부품 번호 동그라미, 커넥터 원 등).
    구멍 면적이 같은 크기 타원 면적과 거의 같고(0.88~1.08), 볼록하며, 글자 구멍보다 큰 것.
    선 추출보다 먼저 실행해야 원호가 짧은 가로/세로 선 조각으로 잘못 잡히지 않는다."""
    _, labels = cv2.connectedComponents(binary, connectivity=8)
    circles = []
    mask = np.zeros_like(binary)
    for c in _holes(binary):
        x, y, w, h = cv2.boundingRect(c)
        if min(w, h) < char_h * 1.2 or not (0.75 < w / h < 1.33):
            continue
        area = cv2.contourArea(c)
        ellipse_ratio = area / (np.pi * w * h / 4)
        solidity = area / max(cv2.contourArea(cv2.convexHull(c)), 1)
        if 0.88 < ellipse_ratio < 1.08 and solidity > 0.95:
            circles.append({'type': 'circle', 'bbox': (x, y, w, h),
                            'center': (x + w // 2, y + h // 2), 'r': (w + h) // 4})
            _erase_outline(mask, c, labels, stroke_w)
    return circles, mask


def _is_char_sized_owner(binary_labels, stats, contour, limit):
    """구멍을 둘러싼 잉크 덩어리가 글자 크기인지(=글자 속 구멍인지)"""
    x, y, w, h = cv2.boundingRect(contour)
    # 구멍 바로 왼쪽 바깥 픽셀의 라벨 = 구멍을 둘러싼 덩어리
    px, py = max(x - 1, 0), y + h // 2
    for dx in range(0, 4):
        lab = binary_labels[py, max(px - dx, 0)]
        if lab > 0:
            _, _, ow, oh, _ = stats[lab]
            return max(ow, oh) < limit
    return True


def detect_polygons(binary, char_h, stroke_w):
    """네모/삼각형/다각형 분류 (구멍 윤곽 기준). 글자 크기 덩어리의 구멍('D', 'A', 'B' 속)은 제외.
    지우기는 하지 않는다: 네모의 변은 대부분 선 단계에서 지워지고,
    짧아서 남은 변은 '그래픽에 붙은 잔재' 규칙(symbols.py)으로 정리된다."""
    _, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    limit = char_h * 2.5
    min_side = max(stroke_w * 3, char_h * 0.25)

    polys = []
    for c in _holes(binary):
        x, y, w, h = cv2.boundingRect(c)
        if min(w, h) < min_side:
            continue
        if _is_char_sized_owner(labels, stats, c, limit):
            continue
        area = cv2.contourArea(c)
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        n = len(approx)
        solidity = area / max(cv2.contourArea(cv2.convexHull(c)), 1)
        if n == 4 and area / (w * h) > 0.85:
            kind = 'rect'
        elif n == 3 and solidity > 0.9:
            kind = 'triangle'
        elif 5 <= n <= 10 and solidity > 0.9:
            kind = 'polygon'
        else:
            continue
        polys.append({'type': kind, 'bbox': (x, y, w, h),
                      'points': approx.reshape(-1, 2).tolist()})
    return polys
