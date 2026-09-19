import cv2
import numpy as np


def classify_residual(binary, residual, char_h, stroke_w):
    """선/도형을 지우고 남은 연결요소를 분류한다.

    - remnant: 원본에서 지운 그래픽(선/원/도형)과 같은 덩어리였던 조각.
               글자는 선과 떨어져 있으므로, 그래픽에 붙어 있던 조각은 기호의 일부로 본다.
               그중 속이 찬 작은 원 = 접속점(dot), 속이 찬 삼각형 = 화살촉(arrowhead)
    - dash:    독립된 가늘고 속이 꽉 찬 막대 (점선 조각 중 연결되지 못한 것)
    - text:    나머지 = 글자 후보
    """
    graphics = cv2.bitwise_and(binary, cv2.bitwise_not(residual))
    _, orig_labels = cv2.connectedComponents(binary, connectivity=8)
    touched = np.zeros(orig_labels.max() + 1, bool)
    touched[np.unique(orig_labels[graphics > 0])] = True
    touched[0] = False

    n, labels, stats, _ = cv2.connectedComponentsWithStats(residual, connectivity=8)
    max_thick = stroke_w * 2 + 1
    dot_max = max(char_h * 0.5, stroke_w * 4)

    items = {'dot': [], 'arrowhead': [], 'remnant': [], 'dash': [], 'text': []}
    masks = {k: np.zeros_like(residual) for k in ('symbol', 'text')}

    for i in range(1, n):
        x, y, w, h, area = stats[i]
        crop = labels[y:y + h, x:x + w] == i
        owner_ids = orig_labels[y:y + h, x:x + w][crop]
        attached = touched[owner_ids].any()
        fill = area / max(w * h, 1)
        box = (int(x), int(y), int(w), int(h))

        if attached:
            if max(w, h) <= dot_max and fill > 0.6 and 0.6 < w / h < 1.6:
                kind = 'dot'
            elif max(w, h) <= char_h and 0.35 < fill < 0.65 and _is_triangle(crop):
                kind = 'arrowhead'
            else:
                kind = 'remnant'
        elif fill > 0.85 and min(w, h) <= max_thick and max(w, h) >= 2 * min(w, h):
            kind = 'dash'
        else:
            kind = 'text'

        items[kind].append(box)
        target = masks['text'] if kind == 'text' else masks['symbol']
        target[y:y + h, x:x + w][crop] = 255

    return items, masks['symbol'], masks['text']


def _is_triangle(crop):
    contours, _ = cv2.findContours(crop.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    c = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(c, 0.08 * cv2.arcLength(c, True), True)
    return len(approx) == 3
