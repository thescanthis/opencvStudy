import cv2
import numpy as np

# (키, 표시 이름, BGR 색, 그리는 방식)
#   mask: 해당 픽셀을 색칠 / box: 얇은 사각형 / textbox: 굵은 사각형
LAYERS = [
    ('circle',    '원 테두리',        (0, 140, 255),  'mask'),
    ('h_line',    '가로선',           (0, 0, 255),    'mask'),
    ('v_line',    '세로선',           (255, 0, 0),    'mask'),
    ('dash_line', '점선',             (0, 170, 0),    'mask'),
    ('diag_line', '사선(지시선)',     (255, 0, 255),  'mask'),
    ('poly',      '네모/다각형 영역', (200, 200, 0),  'box'),
    ('dot',       '접속점',           (0, 215, 255),  'box'),
    ('arrowhead', '화살촉',           (128, 0, 128),  'box'),
    ('remnant',   '그래픽 잔재',      (42, 42, 165),  'box'),
    ('dash',      '떨어진 대시',      (255, 128, 0),  'box'),
    ('text',      '텍스트 박스',      (200, 60, 0),   'textbox'),
]
GRAPHIC_LAYERS = {'circle', 'h_line', 'v_line', 'dash_line', 'diag_line', 'poly'}
SYMBOL_TEXT_LAYERS = {'dot', 'arrowhead', 'remnant', 'dash', 'text'}


def bgr_to_hex(bgr):
    b, g, r = bgr
    return f"#{r:02x}{g:02x}{b:02x}"


def _faded_binary(binary):
    """원본 잉크를 옅은 회색으로"""
    base = np.full(binary.shape + (3,), 255, np.uint8)
    base[binary > 0] = (210, 210, 210)
    return base


def _faded_image(image):
    return cv2.addWeighted(image, 0.5, np.full_like(image, 255), 0.5, 0)


def _layer_masks(res):
    ln = res['lines']
    return {
        'circle': res['circle_mask'],
        'h_line': ln['h_mask'],
        'v_line': ln['v_mask'],
        'dash_line': ln['dash_mask'],
        'diag_line': ln['diag_mask'],
    }


def _layer_boxes(res):
    boxes = {'poly': [p['bbox'] for p in res['polys']], 'text': res['text_boxes']}
    for k in ('dot', 'arrowhead', 'remnant', 'dash'):
        boxes[k] = res['items'][k]
    return boxes


def render_overlay(res, enabled, base='faded_binary'):
    """enabled: 켜진 레이어 키 집합. base: 'faded_binary' | 'faded_image'"""
    out = _faded_binary(res['binary']) if base == 'faded_binary' else _faded_image(res['image'])
    masks, boxes = _layer_masks(res), _layer_boxes(res)
    for key, _, color, kind in LAYERS:
        if key not in enabled:
            continue
        if kind == 'mask':
            out[masks[key] > 0] = color
        elif kind == 'box':
            for x, y, w, h in boxes[key]:
                cv2.rectangle(out, (x, y), (x + w, y + h), color, 1)
        else:
            for x, y, w, h in boxes[key]:
                cv2.rectangle(out, (x - 2, y - 2), (x + w + 2, y + h + 2), color, 2)
    return out


def render_view(res, mode, enabled=None):
    """대시보드/배치 공용 뷰.
    original | binary | graphics | residual | text_only | overlay"""
    if mode == 'original':
        return res['image']
    if mode == 'binary':
        return cv2.cvtColor(cv2.bitwise_not(res['binary']), cv2.COLOR_GRAY2BGR)
    if mode == 'graphics':
        return render_overlay(res, GRAPHIC_LAYERS)
    if mode == 'residual':
        return cv2.cvtColor(cv2.bitwise_not(res['residual']), cv2.COLOR_GRAY2BGR)
    if mode == 'text_only':
        return cv2.cvtColor(cv2.bitwise_not(res['text_mask']), cv2.COLOR_GRAY2BGR)
    if mode == 'text_boxes':
        return render_overlay(res, SYMBOL_TEXT_LAYERS, base='faded_image')
    if mode == 'overlay':
        return render_overlay(res, enabled if enabled is not None else {k for k, *_ in LAYERS})
    raise ValueError(mode)
