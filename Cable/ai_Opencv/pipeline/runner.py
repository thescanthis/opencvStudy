import cv2

from pipeline.preprocess import preprocess
from pipeline.shapes import detect_circles, detect_polygons
from pipeline.lines import extract_lines
from pipeline.symbols import classify_residual
from pipeline.text import extract_text_boxes


def run_pipeline(image):
    """전체 단계를 순서대로 실행하고 중간 결과를 모두 돌려준다."""
    pre = preprocess(image)
    binary, char_h, stroke_w = pre['binary'], pre['char_h'], pre['stroke_w']

    # 1. 원 (곡선 도형은 선보다 먼저 빼야 원호가 선 조각으로 잡히지 않는다)
    circles, circle_mask = detect_circles(binary, char_h, stroke_w)
    no_circles = cv2.bitwise_and(binary, cv2.bitwise_not(circle_mask))

    # 2. 선 (가로/세로/점선/사선)
    lines = extract_lines(no_circles, char_h, stroke_w, owner_source=binary)

    # 3. 네모/다각형 분류 (지우기는 선 단계와 잔재 규칙이 담당)
    polys = detect_polygons(binary, char_h, stroke_w)
    residual = lines['residual']

    # 4. 잔여 조각 분류 (접속점/화살촉/그래픽 잔재/대시/글자 후보)
    items, symbol_mask, text_mask = classify_residual(binary, residual, char_h, stroke_w)

    # 5. 글자 → 텍스트 박스
    text_boxes, _ = extract_text_boxes(items['text'], items['dash'], char_h)

    return {
        'image': image,
        'binary': binary,
        'thresh_val': pre['thresh_val'],
        'char_h': char_h,
        'stroke_w': stroke_w,
        'circles': circles,
        'circle_mask': circle_mask,
        'lines': lines,
        'polys': polys,
        'residual': residual,
        'items': items,
        'symbol_mask': symbol_mask,
        'text_mask': text_mask,
        'text_boxes': text_boxes,
    }


def summarize(res):
    ln = res['lines']
    kinds = {}
    for p in res['polys']:
        kinds[p['type']] = kinds.get(p['type'], 0) + 1
    return {
        'char_h': res['char_h'],
        'stroke_w': res['stroke_w'],
        'circle': len(res['circles']),
        'h_line': len(ln['h_segs']),
        'v_line': len(ln['v_segs']),
        'dash_line': len(ln['dash_segs']),
        'diag_line': len(ln['diag_segs']),
        'rect': kinds.get('rect', 0),
        'triangle': kinds.get('triangle', 0),
        'polygon': kinds.get('polygon', 0),
        'dot': len(res['items']['dot']),
        'arrowhead': len(res['items']['arrowhead']),
        'remnant': len(res['items']['remnant']),
        'dash': len(res['items']['dash']),
        'text': len(res['text_boxes']),
    }
