def _union(a, b):
    x0, y0 = min(a[0], b[0]), min(a[1], b[1])
    x1 = max(a[0] + a[2], b[0] + b[2])
    y1 = max(a[1] + a[3], b[1] + b[3])
    return (x0, y0, x1 - x0, y1 - y0)


def _merge_until_stable(boxes, should_merge):
    boxes = list(boxes)
    changed = True
    while changed:
        changed = False
        out = []
        while boxes:
            cur = boxes.pop()
            i = 0
            while i < len(boxes):
                if should_merge(cur, boxes[i]):
                    cur = _union(cur, boxes.pop(i))
                    changed = True
                    i = 0
                else:
                    i += 1
            out.append(cur)
        boxes = out
    return boxes


def merge_glyphs(boxes, char_h):
    """한 글자가 여러 조각인 경우(한글 자모 'ㅎ'+'ㅗ', 'i'의 점 등)를 한 글자로 합친다.
    가로로 많이 겹치고 세로 간격이 아주 좁으며, 합쳐도 글자 높이를 크게 넘지 않을 때만."""
    max_gap = max(2, char_h * 0.2)
    max_h = char_h * 1.5

    def should_merge(a, b):
        ov = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
        if ov < 0.5 * min(a[2], b[2]):
            return False
        gap = max(a[1], b[1]) - min(a[1] + a[3], b[1] + b[3])
        if gap > max_gap:
            return False
        return max(a[1] + a[3], b[1] + b[3]) - min(a[1], b[1]) <= max_h

    return _merge_until_stable(boxes, should_merge)


def group_text_lines(glyphs, char_h, gap_factor=1.2):
    """같은 줄(세로로 절반 이상 겹침)에서 가로 간격이 글자 높이 × gap_factor 이내인 글자들을 한 박스로.
    1.2배: 선에 닿아 지워진 밑줄('SIG_LIMIT' 의 '_') 자리만큼 벌어져도 한 단어로 유지되도록."""
    max_gap = char_h * gap_factor

    def should_merge(a, b):
        v_ov = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
        if v_ov < 0.5 * min(a[3], b[3]):
            return False
        h_gap = max(a[0], b[0]) - min(a[0] + a[2], b[0] + b[2])
        return h_gap <= max_gap

    return _merge_until_stable(glyphs, should_merge)


def _contains(outer, inner):
    return (outer[0] <= inner[0] and outer[1] <= inner[1]
            and inner[0] + inner[2] <= outer[0] + outer[2]
            and inner[1] + inner[3] <= outer[1] + outer[3])


def extract_text_boxes(text_items, dash_items, char_h):
    """text_items: 글자 후보 박스 목록 (x, y, w, h)
    dash_items: 막대 모양 조각. 'I', '1', '_', '-' 도 막대라서 여기로 분류되므로,
                글자 옆 같은 줄에 있으면 텍스트에 합친다.
                세로 막대는 글자 높이의 0.7배 이상일 때만('I', '1') 후보로 쓴다(짧은 점선 조각 제외).
    대시만으로 이뤄진 그룹은 버린다."""
    dashes = [d for d in dash_items if d[2] > d[3] or d[3] >= char_h * 0.7]
    glyphs = merge_glyphs(list(text_items) + dashes, char_h)
    lines = group_text_lines(glyphs, char_h)
    lines = [b for b in lines if any(_contains(b, t) for t in text_items)]
    return sorted(lines, key=lambda b: (b[1], b[0])), glyphs
