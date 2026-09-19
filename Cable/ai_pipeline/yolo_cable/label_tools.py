"""
1차 라벨(박스) 생성/시각화 유틸.

Claude가 도면 이미지를 보고 박스 좌표를 픽셀 단위 dict 로 입력하면,
- YOLO txt (정규화 좌표) 로 저장
- 박스를 그린 미리보기 PNG 생성 (검수용)

YOLO txt 형식: 각 줄 "cls cx cy w h" (모두 이미지 대비 0~1 비율)
"""
import os
import cv2

CLASSES = ["connector", "branch_sleeve", "branch_bare", "elbow", "dimension", "balloon"]
COLORS = {
    "connector": (0, 180, 0),
    "branch_sleeve": (0, 0, 220),
    "branch_bare": (200, 0, 200),
    "elbow": (220, 140, 0),
    "dimension": (150, 150, 150),
    "balloon": (0, 200, 220),
}

IMG_DIR = os.path.join(os.path.dirname(__file__), "images", "all")
LBL_DIR = os.path.join(os.path.dirname(__file__), "labels", "all")
PREVIEW_DIR = os.path.join(os.path.dirname(__file__), "preview")


def save_label(name, boxes):
    """boxes: [(cls_name, x1, y1, x2, y2), ...]  픽셀 좌표(좌상단, 우하단).
    name: 확장자 없는 도면명 (예: 'A20016147')."""
    img = cv2.imread(os.path.join(IMG_DIR, f"{name}.png"))
    if img is None:
        raise FileNotFoundError(f"{name}.png")
    H, W = img.shape[:2]
    os.makedirs(LBL_DIR, exist_ok=True)
    lines = []
    for cls, x1, y1, x2, y2 in boxes:
        ci = CLASSES.index(cls)
        cx = (x1 + x2) / 2 / W
        cy = (y1 + y2) / 2 / H
        w = abs(x2 - x1) / W
        h = abs(y2 - y1) / H
        lines.append(f"{ci} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    with open(os.path.join(LBL_DIR, f"{name}.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # preview
    os.makedirs(PREVIEW_DIR, exist_ok=True)
    vis = img.copy()
    for cls, x1, y1, x2, y2 in boxes:
        c = COLORS[cls]
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), c, 2)
        cv2.putText(vis, cls, (int(x1), int(y1) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(PREVIEW_DIR, f"{name}_boxes.png"), vis)
    print(f"{name}: {len(boxes)} boxes -> label + preview")


def load_label(name):
    """저장된 라벨을 픽셀 박스 리스트로 되읽는다(수정/재저장용)."""
    img = cv2.imread(os.path.join(IMG_DIR, f"{name}.png"))
    H, W = img.shape[:2]
    out = []
    p = os.path.join(LBL_DIR, f"{name}.txt")
    if not os.path.exists(p):
        return out
    for line in open(p):
        parts = line.split()
        if len(parts) != 5:
            continue
        ci, cx, cy, w, h = int(parts[0]), *map(float, parts[1:])
        x1 = (cx - w / 2) * W
        y1 = (cy - h / 2) * H
        x2 = (cx + w / 2) * W
        y2 = (cy + h / 2) * H
        out.append((CLASSES[ci], x1, y1, x2, y2))
    return out
