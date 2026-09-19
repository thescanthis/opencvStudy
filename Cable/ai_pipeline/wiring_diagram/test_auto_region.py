"""배선도/회로도/결선도 같은 제목 텍스트를 EasyOCR로 찾아서, 그 위치를
기준으로 다이어그램 영역을 자동 크롭할 수 있는지 검증한다.
도면은 보통 4분할 중 좌측 하단에 이런 제목이 붙는다는 사용자 관찰을 활용."""
import cv2
import easyocr

TITLE_KEYWORDS = ["배선도", "회로도", "결선도", "배선", "회로", "결선"]
TITLE_EXCLUDE = ["공차"]


def find_diagram_title(reader, img):
    h, w = img.shape[:2]
    # 좌측 하단~중단(y: 0.35h~h)만 탐색 (전체 스캔보다 빠르고 오검출도 적음)
    roi_x0, roi_y0 = 0, int(h * 0.35)
    roi = img[roi_y0:h, roi_x0: w // 2]
    results = reader.readtext(roi)
    candidates = []
    for bbox, text, prob in results:
        clean = text.strip()
        if any(kw in clean for kw in TITLE_EXCLUDE):
            continue
        if any(kw in clean for kw in TITLE_KEYWORDS):
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            x0, x1 = min(xs) + roi_x0, max(xs) + roi_x0
            y0, y1 = min(ys) + roi_y0, max(ys) + roi_y0
            candidates.append((clean, prob, (x0, y0, x1, y1)))
    return candidates


def crop_above_title(img, title_bbox, up_ratio=0.16, side_ratio=0.30):
    """제목 텍스트 bbox를 기준으로 그 위쪽 영역을 다이어그램으로 크롭한다.
    up_ratio: 페이지 전체 높이 대비 위로 확장할 비율
    side_ratio: 페이지 전체 폭 대비 좌우로 확장할 비율 (제목 중심 기준)"""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = title_bbox
    cx = (x0 + x1) / 2

    crop_y1 = int(y0)  # 제목 바로 위까지
    crop_y0 = max(0, int(y0 - h * up_ratio))
    crop_x0 = max(0, int(cx - w * side_ratio))
    crop_x1 = min(w, int(cx + w * side_ratio))
    return img[crop_y0:crop_y1, crop_x0:crop_x1]


if __name__ == "__main__":
    img = cv2.imread("out/_a60023104_full.png")
    reader = easyocr.Reader(["ko", "en"], gpu=False)
    candidates = find_diagram_title(reader, img)
    print("찾은 제목 후보:")
    for clean, prob, bbox in candidates:
        print(f"  text={clean!r} prob={prob:.2f} bbox={bbox}")

    if candidates:
        # 가장 확신도 높은 것 사용
        best = max(candidates, key=lambda c: c[1])
        print(f"\n선택: {best[0]!r} @ {best[2]}")
        crop = crop_above_title(img, best[2])
        cv2.imwrite("out/_auto_title_crop.png", crop)
        print("저장: out/_auto_title_crop.png", crop.shape)
    else:
        print("제목을 찾지 못함")
