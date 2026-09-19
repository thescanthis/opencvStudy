import cv2
import numpy as np


def load_image(path):
    """한글 경로도 읽을 수 있도록 imdecode 사용"""
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)


def binarize(image):
    """벡터 스케치 탭과 동일한 이진화.
    채널 최대값으로 흑백 변환(빨간 주석 등 컬러 표시 제거) → CLAHE → Otsu.
    반환: 잉크=255, 배경=0"""
    gray = np.max(image, axis=2).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    thresh_val, binary = cv2.threshold(clahe.apply(gray), 0, 255,
                                       cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return binary, thresh_val


def estimate_stroke_width(binary, max_w=20):
    """가로/세로 런(연속 잉크 구간) 길이의 최빈값 = 대부분의 획 두께.
    가로선은 세로 런이, 세로선과 글자 획은 가로 런이 두께를 알려준다."""
    runs = []
    for arr in (binary, binary.T):
        b = (arr > 0).astype(np.int8)
        d = np.diff(b, axis=1, prepend=0, append=0)
        starts = np.nonzero(d == 1)[1]
        ends = np.nonzero(d == -1)[1]
        runs.append(ends - starts)
    runs = np.concatenate(runs)
    runs = runs[(runs > 0) & (runs <= max_w)]
    if len(runs) == 0:
        return 1
    return int(np.bincount(runs).argmax())


def estimate_char_height(binary, stroke_w):
    """글자처럼 생긴 연결요소(작고, 너무 길쭉하지 않고, 속이 꽉 차지 않은 것)의 높이 중앙값.
    선/박스(큼), 점선 조각(속이 꽉 찬 막대), 점(작음)은 제외된다."""
    h_img, w_img = binary.shape
    _, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    w, h, area = stats[1:, 2], stats[1:, 3], stats[1:, 4]
    fill = area / np.maximum(w * h, 1)
    ok = ((h >= stroke_w * 3) & (h <= min(h_img, w_img) * 0.1)
          & (w <= h * 1.5) & (fill < 0.85))
    if not ok.any():
        return 20
    return int(np.median(h[ok]))


def preprocess(image):
    binary, thresh_val = binarize(image)
    stroke_w = estimate_stroke_width(binary)
    char_h = estimate_char_height(binary, stroke_w)
    return {
        'binary': binary,
        'thresh_val': thresh_val,
        'stroke_w': stroke_w,
        'char_h': char_h,
    }
