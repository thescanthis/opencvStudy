"""
도면 이미지에 픽셀 좌표 그리드를 오버레이해서, Claude가 박스 좌표를 정확히
읽을 수 있게 한다. 100px 마다 굵은 선 + 좌표 숫자, 50px 마다 가는 선.
"""
import os
import sys
import cv2

IMG_DIR = os.path.join(os.path.dirname(__file__), "images", "all")
OUT_DIR = os.path.join(os.path.dirname(__file__), "preview")


def grid(name, x0=0, y0=0, x1=None, y1=None, scale=1.0, step=100):
    img = cv2.imread(os.path.join(IMG_DIR, f"{name}.png"))
    H, W = img.shape[:2]
    x1 = x1 or W
    y1 = y1 or H
    crop = img[y0:y1, x0:x1].copy()
    if scale != 1.0:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    ch, cw = crop.shape[:2]
    # 그리드: 원본좌표 기준 step 마다
    gx = ((x0 // step) + 1) * step
    while gx < x1:
        px = int((gx - x0) * scale)
        cv2.line(crop, (px, 0), (px, ch), (0, 0, 255), 1)
        cv2.putText(crop, str(gx), (px + 2, 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 0, 255), 1, cv2.LINE_AA)
        gx += step
    gy = ((y0 // step) + 1) * step
    while gy < y1:
        py = int((gy - y0) * scale)
        cv2.line(crop, (0, py), (cw, py), (0, 0, 255), 1)
        cv2.putText(crop, str(gy), (2, py - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 0, 255), 1, cv2.LINE_AA)
        gy += step
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"_grid_{name}.png")
    cv2.imwrite(out, crop)
    print(f"-> {out}  crop=({x0},{y0})-({x1},{y1}) scale={scale}")


if __name__ == "__main__":
    a = sys.argv[1:]
    name = a[0]
    kw = dict(x0=0, y0=0, x1=None, y1=None, scale=1.0, step=100)
    for p in a[1:]:
        k, v = p.split("=")
        kw[k] = float(v) if k == "scale" else int(v)
    grid(name, **kw)
