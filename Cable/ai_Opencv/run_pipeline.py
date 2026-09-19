"""단계별 파이프라인 실행 + 디버그 이미지 저장

사용법:
    python run_pipeline.py                  # PDF_File/Image 전체
    python run_pipeline.py A20016147.png    # 특정 파일만 (PDF_File/Image 기준 경로 또는 절대경로)
결과: output/<이미지이름>/stageN_*.png
"""
import os
import sys
import glob
import cv2

from pipeline.preprocess import load_image
from pipeline.runner import run_pipeline, summarize
from pipeline.visualize import render_view

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "PDF_File", "Image")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

OUTPUTS = [
    ("stage0_binary.png", "binary"),
    ("stage1_graphics.png", "graphics"),
    ("stage1_residual.png", "residual"),
    ("stage2_text_only.png", "text_only"),
    ("stage3_text_boxes.png", "text_boxes"),
]


def save(path, img):
    """한글 경로도 저장되도록 imencode 사용"""
    ok, buf = cv2.imencode(".png", img)
    if ok:
        buf.tofile(path)


def run(path):
    name = os.path.splitext(os.path.basename(path))[0]
    out_dir = os.path.join(OUTPUT_DIR, name)
    os.makedirs(out_dir, exist_ok=True)

    res = run_pipeline(load_image(path))
    for filename, mode in OUTPUTS:
        save(os.path.join(out_dir, filename), render_view(res, mode))

    s = summarize(res)
    print(f"{name}: char_h={s['char_h']} stroke={s['stroke_w']} | circle={s['circle']} "
          f"H={s['h_line']} V={s['v_line']} dashed={s['dash_line']} diag={s['diag_line']} | "
          f"rect={s['rect']} tri={s['triangle']} poly={s['polygon']} | "
          f"dot={s['dot']} arrow={s['arrowhead']} remnant={s['remnant']} dash={s['dash']} | "
          f"text_boxes={s['text']}")


def main():
    if len(sys.argv) > 1:
        paths = [p if os.path.isabs(p) else os.path.join(IMAGE_DIR, p) for p in sys.argv[1:]]
    else:
        paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.png")))
    for p in paths:
        run(p)


if __name__ == "__main__":
    main()
