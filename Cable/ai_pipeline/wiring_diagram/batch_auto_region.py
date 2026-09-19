"""PDF_File 폴더 안의 모든 Cable.pdf에 대해 '배선도/회로도/결선도' 영역을
자동 검출해서 크롭 이미지를 저장한다. template_gui.py의 find_diagram_title/
crop_diagram_region을 그대로 import해서 쓴다 - 예전에는 이 스크립트가 같은
로직을 복제해서 갖고 있었는데, GUI만 고치고 여기는 안 고치는 일이 반복돼서
검출 결과가 서로 어긋났다(실측). 로직은 항상 template_gui.py 한 곳에서만
수정한다."""
import os
import glob
import cv2
import numpy as np
import fitz
import easyocr
import sys

sys.path.insert(0, os.path.dirname(__file__) + r"\..")
from template_gui import find_diagram_title, crop_diagram_region

SRC_DIR = r"C:\AIND_Library\Git_PredicTest\x64_Cami_Lib\PSCD1500\PSCD1500\Cable\TestData\PDF_File"
OUT_DIR = os.path.join(os.path.dirname(__file__), "out", "batch_auto_region")


def render_pdf_first_page(pdf_path, zoom=3.0):
    doc = fitz.open(pdf_path)
    page = doc[0]
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img_data, cv2.COLOR_RGBA2BGR)
    elif pix.n == 3:
        img = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
    else:
        img = cv2.cvtColor(img_data, cv2.COLOR_GRAY2BGR)
    doc.close()
    return img


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    pdf_files = sorted(glob.glob(os.path.join(SRC_DIR, "*.pdf")))
    print(f"총 {len(pdf_files)}개 PDF 발견")

    reader = easyocr.Reader(["ko", "en"], gpu=False)

    ok_count = 0
    fail_list = []

    for i, pdf_path in enumerate(pdf_files):
        base = os.path.splitext(os.path.basename(pdf_path))[0]
        print(f"[{i+1}/{len(pdf_files)}] {base} ...", end=" ", flush=True)
        try:
            img = render_pdf_first_page(pdf_path)
            candidates = find_diagram_title(reader, img)
            if not candidates:
                print("실패 (제목 못 찾음)")
                fail_list.append((base, "제목 못 찾음"))
                continue
            best = max(candidates, key=lambda c: c[1])
            x0, y0, x1, y1 = crop_diagram_region(img, best[2])
            crop = img[y0:y1, x0:x1]
            if crop.size == 0:
                print("실패 (빈 크롭)")
                fail_list.append((base, "빈 크롭"))
                continue
            out_path = os.path.join(OUT_DIR, f"{base}.png")
            cv2.imwrite(out_path, crop)
            print(f"OK ('{best[0]}' 신뢰도 {best[1]:.2f}, {crop.shape[1]}x{crop.shape[0]})")
            ok_count += 1
        except Exception as e:
            print(f"예외: {e}")
            fail_list.append((base, f"예외: {e}"))

    print(f"\n=== 완료: {ok_count}/{len(pdf_files)} 성공 ===")
    if fail_list:
        print("실패 목록:")
        for base, reason in fail_list:
            print(f"  - {base}: {reason}")
