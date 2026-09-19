"""Cable.pdf(또는 지정 pdf)에서 가장 큰 임베드 이미지(=도면 본문 스캔본)를
원본 해상도 그대로 추출해 저장한다."""
import os
import sys
import fitz


def extract_main(pdf_path, out_path):
    doc = fitz.open(pdf_path)
    page = doc[0]
    imgs = page.get_images(full=True)
    if not imgs:
        raise RuntimeError(f"no embedded image in {pdf_path}")
    best = max(imgs, key=lambda im: im[2] * im[3])  # width*height 최대
    xref = best[0]
    pix = fitz.Pixmap(doc, xref)
    if pix.n - pix.alpha >= 4:
        pix = fitz.Pixmap(fitz.csRGB, pix)
    pix.save(out_path)
    print(f"{pdf_path} -> {out_path} ({pix.width}x{pix.height})")
    return pix.width, pix.height


if __name__ == "__main__":
    extract_main(sys.argv[1], sys.argv[2])
