"""
케이블 PDF 도면에서 "코넥터 핀 후면투영 상세도" 영역을 OCR로 찾아
크롭 이미지로 저장하는 스크립트.

배경: PDF 본문이 통째로 스캔된 래스터 이미지라(다른 조사에서 확인됨) 좌표를
직접 못 읽는다. Hough Circle로 원형 패턴을 찾는 방식은 스캔 노이즈 때문에
실패했다(param2를 150까지 올려도 실제 1개 대비 200개 이상 오탐).

대신 OCR(Tesseract, kor 언어팩)로 "상세도"/"투영" 같은 라벨 텍스트의 위치를
찾고, 그 라벨 바로 위쪽(도면 관례상 그림이 라벨 위에 있음) 영역을 크롭한다.
OCR은 영숫자 규격명(MS3475 등)을 정확히 읽지 못하는 경우가 많으므로, 이
스크립트는 "위치 후보를 찾아 크롭"까지만 하고, 실제 규격 판독은 크롭 이미지를
사람(또는 Claude vision)이 보고 connector_specs.py에 채워 넣는 별도 단계로
남긴다.
"""
import os
import glob
import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# 도면에 흔히 쓰이는 라벨 키워드. 하나라도 매치되면 후보로 채택한다.
LABEL_KEYWORDS = ["상세도", "투영", "후면", "코넥터", "커넥터"]


def find_label_positions(img: Image.Image, zoom_for_ocr: float = 1.0):
    """이미지에서 라벨 키워드가 포함된 텍스트 박스 위치를 찾는다.

    반환: [(text, left, top, width, height, conf), ...]
    """
    data = pytesseract.image_to_data(img, lang="kor", output_type=pytesseract.Output.DICT)
    matches = []
    n = len(data["text"])
    for i in range(n):
        t = data["text"][i].strip()
        if not t:
            continue
        if any(kw in t for kw in LABEL_KEYWORDS):
            matches.append((t, data["left"][i], data["top"][i], data["width"][i], data["height"][i], data["conf"][i]))
    return matches



# OCR 정확도가 렌더링 해상도(zoom)에 크게 좌우됨을 확인했다 - 목표는
# 도면 텍스트가 픽셀상 충분히 커지는 것이므로, 배율을 고정값으로 두지 않고
# "렌더링 결과의 긴 변이 이 픽셀 수에 가깝도록" 동적으로 계산한다. 페이지가
# 이미 큰 PDF(예: 4680x3300pt)에 배율까지 곱하면 MuPDF의 픽셀맵 크기 한계
# (Overly large image)를 넘을 수 있으므로, 각 후보에 상한(max_zoom)도 둔다.
_TARGET_LONG_SIDE_CANDIDATES = (6800, 9200, 5000)
_MAX_PIXELS = 300_000_000  # MuPDF 픽셀맵 한계보다 여유 있게 낮춘 상한


def _zoom_candidates_for_page(page) -> list:
    rect = page.rect
    long_side = max(rect.width, rect.height)
    if long_side <= 0:
        return [3.0]

    zooms = []
    for target in _TARGET_LONG_SIDE_CANDIDATES:
        z = target / long_side
        total_pixels = (rect.width * z) * (rect.height * z)
        if total_pixels > _MAX_PIXELS:
            z = (_MAX_PIXELS / (rect.width * rect.height)) ** 0.5
        zooms.append(max(0.5, z))
    return zooms


def extract_connector_crops(pdf_path: str, out_dir: str) -> list:
    """PDF에서 커넥터 핀 배치도 라벨 근처 영역을 찾아 크롭 PNG로 저장한다."""
    import fitz

    os.makedirs(out_dir, exist_ok=True)
    saved = []

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"  [Skip] Cannot open {pdf_path}: {e}")
        return saved

    for page_idx, page in enumerate(doc):
        matches = []
        img = None
        zoom_candidates = _zoom_candidates_for_page(page)

        for zoom in zoom_candidates:
            try:
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat)
                img = Image.frombytes("RGB" if pix.n >= 3 else "L", (pix.width, pix.height), pix.samples)
                if img.mode != "RGB":
                    img = img.convert("RGB")

                matches = find_label_positions(img)
                if matches:
                    break
            except Exception as e:
                print(f"  [Warning] zoom={zoom:.2f} failed on {pdf_path} page {page_idx}: {e}")
                continue

        if not matches:
            continue

        # 같은 라벨 그림 주변에서 여러 키워드가 각각 매치되므로, 위치가
        # 서로 가까운 것끼리는 하나로 묶어 중복 크롭을 피한다.
        merged_centers = []
        for (t, l, top, w, h, conf) in matches:
            cx, cy = l + w / 2, top + h / 2
            is_dup = False
            for (mx, my) in merged_centers:
                if abs(cx - mx) < 300 * zoom and abs(cy - my) < 150 * zoom:
                    is_dup = True
                    break
            if not is_dup:
                merged_centers.append((cx, cy))

        for i, (cx, cy) in enumerate(merged_centers):
            # 라벨은 보통 그림 아래에 있으므로, 라벨 기준 위쪽으로 넉넉히 크롭
            crop_w, crop_h = int(500 * zoom), int(700 * zoom)
            x0 = max(0, int(cx - crop_w / 2))
            y0 = max(0, int(cy - crop_h * 0.9))
            x1 = min(img.width, x0 + crop_w)
            y1 = min(img.height, int(cy + crop_h * 0.15))
            crop = img.crop((x0, y0, x1, y1))

            cable_id = os.path.basename(os.path.dirname(pdf_path))
            out_name = f"{cable_id}__p{page_idx}_label{i}.png"
            out_path = os.path.join(out_dir, out_name)
            crop.save(out_path)
            saved.append(out_path)

    return saved


def scan_dataset(target_dirs: list, out_dir: str):
    total_pdfs = 0
    total_crops = 0
    no_match = []
    for target_dir in target_dirs:
        if not os.path.isdir(target_dir):
            print(f"[Warning] Not found: {target_dir}")
            continue
        for cable_folder in sorted(os.listdir(target_dir)):
            folder_path = os.path.join(target_dir, cable_folder)
            if not os.path.isdir(folder_path):
                continue
            pdf_candidates = glob.glob(os.path.join(folder_path, "Cable.pdf")) or \
                glob.glob(os.path.join(folder_path, "*.pdf"))
            if not pdf_candidates:
                continue
            pdf_path = pdf_candidates[0]
            total_pdfs += 1
            try:
                crops = extract_connector_crops(pdf_path, out_dir)
            except Exception as e:
                print(f"[{cable_folder}] [Error] {e}")
                no_match.append(cable_folder)
                continue
            total_crops += len(crops)
            if crops:
                print(f"[{cable_folder}] {len(crops)} candidate(s)")
            else:
                no_match.append(cable_folder)
                print(f"[{cable_folder}] no label match")

    print(f"\nDone. Scanned {total_pdfs} PDFs, saved {total_crops} crops to {out_dir}")
    if no_match:
        print(f"No match ({len(no_match)}): {no_match}")


if __name__ == "__main__":
    target_dirs = [
        r"C:\AIND_Library\Git_PredicTest\x64_Cami_Lib\PSCD1500\PSCD1500\Cable\TestData",
        r"C:\AIND_Library\Git_PredicTest\x64_Cami_Lib\PSCD1500\PSCD1500\Cable\TestData2",
    ]
    out_dir = r"C:\AIND_Library\Git_PredicTest\x64_Cami_Lib\PSCD1500\PSCD1500\Cable\ai_pipeline\test_output\connector_crops"
    scan_dataset(target_dirs, out_dir)
