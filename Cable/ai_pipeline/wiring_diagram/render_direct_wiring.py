"""
범용 렌더러: 두 커넥터가 "같은 핀 이름끼리 1:1 직결"되는 가장 단순한
배선도 스타일 (50073721 에서 처음 확인된 구조: J14<->J26, A-A, B-B, ...).

사용법:
    from render_direct_wiring import render
    render(
        title="50073721 - Wiring Diagram (J14 <-> J26, direct 1:1)",
        left_name="J14", right_name="J26",
        pins=["A","B","C", ...],
        out_stem="out/50073721_clean_wiring",
    )
PDF(벡터, matplotlib)를 원본으로 만들고 PNG는 그 PDF를 고해상도로
rasterize 한 결과이므로 두 파일의 내용이 항상 100% 일치한다.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(__file__)
BLACK = "black"
GRAY = "0.55"


def build_figure(title, left_name, right_name, pins, row_h=40, box_w=40, box_h=28,
                  left_x=60, right_x=800, margin_top=70):
    n = len(pins)
    W = right_x + box_w + 80
    H = margin_top + n * row_h + 40
    fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=150)
    ax.set_xlim(0, W)
    ax.set_ylim(-H, 0)
    ax.axis("off")
    ax.text(20, -20, title, fontsize=9)

    ax.text(left_x, -(margin_top - 15), left_name, fontsize=9)
    ax.text(right_x, -(margin_top - 15), right_name, fontsize=9)

    for i, pin in enumerate(pins):
        y = margin_top + i * row_h
        ax.add_patch(Rectangle((left_x, -(y + box_h)), box_w, box_h, fill=False, edgecolor=BLACK, linewidth=0.8))
        ax.text(left_x + box_w / 2, -(y + box_h / 2), pin, ha="center", va="center", fontsize=8)
        ax.add_patch(Rectangle((right_x, -(y + box_h)), box_w, box_h, fill=False, edgecolor=BLACK, linewidth=0.8))
        ax.text(right_x + box_w / 2, -(y + box_h / 2), pin, ha="center", va="center", fontsize=8)
        cy = y + box_h / 2
        ax.plot([left_x + box_w, right_x], [-cy, -cy], color=GRAY, linewidth=0.7)

    return fig


def render(title, left_name, right_name, pins, out_stem, zoom=2):
    fig = build_figure(title, left_name, right_name, pins)
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    pdf_path = os.path.join(HERE, out_stem + ".pdf")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"-> {pdf_path}")

    import fitz
    doc = fitz.open(pdf_path)
    pix = doc[0].get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    png_path = os.path.join(HERE, out_stem + ".png")
    pix.save(png_path)
    print(f"-> {png_path}")
    return pdf_path, png_path


if __name__ == "__main__":
    # 예시: 50073721 재생성
    render(
        title="50073721 - Wiring Diagram (J14 <-> J26, direct 1:1)",
        left_name="J14", right_name="J26",
        pins=["A", "B", "C", "D", "E", "F", "G", "H", "J", "K",
              "L", "M", "N", "P", "R", "S", "T", "U"],
        out_stem="out/50073721_clean_wiring",
    )
