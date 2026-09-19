"""
A60024600 케이블 결선도를 원본 레이아웃 그대로 재구성한다.

원본 구조(고해상도 이미지 + 사용자 확인으로 최종 확정):
  - 위쪽 구역: P1(좌, A,C,E,G,J,B,D,F,H,K 10핀) <-> P2(우, 같은 순서 10핀),
    전부 1:1 직결, 라벨은 "AWG 16".
  - 아래쪽 구역: P3(K,H,F,D,B,J,G,E,C,A 10칸)로 P1, P2 양쪽에서 각각
    5개씩 분기된다.
      P1 의 B,D,F,H,K -> P3 의 B,D,F,H,K (그대로, 반전 없음)
      P2 의 B,D,F,H,K -> P3 의 J,G,E,C,A
  - 각 P1/P2/P3 옆에는 원형 커넥터 핀 배치도가 있으나, 이 스크립트는
    결선 관계 자체에 집중하고 원형 다이어그램은 생략한다.
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
GRAY = "0.5"
RED = "#c80000"

P1_PINS = ["A", "C", "E", "G", "J", "B", "D", "F", "H", "K"]
P2_PINS = ["A", "C", "E", "G", "J", "B", "D", "F", "H", "K"]
P3_PINS = ["K", "H", "F", "D", "B", "J", "G", "E", "C", "A"]

RTN_PINS = ["B", "D", "F", "H", "K"]
P1_TO_P3 = {"B": "B", "D": "D", "F": "F", "H": "H", "K": "K"}
P2_TO_P3 = {"B": "J", "D": "G", "F": "E", "H": "C", "K": "A"}


def build_figure():
    row_h = 40
    box_w, box_h = 45, 30

    W = 1700
    H = 1050
    fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=150)
    ax.set_xlim(0, W)
    ax.set_ylim(-H, 0)
    ax.axis("off")
    ax.text(20, -20, "A60024600 - Cable Wiring Diagram (clean re-render, verified against source)", fontsize=9)
    ax.text(600, -55, "케이블 결선도", fontsize=11)

    def line(x1, y1, x2, y2, color=GRAY, lw=0.7):
        ax.plot([x1, x2], [-y1, -y2], color=color, linewidth=lw)

    def label(x, y, text, color=RED, fs=6.5, ha="left"):
        ax.text(x, -y, text, color=color, fontsize=fs, va="center", ha=ha)

    def box(x, y, w, h, text, fs=7.5):
        ax.add_patch(Rectangle((x, -(y + h)), w, h, fill=False, edgecolor=BLACK, linewidth=0.8))
        ax.text(x + w / 2, -(y + h / 2), text, ha="center", va="center", fontsize=fs)

    # ---- P1 (좌상단) ----
    p1_x, y0 = 90, 90
    p1_y = {}
    for i, pin in enumerate(P1_PINS):
        y = y0 + i * row_h
        box(p1_x, y, box_w, box_h, pin)
        p1_y[pin] = y + box_h / 2
    ax.text(p1_x, -(y0 - 15), "P1", fontsize=9)

    # ---- P2 (우상단, P1과 같은 y) ----
    p2_x = 1180
    p2_y = {}
    for i, pin in enumerate(P2_PINS):
        y = y0 + i * row_h
        box(p2_x, y, box_w, box_h, pin)
        p2_y[pin] = y + box_h / 2
    ax.text(p2_x, -(y0 - 15), "P2", fontsize=9)

    # ---- P1 -> P2 연결선 (수평, 1:1). B~K(RTN_PINS)는 P3 로만 분기되고
    # P1<->P2 직결선은 없으므로 그 구간은 긋지 않는다. ----
    for pin in P1_PINS:
        if pin in RTN_PINS:
            continue
        y = p1_y[pin]
        line(p1_x + box_w, y, p2_x, y)
        label((p1_x + box_w + p2_x) / 2 - 60, y - 8, "AWG 16")

    # ---- P3 (하단, 10칸: K,H,F,D,B,J,G,E,C,A) ----
    p3_y0 = y0 + len(P1_PINS) * row_h + 260
    # P3 표 전체 폭의 중심을 P1~P2 사이 중앙에 맞춘다(좌측으로 쏠리지
    # 않도록 자동 계산).
    p3_total_w = len(P3_PINS) * box_w + (len(P3_PINS) - 1) * 5
    center_x = (p1_x + box_w + p2_x) / 2
    p3_x = center_x - p3_total_w / 2
    p3_pos_x = {}
    for i, pin in enumerate(P3_PINS):
        x = p3_x + i * (box_w + 5)
        box(x, p3_y0, box_w, box_h, pin)
        p3_pos_x[pin] = x + box_w / 2
    ax.text(p3_x, -(p3_y0 - 15), "P3", fontsize=9)

    def fan_out(src_positions, mapping, side_gap_start, gap_x=20, mid_y_gap=40):
        """src_positions: {pin: (x,y)} 소스 박스 우측 리드선 시작점.
        mapping: {src_pin: dst_pin}. 각 선은 박스에서 옆으로 나온 뒤,
        "목적지 박스 바로 위(x = dst_x)"까지 먼저 수평 이동을 끝내고,
        그 다음은 오직 그 x 하나로만 수직으로 곧장 내려가 박스에 닿는다.
        여러 선이 같은 y에서 옆으로 미끄러지며 섞이는 구간이 없으므로
        어느 선이 어느 박스로 가는지 한눈에 구분된다."""
        for i, (src_pin, dst_pin) in enumerate(mapping.items()):
            sx, sy = src_positions[src_pin]
            dst_x = p3_pos_x[dst_pin]
            # 각 선마다 다른 높이에서 수평 이동을 끝내야 서로 겹치지 않는다.
            turn_y = sy + side_gap_start + i * mid_y_gap
            line(sx, sy, sx, turn_y)          # 박스에서 조금 아래로
            line(sx, turn_y, dst_x, turn_y)   # 목적지 x까지 수평 이동(여기서만)
            line(dst_x, turn_y, dst_x, p3_y0)  # 이후로는 목적지 x 하나로만 수직 하강
            label(dst_x + 4, turn_y - 8, "AWG 16", fs=6)

    p1_src = {pin: (p1_x + box_w, p1_y[pin]) for pin in RTN_PINS}
    fan_out(p1_src, P1_TO_P3, side_gap_start=20, gap_x=20, )

    p2_src = {pin: (p2_x + box_w, p2_y[pin]) for pin in RTN_PINS}
    fan_out(p2_src, P2_TO_P3, side_gap_start=20, gap_x=20, )

    return fig


def render(out_stem="out/A60024600_clean_wiring", zoom=2):
    fig = build_figure()
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
    render()
