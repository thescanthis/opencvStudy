"""
render_clean_wiring.py 와 동일한 좌표/연결관계 데이터로, 래스터(PNG)
대신 진짜 벡터 PDF를 생성한다 - matplotlib 은 vector backend(PDF)를
기본 지원하므로 선/텍스트/사각형이 픽셀이 아니라 도형 그대로 저장된다.
(확대해도 계단현상 없이 깨끗함, 인쇄/편집에도 적합.)

좌표계 차이만 주의: OpenCV 는 y가 아래로 증가하지만 matplotlib 기본
axes 는 y가 위로 증가하므로, 모든 y좌표를 -y 로 변환해서 그대로
뒤집어 그린다(원본 스크립트의 pixel-y 값을 그대로 가져다 쓰되 부호만
반전 - 상대적 배치는 동일하게 유지됨).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = os.path.dirname(__file__)

P1_PINS = ["S", "Q", "P", "N", "M", "L", "K", "J", "H", "G",
           "F", "E", "C", "D", "B", "A", "T", "U", "R", "V"]
P1_WIRES = ["K-16", "K-15", "K-14", "K-13", "K-12", "K-11", "K-10", "K-9",
            "K-8", "K-7", "K-6", "K-5", "K-4", "K-3", "K-2", "K-1",
            "K-17", "K-18", "K-19", "K-20"]

P2_TOP_PINS = ["N", "I", "G", "J", "H", "F", "E", "D", "C", "B", "A", "K", "L", "M"]
P2_WIRE_TO_PIN = {
    "K-16": "J", "K-15": "H", "K-14": "F", "K-13": "E", "K-12": "D",
    "K-11": "C", "K-10": "B", "K-9": "A", "K-21": "M", "K-22": "L", "K-23": "K",
}

P3_PINS = ["H", "F", "E", "D", "G", "A", "C", "B"]
P3_WIRE_TO_PIN = {
    "K-21": "H", "K-22": "F", "K-23": "E", "K-8": "D",
    "K-7": "G", "K-6": "A", "K-5": "C", "K-4": "B",
}
P4_PINS = ["A", "B", "C"]
P4_WIRE_TO_PIN = {"K-18": "A", "K-19": "B", "K-20": "C"}
P5_PINS = ["A", "C", "B"]
P5_WIRE_TO_PIN = {"K-2": "A", "K-1": "C", "K-17": "B"}  # K-2 = GND

GRAY = "0.55"
RED = "#c80000"
BLACK = "black"


def box(ax, x, y, w, h, text, fontsize=8):
    """(x,y) = 좌상단, OpenCV 와 동일한 좌표계(y 아래로 증가)를 그대로 받아
    matplotlib 좌표(y 위로 증가)로 변환해서 그린다."""
    rect = Rectangle((x, -(y + h)), w, h, fill=False, edgecolor=BLACK, linewidth=0.8)
    ax.add_patch(rect)
    ax.text(x + w / 2, -(y + h / 2), text, ha="center", va="center", fontsize=fontsize)


def line(ax, x1, y1, x2, y2, color=GRAY, lw=0.7):
    ax.plot([x1, x2], [-y1, -y2], color=color, linewidth=lw)


def label(ax, x, y, text, color=RED, fontsize=7, ha="left"):
    ax.text(x, -y, text, color=color, fontsize=fontsize, ha=ha, va="center")


def render(out_path="out/60309822_clean_wiring.pdf"):
    W, H = 2000, 1300
    fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
    ax.set_xlim(0, W)
    ax.set_ylim(-H, 0)
    ax.axis("off")

    ax.text(30, -20, "60309822 - Wiring Diagram (clean re-render, verified against source)",
            fontsize=10, va="center")

    # ---- P1 좌측 세로열 ----
    p1_x = 60
    p1_y0 = 260
    row_h = 42
    box_w, box_h = 40, 30
    p1_positions = {}
    for i, pin in enumerate(P1_PINS):
        y = p1_y0 + i * row_h
        box(ax, p1_x, y, box_w, box_h, pin)
        p1_positions[pin] = (p1_x + box_w, y + box_h / 2)

    # ---- P2 상단 가로열 ----
    p2_x0 = 700
    p2_y = 40
    p2_cell_w = 44
    p2_lead_len = 55
    p2_used_pins = set(P2_WIRE_TO_PIN.values())
    p2_positions = {}
    p2_lead_positions = {}
    for i, pin in enumerate(P2_TOP_PINS):
        x = p2_x0 + i * p2_cell_w
        box(ax, x, p2_y, p2_cell_w - 4, 40, pin)
        cx = x + (p2_cell_w - 4) / 2
        p2_positions[pin] = (cx, p2_y + 40)
        p2_lead_positions[pin] = (cx, p2_y + 40 + p2_lead_len)
        if pin in p2_used_pins:
            line(ax, *p2_positions[pin], *p2_lead_positions[pin])

    # ---- P3 우측열 ----
    p3_x = 1750
    p3_positions = {}
    same_row_map = {"D": "H", "G": "G", "A": "F", "C": "E", "B": "C"}
    for pin in ("D", "G", "A", "C", "B"):
        p1_pin = same_row_map[pin]
        _, y1 = p1_positions[p1_pin]
        p3_positions[pin] = (p3_x, y1)
    top_y0 = p3_positions["D"][1] - 3 * row_h
    for i, pin in enumerate(("H", "F", "E")):
        p3_positions[pin] = (p3_x, top_y0 + i * row_h)
    for pin in P3_PINS:
        _, y = p3_positions[pin]
        box(ax, p3_x, y - box_h / 2, box_w, box_h, pin)

    # ---- P4 ----
    p4_x = 1750
    p4_same_row = {"A": "U", "B": "R", "C": "V"}
    p4_positions = {}
    for pin, p1_pin in p4_same_row.items():
        _, y1 = p1_positions[p1_pin]
        p4_positions[pin] = (p4_x, y1)
        box(ax, p4_x, y1 - box_h / 2, box_w, box_h, pin)

    # ---- P5 ----
    p5_x = 1750
    p5_same_row = {"A": "B", "C": "A", "B": "T"}
    p5_positions = {}
    for pin, p1_pin in p5_same_row.items():
        _, y1 = p1_positions[p1_pin]
        p5_positions[pin] = (p5_x, y1)
        box(ax, p5_x, y1 - box_h / 2, box_w, box_h, pin)
    p4_y0 = p4_positions["A"][1]
    p5_y0 = p5_positions["A"][1]

    label(ax, p1_x, p1_y0 + len(P1_PINS) * row_h + 25, "P1 28-16P", color=BLACK, fontsize=8)
    label(ax, p2_x0, p2_y - 15, "P2 20-27P", color=BLACK, fontsize=8)
    label(ax, p3_x + 50, p3_positions["H"][1] - 15, "P3 20-7P", color=BLACK, fontsize=8)
    label(ax, p4_x + 50, p4_y0 - 15, "P4 16S-5S", color=BLACK, fontsize=8)
    label(ax, p5_x + 50, p5_y0 - 15, "P5 10SL-3SN", color=BLACK, fontsize=8)

    # ---- 배선 ----
    for pin, wire in zip(P1_PINS, P1_WIRES):
        x1, y1 = p1_positions[pin]
        line(ax, x1, y1, x1 + 40, y1)
        label(ax, x1 + 45, y1, wire, fontsize=6.5)

        if wire in P2_WIRE_TO_PIN:
            dest_pin = P2_WIRE_TO_PIN[wire]
            dx, dy = p2_lead_positions[dest_pin]
            line(ax, x1 + 40, y1, dx, y1)
            line(ax, dx, y1, dx, dy)
        else:
            for dests, positions in ((P3_WIRE_TO_PIN, p3_positions),
                                      (P4_WIRE_TO_PIN, p4_positions),
                                      (P5_WIRE_TO_PIN, p5_positions)):
                if wire in dests:
                    dx, dy = positions[dests[wire]]
                    line(ax, x1 + 40, y1, dx, dy)
                    label(ax, dx - 90, dy, wire, fontsize=6.5)
                    break

    # ---- P2 -> P3 (K-21,22,23) ----
    p2_to_p3 = {"K-21": ("M", "H"), "K-22": ("L", "F"), "K-23": ("K", "E")}
    for wire, (p2pin, p3pin) in p2_to_p3.items():
        sx, sy = p2_positions[p2pin]
        dx, dy = p3_positions[p3pin]
        line(ax, sx, sy, sx, dy)
        line(ax, sx, dy, dx, dy)
        label(ax, sx + 5, min(sy, dy) + 15, wire, fontsize=6)

    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    full_path = os.path.join(HERE, out_path)
    fig.savefig(full_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"-> {full_path}")


if __name__ == "__main__":
    render()
