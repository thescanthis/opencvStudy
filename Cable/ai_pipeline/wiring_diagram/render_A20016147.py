"""
A20016147 도면의 "회로도"를 원본 레이아웃 그대로 재구성한다.
좌표는 wiring_layout.Column 엔진으로 자동 계산한다(손으로 숫자를
박아넣지 않음 - 항목이 겹치면 자동으로 최소 간격을 보장해 재배치).

원본 구조(고해상도 원본 이미지 직접 확대 대조로 확정):
  - 모든 커넥터 박스 크기는 "핀 개수에 비례"한다: P1(5핀), P4(5핀)는
    리드선 전체 높이를 감싸는 큰 직사각형이고, P3(2핀)는 중간 크기,
    P2(핀 구분 없는 28V 2가닥)만 작은 정사각형이다. (처음에는 P2/P3/P4를
    전부 같은 작은 정사각형으로 잘못 그렸었음 - 원본 재확인으로 정정)
  - 각 리드선은 나가면서 체인링크 실드 표시(원 2개)를 지난다: 1열은
    개별 원형, 2열은 그룹별로 하나의 긴 세로 타원(점선)으로 묶인다.
  - 각 커넥터의 리드선들은 맨 아래에서 전부 하나의 가로선으로 합쳐지고
    화살표로 커넥터 박스를 가리킨다(케이블 실드가 커넥터 쉘에 접지).

확정된 연결관계:
  P1-A -> P2 (28V, 28V 두 가닥)
  P1-B -> P4-A (SOL_LIMIT_ELE_RTN)
  P1-C -> P4-B (SIG_LIMIT_ELE)
  P1-D -> P4-C (SIG_LIMIT_ELE_RTN)
  P1-E -> P4-E (ELE_BACKUP), 같은 라인이 P3-B 로도 분기
  (SOL_LIMIT_ELE, P1 에 없는 별도 라인) -> P4-D, P3-A 로 분기
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse

from wiring_layout import Column, place_after

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(__file__)
BLACK = "black"
GRAY = "0.4"
RED = "#c80000"

ROW_H = 60


def build_figure():
    W, H = 1750, 850
    fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=150)
    ax.set_xlim(0, W)
    ax.set_ylim(-H, 0)
    ax.axis("off")
    ax.text(20, -20, "A20016147 - Circuit Diagram (clean re-render, verified against source)", fontsize=9)

    def line(x1, y1, x2, y2, color=GRAY, lw=0.8):
        ax.plot([x1, x2], [-y1, -y2], color=color, linewidth=lw)

    def label(x, y, text, color=RED, fs=6.5):
        ax.text(x, -y, text, color=color, fontsize=fs, va="center")

    def chain_link(x, y, r=4.5):
        ax.add_patch(Ellipse((x, -y), r * 1.3, r * 2, fill=False, edgecolor=BLACK, linewidth=0.7))

    def shield_group(x, y_top, y_bottom, r=4.5):
        cy = (y_top + y_bottom) / 2
        height = (y_bottom - y_top) + r * 3
        ax.add_patch(Ellipse((x, -cy), r * 2.3, height, fill=False,
                              edgecolor=BLACK, linewidth=0.8, linestyle=(0, (2, 2))))

    def arrow_left(x, y, size=9):
        ax.annotate("", xy=(x - size, -y), xytext=(x, -y),
                     arrowprops=dict(arrowstyle="-|>", color=BLACK, lw=0.8, mutation_scale=8))

    def small_box(x, y, w, h, text):
        ax.add_patch(Rectangle((x, -(y + h)), w, h, fill=False, edgecolor=BLACK, linewidth=1.0))
        ax.text(x + w / 2, -(y + h / 2), text, ha="center", va="center", fontsize=9)

    def big_box_left(right_edge_x, y_top, y_bottom, w, text):
        """리드선이 왼쪽에서 들어오는 큰 커넥터 박스(P1, P4 스타일).
        right_edge_x: 박스 오른쪽 끝(리드선이 나가는 x좌표는 이 박스 왼쪽).
        y_top/y_bottom: 리드선들의 y 범위. 박스는 그 위아래로 살짝 여유를 둔다."""
        pad = ROW_H / 2
        h = (y_bottom - y_top) + pad * 2
        y0 = y_top - pad
        x0 = right_edge_x - w
        ax.add_patch(Rectangle((x0, -(y0 + h)), w, h, fill=False, edgecolor=BLACK, linewidth=1.0))
        ax.text(x0 + w / 2, -(y0 + h / 2), text, ha="center", va="center", fontsize=9)
        return x0, y0, w, h

    def draw_shield_column(x0, ys, groups=None, ring_off=(60, 95)):
        rx1, rx2 = x0 + ring_off[0], x0 + ring_off[1]
        for y in ys:
            chain_link(rx1, y)
            chain_link(rx2, y)
        if groups:
            for y_top, y_bottom in groups:
                shield_group(rx2, y_top, y_bottom)
        return rx1, rx2

    # ============ P1 (좌측 큰 박스, A~E) - 좌표는 Column 이 자동 계산 ============
    p1_x, p1_y0 = 90, 130
    p1 = Column("P1", ["A", "B", "C", "D", "E"], x=p1_x + 105, y0=p1_y0, row_h=ROW_H)
    p1_w, p1_h = 70, p1.height() + ROW_H  # 마지막 핀 아래 여백 포함
    ax.add_patch(Rectangle((p1_x, -(p1_y0 + p1_h)), p1_w, p1_h, fill=False, edgecolor=BLACK, linewidth=1.0))
    ax.text(p1_x + p1_w / 2, -(p1_y0 + p1_h / 2), "P1", ha="center", va="center", fontsize=9)
    for pin in p1.pins:
        x, y = p1.pos(pin)
        ax.text(p1_x + p1_w + 15, -y, pin, fontsize=9, va="center")

    ax.annotate("", xy=(p1_x + 10, -(p1_y0 + p1_h + 15)), xytext=(p1_x + 30, -(p1_y0 + p1_h)),
                 arrowprops=dict(arrowstyle="-|>", color=BLACK, lw=0.8, mutation_scale=8))

    right_x = 1550
    box_w, box_h = 55, 45

    # ============ P4: 핀 순서(A~E) 고정, y는 대응하는 P1 신호에 맞춰 자동 정렬 ============
    p4 = Column("P4", ["A", "B", "C", "D", "E"], x=right_x, y0=p1.y("B"), row_h=ROW_H)
    p4.set_y("A", p1.y("B"))
    p4.set_y("B", p1.y("C"))
    p4.set_y("C", p1.y("D"))
    p4.set_y("E", p1.y("E"))
    # D(SOL_LIMIT_ELE)는 C,E 사이에 넣되 겹치면 Column이 자동으로 벌려줌
    p4.set_y("D", (p4.y("C") + p4.y("E")) / 2)
    p4_box_x0 = right_x + 110  # 박스 왼쪽 끝(= 리드선이 도달하는 x, right_x 와 일치)
    for pin in p4.pins:
        x, y = p4.pos(pin)
        ax.text(right_x - 25, -y, pin, fontsize=9, va="center", ha="right")
    p4_lo, p4_hi = p4.y_range()
    big_box_left(p4_box_x0 + 90, p4_lo, p4_hi, 90, "P4")

    # ============ P3: A,B - P1 아래쪽에 자동 배치 ============
    p3_y0 = place_after(p1, gap=120)
    p3 = Column("P3", ["A", "B"], x=right_x, y0=p3_y0, row_h=ROW_H)
    for pin in p3.pins:
        x, y = p3.pos(pin)
        ax.text(right_x - 25, -y, pin, fontsize=9, va="center", ha="right")
    p3_lo, p3_hi = p3.y_range()
    big_box_left(right_x + 200, p3_lo, p3_hi, 90, "P3")

    # ============ P2: 28V x2, P1-A 근처에 배치 ============
    p2_y0 = p1.y("A") - 30
    p2 = Column("P2", ["1", "2"], x=right_x, y0=p2_y0, row_h=16)
    p2_lo, p2_hi = p2.y_range()
    small_box(right_x + 55, (p2_lo + p2_hi) / 2 - box_h / 2, box_w, box_h, "P2")

    # ============ P1-A -> P2 (28V x2) ============
    ax_, ay = p1.pos("A")
    rx1, rx2 = draw_shield_column(ax_, [ay - 5, ay + 5], groups=[(ay - 5, ay + 5)])
    line(ax_, ay - 5, rx1 - 7, ay - 5)
    line(ax_, ay + 5, rx1 - 7, ay + 5)
    line(rx2 + 7, ay - 5, right_x, p2.y("1"))
    line(rx2 + 7, ay + 5, right_x, p2.y("2"))
    label(rx2 + 15, ay - 9, "28V")
    label(rx2 + 15, ay + 15, "28V")

    # ============ P1-B,C,D,E -> P4 (수평, 자동 정렬 덕분에 모두 일직선) ============
    sig_names = {"B": "SOL_LIMIT_ELE_RTN", "C": "SIG_LIMIT_ELE", "D": "SIG_LIMIT_ELE_RTN", "E": "ELE_BACKUP"}
    p1_to_p4 = {"B": "A", "C": "B", "D": "C", "E": "E"}
    bx0 = p1.x
    b_ys = [p1.y(p) for p in ("B", "C", "D", "E")]
    rx1b, rx2b = draw_shield_column(bx0, b_ys, groups=[
        (p1.y("B") - 15, p1.y("B") + 15),
        (p1.y("C") - 5, p1.y("E") + 5),
    ])
    for p1pin, p4pin in p1_to_p4.items():
        x1, y1 = p1.pos(p1pin)
        line(x1, y1, rx1b - 7, y1)
        line(rx2b + 7, y1, right_x, p4.y(p4pin))
        label(rx2b + 15, y1 - 8, sig_names[p1pin])

    # P4 쪽 체인(A~E 전체 한 그룹) - 체인링크는 right_x 왼쪽, 그 뒤 박스 왼쪽 끝까지 연장
    p4_ys = [p4.y(p) for p in p4.pins]
    p4rx1, p4rx2 = draw_shield_column(right_x, p4_ys, groups=[(min(p4_ys), max(p4_ys))], ring_off=(-95, -60))
    for pin in p4.pins:
        x, y = p4.pos(pin)
        line(p4rx2 + 7, y, p4_box_x0, y)
    bottom_y4 = max(p4_ys) + 25
    line(p4rx1, min(p4_ys), p4rx1, bottom_y4)
    line(p4rx1, bottom_y4, p4_box_x0 + 20, bottom_y4)
    arrow_left(p4_box_x0 + 20, bottom_y4)

    # ============ ELE_BACKUP(P1-E) 분기 -> P3-B ============
    ex, ey = p1.pos("E")
    bx, by = p3.pos("B")
    bend_x1 = 620
    line(ex, ey + 15, bend_x1, ey + 15)
    line(bend_x1, ey + 15, bend_x1, by)

    # ============ SOL_LIMIT_ELE (P1에 없는 별도 라인) -> P4-D, P3-A ============
    sol_x, sol_y = 500, p4.y("D")
    label(sol_x, sol_y - 8, "SOL_LIMIT_ELE")
    line(sol_x + 10, sol_y, sol_x + 90, sol_y)

    # P3 쪽 체인
    p3_box_x0 = right_x + 200
    p3_ys = [p3.y(p) for p in p3.pins]
    p3rx1, p3rx2 = draw_shield_column(right_x, p3_ys, groups=[(min(p3_ys), max(p3_ys))], ring_off=(-95, -60))
    for pin in p3.pins:
        x, y = p3.pos(pin)
        line(p3rx2 + 7, y, p3_box_x0, y)
    bottom_y3 = max(p3_ys) + 25
    line(p3rx1, min(p3_ys), p3rx1, bottom_y3)
    line(p3rx1, bottom_y3, p3_box_x0 + 20, bottom_y3)
    arrow_left(p3_box_x0 + 20, bottom_y3)

    line(bend_x1, by, p3rx1 - 7, by)
    line(sol_x + 90, sol_y, p4rx1 - 7, sol_y)
    line(sol_x + 90, sol_y, sol_x + 90, p3.y("A"))
    line(sol_x + 90, p3.y("A"), p3rx1 - 7, p3.y("A"))

    label(p1_x, p1_y0 + p1_h + 40, "P1: 장전제어기 J1에 연결 (MS3474W14-5PN)", BLACK, 7)
    label(right_x + 130, p2_y0 - 55, "P2: 케이블조립체(A60025769) 회로번호100 (M27144-2)", BLACK, 6.5)
    label(right_x + 130, p2_y0 - 40, "P4: 고저제한스위치에 연결 (MS3111E10-6P)", BLACK, 6.5)
    label(right_x + 130, p3_y0 - 20, "P3: 제한밸브 J1에 연결 (MS3111E12-3P)", BLACK, 6.5)

    return fig


def render_pdf(out_path="out/A20016147_clean_wiring.pdf"):
    fig = build_figure()
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    full_path = os.path.join(HERE, out_path)
    fig.savefig(full_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"-> {full_path}")
    return full_path


def render_png_from_pdf(pdf_path, out_path="out/A20016147_clean_wiring.png", zoom=2):
    import fitz
    doc = fitz.open(pdf_path)
    pix = doc[0].get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    full_out = os.path.join(HERE, out_path)
    pix.save(full_out)
    print(f"-> {full_out}")


if __name__ == "__main__":
    pdf_path = render_pdf()
    render_png_from_pdf(pdf_path)
