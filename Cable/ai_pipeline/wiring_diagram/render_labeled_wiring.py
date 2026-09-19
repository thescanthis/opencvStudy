"""
범용 렌더러: "소스 커넥터 세로열 -> 여러 목적지 커넥터, 각 핀마다 라벨과
함께 연결선"이라는 뼈대를 공유하는 배선도 스타일들을 하나의 데이터
구조로 표현해서 그린다.

이 뼈대를 공유하는 것으로 확인된 도면:
  - 60309822: 라벨 = "K-16" 같은 배선 코드
  - A60024600: 라벨 = "AWG 16" (+ 핀 옆에 "24V WHITE" 같은 보조 텍스트 컬럼)
두 도면 모두 "핀 옆에 라벨을 적고, 목적지 핀까지 선을 긋는다"는 점은
동일하고, 라벨의 문자열 내용만 다르다 - 렌더러 입장에서는 같은 구조.

데이터 모델
-----------
SourceConnector: 이름, 핀 목록(위에서 아래 순서), 각 핀 옆에 붙는 보조
  텍스트 컬럼(옵션, A60024600 의 "24V WHITE" 같은 것).
Destination: 이름, 핀 목록, 배치 방식(가로 상단 그룹 / 세로 우측 그룹).
Connection: (source_pin, label, dest_name, dest_pin).
ShieldGroup: 실드로 묶이는 핀 구간(옵션).

사용 흐름: DiagramSpec 하나를 채우고 render() 를 호출하면 PDF(벡터)를
만들고 그 PDF를 rasterize 해서 PNG 도 함께 만든다(둘 다 100% 동일 내용).
"""
import os
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse

from wiring_layout import Column, place_after

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(__file__)
BLACK = "black"
GRAY = "0.5"
RED = "#c80000"


@dataclass
class DestGroup:
    """목적지 커넥터 하나(세로 핀 목록). y 는 소스 핀 y를 그대로 따라가도록
    dest_pin -> source_pin 매핑을 주거나, 독립적인 세로열로 그릴 수 있다."""
    name: str
    pins: list
    label: str = ""          # 박스 아래/옆에 적을 설명(예: "20-27P")


@dataclass
class DiagramSpec:
    title: str
    source_name: str
    source_pins: list                       # 위->아래 순서
    source_side_cols: list = field(default_factory=list)  # 핀 옆 보조 텍스트 컬럼들(선택)
    destinations: dict = field(default_factory=dict)      # name -> DestGroup
    connections: list = field(default_factory=list)       # (src_pin, label, dest_name, dest_pin)
    shield_groups_source: list = field(default_factory=list)  # [(pin_top, pin_bottom), ...] source 쪽 실드 그룹(핀 이름 기준)


def build_figure(spec: DiagramSpec, row_h=42, box_w=45, box_h=32):
    src_pins = spec.source_pins
    n = len(src_pins)
    # 목적지가 여러 개면 세로로 쌓이므로, 전체 핀 개수(소스 vs 목적지 합)
    # 중 더 큰 쪽 기준으로 캔버스 높이를 계산해야 마지막 박스가 안 잘린다.
    dest_pin_total = sum(len(g.pins) for g in spec.destinations.values())
    n_for_height = max(n, dest_pin_total + len(spec.destinations))

    W = 2000
    H = 140 + n_for_height * row_h + 250
    fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=150)
    ax.set_xlim(0, W)
    ax.set_ylim(-H, 0)
    ax.axis("off")
    ax.text(20, -20, spec.title, fontsize=9)

    def line(x1, y1, x2, y2, color=GRAY, lw=0.7):
        ax.plot([x1, x2], [-y1, -y2], color=color, linewidth=lw)

    def label(x, y, text, color=RED, fs=6.5, ha="left"):
        ax.text(x, -y, text, color=color, fontsize=fs, va="center", ha=ha)

    def box(x, y, w, h, text, fs=8):
        ax.add_patch(Rectangle((x, -(y + h)), w, h, fill=False, edgecolor=BLACK, linewidth=0.8))
        ax.text(x + w / 2, -(y + h / 2), text, ha="center", va="center", fontsize=fs)

    # ---- 소스 커넥터 세로열 ----
    src_x0 = 70
    src_y0 = 140
    src = Column(spec.source_name, src_pins, x=src_x0 + box_w, y0=src_y0, row_h=row_h)
    for pin in src_pins:
        x, y = src.pos(pin)
        box(src_x0, y - box_h / 2, box_w, box_h, pin, fs=7.5)
    ax.text(src_x0, -(src_y0 - box_h / 2 - 15), spec.source_name, fontsize=9)

    # 보조 텍스트 컬럼(24V/WHITE 등)과 실드 체인링크는 현재 비활성화 - 라인이
    # 지저분해 보인다는 피드백으로 뺐다. 연결선은 소스 박스 바로 옆에서 시작.
    lead_x = src_x0 + box_w + 10

    # ---- 목적지 커넥터들 배치: 각 목적지 핀의 y는 연결된 소스 핀과 동일하게
    # 맞춰서(수평선이 되도록) 자동 계산한다 ----
    # 목적지가 여러 개면 소스 핀과 같은 y를 우선 시도하되, 이전 목적지와
    # y범위가 겹치면 그 아래로 밀어서 배치한다(같은 x에 두 목적지가
    # 겹쳐 그려지는 걸 방지) - A60024600 처럼 하나의 P1 이 P2, P3 둘 다로
    # 나가는 경우에 필요.
    dest_x = 1650
    dest_cols = {}
    prev_bottom = None
    for dname, dgroup in spec.destinations.items():
        col = Column(dname, dgroup.pins, x=dest_x, y0=src_y0, row_h=row_h)
        for src_pin, lbl, d, dpin in spec.connections:
            if d == dname and dpin in col.pins:
                col.set_y(dpin, src.y(src_pin))
        lo, hi = col.y_range()
        label_gap = row_h + 35  # 목적지 이름 라벨이 박스 위에 들어갈 여유
        if prev_bottom is not None and lo < prev_bottom + label_gap:
            shift = (prev_bottom + label_gap) - lo
            for pin in col.pins:
                col.set_y(pin, col.y(pin) + shift)
        dest_cols[dname] = col
        prev_bottom = col.y_range()[1]

    for dname, col in dest_cols.items():
        dgroup = spec.destinations[dname]
        for pin in col.pins:
            x, y = col.pos(pin)
            box(dest_x, y - box_h / 2, box_w, box_h, pin, fs=7.5)
        lo, hi = col.y_range()
        ax.text(dest_x, -(lo - 25), f"{dname} {dgroup.label}".strip(), fontsize=9)

    # ---- 연결선 그리기: 소스 리드선 끝 -> 목적지 핀(수평 우선, 다르면 L자) ----
    for src_pin, lbl, dname, dpin in spec.connections:
        x1, y1 = src.pos(src_pin)
        line(x1, y1, lead_x, y1)
        label(lead_x + 10, y1 - 8, lbl, fs=6)
        if dname not in dest_cols:
            continue
        x2, y2 = dest_cols[dname].pos(dpin)
        if abs(y1 - y2) < 1:
            line(lead_x + 10, y1, x2, y2)
        else:
            bend_x = x2 - 60
            line(lead_x + 10, y1, bend_x, y1)
            line(bend_x, y1, bend_x, y2)
            line(bend_x, y2, x2, y2)

    return fig


def render(spec: DiagramSpec, out_stem, zoom=2):
    fig = build_figure(spec)
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
