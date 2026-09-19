"""
A20016147 도면 형상을 2D 이미지로 그린다.

핵심 요구사항(사용자): 몸통과 가지가 겹치는 지점(분기점 슬리브 통과 구간)에서
"한쪽만" 선이 이어지는 게 아니라, 겹치는 두 선 다발 모두 슬리브(둥근 사각
튜브) 안에서 자연스럽게 지나가는 것처럼 보여야 한다. 추가로 갈래선(가지)이
시작되는 부분에도 별도의 슬리브를 하나 더 씌운다(분기점 슬리브와 하나로
합쳐 보이지 않아도 무방 - 따로 보여도 된다).

방법: 각 분기점마다
  1) 분기점 슬리브 사각형(둥근 모서리 rounded-rect, 흰 배경 + 검정 테두리)을
     그린다.
  2) 몸통 겹선(두 줄 파이프 형태)이 슬리브를 관통해서 지나가도록 그린다.
  3) 가지 겹선도 슬리브 중심에서 시작해서 슬리브 밖으로 비스듬히 빠져나가게
     그린다.
  4) 가지가 시작되는 지점(분기점에서 조금 떨어진 곳)에 가지 전용 슬리브를
     하나 더 그린다.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.path import Path as MplPath
import matplotlib.patches as mpatches
from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union

OUT_DIR = r"C:\AIND_Library\Git_PredicTest\x64_Cami_Lib\PSCD1500\PSCD1500\Cable\ai_pipeline\test_output"

LINE_W = 3.0           # 케이블 겉선 두께
PIPE_GAP = 34.0        # 겹선(이중선) 사이 간격 - 원본처럼 두꺼운 파이프 느낌
SLEEVE_H = 58.0        # 슬리브 높이 - 진행축 수직 방향 (PIPE_GAP보다 확실히 커야 감싸짐)

# [테스트] 메인(몸통) 슬리브 폭을 먼저 고정하고, 서브(가지) 슬리브 폭을
# 메인 슬리브 폭의 절반으로 맞춰본다 - 이렇게 하면 분기점 중심 기준으로
# 메인 편측 길이(SLEEVE_W/2)와 가지 슬리브 폭(SLEEVE_W/2)이 항상 같은
# 값이 된다(가지 슬리브 안쪽 끝이 중심에 닿아 있으므로 바깥쪽 끝까지의
# 거리도 SLEEVE_W/2로 메인 편측과 정확히 같아진다).
SLEEVE_W = 270.0
BRANCH_SLEEVE_W = SLEEVE_W / 2.0


def double_line(ax, p0, p1, gap=PIPE_GAP, lw=LINE_W, color="black"):
    """몸통/가지를 이중선(파이프처럼 보이는 두 줄) 컨투어로 그린다."""
    p0 = np.array(p0, dtype=float)
    p1 = np.array(p1, dtype=float)
    d = p1 - p0
    length = np.linalg.norm(d)
    if length < 1e-6:
        return
    dirv = d / length
    normal = np.array([-dirv[1], dirv[0]])
    off = normal * gap / 2
    a0, a1 = p0 + off, p1 + off
    b0, b1 = p0 - off, p1 - off
    ax.plot([a0[0], a1[0]], [a0[1], a1[1]], color=color, lw=lw, solid_capstyle="butt")
    ax.plot([b0[0], b1[0]], [b0[1], b1[1]], color=color, lw=lw, solid_capstyle="butt")


def sleeve(ax, center, axis_deg, w=SLEEVE_W, h=SLEEVE_H, lw=LINE_W, color="black"):
    """슬리브(둥근 모서리 사각 튜브)를 axis_deg 방향(진행축)으로 회전시켜
    그린다. 흰 배경으로 먼저 채워서 그 아래 겹치는 선들을 가린 뒤, 테두리만
    그린다."""
    rect = FancyBboxPatch(
        (-w / 2, -h / 2), w, h,
        boxstyle=f"round,pad=0,rounding_size={h*0.28}",
        linewidth=lw, edgecolor=color, facecolor="white", zorder=5,
    )
    t = mpatches.transforms.Affine2D().rotate_deg(axis_deg).translate(*center) + ax.transData
    rect.set_transform(t)
    ax.add_patch(rect)


def branching_sleeve(ax, center, in_deg, branch_degs, w=SLEEVE_W, h=SLEEVE_H,
                      lw=LINE_W, color="black"):
    """분기점 슬리브를, 가지 방향으로 이음매 없이 매끈하게 벌어지는
    하나의 다각형으로 그린다.

    방법: 각 팔(트렁크 1개 + 가지들)을 "선분(center -> 팔 끝)"으로만
    표현해서 하나의 MultiLineString으로 묶은 뒤, shapely
    `buffer(hh, cap_style=round, join_style=round)`를 한 번에 적용한다.
    buffer는 선의 골격 전체에서 일정 거리(hh)만큼 부풀린 외곽선을
    기하학적으로 계산해주므로, 개별 폴리곤을 만들어 union하는 방식과
    달리 팔 사이 이음매(뾰족한 물방울/화살표 매듭)가 전혀 생기지 않고
    조개껍질처럼 매끈하게 한 몸체로 붙는다.

    가지 개수(트렁크 포함 총 팔 수)가 많을수록 슬리브 두께(hh)를 줄인다
    - 가지 2개(단순 Y분기)면 원래 두께(h) 그대로, 가지가 많은 부채꼴일
    수록 점점 얇아져서 서로 과하게 겹쳐 뭉툭해 보이지 않게 한다."""
    center = np.array(center, float)
    half = w / 2.0
    trunk_deg = in_deg + 180.0
    all_degs = [trunk_deg] + list(branch_degs)

    n_arms = len(all_degs)
    hh_max = h / 2.0
    # n_arms<=2(트렁크+가지1개)는 그대로, 이후 가지 하나 늘 때마다 점점
    # 얇아지되 너무 가늘어지진 않게 하한(hh_max*0.4)을 둔다.
    hh = hh_max if n_arms <= 2 else max(hh_max * (2.0 / n_arms) ** 0.5, hh_max * 0.4)

    lines = []
    for deg in all_degs:
        rad = np.radians(deg)
        end = center + half * np.array([np.cos(rad), np.sin(rad)])
        lines.append(LineString([tuple(center), tuple(end)]))

    merged_line = unary_union(lines)
    merged = merged_line.buffer(hh, cap_style=1, join_style=1)

    geoms = merged.geoms if merged.geom_type == "MultiPolygon" else [merged]
    for geom in geoms:
        if geom.is_empty:
            continue
        exterior = np.array(geom.exterior.coords)
        path = MplPath(exterior, [MplPath.MOVETO] + [MplPath.LINETO] * (len(exterior) - 2) + [MplPath.CLOSEPOLY])
        patch = mpatches.PathPatch(path, facecolor="white", edgecolor=color,
                                    linewidth=lw, joinstyle="round", zorder=5)
        ax.add_patch(patch)


def _rounded_rect_polygon(center, axis_deg, w, h, n_corner=8):
    """둥근 모서리 사각형(슬리브와 같은 모양)을 shapely Polygon으로 만든다.
    union 결과 자체가 원래 슬리브 테두리 모양을 그대로 유지하도록, 각
    조각을 처음부터 둥근 모서리로 만들어 union한다(별도 스무딩 없음 -
    두 슬리브를 겹쳐 그렸을 때 서로 교차하는 안쪽 선만 사라지는 것과
    동일한 결과)."""
    r = h * 0.28
    hw, hh = w / 2, h / 2
    corners = [
        (hw - r, hh - r), (-(hw - r), hh - r),
        (-(hw - r), -(hh - r)), (hw - r, -(hh - r)),
    ]
    start_angles = [0, 90, 180, 270]
    pts = []
    for (cx, cy), a0 in zip(corners, start_angles):
        for k in range(n_corner + 1):
            a = np.radians(a0 + 90.0 * k / n_corner)
            pts.append((cx + r * np.cos(a), cy + r * np.sin(a)))

    rad = np.radians(axis_deg)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    world = [(center[0] + x * cos_a - y * sin_a, center[1] + x * sin_a + y * cos_a)
             for x, y in pts]
    return Polygon(world)


def fused_sleeve(ax, parts, lw=LINE_W, color="black", fillet=0.0):
    """parts: [(center, axis_deg, w, h), ...] 슬리브들을 각각 둥근 모서리
    사각형으로 만든 뒤 union만 적용해서 그린다 - 겹치는 부분(교차하는
    안쪽 테두리)만 사라지고 바깥 윤곽은 원래 모양 그대로 남는다.

    서로 다른 각도로 겹치는 두 사각형을 union하면 이음매 안쪽에 뾰족한
    오목 꼭짓점(홈)이 남는데, 이 홈만 작은 반지름(fillet)으로 살짝
    깎아서(모폴로지 open: buffer(-fillet).buffer(+fillet)) 매끈하게
    메운다 - 전체 실루엣을 스무딩하는 게 아니라 오목한 구석만 메우는
    것이라 바깥 볼록 윤곽(진행축 방향 길이 등)은 그대로 유지된다."""
    polys = [_rounded_rect_polygon(c, a, w, h) for c, a, w, h in parts]
    merged = unary_union(polys)
    if fillet:
        merged = merged.buffer(-fillet, join_style=2).buffer(fillet, join_style=2)

    geoms = merged.geoms if merged.geom_type == "MultiPolygon" else [merged]
    for geom in geoms:
        if geom.is_empty:
            continue
        exterior = np.array(geom.exterior.coords)
        path = MplPath(exterior, [MplPath.MOVETO] + [MplPath.LINETO] * (len(exterior) - 2) + [MplPath.CLOSEPOLY])
        patch = mpatches.PathPatch(path, facecolor="white", edgecolor=color,
                                    linewidth=lw, joinstyle="round", zorder=5)
        ax.add_patch(patch)


def draw_bundle(ax, trunk_pts, branches, branch_angle_from_axis=45.0,
                sleeve_on_branch=False):
    """trunk_pts: [(x,y), ...] 몸통 정점. branches: {idx: [length, ...]} -
    그 정점에서 몸통 진행방향 기준 -branch_angle_from_axis 로 꺾여 나가는
    가지들의 길이 리스트.

    분기점은 슬리브 폴리곤을 union 하지 않고, 모든 선(몸통+가지)이 정확히
    한 점(center)에서 만나게만 그린다 - 실제 도면처럼 가지 사이에 배경이
    보이는 V자 홈이 자연히 생긴다. 슬리브(굵은 부품)가 실제로 있는 경우만
    sleeve_on_branch=True 로 개별 가지 위에 독립 슬리브를 하나씩 얹는다
    (다른 가지 슬리브와 합치지 않음)."""
    trunk_pts = [np.array(p, dtype=float) for p in trunk_pts]
    n = len(trunk_pts)

    for i in range(n - 1):
        double_line(ax, trunk_pts[i], trunk_pts[i + 1])

    for idx, lengths in branches.items():
        center = trunk_pts[idx]
        if idx > 0:
            in_dir = center - trunk_pts[idx - 1]
        else:
            in_dir = trunk_pts[idx + 1] - center
        axis_deg = np.degrees(np.arctan2(in_dir[1], in_dir[0]))

        for length in lengths:
            branch_dir_deg = axis_deg - branch_angle_from_axis
            ang = np.radians(branch_dir_deg)
            dirv = np.array([np.cos(ang), np.sin(ang)])
            end = center + length * dirv
            double_line(ax, center, end)

            if sleeve_on_branch:
                # 가지 자체에 슬리브 부품이 있는 경우만, 그 가지 위에 독립
                # 슬리브를 하나 얹는다(다른 가지와 union 하지 않음 - 각
                # 가지가 서로 별개의 부품이므로 사이 틈이 그대로 남는다).
                bc = center + (BRANCH_SLEEVE_W / 2.0) * dirv
                sleeve(ax, bc, branch_dir_deg, w=BRANCH_SLEEVE_W, h=SLEEVE_H)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(19, 5))

    P1 = (0.0, 0.0)
    J1 = (1800.0, 0.0)
    J2 = (3100.0, 0.0)
    P4 = (3470.0, 0.0)
    trunk = [P1, J1, J2, P4]

    branches = {
        1: [1250.0],  # J1에서 P2로
        2: [1250.0],  # J2에서 P3로
    }

    draw_bundle(ax, trunk, branches, sleeve_on_branch=False)
    for idx, lengths in branches.items():
        center = np.array(trunk[idx])
        in_dir = center - np.array(trunk[idx - 1]) if idx > 0 else np.array(trunk[idx + 1]) - center
        axis_deg = np.degrees(np.arctan2(in_dir[1], in_dir[0]))
        branch_degs = [axis_deg - 45.0 for _ in lengths]  # draw_bundle 기본 각도(45도)와 동일
        branching_sleeve(ax, center, axis_deg, branch_degs)

    ax.set_xlim(-150, 5100)
    ax.set_ylim(-1300, 200)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")

    out = os.path.join(OUT_DIR, "a20016147_2d_overlap_sleeve.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
