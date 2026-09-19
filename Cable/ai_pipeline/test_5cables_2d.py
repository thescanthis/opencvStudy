"""
[테스트] 아직 2D로 안 뽑은 케이블 중 어려운(분기 많은) 5종을 골라 2D 이미지로
추출한다. a20016147_2d_shape.py의 확정 슬리브 규칙을 그대로 재사용한다.

규칙(이전 세션에서 확정):
  - 도면 치수 "숫자"는 무시하고, Cable.pdf를 확대 렌더링해 실제 그려진 선의
    픽셀 좌표(길이/각도)를 재서 그 비율을 쓴다.
  - 치수선/숫자, 풍선번호+지시선, 부품표, 배선도(핀맵), 테두리/타이틀블록,
    수정이력, 워터마크는 절대 참고하지 않는다. 케이블 배선 선만.

고른 5종(난이도 순):
  1. A60023110 - P1 + 몸통 + 다중 부채꼴 분기(커넥터 ~26개). 최고 난이도.
  2. A60023111 - 몸통에서 한 지점 대형 부채꼴(P3~P18) + 오른쪽 추가 분기. 최고.
  3. A20016151 - P1 + 긴 몸통 + 이중 부채꼴로 P2~P10(10개).
  4. A60025776 - 중앙 몸통 + 사방(상/하/좌/우) 다중 분기로 P1~P11(11개).
  5. A60026321 - P1 + 몸통 + T자(P4 아래) + Y자(P2/P3). 4개. (제일 다룰만함)

부채꼴 다중 분기(1~4번)는 지금 로직의 미해결 이슈(가지끼리 겹침)가 크게
드러난다 - 형태 근사만 시도하고, 정밀 좌표는 추후 자동 추출 파이프라인으로.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from a20016147_2d_shape import (
    double_line, fused_sleeve, sleeve, branching_sleeve,
    SLEEVE_W, BRANCH_SLEEVE_W, SLEEVE_H,
)

OUT_DIR = r"C:\AIND_Library\Git_PredicTest\x64_Cami_Lib\PSCD1500\PSCD1500\Cable\ai_pipeline\test_output"

# 픽셀 실측값을 슬리브 크기와 어울리게 키우는 배율(형태/비율은 그대로, 절대크기만).
PX = 1.6


MIN_BRANCH_GAP_DEG = 8.0  # 분기점에서 인접한 두 가지 사이 최소 각도 간격


def _spread_close_angles(center, ends, min_gap_deg=MIN_BRANCH_GAP_DEG):
    """center에서 뻗어나가는 여러 가지(ends) 중, 각도가 min_gap_deg보다
    가까운 것들끼리만 그 차이가 min_gap_deg가 되도록 밀어 벌린다. 각 가지의
    길이(center로부터 거리)는 그대로 유지하고 각도만 회전시킨다 - 그래야
    슬리브 근처에서 선끼리 스치듯 교차하던 게 사라진다. 각도가 이미 충분히
    떨어진 가지는 건드리지 않는다."""
    center = np.array(center, float)
    items = []
    for end in ends:
        end = np.array(end, float)
        d = end - center
        L = np.hypot(*d)
        if L < 1e-6:
            items.append((0.0, 0.0, end))
            continue
        ang = np.degrees(np.arctan2(d[1], d[0]))
        items.append((ang, L, end))

    # 각도 순으로 정렬해서 인접 간격만 검사 - 순서 자체(원래 ends 순서)는
    # 반환값에서 다시 맞춰준다.
    order = sorted(range(len(items)), key=lambda i: items[i][0])
    angs = [items[i][0] for i in order]
    for k in range(1, len(angs)):
        gap = angs[k] - angs[k - 1]
        if gap < min_gap_deg:
            push = (min_gap_deg - gap) / 2.0
            angs[k - 1] -= push
            angs[k] += push

    new_ends = [None] * len(items)
    for pos, i in enumerate(order):
        _, L, _ = items[i]
        if L < 1e-6:
            new_ends[i] = items[i][2]
            continue
        a = np.radians(angs[pos])
        new_ends[i] = center + L * np.array([np.cos(a), np.sin(a)])
    return new_ends


def draw_fan(ax, center, in_pt, ends, sleeve_on_trunk=False, sleeve_on_branch=False,
             auto_spread=True):
    """center에서 여러 가지가 갈라지는 분기점을 그린다.
    center: 분기점 좌표. in_pt: 진입 방향을 정하는 이전 점(몸통이 들어온 쪽).
    ends: [end_xy, ...] 각 가지 끝점. 몸통(center-in_pt 선)은 호출측에서 이미
    그렸다고 가정한다.

    모든 선이 center 한 점에서 정확히 만나도록만 그린다(union 없음) - 실제
    도면처럼 가지 사이에 배경이 드러나는 V자 홈이 자연히 생긴다. 슬리브
    부품이 실제로 있는 곳만 sleeve_on_trunk/sleeve_on_branch 로 개별
    슬리브를 하나씩 독립적으로 얹는다(서로 union 하지 않음).

    auto_spread=True(기본)면, 각도가 가까운 가지끼리만 최소 간격만큼
    자동으로 벌려서 슬리브 근처 선 교차를 줄인다(길이는 그대로 유지)."""
    center = np.array(center, float)
    in_pt = np.array(in_pt, float)
    in_dir = center - in_pt
    axis_deg = np.degrees(np.arctan2(in_dir[1], in_dir[0]))

    if auto_spread and len(ends) > 1:
        ends = _spread_close_angles(center, ends)

    branch_degs = []
    for end in ends:
        end = np.array(end, float)
        d = end - center
        L = np.hypot(*d)
        if L < 1e-6:
            continue
        dirv = d / L
        branch_dir_deg = np.degrees(np.arctan2(d[1], d[0]))
        branch_degs.append(branch_dir_deg)
        double_line(ax, center, end)
        if sleeve_on_branch:
            bc = center + (BRANCH_SLEEVE_W / 2.0) * dirv
            sleeve(ax, bc, branch_dir_deg, w=BRANCH_SLEEVE_W, h=SLEEVE_H)

    if sleeve_on_trunk and branch_degs:
        branching_sleeve(ax, center, axis_deg, branch_degs)


def draw_bundle_multi_angle(ax, trunk_pts, branches, sleeve_on_trunk=False,
                             sleeve_on_branch=False, auto_spread=True):
    """trunk_pts: [(x,y), ...] 몸통 정점(순서대로 직선 연결. y는 수학좌표=위가 +).
    branches: {trunk_index: [(angle_from_axis_deg, length), ...]}

    분기점은 union 없이 모든 선이 한 점(center)에서 만나게만 그린다(V자
    홈 유지). 슬리브 부품이 실제로 있는 곳만 sleeve_on_trunk/
    sleeve_on_branch 로 독립 슬리브를 얹는다. auto_spread=True면 각도가
    가까운 가지끼리만 최소 간격만큼 자동으로 벌려서 선 교차를 줄인다."""
    trunk_pts = [np.array(p, dtype=float) for p in trunk_pts]
    n = len(trunk_pts)
    for i in range(n - 1):
        double_line(ax, trunk_pts[i], trunk_pts[i + 1])

    for idx, branch_list in branches.items():
        center = trunk_pts[idx]
        if idx > 0:
            in_dir = center - trunk_pts[idx - 1]
        else:
            in_dir = trunk_pts[idx + 1] - center
        axis_deg = np.degrees(np.arctan2(in_dir[1], in_dir[0]))

        ends = []
        for angle_from_axis, length in branch_list:
            branch_dir_deg = axis_deg - angle_from_axis
            ang = np.radians(branch_dir_deg)
            ends.append(center + length * np.array([np.cos(ang), np.sin(ang)]))
        if auto_spread and len(ends) > 1:
            ends = _spread_close_angles(center, ends)

        branch_degs = []
        for end in ends:
            d = end - center
            L = np.hypot(*d)
            if L < 1e-6:
                continue
            dirv = d / L
            branch_dir_deg = np.degrees(np.arctan2(d[1], d[0]))
            branch_degs.append(branch_dir_deg)
            double_line(ax, center, end)
            if sleeve_on_branch:
                branch_center = center + (BRANCH_SLEEVE_W / 2.0) * dirv
                sleeve(ax, branch_center, branch_dir_deg, w=BRANCH_SLEEVE_W, h=SLEEVE_H)
        if sleeve_on_trunk and branch_degs:
            branching_sleeve(ax, center, axis_deg, branch_degs)


def _save(fig, name):
    fig.patch.set_facecolor("white")
    out = os.path.join(OUT_DIR, f"{name}_2d.png")
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"-> {out}")


def render(name, trunk, branches, xlim, ylim, figsize=(16, 9)):
    fig, ax = plt.subplots(figsize=figsize)
    draw_bundle_multi_angle(ax, trunk, branches, sleeve_on_trunk=True)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_aspect("equal"); ax.axis("off")
    _save(fig, name)


# ---------------------------------------------------------------------------
# 5. A60026321 : P1 - 몸통 - T자(P4 아래) - 몸통 - Y자(P2 위/P3 아래)
#    실측(픽셀, crop 2000px 기준): P1(280,375) T자(685,383) Y자(1120,383)
#    P2(1770,290) P3(1810,460) / P4는 T자에서 수직 아래로 ~330px
# ---------------------------------------------------------------------------
def build_a60026321():
    P1 = (0.0, 0.0)
    J_T = (405.0 * PX, 0.0)          # T자 분기점 (685-280=405)
    J_Y = (840.0 * PX, 0.0)          # Y자 분기점 (1120-280=840)
    trunk = [P1, J_T, J_Y]

    branches = {
        1: [(90.0, 330.0 * PX)],                    # P4: T자에서 수직 아래(진행축0 기준 +90 = 아래)
        2: [(-13.0, 670.0 * PX), (13.0, 700.0 * PX)],  # Y자: P2 위(-13도), P3 아래(+13도)
    }
    render("A60026321", trunk, branches,
           xlim=(-150, 1900 * PX / 1.6 * 1.6), ylim=(-900, 400))


# ---------------------------------------------------------------------------
# 3. A20016151 : P1 - 몸통 - Jm(몸통 끝, 위/아래 두 갈래) - 각 갈래 끝에서
#    부채꼴. 실측(픽셀, crop 2000px 기준):
#      P1->Jm 940px, ~수평
#      Jm->JA 225px @+37도(위), Jm->JB 234px @-34도(아래)
#      상단 부채꼴(JA): P9 450@+91, P8 443@+63, P7 517@+25, P6 710@-12
#      하단 부채꼴(JB): P5 420@+15, P4 460@-19, P3 642@-41, P2 710@-73, P10 373@-98
#    (math_angle = 수학좌표 절대각. draw_bundle_multi_angle은 진입축 기준
#     상대각을 받으므로, 각 부채꼴에서 axis_deg 를 빼서 넣는다.)
# ---------------------------------------------------------------------------
def _polar(origin, deg, r):
    a = np.radians(deg)
    return (origin[0] + r * np.cos(a), origin[1] + r * np.sin(a))


def build_a20016151():
    S = 1.4  # 픽셀 -> 그림 배율

    P1 = np.array([0.0, 0.0])
    Jm = np.array([940.0 * S, 0.0])                       # 몸통 끝 = 위/아래 갈래점
    JA = np.array(_polar(Jm, 37.0, 225.0 * S))            # 상단 부채꼴 분기점
    JB = np.array(_polar(Jm, -34.0, 234.0 * S))           # 하단 부채꼴 분기점

    fig, ax = plt.subplots(figsize=(18, 13))

    # 몸통
    double_line(ax, P1, Jm)
    # Jm -> JA, Jm -> JB 짧은 목
    double_line(ax, Jm, JA)
    double_line(ax, Jm, JB)
    # Jm 분기점 슬리브(몸통 진입 + 두 갈래)
    draw_fan(ax, Jm, P1, [JA, JB], sleeve_on_trunk=True)

    # 상단 부채꼴 (JA에서, 진입 방향 = Jm->JA)
    fanA = [(450, 90.6), (443, 63.1), (517, 24.6), (710, -11.8)]  # P9,P8,P7,P6
    endsA = [_polar(JA, ang, L * S) for L, ang in fanA]
    draw_fan(ax, JA, Jm, endsA, sleeve_on_trunk=True)

    # 하단 부채꼴 (JB에서, 진입 방향 = Jm->JB)
    fanB = [(420, 15.2), (460, -19.0), (642, -40.9), (710, -73.2), (373, -97.7)]  # P5,P4,P3,P2,P10
    endsB = [_polar(JB, ang, L * S) for L, ang in fanB]
    draw_fan(ax, JB, Jm, endsB, sleeve_on_trunk=True)

    ax.set_xlim(-200, 3200); ax.set_ylim(-1900, 1200)
    ax.set_aspect("equal"); ax.axis("off")
    _save(fig, "A20016151")


# ---------------------------------------------------------------------------
# 1. A60023110 : P1 - 몸통 - 분기점1(P2,P3 아래 / P4,P5 위) - 몸통 -
#    분기점2(P6~P16 위쪽 대형 반원 부채꼴 R156) - 긴 몸통(4700) -
#    분기점3(P17~P21 아래쪽 부채꼴 R257~385) - 몸통 - 분기점4(P22~P26).
#    실측(픽셀, crop 2000x950 기준, 몸통 y~590):
#      P1(110) J1(530) J2(1030) J3(1830) J4(1960)
#    각 부채꼴 각도는 반원을 균등 분할한 근사(도면상 커넥터가 등각 배치).
# ---------------------------------------------------------------------------
def build_a60023110():
    S = 1.7
    px = lambda x: (x - 110) * S

    P1 = np.array([0.0, 0.0])
    J1 = np.array([px(530), 0.0])
    J2 = np.array([px(1030), 0.0])
    J3 = np.array([px(1830), 0.0])
    J4 = np.array([px(1960), 0.0])

    fig, ax = plt.subplots(figsize=(24, 12))

    # 몸통 전체
    for a, b in [(P1, J1), (J1, J2), (J2, J3), (J3, J4), (J4, (J4[0] + 200, 0.0))]:
        double_line(ax, np.array(a), np.array(b))

    # 분기점1: P4,P5 위 / P2,P3 아래 (엘보지만 직선 근사)
    ends1 = [_polar(J1, 62.0, 620.0 * S), _polar(J1, 42.0, 700.0 * S),
             _polar(J1, -118.0, 430.0 * S), _polar(J1, -125.0, 400.0 * S)]
    draw_fan(ax, J1, P1, ends1, sleeve_on_trunk=True)

    # 분기점2: P6~P16 위쪽 반원 부채꼴(11가닥). 각도 175도(P16)~62도(P6) 균등.
    n2 = 11
    R2 = 156.0 * S * 2.2
    ends2 = [_polar(J2, 175.0 - (175.0 - 60.0) * i / (n2 - 1), R2) for i in range(n2)]
    draw_fan(ax, J2, J1, ends2, sleeve_on_trunk=True)

    # 분기점3: P17~P21 아래쪽 부채꼴(5가닥). 각도 -100도(P17)~-15도(P21).
    n3 = 5
    ends3 = [_polar(J3, -100.0 + (100.0 - 15.0) * i / (n3 - 1), (300.0 + 90.0 * i) * S)
             for i in range(n3)]
    draw_fan(ax, J3, J2, ends3, sleeve_on_trunk=True)

    # 분기점4: P22~P26 아래쪽 소부채꼴(5가닥).
    n4 = 5
    ends4 = [_polar(J4, -110.0 + (110.0 - 40.0) * i / (n4 - 1), 300.0 * S) for i in range(n4)]
    draw_fan(ax, J4, J3, ends4, sleeve_on_trunk=True)

    ax.set_xlim(-200, px(1960) + 700)
    ax.set_ylim(-1400, 1500)
    ax.set_aspect("equal"); ax.axis("off")
    _save(fig, "A60023110")


# ---------------------------------------------------------------------------
# 2. A60023111 : P1 - 수평 몸통 - 몸통의 서로 다른 지점에서 가지가 순차로
#    갈라져 아래로 거의 나란히 늘어짐(P3,P4 짧게 / P7~P18 길게).
#    실측(픽셀, crop 2000px 기준, 몸통 y~235):
#      (가지명, 몸통상 시작x, 길이px, 수학각도)
# ---------------------------------------------------------------------------
def build_a60023111():
    S = 1.3
    # (시작x_px, 길이px, 각도deg)  - crop 좌표를 P1 원점(0) 기준으로 이동(px-95)
    seq = [
        (330, 231, -83.8), (360, 305, -89.1),
        (430, 673, -78.9), (470, 674, -74.5), (520, 680, -72.9),
        (600, 658, -73.2), (660, 636, -73.6), (710, 622, -66.3),
        (760, 613, -65.9), (810, 613, -65.9), (900, 596, -67.3),
        (960, 586, -64.7), (1050, 591, -61.7), (1130, 588, -58.2),
        (1230, 577, -56.3), (1330, 555, -56.0),
    ]
    trunk_end_x = 1900

    fig, ax = plt.subplots(figsize=(17, 11))

    # 몸통
    P1 = np.array([0.0, 0.0])
    Pend = np.array([(trunk_end_x - 95) * S, 0.0])
    double_line(ax, P1, Pend)

    # 각 가지: 몸통상 시작점 = (x, 0), 끝점 = start + polar(각도, 길이)
    for sx, L, ang in seq:
        c = np.array([(sx - 95) * S, 0.0])
        end = _polar(c, ang, L * S)
        double_line(ax, c, end)
        # 가지 슬리브만 독립적으로 얹는다(트렁크와 union 하지 않음 -
        # 선은 한 점 c에서 만나고, 슬리브는 가지 위에만 별개로 존재).
        d = np.array(end) - c
        bdir = np.degrees(np.arctan2(d[1], d[0]))
        dirv = d / np.hypot(*d)
        bc = c + (BRANCH_SLEEVE_W / 2.0) * dirv
        sleeve(ax, bc, bdir, w=BRANCH_SLEEVE_W, h=SLEEVE_H)

    ax.set_xlim(-150, (trunk_end_x - 95) * S + 150)
    ax.set_ylim(-1300, 300)
    ax.set_aspect("equal"); ax.axis("off")
    _save(fig, "A60023111")


# ---------------------------------------------------------------------------
# 4. A60025776 : 수평 몸통 + 3개 분기점(사방으로 P1~P11).
#    실측(픽셀, trunk crop 2000x808 기준 & L/R crop 보정):
#      몸통: JL(왼쪽분기) -> JC(중앙분기) 560px @0도 -> JR(오른쪽분기) 513px @+6도
#      왼쪽분기 JL: P1 위대각(~+50도), P2 아래대각(~-55도), P11 수평왼쪽(180도)
#      중앙분기 JC: P10 위수직(+90, 620px), P3 아래대각(-52, 524px), P4 아래대각(-29, 651px)
#      오른쪽분기 JR: P5 아래왼쪽엘보(-38, 392px), P6 아래오른대각(-14, 660px),
#                     P7 수평(0, 695px), P8/P9 위대각(+29~32, 530px)
# ---------------------------------------------------------------------------
def build_a60025776():
    S = 1.5

    JL = np.array([0.0, 0.0])
    JC = np.array(_polar(JL, 0.0, 560.0 * S))
    JR = np.array(_polar(JC, 6.0, 513.0 * S))

    fig, ax = plt.subplots(figsize=(19, 11))

    # 몸통 (JL - JC - JR)
    double_line(ax, JL, JC)
    double_line(ax, JC, JR)

    # 왼쪽 분기점: 진입은 오른쪽에서 들어온다고 보고(몸통 반대), P1/P2/P11
    Lext = np.array(_polar(JL, 180.0, 200.0 * S))   # 진입 더미(오른쪽 몸통쪽)
    endsL = [_polar(JL, 128.0, 520.0 * S),   # P1 위 대각
             _polar(JL, -125.0, 430.0 * S),  # P2 아래 대각
             _polar(JL, 180.0, 300.0 * S)]   # P11 수평 왼쪽
    draw_fan(ax, JL, JC, endsL, sleeve_on_trunk=True)

    # 중앙 분기점
    endsC = [_polar(JC, 90.0, 620.0 * S),    # P10 위 수직
             _polar(JC, -52.0, 524.0 * S),   # P3 아래 대각
             _polar(JC, -29.0, 651.0 * S)]   # P4 아래 대각
    draw_fan(ax, JC, JL, endsC, sleeve_on_trunk=True)

    # 오른쪽 분기점
    endsR = [_polar(JR, -140.0, 392.0 * S),  # P5 아래-왼쪽 엘보 방향
             _polar(JR, -14.0, 660.0 * S),   # P6 아래 대각
             _polar(JR, 2.0, 695.0 * S),     # P7 수평
             _polar(JR, 29.0, 530.0 * S),    # P8 위 대각
             _polar(JR, 33.0, 545.0 * S)]    # P9 위 대각(P8 바로 옆)
    draw_fan(ax, JR, JC, endsR, sleeve_on_trunk=True)

    ax.set_xlim(-1100, 2600); ax.set_ylim(-1400, 1200)
    ax.set_aspect("equal"); ax.axis("off")
    _save(fig, "A60025776")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    build_a60026321()
    build_a20016151()
    build_a60025776()
    build_a60023111()
    build_a60023110()


if __name__ == "__main__":
    main()
