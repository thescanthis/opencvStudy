"""
[테스트] 사용자가 지목한 3장(A20016149, A20016151, A60023103)을 새 분기 규칙
(점 분기 + 개별 슬리브, V자 홈 유지)으로 2D 추출한다.

규칙: 도면 치수 숫자는 무시하고 Cable.pdf 확대 렌더링에서 실측한 픽셀
좌표(길이/각도) 비율을 쓴다. 분기점은 트렁크 슬리브 하나가 끊김 없이
이어지고, 가지가 갈라지는 자리에만 V자 홈이 파이도록 그린다(가지마다
별개의 슬리브가 필요하면 sleeve_on_branch로 독립적으로 얹는다).
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from a20016147_2d_shape import double_line, sleeve, SLEEVE_W, BRANCH_SLEEVE_W, SLEEVE_H
from test_5cables_2d import draw_fan, draw_bundle_multi_angle, _polar, _save

OUT_DIR = r"C:\AIND_Library\Git_PredicTest\x64_Cami_Lib\PSCD1500\PSCD1500\Cable\ai_pipeline\test_output"


# ---------------------------------------------------------------------------
# A20016149 : P1 - 트렁크(완만히 꺾이며 하강) - J1(P10) - J2(P9,P8,P2 부채꼴)
#   - J3(P6) - J4(P5,P3,P4 부채꼴)
#   실측(픽셀, 원본 2000px 기준, y는 이미지좌표=아래가 +이므로 그릴 때 반전):
#     P1(190,625) J1(615,625) J2(770,610) J3(880,665) J4(1080,795)
# ---------------------------------------------------------------------------
def build_a20016149():
    S = 1.5
    flip = lambda x, y: (x * S, -y * S)  # 이미지좌표(y아래+) -> 수학좌표(y위+)

    P1 = np.array(flip(0, 0))
    J1 = np.array(flip(615 - 190, 0))
    J2 = np.array(flip(770 - 190, 610 - 625))
    J3 = np.array(flip(880 - 190, 665 - 625))
    J4 = np.array(flip(1080 - 190, 795 - 625))
    trunk = [P1, J1, J2, J3, J4]

    fig, ax = plt.subplots(figsize=(19, 11))
    for a, b in zip(trunk[:-1], trunk[1:]):
        double_line(ax, a, b)

    # J1 : P10 하나 (위쪽 대각)
    P10 = np.array(flip(400 - 190, 460 - 625))
    draw_fan(ax, J1, P1, [P10], sleeve_on_trunk=True)

    # J2 : P9, P8, P2 부채꼴 (위쪽)
    P9 = np.array(flip(920 - 190, 180 - 625))
    P8 = np.array(flip(1150 - 190, 280 - 625))
    P2 = np.array(flip(1290 - 190, 420 - 625))
    draw_fan(ax, J2, J1, [P9, P8, P2], sleeve_on_trunk=True)

    # J3 : P6 하나 (오른쪽 대각 아래로, 이미지상 아래=수학상 -)
    P6 = np.array(flip(1610 - 190, 795 - 625))
    draw_fan(ax, J3, J2, [P6], sleeve_on_trunk=True)

    # J4 : P5(오른쪽 대각 위), P3/P4(아래쪽 대각)
    P5 = np.array(flip(1870 - 190, 795 - 625))
    P3 = np.array(flip(1290 - 190, 950 - 625))
    P4 = np.array(flip(1620 - 190, 950 - 625))
    draw_fan(ax, J4, J3, [P5, P3, P4], sleeve_on_trunk=True)

    ax.set_xlim(-150, 2900); ax.set_ylim(-700, 900)
    ax.set_aspect("equal"); ax.axis("off")
    _save(fig, "A20016149")


# ---------------------------------------------------------------------------
# A20016151 : P1 - 몸통 - Jm(위/아래 두 갈래) - 각 갈래 끝에서 부채꼴.
#   test_5cables_2d.py build_a20016151() 와 동일 실측값, 새 분기 규칙만 적용.
# ---------------------------------------------------------------------------
def build_a20016151():
    S = 1.4

    P1 = np.array([0.0, 0.0])
    Jm = np.array([940.0 * S, 0.0])
    JA = np.array(_polar(Jm, 37.0, 225.0 * S))
    JB = np.array(_polar(Jm, -34.0, 234.0 * S))

    fig, ax = plt.subplots(figsize=(18, 13))

    double_line(ax, P1, Jm)
    double_line(ax, Jm, JA)
    double_line(ax, Jm, JB)
    draw_fan(ax, Jm, P1, [JA, JB], sleeve_on_trunk=True)

    fanA = [(450, 90.6), (443, 63.1), (517, 24.6), (710, -11.8)]  # P9,P8,P7,P6
    endsA = [_polar(JA, ang, L * S) for L, ang in fanA]
    draw_fan(ax, JA, Jm, endsA, sleeve_on_trunk=True)

    fanB = [(420, 15.2), (460, -19.0), (642, -40.9), (710, -73.2), (373, -97.7)]  # P5,P4,P3,P2,P10
    endsB = [_polar(JB, ang, L * S) for L, ang in fanB]
    draw_fan(ax, JB, Jm, endsB, sleeve_on_trunk=True)

    ax.set_xlim(-200, 3200); ax.set_ylim(-1900, 1200)
    ax.set_aspect("equal"); ax.axis("off")
    _save(fig, "A20016151")


# ---------------------------------------------------------------------------
# A60023103 : P1(45도 엘보) - 트렁크 - J_C890K(90도 엘보, P9 위) - 트렁크 -
#   J1(P8/P7 위) - J2(P6 위) - J3(Y자, P5 위/P4 중간/P3 아래) - P2.
#   실측(픽셀, 원본 2000px 기준):
#     P1(60,625) J_C890K(430,455,90도꺾임) J1(760,625) J2(1000,625)
#     J3(1600,625) P2(1900,625)
# ---------------------------------------------------------------------------
def build_a60023103():
    S = 1.3
    flip = lambda x, y: (x * S, -y * S)

    P1 = np.array(flip(0, 0))
    JC = np.array(flip(430 - 60, 455 - 625))    # C-890K 90도 엘보(코너)
    J1 = np.array(flip(760 - 60, 625 - 625))
    J2 = np.array(flip(1000 - 60, 625 - 625))
    J3 = np.array(flip(1600 - 60, 625 - 625))
    Pend = np.array(flip(1900 - 60, 625 - 625))
    trunk = [P1, JC, J1, J2, J3, Pend]

    fig, ax = plt.subplots(figsize=(20, 10))
    for a, b in zip(trunk[:-1], trunk[1:]):
        double_line(ax, a, b)
    for idx in (1, 2, 3):  # JC, J1, J2 : 트렁크 슬리브만(꺾임/T분기)
        pass

    # JC : 코너에서 위로 꺾여 P9 로 가는 엘보 겸 분기 (90도 꺾임 + 별도 가지)
    P9 = np.array(flip(430 - 60, 320 - 625))
    draw_fan(ax, JC, P1, [P9], sleeve_on_trunk=True)

    # J1 : 위로 P8, P7 분기 (KM2A2 쪽)
    P8 = np.array(flip(870 - 60, 280 - 625))
    P7 = np.array(flip(780 - 60, 230 - 625))
    draw_fan(ax, J1, JC, [P8, P7], sleeve_on_trunk=True)

    # J2 : 위로 P6 분기 (KM3 쪽)
    P6 = np.array(flip(1000 - 60, 300 - 625))
    draw_fan(ax, J2, J1, [P6], sleeve_on_trunk=True)

    # J3 : Y자 - P5(오른쪽 위 대각, 가장 김), P4(중간), P3(짧은 대각)
    P5 = np.array(flip(1930 - 60, 130 - 625))
    P4 = np.array(flip(1810 - 60, 150 - 625))
    P3 = np.array(flip(1760 - 60, 190 - 625))
    draw_fan(ax, J3, J2, [P5, P4, P3], sleeve_on_trunk=True)

    ax.set_xlim(-150, 2600); ax.set_ylim(-900, 700)
    ax.set_aspect("equal"); ax.axis("off")
    _save(fig, "A60023103")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    build_a20016149()
    build_a20016151()
    build_a60023103()


if __name__ == "__main__":
    main()
