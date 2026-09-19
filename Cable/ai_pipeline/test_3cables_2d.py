"""
[테스트] 어려운(분기 각도/구조가 까다로운) 케이블 3종을 a20016147_2d_shape.py의
확정 슬리브 규칙(SLEEVE_W/BRANCH_SLEEVE_W, shapely union, 스무딩 없음)을 그대로
재사용해서 2D 이미지로 뽑아본다.

중요: 도면에 적힌 치수 "숫자"(예: 1300.0mm)를 그대로 좌표로 쓰지 않는다.
      대신 원본 Cable.pdf를 6배 확대 렌더링한 뒤, 화면에 실제로 그려진 선의
      픽셀 좌표를 직접 측정해서 그 비율(길이비/각도)을 그대로 쓴다 - 도면
      작성자가 실제로는 축척을 안 지키고 그린 경우가 많아서, 숫자보다 눈에
      보이는 대로의 형태가 우선이라는 요구사항 반영.
      (측정 방법: test_output/*_cablecrop.png 를 Read로 열어서 분기점/끝점
      픽셀 좌표를 눈대중으로 추정 -> math.dist/atan2로 길이·각도 계산)

측정한 3종:
  - A60025763: 분기점에서 가지 2개(P3, P4)가 거의 나란히 갈라짐.
    실측(픽셀): P3 len=758 angle=84.7도, P4 len=1002 angle=96.6도
    (진입축이 0도이므로 branch_angle_from_axis = angle, 즉 P3는 +84.7,
    P4는 +96.6 - 둘 다 거의 수직 아래로, 미세하게 벌어짐)
  - A60025764: T자 몸통(수평+수직) + 순차 분기 5개.
    실측(픽셀, 크롭 이미지 기준 x10 스케일 없이 그대로 비율만 사용):
      J1-J3(수평 몸통 후반) 610, J1-Jv2(수직 몸통, 축약 표시) 390,
      Jv2-Jv3 205, Jv3-P4 335 / P7: J2에서 350 (-90도인데 이미지 y가
      아래로 증가하므로 수학좌표로는 위쪽 = -90도), P6: J3에서 536
      (-16.2도), P5: J3에서 399(+23.7도), P2: Jv2에서 345(0도, 수평),
      P3: Jv3에서 489(17.9도)
  - A60025765: 완만한 비대칭 Y자. 실측(픽셀): P3 len=806 angle=-9.3도,
    P2 len=807 angle=+10.0도 (길이는 그림상 거의 동일)

a20016147_2d_shape.py의 draw_bundle()은 모든 가지가 같은 각도로 갈라지는
경우만 지원하므로, 가지별로 다른 각도를 줄 수 있는 draw_bundle_multi_angle()을
새로 만들어 여기서만 쓴다(본체 파일은 건드리지 않음).
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from a20016147_2d_shape import (
    double_line, fused_sleeve, SLEEVE_W, BRANCH_SLEEVE_W, SLEEVE_H,
)

OUT_DIR = r"C:\AIND_Library\Git_PredicTest\x64_Cami_Lib\PSCD1500\PSCD1500\Cable\ai_pipeline\test_output"


def draw_bundle_multi_angle(ax, trunk_pts, branches):
    """trunk_pts: [(x,y), ...] 몸통 정점(순서대로 직선 연결).
    branches: {trunk_index: [(angle_from_axis_deg, length), ...]} - 그
    정점에서 몸통 진입방향 기준 angle_from_axis_deg(부호 있음, +/- 로
    위아래 구분)만큼 꺾여 나가는 가지들. 가지마다 다른 각도/길이를 줄 수
    있다(도면 실측 픽셀 비율을 그대로 반영하기 위함).

    슬리브 규칙은 a20016147_2d_shape.py에서 확정한 것과 동일: 메인 슬리브
    (SLEEVE_W) + 가지 슬리브(BRANCH_SLEEVE_W = SLEEVE_W/2, 안쪽 끝이 분기점
    중심에 닿음)를 shapely union으로 하나의 외곽선으로 그린다."""
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

        sleeve_parts = [(center, axis_deg, SLEEVE_W, SLEEVE_H)]
        for angle_from_axis, length in branch_list:
            branch_dir_deg = axis_deg - angle_from_axis
            ang = np.radians(branch_dir_deg)
            dirv = np.array([np.cos(ang), np.sin(ang)])
            end = center + length * dirv
            double_line(ax, center, end)

            branch_center = center + (BRANCH_SLEEVE_W / 2.0) * dirv
            sleeve_parts.append((branch_center, branch_dir_deg, BRANCH_SLEEVE_W, SLEEVE_H))

        fused_sleeve(ax, sleeve_parts)


def render(name, trunk, branches, xlim, ylim, figsize=(14, 8)):
    fig, ax = plt.subplots(figsize=figsize)
    draw_bundle_multi_angle(ax, trunk, branches)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")
    out = os.path.join(OUT_DIR, f"{name}_2d.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"-> {out}")


# 실측 단위(픽셀)를 mm 스케일 감으로 쓰기 좋게 배율만 곱해서 쓴다(형태/비율은
# 그대로 유지, 절대 크기만 슬리브와 어울리게 확대).
PX_SCALE = 1.6


def build_a60025763():
    """실측(픽셀, PX_SCALE 적용): 분기점에서 P3(길이 758*스케일, +84.7도),
    P4(길이 1002*스케일, +96.6도)가 거의 나란히 갈라짐. 몸통은 도면 텍스트
    비율(150:380/930 근사)을 그대로 쓴다 - 몸통은 직선이라 형태 왜곡 이슈가
    없으므로 텍스트 치수를 유지."""
    P1 = (0.0, 0.0)
    J = (500.0, 0.0)
    P2 = (930.0, 0.0)
    trunk = [P1, J, P2]

    branches = {
        1: [(84.7, 758.0 * PX_SCALE), (96.6, 1002.0 * PX_SCALE)],
    }
    render("A60025763", trunk, branches, xlim=(-150, 1100), ylim=(-2000, 200))


def build_a60025764():
    """실측(픽셀, PX_SCALE 적용) 그대로 반영. 수평 몸통(P1-J1-J2-J3)과 수직
    몸통(J1-Jv2-Jv3-P4)은 도면 그림에서 측정한 세그먼트 길이 비율을 쓰고,
    각 가지도 실측 길이/각도를 그대로 쓴다."""
    P1 = (0.0, 0.0)
    J1 = (200.0 * PX_SCALE, 0.0)
    J2 = (J1[0] + 90.0 * PX_SCALE, 0.0)     # 실측: J1-J2 구간(도면 200+280 중 일부, 그림상 짧게)
    J3 = (J2[0] + 520.0 * PX_SCALE, 0.0)    # 실측: J2-J3 (J1-J3 총 610px 중 나머지)
    trunk_top = [P1, J1, J2, J3]

    # 실측: P7은 J2에서 수직 위로(그림 y가 위로 갈수록 음수라 -90도) 350px,
    # P6은 J3에서 -16.2도(살짝 위) 536px, P5는 J3에서 +23.7도(아래) 399px.
    branches_top = {
        2: [(-90.0, 350.0 * PX_SCALE)],
        3: [(-16.2, 536.0 * PX_SCALE), (23.7, 399.0 * PX_SCALE)],
    }

    Jv2 = (J1[0], J1[1] - 390.0 * PX_SCALE)   # 실측: J1-Jv2 (도면상 축약 표시된 구간)
    Jv3 = (Jv2[0], Jv2[1] - 205.0 * PX_SCALE)  # 실측: Jv2-Jv3
    P4 = (Jv3[0], Jv3[1] - 335.0 * PX_SCALE)   # 실측: Jv3-P4(수직 몸통 끝)
    trunk_down = [J1, Jv2, Jv3, P4]

    # 실측(재측정, P3zoom2 확대본으로 재확인): Jv2에서 P2는 진행축(아래)
    # 기준 -90도(오른쪽 수평) 345px. Jv3에서 P3도 마찬가지로 거의 완전한
    # 수평(-90도)으로 갈라진다 - 처음 측정 때 커넥터 끝의 45도 엘보 표시를
    # 가지 자체의 진행 각도로 착각해서 -17.9도로 잘못 넣었었다. 확대해서
    # 다시 보면 중심선이 수직 몸통에서 정확히 수평으로 꺾여 나간다.
    branches_down = {
        1: [(-90.0, 345.0 * PX_SCALE)],
        2: [(-90.0, 489.0 * PX_SCALE)],
    }

    fig, ax = plt.subplots(figsize=(14, 12))
    draw_bundle_multi_angle(ax, trunk_top, branches_top)
    draw_bundle_multi_angle(ax, trunk_down, branches_down)
    ax.set_xlim(-300, 2400)
    ax.set_ylim(-1700, 700)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")
    out = os.path.join(OUT_DIR, "A60025764_2d.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"-> {out}")


def build_a60025765():
    """실측(픽셀, PX_SCALE 적용): 분기점에서 P3(길이 806, -9.3도), P2(길이
    807, +10.0도) - 그림상 좌우 거의 대칭인 완만한 Y자."""
    P1 = (0.0, 0.0)
    J1 = (700.0, 0.0)
    trunk = [P1, J1]

    branches = {
        1: [(-9.3, 806.0 * PX_SCALE), (10.0, 807.0 * PX_SCALE)],
    }
    render("A60025765", trunk, branches, xlim=(-150, 2100), ylim=(-800, 800))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    build_a60025763()
    build_a60025764()
    build_a60025765()


if __name__ == "__main__":
    main()
