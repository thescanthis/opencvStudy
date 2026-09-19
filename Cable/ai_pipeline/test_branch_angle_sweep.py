"""
[테스트 전용] 분기 슬리브(a20016147_2d_shape.py의 확정 규칙)를 그대로 재사용해서
분기 각도를 10도 ~ 90도까지 바꿔가며 한 이미지에 그리드로 모아 비교한다.

a20016147_2d_shape.py 본체는 건드리지 않고, 거기서 확정된 sleeve/fused_sleeve
로직만 그대로 import해서 쓴다. 결과는 별도 png로 저장한다.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from a20016147_2d_shape import (
    double_line, fused_sleeve, SLEEVE_W, BRANCH_SLEEVE_W, SLEEVE_H, PIPE_GAP,
)

OUT_DIR = r"C:\AIND_Library\Git_PredicTest\x64_Cami_Lib\PSCD1500\PSCD1500\Cable\ai_pipeline\test_output"

ANGLES = list(range(10, 91, 10))  # 10, 20, ..., 90


PUSH_BACK_REF_ANGLE = 30.0  # 이 각도부터는 밀지 않음(케이블선이 이미 안 겹침)


def _push_back_for_angle(angle_deg):
    """[테스트] 10~20도처럼 얕은 각도에서는 가지 케이블(이중선)이 슬리브
    밖으로 나온 뒤에도 한동안 메인 케이블(이중선)과 겹쳐 보인다. 분기점을
    몸통 뒤쪽으로 미는 건 "보기 좋으라고" 임의로 늘리는 게 아니라, 딱
    두 이중선이 슬리브 바깥에서 서로 겹치지 않게 되는 최소 거리만큼만
    민다.

    분기점(0,0 기준)에서 가지 라인이 메인 라인의 위쪽 평행선(오프셋
    PIPE_GAP/2)과 다시 만나는 지점까지의 거리가, 가지선이 메인선 두께
    범위를 벗어나기 전까지 "겹쳐 보이는" 구간이다. 그 지점을 분기점
    뒤로 미는 최소 push_back 으로 삼는다(각도가 0도에 가까울수록 커지고,
    PUSH_BACK_REF_ANGLE 이상이면 거의 0에 수렴)."""
    if angle_deg >= PUSH_BACK_REF_ANGLE:
        return 0.0
    half_gap = PIPE_GAP / 2.0
    # 분기점에서 각도 angle_deg로 뻗어나가는 직선이, 분기점을 지나는
    # 메인 라인과 나란한 평행선(수직거리 half_gap)에 도달하는 데 필요한
    # 진행축 방향 거리 = half_gap / tan(angle_deg)
    needed = half_gap / np.tan(np.radians(angle_deg))
    return max(0.0, needed)


def draw_one_branch(ax, angle_deg, trunk_len=900.0, branch_len=700.0):
    """수평 몸통 하나 + 그 끝 분기점에서 angle_deg 만큼 꺾여 나가는 가지
    하나를 그린다. 분기점 슬리브 규칙은 a20016147_2d_shape.py와 동일하게
    메인/가지 슬리브를 union해서 그린다(확정 규칙 그대로).

    [테스트] 10~20도처럼 얕은 각도에서는 분기점 자체를 몸통 뒤쪽으로 밀어
    30도일 때와 비슷한 정도로 벌어져 보이게 한다(_push_back_for_angle)."""
    P0 = np.array([0.0, 0.0])
    axis_deg = 0.0  # 몸통이 수평이므로 진입 방향은 0도
    axis_dirv = np.array([np.cos(np.radians(axis_deg)), np.sin(np.radians(axis_deg))])

    push_back = _push_back_for_angle(angle_deg)
    center = np.array([trunk_len, 0.0]) - push_back * axis_dirv
    P_end = np.array([trunk_len + trunk_len * 0.35, 0.0])  # 몸통 끝은 고정

    double_line(ax, P0, center)
    double_line(ax, center, P_end)

    branch_dir_deg = axis_deg - angle_deg
    ang = np.radians(branch_dir_deg)
    dirv = np.array([np.cos(ang), np.sin(ang)])
    end = center + branch_len * dirv
    double_line(ax, center, end)

    branch_center = center + (BRANCH_SLEEVE_W / 2.0) * dirv
    sleeve_parts = [
        (center, axis_deg, SLEEVE_W, SLEEVE_H),
        (branch_center, branch_dir_deg, BRANCH_SLEEVE_W, SLEEVE_H),
    ]
    fused_sleeve(ax, sleeve_parts)

    ax.set_title(f"{angle_deg}\u00b0", fontsize=14)
    ax.set_xlim(-150 - push_back, trunk_len * 1.5)
    ax.set_ylim(-branch_len * 1.15, 300)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    n = len(ANGLES)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = np.array(axes).reshape(-1)

    for i, angle in enumerate(ANGLES):
        draw_one_branch(axes[i], angle)

    for j in range(len(ANGLES), len(axes)):
        axes[j].axis("off")

    fig.patch.set_facecolor("white")
    fig.tight_layout()

    out = os.path.join(OUT_DIR, "test_branch_angle_sweep.png")
    fig.savefig(out, dpi=150, facecolor="white")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
