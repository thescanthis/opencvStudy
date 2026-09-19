"""
[테스트 전용] 한 분기점에서 가지가 여러 개(예: 3개) 동시에 뻗어나가는 경우를
테스트한다. test_branch_angle_sweep.py에서 만든 "얕은 각도 분기점 뒤로 밀기"
로직(_push_back_for_angle, 케이블선이 슬리브 밖에서 겹치지 않는 최소 거리)을
그대로 재사용하되, 가지가 여러 개면 그중 가장 얕은 각도 기준으로 민다
(가장 많이 밀어야 하는 가지에 맞추면 나머지 가지는 자동으로 안 겹침).
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from a20016147_2d_shape import double_line, fused_sleeve, SLEEVE_W, BRANCH_SLEEVE_W, SLEEVE_H
from test_branch_angle_sweep import _push_back_for_angle

OUT_DIR = r"C:\AIND_Library\Git_PredicTest\x64_Cami_Lib\PSCD1500\PSCD1500\Cable\ai_pipeline\test_output"


def draw_multi_branch(ax, angles_deg, trunk_len=900.0, branch_len=700.0, title=""):
    """수평 몸통 하나 + 그 끝 분기점에서 angles_deg 각각으로 꺾여 나가는
    가지 여러 개를 그린다(모두 같은 분기점에서 갈라짐). 슬리브는 메인
    슬리브 1개 + 가지별 슬리브를 전부 union해서 하나로 그린다."""
    P0 = np.array([0.0, 0.0])
    axis_deg = 0.0
    axis_dirv = np.array([1.0, 0.0])

    # 여러 가지 중 가장 많이 밀어야 하는(=가장 얕은 각도) 값 기준으로 push_back 결정
    # (위/아래 어느 쪽으로 꺾이든 얕은 정도는 절댓값으로 판단한다)
    push_back = max(_push_back_for_angle(abs(a)) for a in angles_deg)
    center = np.array([trunk_len, 0.0]) - push_back * axis_dirv
    P_end = np.array([trunk_len + trunk_len * 0.35, 0.0])

    double_line(ax, P0, center)
    double_line(ax, center, P_end)

    sleeve_parts = [(center, axis_deg, SLEEVE_W, SLEEVE_H)]
    for angle_deg in angles_deg:
        branch_dir_deg = axis_deg - angle_deg
        ang = np.radians(branch_dir_deg)
        dirv = np.array([np.cos(ang), np.sin(ang)])
        end = center + branch_len * dirv
        double_line(ax, center, end)

        branch_center = center + (BRANCH_SLEEVE_W / 2.0) * dirv
        sleeve_parts.append((branch_center, branch_dir_deg, BRANCH_SLEEVE_W, SLEEVE_H))

    fused_sleeve(ax, sleeve_parts)

    ax.set_title(title, fontsize=14)
    ax.set_xlim(-150 - push_back, trunk_len * 1.5)
    ax.set_ylim(-branch_len * 1.15, branch_len * 1.15)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    cases = [
        ("3 branches: 10/20/30 deg", [10.0, 20.0, 30.0]),
        ("3 branches: 10/45/80 deg", [10.0, 45.0, 80.0]),
        ("3 branches: mixed -10/20/45", [-10.0, 20.0, 45.0]),
    ]

    fig, axes = plt.subplots(1, len(cases), figsize=(8 * len(cases), 7))
    for ax, (title, angles) in zip(axes, cases):
        draw_multi_branch(ax, angles, title=title)

    fig.patch.set_facecolor("white")
    fig.tight_layout()

    out = os.path.join(OUT_DIR, "test_branch_multi.png")
    fig.savefig(out, dpi=150, facecolor="white")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
