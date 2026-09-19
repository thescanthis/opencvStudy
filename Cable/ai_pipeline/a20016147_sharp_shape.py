"""
A20016147 도면(Cable.pdf)을 OpenCV로 직접 분석해 확인한 실제 형상을 3D
셸(외피)로 만든다.

OpenCV 허프 변환 + 분기점 확대 관찰로 확정한 사실:
  - 몸통(P1 -> 분기점1 -> 분기점2 -> P4)은 처음부터 끝까지 완전한 수평
    직선이다(검출된 수평선: y=2181px, x=1127~5765px, 흔들림 없음).
  - P2, P3 두 가지는 몸통에서 정확히 "45.00도"로 아래로 꺾여 갈라진다
    (검출된 사선 각도 정확히 45.00도).
  - 분기점(J1, J2) 확대본을 보면 몸통이 그 지점에서 굵은 원통형 슬리브
    (커플러 부품)로 확장되고, 그 슬리브에서 가지가 갈라져 나간 뒤 몸통은
    다시 가는 두께로 돌아간다(사용자가 "Y로 나뉠 때 뭔가 끼워진 느낌"이라고
    표현한 부분 - 실제로는 수축튜브가 아니라 굵은 슬리브 부품).

길이는 도면 치수(P1->J1 1800, J1->J2 1300, J2->P4 370, J2->P3 1250mm)를
그대로 비율로 쓴다.
"""
import os
import numpy as np
import trimesh

from wire_builder import create_tube_mesh

OUT = r"C:\AIND_Library\Git_PredicTest\x64_Cami_Lib\PSCD1500\PSCD1500\Cable\ai_pipeline\test_output\6_A20016147_sharp_shape.glb"

THIN_RADIUS = 4.0    # 가는 케이블 외피 반지름
SLEEVE_RADIUS = 7.0  # 분기점 슬리브(커플러) 반지름 - 도면에서 확인된 확장 비율
SHELL_COLOR = [20, 20, 20, 255]
SAMPLES_PER_LEG = 40
SLEEVE_SAMPLES = 12
BRANCH_ANGLE_DEG = 45.0  # OpenCV로 검출된 정확한 분기 각도
SLEEVE_LEN = 110.0       # 슬리브 구간 길이(스케일 적용 전, 도면 비율감으로 추정)


def _leg(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    t = np.linspace(0, 1, n)[:, None]
    return a + t * (b - a)


def _to_3d(xy_list):
    """(x,y) 2D 점열 -> (x, 0, y) 3D 점열 (Z가 도면의 세로축, 아래가 -Z)."""
    return np.array([[p[0], 0.0, p[1]] for p in xy_list])


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    SCALE = 0.5
    P1 = np.array([0.0, 0.0])
    J1 = P1 + np.array([1800.0, 0.0])   # P1 -> 분기점1 (수평)
    J2 = J1 + np.array([1300.0, 0.0])   # 분기점1 -> 분기점2 (수평, 몸통 계속)
    P4 = J2 + np.array([370.0, 0.0])    # 분기점2 -> P4 (수평, 몸통 끝)

    ang = np.radians(-BRANCH_ANGLE_DEG)
    branch_dir = np.array([np.cos(ang), np.sin(ang)])
    P2 = J1 + branch_dir * 1250.0       # 분기점1 -> P2 (45도 아래)
    P3 = J2 + branch_dir * 1250.0       # 분기점2 -> P3 (45도 아래)

    for p in (P1, J1, P2, J2, P4, P3):
        p *= SCALE
    sleeve_len = SLEEVE_LEN * SCALE

    # 각 분기점의 슬리브 구간(±sleeve_len/2)을 몸통 진행 방향 기준으로 계산
    trunk_dir = np.array([1.0, 0.0])
    j1_s0, j1_s1 = J1 - trunk_dir * sleeve_len / 2, J1 + trunk_dir * sleeve_len / 2
    j2_s0, j2_s1 = J2 - trunk_dir * sleeve_len / 2, J2 + trunk_dir * sleeve_len / 2

    segments = []  # (centerline_2d, radius, samples)

    # 몸통: P1 -> J1 슬리브 진입 전 (가는 두께)
    segments.append((_leg(P1, j1_s0, SAMPLES_PER_LEG), THIN_RADIUS))
    # J1 슬리브 구간 (굵은 두께)
    segments.append((_leg(j1_s0, j1_s1, SLEEVE_SAMPLES), SLEEVE_RADIUS))
    # 몸통: J1 슬리브 이후 -> J2 슬리브 진입 전 (가는 두께)
    segments.append((_leg(j1_s1, j2_s0, SAMPLES_PER_LEG), THIN_RADIUS))
    # J2 슬리브 구간 (굵은 두께)
    segments.append((_leg(j2_s0, j2_s1, SLEEVE_SAMPLES), SLEEVE_RADIUS))
    # 몸통: J2 슬리브 이후 -> P4 (가는 두께)
    segments.append((_leg(j2_s1, P4, SAMPLES_PER_LEG), THIN_RADIUS))

    # 가지: 각 슬리브 중심(J1, J2)에서 갈라져 나가는 45도 가지(가는 두께)
    segments.append((_leg(J1, P2, SAMPLES_PER_LEG), THIN_RADIUS))
    segments.append((_leg(J2, P3, SAMPLES_PER_LEG), THIN_RADIUS))

    tubes = []
    for centerline_2d, radius in segments:
        tube = create_tube_mesh(_to_3d(centerline_2d), radius, SHELL_COLOR, sides=24)
        if len(tube.vertices) > 0:
            tubes.append(tube)

    mesh = trimesh.util.concatenate(tubes) if tubes else trimesh.Trimesh()
    print(f"vertices: {len(mesh.vertices)}")
    mesh.export(OUT)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
