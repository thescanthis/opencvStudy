"""
케이블 형태(shape) 원형 4종의 중심선(centerline) 생성기.

실제 케이블의 구체적인 굽은 형태는 제각각이지만, 구조적으로는 다음
네 가지 원형으로 환원된다:
  1. 직선(straight)   - 커넥터 A -> B, 곧게
  2. 커브(curved)      - 커넥터 A -> B, 부드럽게 휨 (S자/아치 등)
  3. 꺾임(bent)         - 커넥터 A -> B, 특정 지점에서 각지게 절곡 (L자 등)
  4. 분기(branching)    - 커넥터 A -> N개 커넥터, 한 지점에서 갈라짐

이 모듈은 셸/배선을 직접 만들지 않고, "케이블이 공간에서 어떤 경로를
지나가는가"만 계산한다. 실제 메쉬 생성은 wire_builder.create_tube_mesh와
CircuitWireBuilder/BranchingCableBuilder가 담당한다 - 이 모듈은 그것들이
쓸 매끈한 중심선(centerline_3d)을 공급하는 역할만 한다.

AI(사진 기반 depth/mask) 대신 순수 기하학적 곡선을 쓰는 이유: 실물 사진의
울퉁불퉁함이 배선 다발을 겹치고 지저분해 보이게 만들었기 때문에, "실제
케이블과 닮은 형태"이되 "매끈하고 정갈한" 곡선이 목표에 더 맞는다.
"""
from typing import List, Tuple
import numpy as np


def straight_centerline(length: float = 200.0, n: int = 60) -> np.ndarray:
    """직선 케이블. X축을 따라 곧게 뻗는다."""
    x = np.linspace(0, length, n)
    y = np.zeros(n)
    z = np.zeros(n)
    return np.stack([x, y, z], axis=1)


def curved_centerline(length: float = 200.0, amplitude: float = 40.0,
                       waves: float = 1.0, n: int = 60) -> np.ndarray:
    """부드럽게 휘는 케이블 (사인 곡선 기반 아치/S자).

    amplitude: 휘는 정도. waves: 몇 번 굽이치는지 (0.5=아치 하나, 1.0=S자).
    """
    t = np.linspace(0, 1, n)
    x = t * length
    y = amplitude * np.sin(t * np.pi * 2 * waves)
    z = np.zeros(n)
    return np.stack([x, y, z], axis=1)


def bent_centerline(seg1_length: float = 100.0, seg2_length: float = 100.0,
                     bend_angle_deg: float = 90.0, corner_radius: float = 15.0,
                     n_per_seg: int = 30, n_corner: int = 15) -> np.ndarray:
    """한 지점에서 각지게 꺾이는 케이블 (L자 등). 2D(XY 평면)에서
    직선-원호-직선을 이어서 만든다.

    완전히 뾰족하게 꺾으면(직각) 튜브 스윕이 왜곡되므로, 모서리에
    corner_radius만큼의 둥근 필렛(fillet)을 넣어 매끄럽게 연결한다.
    bend_angle_deg: 0=안 꺾임(직선), 90=L자, 180=U턴에 가까움.
    """
    angle_rad = np.radians(bend_angle_deg)
    r = min(corner_radius, seg1_length * 0.4, seg2_length * 0.4)

    # 1구간: (0,0) -> (seg1_length - r, 0), +X 방향으로 진행하다 코너 반경만큼 못 미쳐 멈춤
    d1 = seg1_length - r
    t1 = np.linspace(0, d1, n_per_seg)
    p1 = np.stack([t1, np.zeros_like(t1)], axis=1)

    # 코너: 중심 (d1, r), 반경 r인 원호. +X 방향(접선)으로 진입해
    # angle_rad만큼 돌아서 (cos(angle_rad), sin(angle_rad)) 방향(접선)으로 진출.
    # 중심 기준 진입점의 각도는 -90도(= 원의 아래쪽 점이 진입점).
    center = np.array([d1, r])
    start_vec_angle = -np.pi / 2
    corner_pts = _arc_points(center, r, start_vec_angle, angle_rad, n_corner)

    corner_end = corner_pts[-1]
    exit_dir = np.array([np.cos(angle_rad), np.sin(angle_rad)])

    # 2구간: 코너 끝에서 exit_dir 방향으로 seg2_length만큼
    t2 = np.linspace(0, seg2_length - r, max(2, n_per_seg))
    p2 = corner_end + np.outer(t2, exit_dir)

    xy = np.concatenate([p1, corner_pts, p2], axis=0)
    z = np.zeros(len(xy))
    return np.stack([xy[:, 0], xy[:, 1], z], axis=1)


def _arc_points(center: np.ndarray, radius: float, start_angle: float,
                 sweep_angle: float, n: int) -> np.ndarray:
    """center를 중심으로, (반경*[cos,sin](start_angle))에서 시작해
    sweep_angle만큼 회전한 원호 위의 점들을 반환한다."""
    angles = np.linspace(start_angle, start_angle + sweep_angle, n)
    return center + radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)


def fork_branch_centerline(trunk_length: float = 100.0, branch_length: float = 80.0,
                            branch_angle_deg: float = 30.0, n_trunk: int = 30,
                            n_branch: int = 30) -> np.ndarray:
    """트렁크에서 한 가지가 "포크(Y자)"처럼 부드럽게 갈라지는 경로.

    사용자가 제공한 참조 GLB(Cable_1-2.glb)의 형태를 관찰해 확정한 방식:
    bent_centerline(직선+원호, 각지게 꺾임)과 달리, 분기점에서 접선이
    트렁크와 완전히 같은 방향으로 시작해서(꺾이지 않고) 서서히 옆으로
    휘어나간다. curved_split_demo.py에서 확인된
    side_amount = (1-cos(t*pi))/2 공식(t=0에서 기울기 0)을 그대로 쓴다 -
    분기가 각진 코너가 아니라 매끈하게 "벌어지는" 형태로 보이는 핵심 이유.

    반환값은 트렁크(0~trunk_length) + 가지(분기점 이후) 전체를 이은 2D(XY,
    +X가 트렁크 진행 방향) 경로다. branch_angle_deg는 가지 끝에서의 대략적인
    벌어짐 각도(코너가 없어 bent_centerline의 bend_angle_deg와 완전히
    같지는 않다 - 끝점 근방의 실효 각도).
    """
    trunk_x = np.linspace(0, trunk_length, n_trunk)
    trunk = np.stack([trunk_x, np.zeros_like(trunk_x)], axis=1)

    junction = trunk[-1]
    tangent = trunk[-1] - trunk[-2]
    tangent = tangent / (np.linalg.norm(tangent) + 1e-9)
    side = np.array([-tangent[1], tangent[0]])  # 2D에서 90도 회전 = 좌측 법선

    angle_rad = np.radians(branch_angle_deg)
    # 목표: 가지 끝에서 진행 방향이 트렁크 대비 angle_rad만큼 기울어지도록,
    # side 성분의 최종 크기를 branch_length*tan(angle_rad)로 맞춘다.
    side_extent = branch_length * np.tan(angle_rad)

    t = np.linspace(0, 1, n_branch)
    side_amount = (1 - np.cos(t * np.pi)) / 2  # t=0 기울기 0 -> 트렁크 접선과 매끄럽게 연결
    branch = junction + np.outer(t * branch_length, tangent) + np.outer(side_amount * side_extent, side)

    xy = np.concatenate([trunk, branch[1:]], axis=0)
    z = np.zeros(len(xy))
    return np.stack([xy[:, 0], xy[:, 1], z], axis=1)


def branch_directions(n_branches: int, fan_angle_deg: float = 150.0) -> List[np.ndarray]:
    """분기점에서 N개 가지가 뻗어나갈 단위 방향 벡터(XY 평면)를 균등 부채꼴로 계산한다."""
    if n_branches == 1:
        return [np.array([1.0, 0.0, 0.0])]
    fan = np.radians(fan_angle_deg)
    dirs = []
    for i in range(n_branches):
        a = -fan / 2 + fan * (i / (n_branches - 1))
        dirs.append(np.array([np.cos(a), np.sin(a), 0.0]))
    return dirs


if __name__ == "__main__":
    s = straight_centerline()
    c = curved_centerline()
    b = bent_centerline(bend_angle_deg=90.0)
    print("straight:", s.shape, s[0], s[-1])
    print("curved:", c.shape, c[0], c[-1])
    print("bent:", b.shape, b[0], b[-1])
