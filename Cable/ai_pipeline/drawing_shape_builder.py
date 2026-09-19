"""
도면(PDF)에서 직접 읽은 좌표/각도/길이로 케이블 셸(외피)만 3D로 만드는
공용 빌더. A20016147에서 확립한 방식을 재사용한다:

  - 몸통(트렁크) 구간은 완전 직선(각지게, 라운드 없음)으로 잇는다.
  - 각 분기점에는 몸통보다 굵은 슬리브(커플러) 마디를 넣는다 - 도면을
    확대해서 실제로 확인되는 형태(수축튜브가 아니라 굵은 원통 부품).
  - 커넥터 표시, 치수 텍스트는 만들지 않는다(형상만).

사용법: TRUNK(몸통 각 정점 좌표 리스트)와 BRANCHES(분기점에서 뻗어나가는
가지들, 각도/길이)를 도면을 보고 채운 뒤 build_cable_shape()를 부른다.
"""
import numpy as np
import trimesh

from wire_builder import create_tube_mesh

THIN_RADIUS = 4.0
SLEEVE_RADIUS = 7.0
SHELL_COLOR = [20, 20, 20, 255]
SAMPLES_PER_LEG = 40
SLEEVE_SAMPLES = 12


def _leg(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    t = np.linspace(0, 1, n)[:, None]
    return a + t * (b - a)


def _to_3d(xy_list):
    return np.array([[p[0], 0.0, p[1]] for p in xy_list])


def _spindle_mesh(center_3d: np.ndarray, axis_3d: np.ndarray, length: float,
                   max_radius: float, sides: int = 24, samples: int = 20) -> trimesh.Trimesh:
    """분기점 마디를 방추형(양끝이 뾰족해지는 다이아몬드/럭비공 모양)으로
    만든다. 도면을 보면 분기점의 마디가 방향성 없는 구가 아니라, 몸통이
    그 지점에서 부드럽게 굵어졌다가 다시 가늘어지는 회전체이고 그 굵은
    지점에서 가지들이 뻗어 나간다 - 그 형태를 축 방향 반지름 프로파일로
    재현한다(반지름 = max_radius * sin(t*pi), t=0..1일 때 양끝 0).
    """
    axis = axis_3d / (np.linalg.norm(axis_3d) + 1e-9)
    up = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(axis, up)) > 0.9:
        up = np.array([1.0, 0.0, 0.0])
    side = np.cross(axis, up)
    side = side / (np.linalg.norm(side) + 1e-9)
    normal = np.cross(side, axis)

    t = np.linspace(0, 1, samples)
    radius_profile = max_radius * np.sin(t * np.pi)
    centers = center_3d + np.outer((t - 0.5) * length, axis)

    angles = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    cos_a, sin_a = np.cos(angles), np.sin(angles)

    verts = np.empty((samples, sides, 3))
    for i in range(samples):
        verts[i] = centers[i] + radius_profile[i] * (np.outer(cos_a, side) + np.outer(sin_a, normal))
    verts = verts.reshape(-1, 3)

    faces = []
    for i in range(samples - 1):
        for j in range(sides):
            j2 = (j + 1) % sides
            a = i * sides + j
            b = i * sides + j2
            c = (i + 1) * sides + j2
            d = (i + 1) * sides + j
            faces.append([a, b, c])
            faces.append([a, c, d])
    mesh = trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=False)
    mesh.visual.vertex_colors = SHELL_COLOR
    return mesh


def build_cable_shape(trunk_points, branches, out_path, scale=0.5,
                       sleeve_len=110.0, thin_radius=THIN_RADIUS,
                       sleeve_radius=SLEEVE_RADIUS):
    """
    trunk_points: 몸통 정점들의 2D 좌표 리스트(도면 mm 단위), 순서대로 직선
        연결됨. 중간 정점들이 분기점이 된다(마지막/처음 제외).
    branches: {trunk_point_index: [(end_point_2d, ...), ...]} - 그 몸통
        정점에서 갈라지는 가지들의 끝점 목록(2D, mm 단위). 여러 가지가
        같은 정점에서 갈라질 수 있다(리스트로).

    분기점 마디(슬리브)는 몸통 축 방향으로 길쭉하게 뻗은 튜브가 아니라,
    갈라지는 모든 선(몸통 양쪽 + 가지들)이 공통으로 만나는 지점 자체에
    놓인 방향성 없는 구(sphere)로 그린다 - "갈라지는 곳 전체에 마디가
    생겨야 한다"는 요구사항 반영. 몸통 튜브/가지 튜브는 전부 분기점
    좌표(center)까지 그대로 이어지고, 그 위에 구가 덧씌워지는 방식이라
    한쪽으로만 마디가 뻗어 보이는 문제가 없다.
    """
    trunk = [np.array(p, dtype=float) * scale for p in trunk_points]

    segments = []  # (2d centerline, radius)
    nodes = []  # 분기점 마디: (center_2d, axis_2d)

    # 몸통: 각 구간을 있는 그대로 이어서 그린다(마디는 별도로 방추형으로 얹는다).
    n = len(trunk)
    for i in range(n - 1):
        a, b = trunk[i], trunk[i + 1]
        segments.append((_leg(a, b, SAMPLES_PER_LEG), thin_radius))

    # 분기점마다 가지들을 그대로 잇고, 중심에 방추형 마디를 놓는다.
    for idx, branch_list in branches.items():
        center = trunk[idx]
        # 마디의 축 방향은 몸통이 들어오는 방향(양옆에 트렁크가 있으면 그
        # 진행 방향, 없으면 첫 가지 방향)을 쓴다.
        if idx > 0:
            axis = center - trunk[idx - 1]
        elif idx + 1 < n:
            axis = trunk[idx + 1] - center
        else:
            axis = np.array(branch_list[0], dtype=float) * scale - center
        nodes.append((center, axis))
        for end_point in branch_list:
            end = np.array(end_point, dtype=float) * scale
            segments.append((_leg(center, end, SAMPLES_PER_LEG), thin_radius))

    tubes = []
    for centerline_2d, radius in segments:
        tube = create_tube_mesh(_to_3d(centerline_2d), radius, SHELL_COLOR, sides=24)
        if len(tube.vertices) > 0:
            tubes.append(tube)

    for center_2d, axis_2d in nodes:
        center_3d = _to_3d([center_2d])[0]
        axis_3d = _to_3d([axis_2d])[0] - _to_3d([(0.0, 0.0)])[0]
        spindle = _spindle_mesh(center_3d, axis_3d, length=sleeve_radius * 3.2,
                                 max_radius=sleeve_radius)
        tubes.append(spindle)

    mesh = trimesh.util.concatenate(tubes) if tubes else trimesh.Trimesh()
    print(f"vertices: {len(mesh.vertices)}")
    mesh.export(out_path)
    print(f"-> {out_path}")
    return mesh
