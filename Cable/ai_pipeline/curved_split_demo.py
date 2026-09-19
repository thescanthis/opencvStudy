"""
curved_test.glb(2-커넥터 S자 커브 케이블)를 기반으로, 배선을 9/8로 단순히
나눠 커브 중간 지점부터 두 그룹이 서로 다른 파이프(가지)로 갈라지는 데모.

배선을 어떤 기준(핀 번호, 뱅크 등)으로 나눌지는 지금 정하지 않고, TCL의
o@ 목록 순서대로 앞 9개 / 뒤 8개로 단순 분할한다 - "다르게 저장해서 보여줘"
라는 요청에 따라 기준 없이 빠르게 확인하기 위한 용도.

구조:
  - 트렁크: curved_centerline()의 앞쪽 절반(0.0~0.5 구간)
  - 가지 A(원래 목적지, J26): 트렁크 뒤에 이어지는 S자 뒷쪽 절반 그대로
  - 가지 B(새 목적지, J26B): 트렁크 끝에서 다른 방향으로 새로 뻗는 곡선
  - 9가닥은 가지 A로, 8가닥은 가지 B로 보낸다.
"""
import os
import numpy as np
import trimesh

from cable_shapes import curved_centerline, bent_centerline
from wire_builder import create_tube_mesh, _centerline_frames, _compute_curvature_shrink
from tcl_parser import parse_tcl
from connector_layout import layout_pins

TCL_PATH = r"C:\AIND_Library\Git_PredicTest\x64_Cami_Lib\PSCD1500\PSCD1500\Cable\TestData\50073721_5995-37-501-9359\cable.tcl"
OUT_PATH = r"C:\AIND_Library\Git_PredicTest\x64_Cami_Lib\PSCD1500\PSCD1500\Cable\ai_pipeline\test_output\curved_split_demo.glb"

SHELL_RADIUS = 10.0
SHELL_COLOR = [100, 149, 237, 120]
WIRE_RADIUS = max(0.5, SHELL_RADIUS * 0.05)
CONNECTOR_RADIUS = SHELL_RADIUS * 0.65 - WIRE_RADIUS
CONNECTOR_TYPES = {"J14": "MS3475 W14-18P", "J26": "MS3475 W14-18P"}

WIRE_COLOR_A = [255, 69, 0, 255]     # 원래 경로로 가는 9가닥 (주황)
WIRE_COLOR_B = [30, 144, 255, 255]   # 새 파이프로 갈라지는 8가닥 (파랑 계열로 구분)


def _offset_path(centerline, frames, shrink, uv_from, uv_to):
    n = len(centerline)
    t = np.linspace(0, 1, n)
    smooth_t = t * t * (3 - 2 * t)
    u = (uv_from[0] * (1 - smooth_t) + uv_to[0] * smooth_t) * shrink
    v = (uv_from[1] * (1 - smooth_t) + uv_to[1] * smooth_t) * shrink
    path = np.empty_like(centerline)
    for i in range(n):
        normal, binormal = frames[i]
        path[i] = centerline[i] + u[i] * normal + v[i] * binormal
    return path


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    topo = parse_tcl(TCL_PATH)
    wires = topo.wires
    print(f"Total wires: {len(wires)}")
    wires_a = wires[:9]   # 원래 목적지로 가는 9가닥
    wires_b = wires[9:]   # 새 파이프로 갈라지는 나머지 (17개 중 8개)
    print(f"  Group A (stay on original path): {len(wires_a)}")
    print(f"  Group B (split to new pipe): {len(wires_b)}")

    # 전체 S자 커브를 만들고, 절반 지점(중간)을 분기점으로 삼는다.
    full_curve = curved_centerline(length=220.0, amplitude=45.0, waves=1.0, n=80)
    split_idx = len(full_curve) // 2
    trunk = full_curve[:split_idx + 1]
    branch_a = full_curve[split_idx:]  # 원래 경로 뒷부분 그대로 이어감

    # 가지 B: 분기점에서 다른 방향으로 새로 뻗는 곡선.
    # branch_a(주황)는 curved_centerline의 순수 사인 곡선이라 다발 전체가
    # 하나의 통일된 형태로 보인다(사용자가 "자연스럽다"고 확인). 이전에
    # (직진+사인 혼합)과 (직선+원호, bent_centerline) 둘 다 시도했지만 각각
    # "가닥마다 다르게 보임" / "너무 꺾인 직선처럼 보임" 문제가 있었다.
    # 그래서 branch_b도 branch_a와 동일하게 순수 사인 곡선으로 만들어
    # 형태의 성격을 통일한다 - 진행 방향(tangent)만 다른 각도로 돌려서
    # "새 방향으로 휘어나가는 대칭 커브"를 만든다.
    junction = trunk[-1]
    tangent = trunk[-1] - trunk[-2]
    tangent = tangent / (np.linalg.norm(tangent) + 1e-9)
    up = np.array([0.0, 1.0, 0.0])
    side = np.cross(tangent, up)
    if np.linalg.norm(side) < 1e-6:
        side = np.array([0.0, 0.0, 1.0])
    side = side / np.linalg.norm(side)

    branch_b_length = 100.0
    branch_b_amplitude = 45.0
    n_branch_b = len(full_curve) - split_idx
    t_b = np.linspace(0, 1, n_branch_b)
    # sin(t*pi)는 t=0에서 기울기(cos(0)=1)가 최대라, 분기점에서 트렁크
    # 접선과 이 곡선의 시작 접선이 급격히 어긋나 "직각으로 꺾이는" 것처럼
    # 보였다(사용자 피드백). side 성분을 (1-cos(t*pi))/2 형태로 바꾸면
    # t=0에서 기울기가 0이라 트렁크 접선을 그대로 이어받아 출발하고,
    # 뒤로 갈수록 서서히 옆으로 휘어나간다 - 갈라지는 시작이 매끄럽다.
    side_amount = (1 - np.cos(t_b * np.pi)) / 2  # 0 -> 1, 시작 기울기 0
    branch_b = junction + np.outer(t_b * branch_b_length, tangent) \
        + np.outer(side_amount * branch_b_amplitude, side)

    # 셸: 트렁크 + 가지 A + 가지 B (가지 A/B를 색으로 구분해서 렌더링 확인이 쉽게)
    SHELL_COLOR_A = [100, 149, 237, 120]   # 기존 파란색: 원래 경로
    SHELL_COLOR_B = [147, 112, 219, 120]   # 보라색: 새로 갈라지는 경로
    shell_trunk = create_tube_mesh(trunk, SHELL_RADIUS, SHELL_COLOR, sides=16)
    shell_a = create_tube_mesh(branch_a, SHELL_RADIUS, SHELL_COLOR_A, sides=16)
    shell_b = create_tube_mesh(branch_b, SHELL_RADIUS, SHELL_COLOR_B, sides=16)

    # 커넥터 UV
    connectors = topo.connectors  # ["J14", "J26"]
    conn_from, conn_to = connectors[0], connectors[1]
    pins_from = [p.pin_name for p in topo.pins_of(conn_from)]
    pins_to = [p.pin_name for p in topo.pins_of(conn_to)]
    layout_from = layout_pins(pins_from, radius=CONNECTOR_RADIUS, connector_type=CONNECTOR_TYPES.get(conn_from))
    layout_to = layout_pins(pins_to, radius=CONNECTOR_RADIUS, connector_type=CONNECTOR_TYPES.get(conn_to))
    pin_uv = {}
    for pin_name, uv in layout_from.items():
        pin_uv[f"{conn_from}-{pin_name}"] = uv
    for pin_name, uv in layout_to.items():
        pin_uv[f"{conn_to}-{pin_name}"] = uv

    trunk_frames = _centerline_frames(trunk)
    trunk_shrink = _compute_curvature_shrink(trunk, SHELL_RADIUS)
    a_frames = _centerline_frames(branch_a)
    a_shrink = _compute_curvature_shrink(branch_a, SHELL_RADIUS)
    b_frames = _centerline_frames(branch_b)
    b_shrink = _compute_curvature_shrink(branch_b, SHELL_RADIUS)

    meshes = [shell_trunk, shell_a, shell_b]

    def build_wire(wire, branch_centerline, branch_frames, branch_shrink, color):
        uv_from = pin_uv.get(wire.from_pin)
        uv_to = pin_uv.get(wire.to_pin)
        if uv_from is None or uv_to is None:
            return None
        # 트렁크 구간: 시작 핀 위치를 유지한 채 공통 트렁크를 지남
        leg1 = _offset_path(trunk, trunk_frames, trunk_shrink, uv_from, (0.0, 0.0))
        # 가지 구간: 분기점에서 목적지 핀 위치로 수렴
        leg2 = _offset_path(branch_centerline, branch_frames, branch_shrink, (0.0, 0.0), uv_to)
        path = np.concatenate([leg1, leg2[1:]], axis=0)
        mesh = create_tube_mesh(path, WIRE_RADIUS, color)
        return mesh if len(mesh.vertices) > 0 else None

    for wire in wires_a:
        m = build_wire(wire, branch_a, a_frames, a_shrink, WIRE_COLOR_A)
        if m is not None:
            meshes.append(m)

    for wire in wires_b:
        m = build_wire(wire, branch_b, b_frames, b_shrink, WIRE_COLOR_B)
        if m is not None:
            meshes.append(m)

    combined = trimesh.util.concatenate(meshes)
    combined.export(OUT_PATH)
    print(f"Exported {len(combined.vertices)} vertices -> {OUT_PATH}")


if __name__ == "__main__":
    main()
