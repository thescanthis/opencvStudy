"""
TCL 배선 토폴로지 + 커넥터 UV 레이아웃을 이용해 실제 배선 가닥을 3D 튜브로
생성한다.

기존(예전) geometry.py의 CircuitWireBuilder는 TCL을 파싱하지 않고 23가닥을
임의의 나선형으로 흩뿌리기만 했다. 이 모듈이 그 자리를 대체한다: 각 와이어는
반드시 TCL의 o@ 라인이 말하는 실제 From/To 핀 쌍을 따라가며, 그 핀의 위치는
connector_layout의 UV 좌표로 정해진다 - 임의의 값이 섞이지 않는다.

와이어 경로는 케이블 중심선(centerline_3d) 전체를 따라간다 - 양 끝 핀 사이를
직선으로 관통시키면 케이블이 휘어져 있을 때 배선이 외피를 뚫고 나가 보이므로,
중심선의 각 샘플 지점마다 로컬 단면 좌표계(normal/binormal)에서의 오프셋을
시작 핀 위치 -> 끝 핀 위치로 부드럽게 보간해 더하는 방식을 쓴다.
"""
from typing import List, Dict, Tuple
import numpy as np
import trimesh

from tcl_parser import CableTopology, parse_tcl
from connector_layout import layout_pins


def _local_frame(tangent: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """주어진 접선 방향에 수직인 (normal, binormal) 로컬 좌표축을 만든다."""
    up = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(tangent, up)) > 0.99:
        up = np.array([1.0, 0.0, 0.0])
    binormal = np.cross(tangent, up)
    binormal = binormal / (np.linalg.norm(binormal) + 1e-9)
    normal = np.cross(binormal, tangent)
    normal = normal / (np.linalg.norm(normal) + 1e-9)
    return normal, binormal


def _centerline_frames(centerline_3d: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """centerline의 각 점에 대한 (normal, binormal) 로컬 프레임을 계산한다.

    인접한 프레임끼리 급격히 뒤틀리지 않도록(twist-free), 이전 프레임의
    normal을 다음 지점의 접선에 투영해 이어 붙이는 방식(간이 parallel
    transport)을 사용한다.
    """
    n = len(centerline_3d)
    tangents = np.gradient(centerline_3d, axis=0)
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents = np.divide(tangents, norms, out=np.zeros_like(tangents), where=norms > 1e-9)

    frames: List[Tuple[np.ndarray, np.ndarray]] = []
    prev_normal, prev_binormal = _local_frame(tangents[0])
    frames.append((prev_normal, prev_binormal))

    for i in range(1, n):
        t = tangents[i]
        # 이전 normal을 현재 접선에 수직인 평면으로 투영해 twist를 최소화
        proj_normal = prev_normal - np.dot(prev_normal, t) * t
        pn_norm = np.linalg.norm(proj_normal)
        if pn_norm < 1e-6:
            normal, binormal = _local_frame(t)
        else:
            normal = proj_normal / pn_norm
            binormal = np.cross(t, normal)
            b_norm = np.linalg.norm(binormal)
            binormal = binormal / b_norm if b_norm > 1e-9 else binormal
        frames.append((normal, binormal))
        prev_normal, prev_binormal = normal, binormal

    return frames


def _build_pin_uv(
    topo: CableTopology,
    connector_radius: float,
    connector_types: Dict[str, str] = None,
) -> Dict[str, Tuple[float, float]]:
    """각 핀의 (u, v) 로컬 오프셋(월드 좌표 아님)을 계산한다.

    connector_types에 {"J14": "MS3475 W14-18P", ...}처럼 커넥터별 규격명이
    주어지면 실제 도면 기반 배치를, 없으면 균등 원형 근사를 사용한다.
    """
    connectors = topo.connectors
    uv: Dict[str, Tuple[float, float]] = {}
    if len(connectors) < 2:
        return uv

    connector_types = connector_types or {}
    conn_a, conn_b = connectors[0], connectors[1]
    pins_a = [p.pin_name for p in topo.pins_of(conn_a)]
    pins_b = [p.pin_name for p in topo.pins_of(conn_b)]

    layout_a = layout_pins(pins_a, radius=connector_radius, connector_type=connector_types.get(conn_a))
    layout_b = layout_pins(pins_b, radius=connector_radius, connector_type=connector_types.get(conn_b))

    for pin_name, uv_val in layout_a.items():
        uv[f"{conn_a}-{pin_name}"] = uv_val
    for pin_name, uv_val in layout_b.items():
        uv[f"{conn_b}-{pin_name}"] = uv_val

    return uv


def _compute_curvature_shrink(centerline_3d: np.ndarray, shell_radius: float) -> np.ndarray:
    """각 지점의 로컬 곡률에 따라 배선 오프셋을 얼마나 줄여야 하는지(0~1) 계산한다.

    케이블이 급하게 휘는 구간에서는 셸 자체의 안쪽 곡률 반경이 shell_radius보다
    작아질 수 있으므로, 그 구간에서 배선 오프셋을 셸 반경 이하로 강제로 줄여
    배선이 외피를 뚫고 나가지 않도록 한다. 완만한 구간(직선에 가까움)에서는
    축소하지 않는다(shrink=1.0).
    """
    n = len(centerline_3d)
    if n < 3:
        return np.ones(n)

    # 2차 차분으로 각 지점의 근사 곡률 반경을 추정
    d1 = np.gradient(centerline_3d, axis=0)
    d2 = np.gradient(d1, axis=0)
    speed = np.linalg.norm(d1, axis=1)
    cross = np.cross(d1, d2)
    cross_norm = np.linalg.norm(cross, axis=1) if cross.ndim == 2 else np.abs(cross)
    curvature = np.divide(cross_norm, speed ** 3 + 1e-9)
    radius_of_curvature = np.divide(1.0, curvature + 1e-9)

    # 곡률 반경이 (셸 반경 + 배선 다발 여유폭)보다 작아지는 구간에서
    # 그 비율만큼 오프셋을 축소. 안전 마진으로 셸 반경의 1.5배를 기준점으로 삼는다.
    safe_radius = shell_radius * 1.5
    shrink = np.clip(radius_of_curvature / safe_radius, 0.15, 1.0)
    return shrink


def _sample_wire_path_along_centerline(
    uv_from: Tuple[float, float],
    uv_to: Tuple[float, float],
    centerline_3d: np.ndarray,
    frames: List[Tuple[np.ndarray, np.ndarray]],
    curvature_shrink: np.ndarray,
) -> np.ndarray:
    """centerline 전체를 따라가며, 로컬 (u, v) 오프셋을 uv_from -> uv_to로
    부드럽게 보간해 더한 3D 경로를 만든다.

    이렇게 하면 케이블이 휘어져 있어도 배선이 외피를 뚫지 않고 곡률을
    그대로 따라가며, 양 끝에서는 정확히 커넥터 핀 위치로 수렴한다.
    급격히 휘는 구간에서는 curvature_shrink로 오프셋을 줄여 셸을 뚫지 않게 한다.
    """
    n = len(centerline_3d)
    t = np.linspace(0, 1, n)
    # smoothstep으로 부드럽게 보간 (양 끝단에서 속도가 0에 가까워 자연스러움)
    smooth_t = t * t * (3 - 2 * t)

    u = (uv_from[0] * (1 - smooth_t) + uv_to[0] * smooth_t) * curvature_shrink
    v = (uv_from[1] * (1 - smooth_t) + uv_to[1] * smooth_t) * curvature_shrink

    path = np.empty_like(centerline_3d)
    for i in range(n):
        normal, binormal = frames[i]
        path[i] = centerline_3d[i] + u[i] * normal + v[i] * binormal

    return path


def create_tube_mesh(path_3d: np.ndarray, radius: float, color: list, sides: int = 8) -> trimesh.Trimesh:
    """path_3d를 따라가는 튜브(원통) 메쉬를 직접 스위핑해서 만든다.

    이전에는 trimesh.creation.sweep_polygon(shapely 기반)을 썼지만, 경로의
    접선(tangent) 방향이 인접 지점 사이에서 급격히 바뀌는 구간(분기 케이블의
    junction 근처 등)에서 프레넷 프레임 계산이 0-나눗셈을 일으켜 좌표가
    수백 배로 폭발하는 버그를 재현/확인했다. _centerline_frames가 쓰는
    parallel-transport 프레임(급격한 뒤틀림을 스스로 방지)을 그대로 재사용해
    각 지점에 원형 단면 링을 배치하고 인접 링을 사각면으로 잇는 방식으로
    직접 스위핑하면 이 문제가 생기지 않는다.
    """
    try:
        n = len(path_3d)
        if n < 2:
            return trimesh.Trimesh()

        frames = _centerline_frames(path_3d)
        angles = np.linspace(0, 2 * np.pi, sides, endpoint=False)
        cos_a, sin_a = np.cos(angles), np.sin(angles)

        # 각 경로 지점마다 원형 단면의 정점들을 생성
        ring_vertices = np.empty((n, sides, 3), dtype=np.float64)
        for i in range(n):
            normal, binormal = frames[i]
            center = path_3d[i]
            ring_vertices[i] = center + radius * (np.outer(cos_a, normal) + np.outer(sin_a, binormal))

        vertices = ring_vertices.reshape(-1, 3)

        faces = []
        for i in range(n - 1):
            base0 = i * sides
            base1 = (i + 1) * sides
            for s in range(sides):
                s_next = (s + 1) % sides
                a, b = base0 + s, base0 + s_next
                c, d = base1 + s, base1 + s_next
                faces.append([a, b, d])
                faces.append([a, d, c])

        mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(faces), process=False)
        mesh.visual.vertex_colors = color
        return mesh
    except Exception as e:
        print(f"[Tube Creation Error] {e}")
        return trimesh.Trimesh()


class CircuitWireBuilder:
    """TCL의 실제 배선(o@)을 커넥터 UV 레이아웃 + 케이블 중심선을 따라 3D 튜브로 만든다."""

    def __init__(self, tcl_path: str, bounds_min, bounds_max, shell_radius: float = None,
                 wire_radius: float = None, connector_radius: float = None, wire_color: list = None,
                 connector_types: Dict[str, str] = None):
        self.tcl_path = tcl_path
        self.bounds_min = np.array(bounds_min, dtype=np.float32)
        self.bounds_max = np.array(bounds_max, dtype=np.float32)
        span = np.linalg.norm(self.bounds_max - self.bounds_min)
        # shell_radius가 주어지지 않으면 bounds에서 대략 역산 (span의 5% 근사)
        self.shell_radius = shell_radius if shell_radius is not None else max(1.0, span * 0.05)
        self.wire_radius = wire_radius if wire_radius is not None else max(0.5, span * 0.004)
        # 배선 다발 반경(+ 배선 자체 두께)은 셸 반경보다 작아야 하지만,
        # 너무 작으면 셸 정중앙에 파묻혀 화면상 안 보인다(육안 확인함).
        # "안에 선이 지나가는 게 보이는" 느낌을 살리기 위해 0.65배로 셸에
        # 가깝게 붙이고, 급커브 구간은 _compute_curvature_shrink가 자동으로
        # 오프셋을 줄여 뚫리지 않게 한다.
        self.connector_radius = connector_radius if connector_radius is not None else max(
            0.5, self.shell_radius * 0.65 - self.wire_radius
        )
        self.wire_color = wire_color if wire_color is not None else [255, 69, 0, 255]
        # 예: {"J14": "MS3475 W14-18P", "J26": "MS3475 W14-18P"} - 알려진 규격이면
        # connector_specs의 실제 도면 기반 배치를 쓰고, 없으면 균등 원형 근사.
        self.connector_types = connector_types or {}

    def build(self, centerline_3d: np.ndarray) -> trimesh.Trimesh:
        topo = parse_tcl(self.tcl_path)
        if not topo.wires:
            print(f"  [WireBuilder] No o@ (actual wire) entries found in {self.tcl_path}; skipping wires.")
            return trimesh.Trimesh()

        if len(centerline_3d) < 2 or len(topo.connectors) < 2:
            print("  [WireBuilder] Need >=2 connectors and a valid centerline; skipping wires.")
            return trimesh.Trimesh()

        pin_uv = _build_pin_uv(topo, self.connector_radius, self.connector_types)
        frames = _centerline_frames(centerline_3d)
        # 배선 다발 최대 오프셋(connector_radius + wire_radius)이 셸 반경을
        # 넘지 않도록, 급커브 구간에서 오프셋을 줄이는 축소 계수를 계산
        curvature_shrink = _compute_curvature_shrink(centerline_3d, self.shell_radius)

        meshes = []
        skipped = 0
        for wire in topo.wires:
            uv_from = pin_uv.get(wire.from_pin)
            uv_to = pin_uv.get(wire.to_pin)
            if uv_from is None or uv_to is None:
                skipped += 1
                continue
            path = _sample_wire_path_along_centerline(uv_from, uv_to, centerline_3d, frames, curvature_shrink)
            mesh = create_tube_mesh(path, self.wire_radius, self.wire_color)
            if len(mesh.vertices) > 0:
                meshes.append(mesh)

        if skipped:
            print(f"  [WireBuilder] Skipped {skipped}/{len(topo.wires)} wires (pin position not resolved).")

        if not meshes:
            return trimesh.Trimesh()

        return trimesh.util.concatenate(meshes)


if __name__ == "__main__":
    tcl_path = r"C:\AIND_Library\Git_PredicTest\x64_Cami_Lib\PSCD1500\PSCD1500\Cable\TestData\50073721_5995-37-501-9359\cable.tcl"
    # 휘어진 centerline으로 테스트 (직선이면 곡률 반영 여부를 확인할 수 없음)
    t = np.linspace(0, 1, 50)
    x = t * 200 - 100
    y = 30 * np.sin(t * np.pi)  # 위로 볼록하게 휘어진 형태
    z = np.zeros_like(t)
    centerline = np.stack([x, y, z], axis=1)

    builder = CircuitWireBuilder(tcl_path=tcl_path, bounds_min=[-100, -20, -20], bounds_max=[100, 50, 20], shell_radius=12.0)
    mesh = builder.build(centerline)
    print(f"Built wire mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    out_path = r"C:\AIND_Library\Git_PredicTest\x64_Cami_Lib\PSCD1500\PSCD1500\Cable\ai_pipeline\test_output\wires_only_test.glb"
    mesh.export(out_path)
    print(f"Exported to {out_path}")
