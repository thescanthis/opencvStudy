"""
케이블 트리(cable_tree.CableNode)를 따라 셸 + 배선을 3D로 만든다.

branching_wire_builder.py는 "트렁크 1개 + 가지 N개"인 1단 분기만 다뤘다.
이 모듈은 가지가 다시 트렁크가 되어 또 갈라지는 N중 분기를 지원한다 -
분기는 별개의 패턴이 아니라 재귀 구조라는 정리에 따른 것.

설계 원칙(지금까지 확정된 것들):
  1. 각 구간(부모->자식)은 fork_branch_centerline으로 만든다. 분기점에서
     접선이 부모와 같은 방향으로 시작해 서서히 벌어지므로 각지지 않는다.
  2. 배선 한 가닥은 트리에서 자기 출발 커넥터 -> 도착 커넥터까지의 경로를
     통째로 이어 하나의 튜브로 만든다. 조각조각 나누지 않으므로 이음매나
     "허공에서 시작하는 선"이 생기지 않는다.
  3. 어떤 구간을 여러 배선이 함께 지나가면, 그 구간에서 각 배선은 서로 다른
     고정 오프셋(다발 안의 자기 자리)을 갖는다. 자리는 구간 단위로 미리
     배정하고, 구간이 바뀔 때만 부드럽게 옮겨간다 - 그래서 같이 가는 동안은
     서로 겹치지 않는다.
  4. 갈라지는 시점(분기점)에서만 경로가 나뉜다.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import trimesh

from tcl_parser import CableTopology, parse_tcl
from connector_layout import layout_pins
from cable_shapes import fork_branch_centerline
from wire_builder import create_tube_mesh, _centerline_frames, _compute_curvature_shrink
from cable_tree import CableNode, build_cable_tree


SHELL_COLOR = [100, 149, 237, 120]
WIRE_COLOR = [255, 69, 0, 255]


@dataclass
class Segment:
    """트리의 한 간선(부모 커넥터 -> 자식 커넥터)에 해당하는 물리적 구간."""
    parent: Optional[str]
    child: str
    centerline: np.ndarray                       # (N,3) 월드 좌표
    frames: List[Tuple[np.ndarray, np.ndarray]]
    shrink: np.ndarray
    # 이 구간을 지나는 배선들에게 배정된 다발 안 자리: wire_key -> (u,v)
    slots: Dict[str, Tuple[float, float]]


def _place_2d_in_3d(xy: np.ndarray, origin: np.ndarray, direction: np.ndarray,
                     up_hint: np.ndarray) -> np.ndarray:
    """2D(XY, +X가 진행 방향) 경로를 origin에서 direction을 향하도록 배치한다."""
    direction = direction / (np.linalg.norm(direction) + 1e-9)
    side = np.cross(direction, up_hint)
    if np.linalg.norm(side) < 1e-6:
        up_hint = np.array([1.0, 0.0, 0.0]) if abs(direction[1]) > 0.9 else np.array([0.0, 1.0, 0.0])
        side = np.cross(direction, up_hint)
    side = side / (np.linalg.norm(side) + 1e-9)
    plane_up = np.cross(side, direction)
    plane_up = plane_up / (np.linalg.norm(plane_up) + 1e-9)
    return origin + xy[:, 0:1] * direction + xy[:, 1:2] * plane_up


def _fan_angles(n: int, total_deg: float) -> List[float]:
    """n개 가지를 total_deg 폭의 부채꼴로 균등 분산한 각도 목록."""
    if n <= 0:
        return []
    if n == 1:
        return [0.0]
    half = total_deg / 2
    return [-half + (2 * half) * (i / (n - 1)) for i in range(n)]


class TreeCableBuilder:
    """케이블 트리를 따라 셸과 배선을 만든다(N중 분기 지원)."""

    def __init__(self, tcl_path: str, shell_radius: float = 10.0,
                 root_length: float = 120.0, segment_lengths: Dict[str, float] = None,
                 default_segment_length: float = 90.0,
                 fan_angle_deg: float = 80.0, samples: int = 40,
                 connector_types: Dict[str, str] = None,
                 root: str = None, wire_radius: float = None,
                 segment_angles: Dict[str, float] = None):
        """
        root_length: 트렁크 시작 구간(루트가 처음 뻗어나가는 몸통)의 길이.
        segment_lengths: {커넥터: 길이} - 그 커넥터로 들어가는 구간의 길이.
            도면 실측치를 넣을 때 사용. 없으면 default_segment_length.
        fan_angle_deg: 한 분기점에서 가지들이 퍼지는 부채꼴 전체 폭(자동 배치용).
        segment_angles: {커넥터: 각도(도)} - 그 커넥터로 들어가는 구간의 꺾임각을
            직접 지정. 지정된 커넥터는 자동 부채꼴 계산을 건너뛰고 이 값을
            그대로 쓴다(도면 실측 각도를 그대로 반영하고 싶을 때 사용).
        """
        self.tcl_path = tcl_path
        self.shell_radius = shell_radius
        self.root_length = root_length
        self.segment_lengths = segment_lengths or {}
        self.default_segment_length = default_segment_length
        self.fan_angle_deg = fan_angle_deg
        self.samples = samples
        self.connector_types = connector_types or {}
        self.root = root
        self.wire_radius = wire_radius if wire_radius is not None else max(0.3, shell_radius * 0.05)
        self.segment_angles = segment_angles or {}

    # ------------------------------------------------------------------
    # 1단계: 트리를 따라가며 각 구간의 중심선을 만든다
    # ------------------------------------------------------------------
    def _build_segments(self, tree: CableNode) -> Dict[str, Segment]:
        """커넥터 이름 -> 그 커넥터로 들어오는 구간(Segment)."""
        segments: Dict[str, Segment] = {}

        def make_segment(parent: Optional[str], node: CableNode,
                          origin: np.ndarray, direction: np.ndarray,
                          length: float, angle_deg: float,
                          plane_roll_deg: float = 0.0) -> Segment:
            # 분기가 일어나는 평면을 깊이마다 조금씩 돌려준다. 안 그러면 모든
            # 분기가 같은 평면에서만 퍼져서 납작하게 보인다.
            direction_n = direction / (np.linalg.norm(direction) + 1e-9)
            base_up = np.array([0.0, 0.0, 1.0])
            if abs(np.dot(direction_n, base_up)) > 0.9:
                base_up = np.array([0.0, 1.0, 0.0])
            # direction 축을 기준으로 base_up을 plane_roll_deg만큼 회전(로드리게스)
            roll = np.radians(plane_roll_deg)
            k = direction_n
            up_hint = (base_up * np.cos(roll)
                       + np.cross(k, base_up) * np.sin(roll)
                       + k * np.dot(k, base_up) * (1 - np.cos(roll)))

            # 진입부(부모 접선을 이어받는 짧은 구간) + 본체
            xy = fork_branch_centerline(
                trunk_length=length * 0.25,
                branch_length=length * 0.75,
                branch_angle_deg=angle_deg,
                n_trunk=max(4, self.samples // 3),
                n_branch=max(6, self.samples),
            )
            world = _place_2d_in_3d(xy, origin, direction, up_hint)
            frames = _centerline_frames(world)
            shrink = _compute_curvature_shrink(world, self.shell_radius * 1.5)
            return Segment(parent=parent, child=node.connector, centerline=world,
                           frames=frames, shrink=shrink, slots={})

        def recurse(node: CableNode, origin: np.ndarray, direction: np.ndarray, depth: int):
            children = node.children
            if not children:
                return

            angle_for: Dict[int, float] = {}
            roll_for: Dict[int, float] = {}
            # 분기 깊이마다 기준 평면을 돌려 전체가 납작해지지 않게 한다
            base_roll = 55.0 * depth

            if len(children) == 1:
                # 가지가 하나뿐이면 부채꼴이랄 게 없지만, 완전히 직진시키면
                # 부모 구간과 한 줄로 이어져 "분기했다"는 게 안 보인다.
                # 살짝만 휘게 해서 방향이 바뀐 걸 알 수 있게 한다.
                angle_for = {0: self.fan_angle_deg * 0.35}
                roll_for = {0: base_roll}
            elif len(children) <= 3:
                # 2~3갈래는 한 평면에서 부채꼴로 퍼지는 게 실물과 가장 비슷하다.
                angles = _fan_angles(len(children), self.fan_angle_deg)
                # 가지 수가 홀수면 가운데 가지가 각도 0(부모와 일직선)을 받아
                # 몸통이 그대로 관통하는 것처럼 보인다. 전체를 조금 틀어서
                # 어느 가지도 부모와 완전히 일직선이 되지 않게 한다.
                if len(children) % 2 == 1:
                    tilt = self.fan_angle_deg / (2 * (len(children) - 1))
                    angles = [a + tilt for a in angles]
                # 가닥이 많은 가지(=몸통에 가까운 쪽)를 가운데로 두면 자연스럽다
                order = sorted(range(len(children)), key=lambda i: -children[i].wire_count)
                mid = len(children) // 2
                slot_order = sorted(range(len(children)), key=lambda i: abs(i - mid))
                for rank, child_idx in enumerate(order):
                    angle_for[child_idx] = angles[slot_order[rank]]
                    roll_for[child_idx] = base_roll
            else:
                # 4갈래 이상을 한 평면에 늘어놓으면 부챗살처럼 납작해진다.
                # 몸통 축 둘레로 방위각을 돌려가며 원뿔 모양으로 퍼뜨린다 -
                # 실물에서 여러 가닥이 사방으로 뻗는 모습에 가깝다.
                spread = self.fan_angle_deg / 2
                order = sorted(range(len(children)), key=lambda i: -children[i].wire_count)
                for rank, child_idx in enumerate(order):
                    # 가닥이 많은 가지일수록 몸통에 가깝게(작은 각도로) 둔다
                    ratio = rank / max(1, len(children) - 1)
                    angle_for[child_idx] = spread * (0.45 + 0.55 * ratio)
                    roll_for[child_idx] = base_roll + 360.0 * rank / len(children)

            for idx, child in enumerate(children):
                length = self.segment_lengths.get(child.connector, self.default_segment_length)
                angle = self.segment_angles.get(child.connector, angle_for[idx])
                seg = make_segment(node.connector, child, origin, direction,
                                   length, angle, roll_for[idx])
                segments[child.connector] = seg
                # 자식의 끝점/끝 접선이 그 자식에서 갈라질 다음 분기의 시작이 된다
                end = seg.centerline[-1]
                tangent = seg.centerline[-1] - seg.centerline[-2]
                recurse(child, end, tangent, depth + 1)

        # 루트 자신의 몸통 구간(케이블이 루트 커넥터에서 뻗어나오는 부분)
        origin = np.array([0.0, 0.0, 0.0])
        direction = np.array([1.0, 0.0, 0.0])
        root_seg = make_segment(None, tree, origin, direction, self.root_length, 0.0)
        segments[tree.connector] = root_seg

        recurse(tree, root_seg.centerline[-1],
                root_seg.centerline[-1] - root_seg.centerline[-2], depth=0)
        return segments

    # ------------------------------------------------------------------
    # 2단계: 배선마다 트리 상의 경로(지나가는 구간 목록)를 구한다
    # ------------------------------------------------------------------
    @staticmethod
    def _path_between(tree: CableNode, a: str, b: str) -> List[str]:
        """트리에서 커넥터 a -> b로 가는 경로를 커넥터 이름 목록으로 반환한다."""
        def find_chain(node: CableNode, target: str, acc: List[str]) -> Optional[List[str]]:
            acc = acc + [node.connector]
            if node.connector == target:
                return acc
            for child in node.children:
                found = find_chain(child, target, acc)
                if found:
                    return found
            return None

        chain_a = find_chain(tree, a, [])
        chain_b = find_chain(tree, b, [])
        if chain_a is None or chain_b is None:
            return []
        # 공통 조상까지 거슬러 올라갔다가 내려간다
        common = 0
        while common < min(len(chain_a), len(chain_b)) and chain_a[common] == chain_b[common]:
            common += 1
        up = list(reversed(chain_a[common:]))
        down = chain_b[common:]
        junction = chain_a[common - 1] if common > 0 else None
        path = up + ([junction] if junction else []) + down
        return path

    # ------------------------------------------------------------------
    # 3단계: 각 구간에서 배선들에게 다발 안 자리를 배정한다
    # ------------------------------------------------------------------
    def _assign_slots(self, segments: Dict[str, Segment], wire_routes: Dict[str, List[str]]):
        """배선마다 다발 안의 "자기 각도"를 하나 정하고, 모든 구간에서 그
        각도를 그대로 유지한다.

        예전에는 구간마다 "그 구간을 지나는 배선들"만 모아 다시 원형 배치를
        했는데, 구간마다 지나는 배선 수가 다르므로(몸통 15가닥 -> 가지 9가닥
        -> 끝 3가닥) 같은 배선의 자리가 구간이 바뀔 때마다 확 달라졌다.
        그래서 케이블 안에서 가닥들이 서로 자리를 맞바꾸며 빙빙 도는(뒤틀리는)
        모양이 나왔다(사용자 피드백).

        각도를 전역으로 한 번만 정해두면 배선은 몸통에서든 가지에서든 늘
        같은 방향에 있으므로, 다발이 통째로 나란히 흐르고 뒤틀리지 않는다.
        반지름만 구간마다 조절한다 - 가닥이 적은 구간에서는 다발이 얇아지는
        게 자연스럽기 때문.
        """
        radius = max(0.5, self.shell_radius * 0.6 - self.wire_radius)

        # 1) 배선별 전역 각도: 키 정렬 순서대로 원을 균등 분할해 하나씩 준다.
        all_keys = sorted(wire_routes.keys())
        total = len(all_keys)
        wire_angle: Dict[str, float] = {}
        wire_ring: Dict[str, float] = {}
        for i, key in enumerate(all_keys):
            wire_angle[key] = 2 * np.pi * i / max(1, total)
            # 가닥이 아주 많으면 한 겹으로는 빽빽하므로 안팎 두 겹으로 나눈다
            wire_ring[key] = 1.0 if total <= 12 else (1.0 if i % 2 == 0 else 0.62)

        # 2) 구간별로 "이 구간을 지나는 배선이 몇 개인지"만 세어 반지름을 정한다.
        users: Dict[str, List[str]] = {name: [] for name in segments}
        for wire_key, route in wire_routes.items():
            for connector in route:
                if connector in users and wire_key not in users[connector]:
                    users[connector].append(wire_key)

        for connector, wire_keys in users.items():
            seg = segments[connector]
            count = len(wire_keys)
            if count == 0:
                continue
            # 각도뿐 아니라 반지름도 구간에 상관없이 고정한다. 구간마다
            # 다발 굵기를 바꾸면 배선이 안팎으로 들락거려 직선으로 가지
            # 못한다(사용자 요구: 꼬여도 좋으니 최대한 직선으로, 겹치지 않게).
            for key in wire_keys:
                angle = wire_angle[key]
                r = radius * wire_ring[key]
                seg.slots[key] = (r * np.cos(angle), r * np.sin(angle))

    # ------------------------------------------------------------------
    # 4단계: 배선 경로를 실제 3D 점열로 만든다
    # ------------------------------------------------------------------
    def _wire_points(self, segments: Dict[str, Segment], route: List[str],
                      wire_key: str, uv_start: Tuple[float, float],
                      uv_end: Tuple[float, float]) -> Optional[np.ndarray]:
        """배선 한 가닥이 지나는 구간들을 이어 하나의 연속된 점열로 만든다."""
        if not route:
            return None

        pieces: List[np.ndarray] = []
        # 각 구간에서 이 배선이 앉을 자리(uv). 양 끝은 실제 핀 위치를 쓴다.
        stops: List[Tuple[Segment, Tuple[float, float], bool]] = []
        for i, connector in enumerate(route):
            seg = segments.get(connector)
            if seg is None:
                continue
            reverse = False
            # route에서 부모 방향으로 거슬러 올라가는 구간은 뒤집어 써야 한다
            if i + 1 < len(route) and seg.parent == route[i + 1]:
                reverse = True
            stops.append((seg, seg.slots.get(wire_key, (0.0, 0.0)), reverse))

        if not stops:
            return None

        for idx, (seg, uv, reverse) in enumerate(stops):
            centerline = seg.centerline
            frames = seg.frames
            shrink = seg.shrink
            n = len(centerline)
            t = np.linspace(0, 1, n)
            smooth = t * t * (3 - 2 * t)

            # 이 구간 안에서 시작/끝 오프셋: 앞 구간에서 이어받고, 뒤 구간으로 넘긴다
            uv_in = stops[idx - 1][1] if idx > 0 else (uv_start if not reverse else uv)
            uv_out = stops[idx + 1][1] if idx + 1 < len(stops) else (uv_end if not reverse else uv)
            if idx == 0:
                uv_in = uv_start
            if idx == len(stops) - 1:
                uv_out = uv_end

            # (u,v)를 직선으로 보간하면 다발 반대편으로 옮겨갈 때 원의 중심을
            # 가로질러 지나가 경로가 각지게 꺾인다("네모를 그리며 간다").
            # 극좌표로 보간해 각도는 최단 방향으로 돌고 반지름만 늘었다 줄게
            # 하면, 배선이 케이블 단면의 둘레를 따라 부드럽게 옮겨간다.
            # 각도를 전역으로 고정해뒀으므로 대부분의 구간에서는 각도 변화가
            # 0이고, 실제로 도는 것은 양 끝(핀에 꽂히는 부분)뿐이다.
            r_in = float(np.hypot(uv_in[0], uv_in[1]))
            r_out = float(np.hypot(uv_out[0], uv_out[1]))
            a_in = float(np.arctan2(uv_in[1], uv_in[0])) if r_in > 1e-9 else None
            a_out = float(np.arctan2(uv_out[1], uv_out[0])) if r_out > 1e-9 else None
            if a_in is None and a_out is None:
                u = np.zeros(n)
                v = np.zeros(n)
            else:
                # 한쪽이 중심(반지름 0)이면 반대쪽 각도를 그대로 쓴다
                if a_in is None:
                    a_in = a_out
                if a_out is None:
                    a_out = a_in
                d_angle = (a_out - a_in + np.pi) % (2 * np.pi) - np.pi
                angle = a_in + d_angle * smooth
                radius_t = r_in * (1 - smooth) + r_out * smooth
                u = radius_t * np.cos(angle) * shrink
                v = radius_t * np.sin(angle) * shrink

            pts = np.empty_like(centerline)
            for i in range(n):
                normal, binormal = frames[i]
                pts[i] = centerline[i] + u[i] * normal + v[i] * binormal
            if reverse:
                pts = pts[::-1]
            # 구간 경계에서 점이 겹치지 않게 이어붙이기
            pieces.append(pts if not pieces else pts[1:])

        if not pieces:
            return None
        return np.concatenate(pieces, axis=0)

    # ------------------------------------------------------------------
    def build(self, tree: Optional[CableNode] = None) -> trimesh.Trimesh:
        """tree를 직접 넘기면(도면 실측 형상을 손으로 지정한 경우 등) 그것을
        쓰고, 안 넘기면 TCL 배선 수로 자동 복원한 트리(cable_tree)를 쓴다.
        """
        topo = parse_tcl(self.tcl_path)
        if tree is None:
            tree = build_cable_tree(topo, root=self.root)
        if tree is None:
            return trimesh.Trimesh()

        segments = self._build_segments(tree)

        # 커넥터별 핀 UV(실제 핀 위치) - 배선의 양 끝에서 사용
        connector_radius = max(0.5, self.shell_radius * 0.6 - self.wire_radius)
        pin_uv: Dict[str, Tuple[float, float]] = {}
        for connector in topo.connectors:
            pins = [p.pin_name for p in topo.pins_of(connector)]
            layout = layout_pins(pins, radius=connector_radius,
                                 connector_type=self.connector_types.get(connector))
            for pin_name, uv in layout.items():
                pin_uv[f"{connector}-{pin_name}"] = uv

        # 배선마다 트리 경로 계산
        wire_routes: Dict[str, List[str]] = {}
        wire_ends: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]] = {}
        skipped = 0
        for i, wire in enumerate(topo.wires):
            a = topo.connector_of_full_pin(wire.from_pin)
            b = topo.connector_of_full_pin(wire.to_pin)
            if a is None or b is None:
                skipped += 1
                continue
            key = f"w{i:03d}_{wire.from_pin}->{wire.to_pin}"
            if a == b:
                route = [a]
            else:
                route = self._path_between(tree, a, b)
            if not route:
                skipped += 1
                continue
            # 핀 이름 정규화(단일핀 커넥터는 "P2 "처럼 하이픈이 없다)
            pin_a = wire.from_pin[len(a):].lstrip("-").strip() or " "
            pin_b = wire.to_pin[len(b):].lstrip("-").strip() or " "
            uv_a = pin_uv.get(f"{a}-{pin_a}") or (0.0, 0.0)
            uv_b = pin_uv.get(f"{b}-{pin_b}") or (0.0, 0.0)
            wire_routes[key] = route
            wire_ends[key] = (uv_a, uv_b)

        self._assign_slots(segments, wire_routes)

        meshes: List[trimesh.Trimesh] = []

        # 셸: 각 구간마다 하나씩
        for seg in segments.values():
            shell = create_tube_mesh(seg.centerline, self.shell_radius, SHELL_COLOR, sides=16)
            if len(shell.vertices) > 0:
                meshes.append(shell)

        # 배선
        for key, route in wire_routes.items():
            uv_a, uv_b = wire_ends[key]
            pts = self._wire_points(segments, route, key, uv_a, uv_b)
            if pts is None or len(pts) < 2:
                skipped += 1
                continue
            mesh = create_tube_mesh(pts, self.wire_radius, WIRE_COLOR)
            if len(mesh.vertices) > 0:
                meshes.append(mesh)

        if skipped:
            print(f"  [TreeCableBuilder] skipped {skipped} wires")
        if not meshes:
            return trimesh.Trimesh()
        return trimesh.util.concatenate(meshes)


if __name__ == "__main__":
    import os
    base = r"C:\AIND_Library\Git_PredicTest\x64_Cami_Lib\PSCD1500\PSCD1500\Cable\TestData"
    tcl = os.path.join(base, "A20016148_6145-37-520-5297", "cable.tcl")
    builder = TreeCableBuilder(tcl_path=tcl, shell_radius=10.0, root_length=120.0,
                               default_segment_length=100.0, fan_angle_deg=80.0)
    mesh = builder.build()
    print("vertices:", len(mesh.vertices))
    out = r"C:\AIND_Library\Git_PredicTest\x64_Cami_Lib\PSCD1500\PSCD1500\Cable\ai_pipeline\test_output\tree_test.glb"
    mesh.export(out)
    print("exported ->", out)
