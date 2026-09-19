"""
커넥터가 3개 이상인 분기(Y자/트리형) 케이블의 배선을 3D 튜브로 생성한다.

핵심 아이디어(사용자 피드백 반영): 분기 케이블은 "꺾인 파이프(cable_shapes.
bent_centerline)가 여러 개, 트렁크 구간을 공유하며 겹쳐 있는 것"으로 취급한다.
별도의 "트렁크에서 배선이 뭉쳐있다가 막판에 갈라지는" 특수 로직을 쓰지 않고,
가지마다 독립적인 bent_centerline() 경로 하나씩을 통째로 만들어, 그 가지에
배정된 배선들이 그 경로를 그대로(가지 전용 UV 오프셋만 얹어) 따라가게 한다.
모든 가지의 bent_centerline은 트렁크 구간(1구간)이 서로 완전히 동일하므로,
자연히 "같은 몸통 안에서 나란히 가다가 꺾이는 지점에서 각자 갈라지는" 형태가
된다.

wire_builder.py는 커넥터가 정확히 2개인 점대점 케이블만 다룬다. 이 모듈은
커넥터가 3개 이상인 경우(하나의 트렁크에서 N개 커넥터로 갈라짐)를 다룬다.
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import trimesh

from tcl_parser import CableTopology, parse_tcl
from connector_layout import layout_pins
from wire_builder import create_tube_mesh, _centerline_frames, _compute_curvature_shrink
from cable_shapes import bent_centerline, fork_branch_centerline


@dataclass
class BranchPath:
    """하나의 가지(트렁크 공유 구간 + 그 가지만의 꺾인 구간) 전체 중심선."""
    connector: str
    centerline: np.ndarray  # (N, 3), 월드 좌표
    frames: List[Tuple[np.ndarray, np.ndarray]]
    curvature_shrink: np.ndarray


def _place_2d_path_in_3d(xy_path: np.ndarray, origin: np.ndarray,
                          trunk_dir: np.ndarray, up_hint: np.ndarray) -> np.ndarray:
    """bent_centerline()이 만든 2D(XY 평면, +X가 트렁크 진행 방향) 경로를,
    실제 3D 공간에서 origin을 시작점으로 하고 trunk_dir을 향하도록 회전/이동한다.
    """
    trunk_dir = trunk_dir / (np.linalg.norm(trunk_dir) + 1e-9)
    side = np.cross(trunk_dir, up_hint)
    side = side / (np.linalg.norm(side) + 1e-9)
    plane_up = np.cross(side, trunk_dir)
    plane_up = plane_up / (np.linalg.norm(plane_up) + 1e-9)

    # xy_path의 x -> trunk_dir 축, y -> plane_up 축
    x = xy_path[:, 0:1]
    y = xy_path[:, 1:2]
    world = origin + x * trunk_dir + y * plane_up
    return world


def _build_branch_paths(
    connectors: List[str],
    trunk_connector: str,
    origin: np.ndarray,
    trunk_dir: np.ndarray,
    trunk_length: float,
    branch_lengths: Dict[str, float],
    corner_radius: float,
    max_fan_angle_deg: float = 150.0,
    samples_per_seg: int = 30,
) -> Dict[str, BranchPath]:
    """각 non-trunk 커넥터마다, 트렁크 구간을 공유하는 독립적인 bent_centerline
    경로를 만든다. 부채꼴로 균등 분산된 각도를 bend_angle로 사용하고, 가지
    길이는 branch_lengths[connector]에서 개별적으로 가져온다(도면 실측 반영).

    트렁크 자신(trunk_connector)은 꺾이지 않는 직선 경로(bend_angle=0)로 취급해,
    "몸통이 곧게 들어와서 여기서 갈라진다"는 느낌을 그대로 살린다.

    max_fan_angle_deg는 "정면 기준 좌우"가 아니라 전체 부채꼴 폭이다
    (예: 150이면 -75~+75도). 가지 수가 많을 때 파이프끼리 겹치지 않으려면
    75(기존 기본값)보다 넓게 잡아야 한다.
    """
    up_hint = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(trunk_dir / (np.linalg.norm(trunk_dir) + 1e-9), up_hint)) > 0.9:
        up_hint = np.array([1.0, 0.0, 0.0])

    other_connectors = [c for c in connectors if c != trunk_connector]
    n_branches = max(1, len(other_connectors))
    half_fan = max_fan_angle_deg / 2

    paths: Dict[str, BranchPath] = {}

    def make_path(connector: str, branch_angle_deg: float, branch_length: float) -> BranchPath:
        # 참조 GLB(Cable_1-2.glb)에서 확인된 형태: 각진 원호 코너(bent_centerline)
        # 대신, 분기점에서 접선이 트렁크와 같은 방향으로 시작해 서서히
        # 벌어지는 부드러운 포크(fork_branch_centerline)를 쓴다.
        xy = fork_branch_centerline(
            trunk_length=trunk_length, branch_length=branch_length,
            branch_angle_deg=branch_angle_deg,
            n_trunk=samples_per_seg, n_branch=samples_per_seg,
        )
        world = _place_2d_path_in_3d(xy, origin, trunk_dir, up_hint)
        frames = _centerline_frames(world)
        # 급하게 휘는 구간에서 배선 오프셋이 셸을 뚫지 않도록 곡률 기반 축소를 재사용
        shrink = _compute_curvature_shrink(world, corner_radius)
        return BranchPath(connector=connector, centerline=world, frames=frames, curvature_shrink=shrink)

    # 트렁크 자신: 꺾이지 않는 짧은 캡 연장선(몸통이 뭉툭하게 끝나지 않고
    # 살짝 이어지는 느낌만 준다). 예전엔 "가장 긴 가지와 비슷한 길이"로
    # 늘렸었는데, 그러면 트렁크 자신의 경로가 실제 가지(예: P4)와 같은
    # 공간을 길게 나란히 지나가면서 물리적으로 겹쳐 보이는 문제가 있었다
    # (사용자 피드백: "직진으로 가는 굵은 다발"이 갈래 다발과 뒤섞여 보임).
    # 트렁크 자신은 어차피 목적지가 없는 커넥터이므로 짧은 캡만 있으면 된다.
    trunk_cap_length = min(branch_lengths.values(), default=trunk_length) * 0.15 if branch_lengths else trunk_length * 0.15
    paths[trunk_connector] = make_path(trunk_connector, branch_angle_deg=0.0, branch_length=max(trunk_cap_length, 1e-3))

    for i, connector in enumerate(other_connectors):
        if n_branches == 1:
            angle = half_fan / 2
        else:
            angle = -half_fan + (2 * half_fan) * (i / (n_branches - 1))
        branch_length = branch_lengths.get(connector, trunk_length * 0.7)
        paths[connector] = make_path(connector, branch_angle_deg=angle, branch_length=branch_length)

    return paths


class BranchingCableBuilder:
    """분기형(3개 이상 커넥터) 케이블의 셸 + 배선을 생성한다.

    각 가지는 "트렁크 구간 + 그 가지만의 꺾인 구간"을 통째로 가진 독립
    경로(BranchPath)이며, 모든 가지의 트렁크 구간은 기하학적으로 동일하다
    (같은 origin, trunk_dir, trunk_length로 생성됨).
    """

    def __init__(self, tcl_path: str, shell_radius: float = 8.0, trunk_length: float = 90.0,
                 branch_length: float = 70.0, branch_lengths: Dict[str, float] = None,
                 corner_radius: float = None, max_fan_angle_deg: float = 150.0,
                 wire_radius: float = None, connector_types: Dict[str, str] = None,
                 trunk_connector: str = None, samples_per_seg: int = 30):
        """
        branch_length: 가지별 길이가 없을 때 쓰는 공통 기본값.
        branch_lengths: {connector_name: length} - 도면 실측 등 가지별로 다른
            길이를 줄 때 사용. 지정 안 된 커넥터는 branch_length로 대체.
        max_fan_angle_deg: 가지들이 퍼지는 전체 부채꼴 폭(도). 가지 수가
            많으면 좁은 각도(기본 이전 값 75)에서 파이프끼리 겹치므로 기본을
            150으로 넓혔다.
        trunk_connector: 트렁크로 쓸 커넥터 이름을 명시. 지정하지 않으면 핀 수가
            가장 많은 커넥터를 자동 선택하는데, 핀 수가 동률인 경우(예: 두
            커넥터가 둘 다 6핀) 결과가 불안정할 수 있어 실물/도면상 명백한
            경우 명시적으로 지정하는 것을 권장한다.
        """
        self.tcl_path = tcl_path
        self.shell_radius = shell_radius
        self.trunk_length = trunk_length
        self.branch_length = branch_length
        self.branch_lengths = branch_lengths or {}
        self.corner_radius = corner_radius if corner_radius is not None else shell_radius * 2.5
        self.max_fan_angle_deg = max_fan_angle_deg
        self.wire_radius = wire_radius if wire_radius is not None else max(0.3, shell_radius * 0.05)
        self.connector_types = connector_types or {}
        self.trunk_connector = trunk_connector
        self.samples_per_seg = samples_per_seg

    def build(self) -> trimesh.Trimesh:
        topo = parse_tcl(self.tcl_path)
        if len(topo.connectors) < 3:
            print(f"  [BranchingCableBuilder] {self.tcl_path} has <3 connectors; not a branching cable.")
            return trimesh.Trimesh()

        trunk_connector = self.trunk_connector or max(topo.connectors, key=lambda c: len(topo.pins_of(c)))

        # 가지별 길이 dict를 완성: 지정 안 된 커넥터는 공통 기본값으로 채움
        resolved_lengths = {
            c: self.branch_lengths.get(c, self.branch_length)
            for c in topo.connectors if c != trunk_connector
        }

        origin = np.array([0.0, 0.0, 0.0])
        trunk_dir = np.array([1.0, 0.0, 0.0])
        paths = _build_branch_paths(
            topo.connectors, trunk_connector, origin, trunk_dir,
            self.trunk_length, resolved_lengths, self.corner_radius,
            max_fan_angle_deg=self.max_fan_angle_deg,
            samples_per_seg=self.samples_per_seg,
        )

        meshes = []

        # 1. 셸: 트렁크 커넥터 경로 하나만 대표로 그려서 "몸통 하나가 갈라지는"
        # 형태를 준다. 각 가지 끝부분은 배선 셸 대신 아래에서 개별 캡으로 처리.
        shell_color = [100, 149, 237, 120]
        trunk_shell = create_tube_mesh(paths[trunk_connector].centerline, self.shell_radius, shell_color, sides=16)
        if len(trunk_shell.vertices) > 0:
            meshes.append(trunk_shell)
        for connector, bp in paths.items():
            if connector == trunk_connector:
                continue
            # 가지 구간(코너 이후)만 셸로 그린다 - 트렁크 구간은 이미 그렸으므로 중복 방지
            branch_only = self._branch_only_segment(bp.centerline)
            if len(branch_only) >= 2:
                shell = create_tube_mesh(branch_only, self.shell_radius, shell_color, sides=16)
                if len(shell.vertices) > 0:
                    meshes.append(shell)

        # 2. 커넥터별 핀 UV
        connector_radius = max(0.5, self.shell_radius * 0.65 - self.wire_radius)
        pin_uv: Dict[str, Tuple[float, float]] = {}
        for connector in topo.connectors:
            pins = [p.pin_name for p in topo.pins_of(connector)]
            layout = layout_pins(pins, radius=connector_radius, connector_type=self.connector_types.get(connector))
            for pin_name, uv in layout.items():
                pin_uv[f"{connector}-{pin_name}"] = uv

        # 3. 배선: from/to 핀이 속한 가지의 BranchPath를 그대로 따라간다
        skipped = 0
        unique_legs = {}
        for wire in topo.wires:
            from_connector = topo.connector_of_full_pin(wire.from_pin)
            to_connector = topo.connector_of_full_pin(wire.to_pin)
            if from_connector is None or to_connector is None:
                skipped += 1
                continue
            # wire.from_pin/to_pin은 TCL 원문 그대로(예: "P2 ", 하이픈 없이
            # 커넥터명=핀 전체인 단일핀 커넥터 케이스 포함)라 pin_uv 키
            # ("커넥터-핀명")과 형식이 다를 수 있다. 커넥터/핀명을 정규화해서 찾는다.
            from_pin_name = wire.from_pin[len(from_connector):].lstrip("-").strip() or " "
            to_pin_name = wire.to_pin[len(to_connector):].lstrip("-").strip() or " "
            uv_from = pin_uv.get(f"{from_connector}-{from_pin_name}") or pin_uv.get(wire.from_pin)
            uv_to = pin_uv.get(f"{to_connector}-{to_pin_name}") or pin_uv.get(wire.to_pin)
            if uv_from is None or uv_to is None:
                skipped += 1
                continue

            legs = self._route_wire_legs(paths, trunk_connector, from_connector, to_connector, uv_from, uv_to)
            if not legs:
                skipped += 1
                continue
            for key, path in legs:
                unique_legs[key] = path

        # Generate unique meshes
        for key, path in unique_legs.items():
            mesh = create_tube_mesh(path, self.wire_radius, [255, 69, 0, 255])
            if len(mesh.vertices) > 0:
                meshes.append(mesh)

        if skipped:
            print(f"  [BranchingCableBuilder] Skipped {skipped}/{len(topo.wires)} wires.")

        if not meshes:
            return trimesh.Trimesh()
        return trimesh.util.concatenate(meshes)

    def _branch_only_segment(self, centerline: np.ndarray) -> np.ndarray:
        """BranchPath.centerline에서, 트렁크와 겹치지 않는 가지 전용 후반부만
        잘라 반환한다. fork_branch_centerline은 트렁크(n_trunk개) + 가지
        (n_branch개, 첫 점이 분기점과 중복되어 실제로는 n_branch-1개 추가)
        순서로 점을 이어붙이므로, 정확히 트렁크 샘플 수만큼을 건너뛴다.
        """
        n_trunk = self.samples_per_seg
        n = len(centerline)
        start_idx = min(max(0, n_trunk - 1), n - 1)
        return centerline[start_idx:]

    def _offset_path(self, bp: BranchPath, uv_from: Tuple[float, float], uv_to: Tuple[float, float],
                      branch_only: bool = False, full_span: bool = False) -> np.ndarray:
        """BranchPath를 따라가며 (u,v) 오프셋을 uv_from -> uv_to로 보간한다.

        bp.centerline은 항상 "트렁크 구간(n_trunk개) + 가지 구간"이 이어붙은
        경로다(branch_only=True면 트렁크 구간을 잘라낸 후반부만). 트렁크
        구간에서는 오프셋을 uv_from으로 고정하고(다른 배선들과 나란히, 같은
        물리적 다발 안을 지나감), 가지 구간에 들어서야만 서서히 uv_to로
        수렴시킨다. 이걸 트렁크가 끝나는 실제 인덱스(n_trunk) 기준으로
        정확히 나눠야, 트렁크 UV와 가지 UV가 서로 다른 독립 원형 레이아웃
        이라도 다발이 교차하지 않고(트렁크에서는 완전히 고정), 가지 진입
        후에는 매끄럽게 벌어진다(사용자 피드백: 교차 문제, 급격한 꺾임 문제
        둘 다 이 경계를 명확히 해서 해결).

        branch_only=True면 반환 경로 자체가 트렁크 구간 없이 가지 구간만.
        """
        n_trunk = self.samples_per_seg
        if branch_only:
            centerline = self._branch_only_segment(bp.centerline)
            offset_in_full = len(bp.centerline) - len(centerline)
            frames = bp.frames[offset_in_full:]
            shrink = bp.curvature_shrink[offset_in_full:]
            # 잘려나간 트렁크 구간 길이만큼을 빼서, 이 조각 안에서
            # "가지가 시작되는 인덱스"가 몇 번째인지 다시 계산한다.
            trunk_end_idx = max(0, n_trunk - 1 - offset_in_full)
        else:
            centerline = bp.centerline
            frames = bp.frames
            shrink = bp.curvature_shrink
            trunk_end_idx = n_trunk - 1

        n = len(centerline)
        if full_span:
            # 트렁크 개념이 없는 경로(같은 커넥터 내부 배선 등): 처음부터
            # 끝까지 단순하게 uv_from -> uv_to로 보간한다.
            t = np.linspace(0, 1, n)
            smooth_t = t * t * (3 - 2 * t)
        else:
            trunk_end_idx = min(max(0, trunk_end_idx), n - 1)
            idx = np.arange(n)
            # 트렁크 구간(idx <= trunk_end_idx): t=0 고정 -> uv_from 그대로.
            # 가지 구간(idx > trunk_end_idx): 0->1로 부드럽게 진행 -> uv_to로 수렴.
            branch_span = max(1, n - 1 - trunk_end_idx)
            t = np.clip((idx - trunk_end_idx) / branch_span, 0.0, 1.0)
            smooth_t = t * t * (3 - 2 * t)
        # (u,v) 평면에서 uv_from -> uv_to로 직선 보간한다. 한때 "다발 안
        # 상대 순서 유지"를 노려 극좌표(각도, 반지름) 보간을 써봤는데,
        # 그러면 목적지 각도가 트렁크와 조금만 달라도 센터라인이 완전
        # 직선인 가지에서조차 배선이 나사선처럼 불필요하게 빙 도는 곡선을
        # 그렸다(사용자 피드백: "직진하는 애들이 왜 커브가 생기나"). 실제로
        # 필요한 건 최단 직선 이동이지 회전이 아니므로 원래의 (u,v) 직선
        # 보간으로 되돌린다. 분기점에서 몇몇 가닥이 서로 교차하는 것은
        # 배선 설계 자체(핀 순서 재배치)에 따른 물리적으로 자연스러운
        # 현상으로 보고 그대로 둔다.
        u = (uv_from[0] * (1 - smooth_t) + uv_to[0] * smooth_t) * shrink
        v = (uv_from[1] * (1 - smooth_t) + uv_to[1] * smooth_t) * shrink

        path = np.empty_like(centerline)
        for i in range(n):
            normal, binormal = frames[i]
            path[i] = centerline[i] + u[i] * normal + v[i] * binormal
        return path

    def _route_wire_legs(self, paths, trunk_connector, from_connector, to_connector, uv_from, uv_to):
        legs = []
        if from_connector == to_connector:
            bp = paths.get(from_connector)
            if bp is None: return []
            legs.append((f"direct_{from_connector}_{uv_from}_{uv_to}", self._offset_path(bp, uv_from, uv_to, full_span=True)))
            return legs

        trunk_bp = paths.get(trunk_connector)
        if trunk_bp is None: return []

        if from_connector == trunk_connector:
            branch_bp = paths.get(to_connector)
            if branch_bp is None: return []
            # 트렁크 구간과 가지 구간을 branch_bp 하나(연속된 같은 경로)로
            # 통째로 그린다. 예전엔 트렁크 구간을 별도의 trunk_bp(P1 자신의
            # 짧은 캡 경로)로 그렸는데, 그 캡이 분기점보다 훨씬 짧게 끝나서
            # 셸이 안 그려지는 구간에서 배선이 "허공에서 갑자기 시작"하는
            # 것처럼 보였다(사용자 피드백). branch_bp는 트렁크 전체 구간을
            # 포함하고 있으므로 그대로 처음부터 끝까지 쓰면 이음매가 없다.
            leg = self._offset_path(branch_bp, uv_from, uv_to, branch_only=False)
            legs.append((f"full_{to_connector}_{uv_to[0]:.3f}_{uv_to[1]:.3f}_from_{uv_from[0]:.3f}_{uv_from[1]:.3f}", leg))
            return legs

        if to_connector == trunk_connector:
            branch_bp = paths.get(from_connector)
            if branch_bp is None: return []
            # from->trunk 케이스와 대칭: branch_bp 하나(트렁크+가지 통짜
            # 경로)를 끝에서부터 거꾸로 훑어, 가지 쪽(uv_from)에서 시작해
            # 트렁크 쪽(uv_to)으로 진행하도록 반전한다.
            leg = self._offset_path(branch_bp, uv_to, uv_from, branch_only=False)[::-1]
            legs.append((f"full_{from_connector}_{uv_from[0]:.3f}_{uv_from[1]:.3f}_to_{uv_to[0]:.3f}_{uv_to[1]:.3f}", leg))
            return legs

        # 두 가지(non-trunk) 사이의 직접 배선(예: P3-A -> P4-D): 이 배선도
        # 다른 모든 배선과 마찬가지로 실제로는 트렁크(P1) 다발 안에 처음부터
        # 들어있다가 분기점에서 갈라지는 것이어야 한다. 예전엔 트렁크 구간을
        # 아예 그리지 않고 가지 전용 구간만 이어서, 이 배선만 "트렁크 다발과
        # 무관하게 분기점에서 갑자기 새로 생겨난 선"처럼 보였다(사용자
        # 피드백: "저게 갈래선이면 원래 선에서 나뉘어져야 하는데 아니잖아").
        # 트렁크 구간에서는 다발 중심(0,0)을 지나는 것으로 그려(이 배선은
        # P1의 실제 핀에 속하지 않으므로 특정 핀 오프셋을 가질 근거가 없다),
        # 분기점부터 각자 가지를 타고 자기 목적지 핀으로 갈라지게 한다.
        bp_from = paths.get(from_connector)
        bp_to = paths.get(to_connector)
        if bp_from is None or bp_to is None: return []
        trunk_leg = self._offset_path(bp_from, (0.0, 0.0), (0.0, 0.0), branch_only=False)
        n_trunk_pts = self.samples_per_seg
        trunk_leg = trunk_leg[:n_trunk_pts]
        leg1 = self._offset_path(bp_from, (0.0, 0.0), uv_from, branch_only=True)[::-1]
        leg2 = self._offset_path(bp_to, (0.0, 0.0), uv_to, branch_only=True)
        legs.append((f"trunk_X_{from_connector}_{to_connector}", trunk_leg))
        legs.append((f"branch_{from_connector}_{uv_from[0]:.3f}_{uv_from[1]:.3f}_X", leg1))
        legs.append((f"branch_{to_connector}_{uv_to[0]:.3f}_{uv_to[1]:.3f}_X", leg2))
        return legs


if __name__ == "__main__":
    tcl_path = r"C:\AIND_Library\Git_PredicTest\x64_Cami_Lib\PSCD1500\PSCD1500\Cable\TestData\60309822_5995-37-502-9688\cable.tcl"
    # 도면(60309822)에 적힌 실측 가지 길이(mm): P2=400, P3=1500, P4=900, P5=1000.
    # P6(GND)은 도면상 물리적 가지 치수가 아니라 표1의 단자 마커(32번)로
    # 보여 실제 길이 정보가 없다 - 짧은 임의값으로 처리.
    # 그대로 쓰면 다른 파라미터(shell_radius=8 등)와 스케일이 안 맞으므로
    # 1/10로 축소해 비율만 유지한다.
    real_lengths_mm = {"P2": 400.0, "P3": 1500.0, "P4": 900.0, "P5": 1000.0, "P6": 150.0}
    branch_lengths = {k: v * 0.1 for k, v in real_lengths_mm.items()}
    builder = BranchingCableBuilder(
        tcl_path=tcl_path, shell_radius=8.0, trunk_length=90.0,
        branch_lengths=branch_lengths, max_fan_angle_deg=150.0,
    )
    mesh = builder.build()
    print(f"Built branching cable mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    out_path = r"C:\AIND_Library\Git_PredicTest\x64_Cami_Lib\PSCD1500\PSCD1500\Cable\ai_pipeline\test_output\branching_test.glb"
    mesh.export(out_path)
    print(f"Exported to {out_path}")
