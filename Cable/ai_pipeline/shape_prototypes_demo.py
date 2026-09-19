"""
케이블 형태 원형 4종(직선/커브/꺾임/분기) 프로토타입 생성 스크립트.

각 프로토타입은:
  - AI(사진 기반 depth/mask) 없이, cable_shapes.py의 순수 기하학적 중심선으로
    매끈한 셸을 만들고,
  - TCL의 실제 배선(o@)을 connector_specs.py의 실제 커넥터 핀 배치에 따라
    그 안에 정갈하게 배치한다 (curved_test.glb에서 확인된, 가닥이 서로
    겹치지 않는 방식).

사용하는 TCL/커넥터 규격은 기존에 확보한 실제 데이터(50073721 케이블,
MS3475 W14-18P, 17가닥 배선)를 그대로 재사용해 네 형태를 비교할 수 있게 한다.
"""
import os
import numpy as np
import trimesh

from cable_shapes import straight_centerline, curved_centerline, bent_centerline
from wire_builder import CircuitWireBuilder, create_tube_mesh
from branching_wire_builder import BranchingCableBuilder

TCL_2CONN = r"C:\AIND_Library\Git_PredicTest\x64_Cami_Lib\PSCD1500\PSCD1500\Cable\TestData\50073721_5995-37-501-9359\cable.tcl"
TCL_BRANCH = r"C:\AIND_Library\Git_PredicTest\x64_Cami_Lib\PSCD1500\PSCD1500\Cable\TestData\60309822_5995-37-502-9688\cable.tcl"
TCL_BRANCH2 = r"C:\AIND_Library\Git_PredicTest\x64_Cami_Lib\PSCD1500\PSCD1500\Cable\TestData\A20016147_6150-37-520-5295\cable.tcl"

OUT_DIR = r"C:\AIND_Library\Git_PredicTest\x64_Cami_Lib\PSCD1500\PSCD1500\Cable\ai_pipeline\test_output\shape_prototypes"

SHELL_RADIUS = 10.0
SHELL_COLOR = [100, 149, 237, 120]
CONNECTOR_TYPES_2CONN = {"J14": "MS3475 W14-18P", "J26": "MS3475 W14-18P"}


def build_two_connector_shape(name: str, centerline: np.ndarray) -> trimesh.Trimesh:
    """직선/커브/꺾임 공통: 셸 + TCL 실배선(2-커넥터)을 centerline을 따라 만든다."""
    shell = create_tube_mesh(centerline, SHELL_RADIUS, SHELL_COLOR, sides=16)

    wire_builder = CircuitWireBuilder(
        tcl_path=TCL_2CONN,
        bounds_min=centerline.min(axis=0) - SHELL_RADIUS,
        bounds_max=centerline.max(axis=0) + SHELL_RADIUS,
        shell_radius=SHELL_RADIUS,
        connector_types=CONNECTOR_TYPES_2CONN,
    )
    wires = wire_builder.build(centerline)

    parts = [shell]
    if len(wires.vertices) > 0:
        parts.append(wires)
    return trimesh.util.concatenate(parts)


def build_branching_shape() -> trimesh.Trimesh:
    """분기: BranchingCableBuilder가 트렁크+가지 구조를 자체적으로 만든다.

    도면(60309822)에 적힌 실측 가지 길이(mm)를 반영: P2=400, P3=1500,
    P4=900, P5=1000. P6(GND)은 도면상 물리적 가지 치수가 없어 임의 짧은 값.
    다른 파라미터(shell_radius 등)와 스케일을 맞추기 위해 1/10로 축소.
    """
    real_lengths_mm = {"P2": 400.0, "P3": 1500.0, "P4": 900.0, "P5": 1000.0, "P6": 150.0}
    branch_lengths = {k: v * 0.1 for k, v in real_lengths_mm.items()}
    builder = BranchingCableBuilder(
        tcl_path=TCL_BRANCH,
        shell_radius=SHELL_RADIUS,
        trunk_length=90.0,
        branch_lengths=branch_lengths,
        max_fan_angle_deg=150.0,
    )
    return builder.build()


def build_branching_shape2() -> trimesh.Trimesh:
    """분기 2번째 예시(A20016147): 실물 사진과 도면(Cable.pdf) 모두 확인한
    3갈래 케이블. 실제로는 P1 트렁크 -> (1차 분기: P2 갈라짐) -> (2차 분기:
    P3/P4 갈라짐)인 2단 분기지만, 이 예시는 단순화해 P1에서 P2/P3/P4가
    동시에 3갈래로 갈라지는 것으로 근사한다(BranchingCableBuilder가 아직
    중첩 분기를 지원하지 않음).

    도면 실측 치수(mm): P1-트렁크 1800, 1차분기-2차분기 1300, 2차분기-P4 370,
    2차분기-P3 1250. P2 가지는 1차분기 지점에서 갈라지므로 트렁크 관점에서는
    "P1~1차분기(1800)"보다 더 길게 뻗어 나간 것처럼 보이도록 트렁크+1300
    근방 길이를 준다. 스케일은 다른 예시와 맞추기 위해 1/40로 축소.
    P1이 6핀으로 P4와 동률이라 트렁크 자동판별이 불안정 -> trunk_connector
    명시.
    """
    real_lengths_mm = {"P2": 1300.0 + 300.0, "P3": 1300.0 + 1250.0, "P4": 1300.0 + 370.0}
    scale = 1.0 / 15.0
    branch_lengths = {k: v * scale for k, v in real_lengths_mm.items()}
    trunk_length_scaled = 1800.0 * scale
    builder = BranchingCableBuilder(
        tcl_path=TCL_BRANCH2,
        shell_radius=SHELL_RADIUS,
        trunk_length=trunk_length_scaled,
        branch_lengths=branch_lengths,
        corner_radius=SHELL_RADIUS * 1.2,
        max_fan_angle_deg=90.0,
        trunk_connector="P1",
    )
    return builder.build()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    shapes = {
        "1_straight": build_two_connector_shape("straight", straight_centerline(length=220.0)),
        "2_curved": build_two_connector_shape("curved", curved_centerline(length=220.0, amplitude=45.0, waves=1.0)),
        "3_bent": build_two_connector_shape("bent", bent_centerline(seg1_length=120.0, seg2_length=120.0, bend_angle_deg=90.0, corner_radius=25.0)),
        "4_branching": build_branching_shape(),
        "5_branching_A20016147": build_branching_shape2(),
    }

    for name, mesh in shapes.items():
        out_path = os.path.join(OUT_DIR, f"{name}.glb")
        mesh.export(out_path)
        print(f"[{name}] {len(mesh.vertices)} vertices -> {out_path}")


if __name__ == "__main__":
    main()
