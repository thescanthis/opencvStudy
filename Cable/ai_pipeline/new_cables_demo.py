"""
실제 TCL/도면을 보고 새 케이블 5종을 GLB로 생성한다.

지금까지 확정된 4가지 요구사항을 지키는 것이 목표:
  1. 배선은 처음부터 서로 겹치지 않게 (구간마다 다발 안 자리를 미리 배정)
  2. 분기는 최대한 자연스럽게 (fork_branch_centerline: 접선 연속 포크)
  3. 커브는 예시(curved_test.glb)처럼 다발이 하나로 뭉쳐 따라가게
  4. 갈래선은 갈라지는 시점에서 나뉘게 (트리 경로를 통째로 이어 하나의 튜브)

고른 5종(패턴이 서로 겹치지 않게):
  1) A60025772 - 2커넥터, 배선 1가닥. 가장 단순한 커브.
  2) A60025765 - 3커넥터 Y자. 10가닥이 5+5로 정확히 반씩 갈라짐.
  3) A60025763 - 4커넥터. P2에서 다시 갈라지는 2중 분기.
  4) A20016148 - 5커넥터. P1->P3->P5로 이어지는 2중 분기 + 곁가지 2개.
  5) A60025764 - 7커넥터. 한 점에서 6갈래로 퍼지는 부채꼴.
"""
import os
import numpy as np
import trimesh

from tcl_parser import parse_tcl
from cable_tree import build_cable_tree, describe_tree
from tree_cable_builder import TreeCableBuilder
from wire_builder import CircuitWireBuilder, create_tube_mesh
from cable_shapes import curved_centerline

BASE = r"C:\AIND_Library\Git_PredicTest\x64_Cami_Lib\PSCD1500\PSCD1500\Cable\TestData"
OUT_DIR = r"C:\AIND_Library\Git_PredicTest\x64_Cami_Lib\PSCD1500\PSCD1500\Cable\ai_pipeline\test_output\new_cables"

SHELL_RADIUS = 10.0
SHELL_COLOR = [100, 149, 237, 120]


def tcl_of(folder: str) -> str:
    return os.path.join(BASE, folder, "cable.tcl")


def build_two_connector_curved(folder: str, length: float = 240.0,
                                amplitude: float = 55.0, waves: float = 1.0) -> trimesh.Trimesh:
    """커넥터가 2개뿐인 케이블: 트리랄 게 없으니 곡선 하나에 배선을 태운다.

    curved_test.glb에서 확인된 방식 그대로 - 다발이 곡선을 따라 하나로
    뭉쳐서 가고, 양 끝에서만 각자 핀 위치로 흩어진다(요구사항 3).
    """
    centerline = curved_centerline(length=length, amplitude=amplitude, waves=waves, n=90)
    shell = create_tube_mesh(centerline, SHELL_RADIUS, SHELL_COLOR, sides=16)
    builder = CircuitWireBuilder(
        tcl_path=tcl_of(folder),
        bounds_min=centerline.min(axis=0) - SHELL_RADIUS,
        bounds_max=centerline.max(axis=0) + SHELL_RADIUS,
        shell_radius=SHELL_RADIUS,
    )
    wires = builder.build(centerline)
    parts = [shell]
    if len(wires.vertices) > 0:
        parts.append(wires)
    return trimesh.util.concatenate(parts)


def build_tree(folder: str, **kwargs) -> trimesh.Trimesh:
    """커넥터가 3개 이상인 케이블: 트리 구조를 복원해 재귀적으로 만든다."""
    builder = TreeCableBuilder(tcl_path=tcl_of(folder), shell_radius=SHELL_RADIUS, **kwargs)
    return builder.build()


CABLES = [
    # (출력이름, 폴더, 빌더, 파라미터)
    ("1_A60025772_simple_curve", "A60025772_6150-37-520-1801", "curve",
     dict(length=260.0, amplitude=60.0, waves=1.0)),

    ("2_A60025765_Y_split", "A60025765_6150-37-520-1795", "tree",
     dict(root_length=130.0, default_segment_length=120.0, fan_angle_deg=70.0)),

    ("3_A60025763_double_branch", "A60025763_6150-37-520-1796", "tree",
     dict(root_length=110.0, default_segment_length=110.0, fan_angle_deg=85.0)),

    ("4_A20016148_deep_tree", "A20016148_6145-37-520-5297", "tree",
     dict(root_length=130.0, default_segment_length=105.0, fan_angle_deg=85.0,
          segment_lengths={"P2": 70.0, "P4": 120.0, "P3": 115.0, "P5": 95.0})),

    # 6갈래는 부채꼴이 좁으면 가지들이 서로 붙어 뭉쳐 보인다. 각도를 넓히고
    # 가지마다 길이를 달리 해서 끝단이 한 줄로 늘어서지 않게 한다.
    ("5_A60025764_fan6", "A60025764_6150-37-520-1794", "tree",
     dict(root_length=150.0, default_segment_length=110.0, fan_angle_deg=155.0,
          segment_lengths={"P2": 95.0, "P3": 130.0, "P4": 105.0,
                           "P5": 140.0, "P6": 115.0, "P7": 85.0})),
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, folder, kind, params in CABLES:
        topo = parse_tcl(tcl_of(folder))
        print(f"=== {name}")
        print(f"    커넥터 {len(topo.connectors)}개, 배선 {len(topo.wires)}가닥")
        if kind == "tree":
            tree = build_cable_tree(topo)
            print("    " + describe_tree(tree).replace("\n", "\n    "))
            mesh = build_tree(folder, **params)
        else:
            mesh = build_two_connector_curved(folder, **params)

        if len(mesh.vertices) == 0:
            print("    !! 메쉬 생성 실패")
            continue
        out_path = os.path.join(OUT_DIR, f"{name}.glb")
        mesh.export(out_path)
        print(f"    -> {len(mesh.vertices)} vertices, {out_path}")
        print()


if __name__ == "__main__":
    main()
