"""
A20016147 도면(Cable.pdf) 실제 형상을 최대한 그대로 재현한다.

[셸 전용 모드] 사용자가 "내부 배선은 나중에 뽑을 테니 지금은 도면 겉모습
(케이블 갖데기 형상)만 그대로 만들어달라"고 요청함에 따라, 배선(와이어)
튜브는 생성하지 않고 셸(외피) 튜브만 만든다. TreeCableBuilder.build()는
셸+배선을 함께 만들므로, 여기서는 그 내부의 _build_segments()만 호출해
센터라인을 얻은 뒤 셸만 튜브화한다.

지금까지 자동 트리 복원(cable_tree)은 "커넥터 사이 배선 가닥 수"만 보고
P1 -> {P2, P3, P4} 1단 3갈래로 판단했다. 하지만 도면을 보면 실제로는:

  P1 --(직선 1800)--> 분기점 --(1250)--> P2
                          \\--(1300)--> 분기점2 --(370)--> P4
                                          \\-- 근처에서 P3로도 짧게 이어짐

즉 P1이 한 번 꺾인 뒤 먼저 P2로 한 가지가 갈라지고, 나머지 한 가지가 또
P3/P4로 갈라지는 2단 구조다. TCL의 배선 가닥 수만으로는 이 구조가 안
드러나므로(P4가 5가닥으로 가장 굵어 자동 로직은 P4를 우선시함), 도면을
보고 트리와 꺾임각을 직접 지정한다.

길이는 도면 실측 mm를 1/15로 축소해 기존 A20016147 예시와 스케일을 맞춘다.
"""
import os
import trimesh
from cable_tree import CableNode, describe_tree
from tree_cable_builder import TreeCableBuilder, SHELL_COLOR
from wire_builder import create_tube_mesh

TCL = r"C:\AIND_Library\Git_PredicTest\x64_Cami_Lib\PSCD1500\PSCD1500\Cable\TestData\A20016147_6150-37-520-5295\cable.tcl"
OUT = r"C:\AIND_Library\Git_PredicTest\x64_Cami_Lib\PSCD1500\PSCD1500\Cable\ai_pipeline\test_output\6_A20016147_drawing_shape.glb"
OUT_SHELL_ONLY = r"C:\AIND_Library\Git_PredicTest\x64_Cami_Lib\PSCD1500\PSCD1500\Cable\ai_pipeline\test_output\6_A20016147_shell_only.glb"

SCALE = 1.0 / 15.0


def build_tree() -> CableNode:
    """도면(Cable.pdf)에 그려진 실제 트리 구조 (사용자가 도면을 다시 짚어준
    버전).

    P1(루트) --1800--> 분기점1 --1300--> 분기점2 --370--> P4
                          \\                  \\--1250--> P3
                           \\--1300--> P2

    즉 분기점1에서 P2가 먼저 갈라지고(도면상 P2쪽 사선), 본선은 계속
    이어지다가 분기점2에서 P3/P4가 동시에(형제로) 갈라진다. 분기점2 자체는
    실제 커넥터가 아닌 가상 마디이므로, TCL에 없는 이름("J2")으로 트리에만
    넣고 배선/핀 UV 단계에서는 무시되도록 wire_routes 쪽에서 걸러진다.
    """
    root = CableNode(connector="P1")
    p2 = CableNode(connector="P2", parent="P1", wire_count=1)
    junction2 = CableNode(connector="J2", parent="P1", wire_count=7)
    p3 = CableNode(connector="P3", parent="J2", wire_count=2)
    p4 = CableNode(connector="P4", parent="J2", wire_count=5)
    junction2.children = [p4, p3]
    root.children = [p2, junction2]
    return root


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tree = build_tree()
    print(describe_tree(tree))

    builder = TreeCableBuilder(
        tcl_path=TCL,
        shell_radius=10.0,
        root_length=1800.0 * SCALE,
        segment_lengths={
            "P2": 1250.0 * SCALE,   # 분기점1 -> P2 (도면 "1250.0" 치수)
            "J2": 1300.0 * SCALE,   # 분기점1 -> 분기점2 (도면 "1300.0" 치수)
            "P4": 370.0 * SCALE,    # 분기점2 -> P4 (도면 "370.0" 치수, 계속 직진)
            "P3": 700.0 * SCALE,    # 분기점2 -> P3 (도면에 전체 길이 명시 없음,
                                     # 사선 길이가 P4쪽보다 짧게 그려진 비율로 추정)
        },
        # 도면상 P1 직선 구간 끝(분기점1)에서 P2가 아래로 크게 꺾여 나가고,
        # 본선(J2)은 완만하게 이어진다. 분기점2에서는 P4가 거의 직진을
        # 유지하고 P3가 아래로 짧게 꺾여 빠진다.
        segment_angles={
            "P2": -42.0,   # 분기점1에서 아래로 (도면 사선 각도)
            "J2": 10.0,    # 본선은 오른쪽 위로 완만하게
            "P4": 3.0,     # 분기점2에서도 계속 거의 직진(수평에 가깝게)
            "P3": -50.0,   # 분기점2에서 아래로 급하게 꺾여 짧게 빠짐
        },
        fan_angle_deg=90.0,
        default_segment_length=100.0 * SCALE,
    )
    # 셸 전용: 배선(와이어)은 만들지 않고, 트리 구조로 얻은 각 구간의
    # 중심선을 그대로 튜브화해 케이블 겉모습(갖데기)만 만든다.
    segments = builder._build_segments(tree)
    meshes = []
    for seg in segments.values():
        shell = create_tube_mesh(seg.centerline, builder.shell_radius, SHELL_COLOR, sides=24)
        if len(shell.vertices) > 0:
            meshes.append(shell)
    mesh = trimesh.util.concatenate(meshes) if meshes else trimesh.Trimesh()
    print(f"vertices: {len(mesh.vertices)}")
    mesh.export(OUT_SHELL_ONLY)
    print(f"-> {OUT_SHELL_ONLY}")


if __name__ == "__main__":
    main()
