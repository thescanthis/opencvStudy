"""
어제 테스트한 5종 케이블을 A20016147과 같은 방식(도면 실측 각도/길이,
각진 몸통, 분기점 슬리브 마디, 커넥터/치수 제외)으로 다시 만든다.

각 케이블의 구조는 도면(Cable.pdf)을 직접 읽어 파악했다:

1) A60025772: P1-P2 완전 직선, 분기 없음. 전체 2900mm.
2) A60025765: P1에서 직선(700mm) 후 정확히 대칭 Y자로 P2/P3 분기
   (도면 주기: "90도로 구부릴 것"). P2/P3 각 450mm.
3) A60025763: P1에서 직선(150mm) 몸통이 오른쪽 P2(930mm)까지 이어지고,
   중간(500mm 지점)에서 P3/P4 두 가지가 아래로 거의 나란히 갈라짐
   (P3 1300mm, P4는 P3 경로에서 다시 300mm 더 - 실제로는 P3->P4 순차
   분기로 보임, 도면 배선도 참고).
4) A20016148: P1에서 직선(2100mm)으로 가다 P2(1700mm 아래)가 갈라지고,
   몸통 계속(200mm) 가다 P5(2900mm 위)가 갈라지고, 몸통 계속(1700mm)
   가다 P4(1150mm 위)가 갈라지고 P3(엘보)로 끝.
5) A60025764: P1에서 시작해 여러 단계로 P7/P6/P5/P4/P2/P3가 순차적으로
   갈라지는 복잡한 트리.
"""
import os
import numpy as np

from drawing_shape_builder import build_cable_shape

OUT_DIR = r"C:\AIND_Library\Git_PredicTest\x64_Cami_Lib\PSCD1500\PSCD1500\Cable\ai_pipeline\test_output\drawing_shapes"


def branch_point(origin, angle_deg, length):
    ang = np.radians(angle_deg)
    return (origin[0] + length * np.cos(ang), origin[1] + length * np.sin(ang))


def build_a60025772():
    """P1-P2 완전 직선, 분기 없음."""
    trunk = [(0.0, 0.0), (2900.0, 0.0)]
    branches = {}
    out = os.path.join(OUT_DIR, "1_A60025772_straight.glb")
    build_cable_shape(trunk, branches, out, sleeve_len=0)  # 직선이라 슬리브 불필요


def build_a60025765():
    """P1(90도 엘보) -> 직선(700) -> 완만한 Y자로 P2/P3.
    OpenCV 허프 변환으로 도면을 직접 측정한 각도: 위쪽(P3) 약 -14도,
    아래쪽(P2) 약 +15도(몸통 기준) - 45도 대칭이 아니라 훨씬 완만하다."""
    P1 = (0.0, 0.0)
    J1 = (700.0, 0.0)
    trunk = [P1, J1]
    P3 = branch_point(J1, 14.0, 450.0)    # 위로 14도(완만)
    P2 = branch_point(J1, -15.0, 450.0)   # 아래로 15도(완만)
    branches = {1: [P3, P2]}
    out = os.path.join(OUT_DIR, "2_A60025765_Y_symmetric.glb")
    build_cable_shape(trunk, branches, out)


def build_a60025763():
    """P1(90도 엘보) -> 직선(150, 몸통 계속 930까지 P2) 중간(500)에서
    P3/P4가 거의 수직(-81도, -82도, OpenCV 허프 변환 측정치)으로 나란히
    아래로 갈라짐 - 이전 추정치(-50/-60도)보다 훨씬 가파르다."""
    P1 = (0.0, 0.0)
    J = (500.0, 0.0)
    P2 = (930.0, 0.0)
    trunk = [P1, J, P2]
    P3 = branch_point(J, -81.0, 1300.0)  # 도면 "1300.0" 치수(P3 가지 길이)
    P4 = branch_point(J, -82.0, 1300.0 + 300.0)  # 도면 "300.0" 만큼 P3보다 더 김
    branches = {1: [P3, P4]}
    out = os.path.join(OUT_DIR, "3_A60025763_close_pair.glb")
    build_cable_shape(trunk, branches, out)


def build_a20016148():
    """P1 -> 직선(2100) -> J1(P2 분기, 아래로) -> 직선(200) -> J2(P5 분기,
    위로) -> 직선(1700) -> J3(P4 분기, 위로) -> P3.
    OpenCV 허프 변환으로 도면을 직접 측정한 각도: 세 분기 전부 정확히
    45.00도(P2는 -45, P5/P4는 +45) - 이전 추정치(-40/35/35)는 부정확했다."""
    P1 = (0.0, 0.0)
    J1 = (2100.0, 0.0)
    J2 = (2100.0 + 200.0, 0.0)
    J3 = (2100.0 + 200.0 + 1700.0, 0.0)
    P3 = (2100.0 + 200.0 + 1700.0 + 150.0, 0.0)
    trunk = [P1, J1, J2, J3, P3]

    P2 = branch_point(J1, -45.0, 1700.0)
    P5 = branch_point(J2, 45.0, 2900.0)
    P4 = branch_point(J3, 45.0, 1150.0)

    branches = {1: [P2], 2: [P5], 3: [P4]}
    out = os.path.join(OUT_DIR, "4_A20016148_triple_branch.glb")
    build_cable_shape(trunk, branches, out)


def build_a60025764():
    """도면을 다시 정밀하게 읽어 바로잡은 구조(이전 버전은 완전히 틀렸음).

    실제 형태: 수평 몸통(P1->J1->J2->...)이 J1에서 T자로 갈라져 수직
    몸통(아래로)이 시작된다. P7은 J2에서 위로 갈라지는 가지고, P6/P5는
    그보다 더 오른쪽의 분기점에서 갈라진다. 수직 몸통은 J1에서 아래로
    내려가며 P2, P3, P4가 순서대로 갈라진다(P4는 수직 몸통의 끝점 자체).

    도면 치수: P1->J1 200, J1->J2 280, J2->J3(P7분기) ... 이후 1350 더
    가서 P6/P5 분기점. J1->P2분기 3900, P2분기->P3분기 220, P3분기->P4
    1100."""
    P1 = (0.0, 0.0)
    J1 = (200.0, 0.0)                 # 수직 몸통이 갈라지는 T자
    J2 = (200.0 + 280.0, 0.0)         # P7이 위로 갈라지는 지점
    J3 = (200.0 + 280.0 + 1350.0, 0.0)  # P6/P5가 갈라지는 지점

    trunk_top = [P1, J1, J2, J3]

    P7 = branch_point(J2, 90.0, 350.0)
    P6 = branch_point(J3, 25.0, 200.0)
    P5 = branch_point(J3, 50.0, 150.0)

    # 수직 몸통: J1에서 아래로 꺾여 시작. 도면 치수: J1 -> P2분기 3900,
    # P2분기 -> P3분기 220, P3분기 -> P4(끝) 1100.
    Jv2 = (J1[0], J1[1] - 3900.0)     # P2가 갈라지는 지점
    P2 = (Jv2[0] + 150.0, Jv2[1])     # P2는 거의 수평으로 오른쪽(150mm)

    Jv3 = (Jv2[0], Jv2[1] - 220.0)    # P3가 갈라지는 지점
    P3 = branch_point(Jv3, -45.0, 470.0)  # P3는 45도 엘보 커넥터로 오른쪽 아래

    P4 = (Jv3[0], Jv3[1] - 1100.0)    # 수직 몸통의 끝(P4)

    trunk_down = [J1, Jv2, Jv3, P4]

    import trimesh
    from drawing_shape_builder import build_cable_shape as _build

    out = os.path.join(OUT_DIR, "5_A60025764_complex_tree.glb")

    # J1은 위쪽 체인(수평)에서는 "그대로 지나가는 통과점"이자 수직 몸통이
    # 시작되는 T자다. trunk_top에서 J1은 분기점이 아니라 몸통이 그대로
    # 이어지는 정점이므로 branches에 넣지 않고, 대신 trunk_down의 시작을
    # 여기 연결한다(두 메쉬가 J1에서 같은 좌표로 만나 자연스럽게 이어짐).
    mesh1 = _build(trunk_top, {2: [P7], 3: [P6, P5]}, out + ".part1.glb")
    mesh2 = _build(trunk_down, {1: [P2], 2: [P3]}, out + ".part2.glb")
    merged = trimesh.util.concatenate([mesh1, mesh2])
    merged.export(out)
    print(f"merged -> {out}")
    os.remove(out + ".part1.glb")
    os.remove(out + ".part2.glb")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    build_a60025772()
    build_a60025765()
    build_a60025763()
    build_a20016148()
    build_a60025764()


if __name__ == "__main__":
    main()
