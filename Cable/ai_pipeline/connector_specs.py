"""
실제 커넥터 규격의 핀 배치(UV) 좌표 테이블.

출처와 검증:
- 도면(50073721, CX-9790K)의 MS3475 W14-18P (18개 20콘텍트)
- 도면(A20024923, C432-2)의 D38999/46WD18PN, D38999/46WD18SN (#20-18EA)
두 도면의 "코넥터 핀 후면투영 상세도"를 직접 육안으로 판독한 결과, 제조사/
형번(MS3475 vs D38999)이 다른데도 핀 배치(A~U, 18핀)가 완전히 동일한 5행
육각형 격자였다. 즉 이 배치는 특정 형번 전용이 아니라 "18핀 원형 군용
커넥터"에 공통으로 쓰이는 업계 표준 배열로 보인다 - 그래서 형번 문자열이
아니라 "18핀 육각형"이라는 패턴 이름으로 등록하고, 알려진 형번들을 전부 그
패턴에 매핑한다.

PDF가 통째로 스캔된 래스터 이미지라 프로그램이 자동으로 좌표를 추출할 수
없어 두 도면 모두 수동으로 확인했다. 임의로 지어낸 값이 아니라 도면에 그려진
배치를 격자 단위로 옮긴 것이며, 실제 각도(도 단위)까지 정밀 측정한 것은 아니다.

18핀 육각형 배치:

          A               L
      B       N       M       K
  C       P       U       T       J
      D       R       S       H
          E       F       G

행(row)은 위에서부터 2, 1, 0, -1, -2 (중앙이 0), 열(col)은 좌우 대칭.
정확한 mm 단위 피치 대신, 격자 간격을 1 unit으로 둔 상대 좌표를 쓴다 -
connector_layout에서 실제 반경(connector_radius)으로 스케일링된다.
"""
from typing import Dict, Tuple, Optional, List

# (row, col) 격자 좌표. row: 위(+2) ~ 아래(-2). col: 좌(-2) ~ 우(+2).
# 도면의 실제 배치를 그대로 옮김. MS3475 W14-18P와 D38999/46WD18PN/SN
# 양쪽 도면에서 동일하게 확인된 배치라 형번을 가리지 않는 공용 패턴이다.
HEX_18PIN_GRID: Dict[str, Tuple[float, float]] = {
    "A": (2, -0.5),
    "L": (2, 0.5),
    "B": (1, -1.5),
    "N": (1, -0.5),
    "M": (1, 0.5),
    "K": (1, 1.5),
    "C": (0, -2.0),
    "P": (0, -1.0),
    "U": (0, 0.0),
    "T": (0, 1.0),
    "J": (0, 2.0),
    "D": (-1, -1.5),
    "R": (-1, -0.5),
    "S": (-1, 0.5),
    "H": (-1, 1.5),
    "E": (-2, -1.0),
    "F": (-2, 0.0),
    "G": (-2, 1.0),
}

# 격자 간격(행/열 사이 거리)을 정규화해서 원 반경 대비 실제 도면 비율에 맞춘다.
# 도면상 5행이 커넥터 원 지름의 약 80%를 채우므로, row 범위(-2~2, 총 4)가
# radius*1.6 정도에 대응하도록 스케일한다.
_GRID_TO_UV_SCALE = 0.42


# 출처: 도면(50073731, 케이블 결합체, CX-9719K)의 "코넥터 핀 후면투영 상세도".
# 커넥터 자체 명칭은 OA-7086K-J20 / TNB-J5 처럼 케이블마다 다른 조립체명이지만,
# 핀 배치 패턴(4핀, 정사각형/다이아몬드: A 상단-B 우측-C 하단-D 좌측, 90도
# 균등 간격)은 여러 4핀 커넥터에 공통으로 재사용될 가능성이 높아 "4핀 다이아몬드"
# 라는 일반 패턴으로 등록한다. 이름표(OA-7086K-J20 등)가 아니라 핀 배치
# 형태(4핀/다이아몬드)로 매칭하는 게 이 패턴을 재사용하는 핵심이다.
DIAMOND_4PIN_GRID: Dict[str, Tuple[float, float]] = {
    "A": (1.0, 0.0),    # 상단
    "B": (0.0, 1.0),    # 우측
    "C": (-1.0, 0.0),   # 하단
    "D": (0.0, -1.0),   # 좌측
}


# 출처: 도면(A20024950, C414-2)의 P2 커넥터 "코넥터 핀 후면투영 상세도",
# MS3475W12-10SW (#20-10EA), 10핀. 3행 배치:
#     H   A   B
#   G   K   J   C
#     F   E   D
DIAMOND_10PIN_GRID: Dict[str, Tuple[float, float]] = {
    "H": (1, -1.0),
    "A": (1, 0.0),
    "B": (1, 1.0),
    "G": (0, -1.5),
    "K": (0, -0.5),
    "J": (0, 0.5),
    "C": (0, 1.5),
    "F": (-1, -1.0),
    "E": (-1, 0.0),
    "D": (-1, 1.0),
}
_DIAMOND_10PIN_SCALE = 0.42


# 출처: 도면(A20024959, C410-2)의 "코넥터 핀 후면투영 상세도",
# D38999/46WC35PN / D38999/46WC35SN (#22D-22EA), 22콘텍트, 3중 동심원 배치:
#   외곽 링(반경 1.0): 1~14 (14개), 12시부터 시계방향 균등 분포
#   중간 링(반경 0.55): 15~21 (7개), 12시부터 시계방향 균등 분포
#   중앙(반경 0): 22 (1개)
# 도면에서 핀 번호 각각의 정확한 각도까지 자로 측정한 것은 아니고, "14개/7개가
# 각 링에 균등 분포"하는 것으로 근사했다 - 실측 각도가 아니라 균등 분포 근사임을
# 명시한다.
def _build_concentric_22pin_grid() -> Dict[str, Tuple[float, float]]:
    import math
    grid: Dict[str, Tuple[float, float]] = {}

    outer_pins = [str(i) for i in range(1, 15)]  # 1~14
    for i, pin in enumerate(outer_pins):
        angle = math.pi / 2 - (2 * math.pi * i / len(outer_pins))
        grid[pin] = (math.cos(angle), math.sin(angle))

    mid_pins = [str(i) for i in range(15, 22)]  # 15~21
    for i, pin in enumerate(mid_pins):
        angle = math.pi / 2 - (2 * math.pi * i / len(mid_pins))
        grid[pin] = (0.55 * math.cos(angle), 0.55 * math.sin(angle))

    grid["22"] = (0.0, 0.0)
    return grid


CONCENTRIC_22PIN_GRID: Dict[str, Tuple[float, float]] = _build_concentric_22pin_grid()


# 출처: 도면(A20024962, C403-2 / A20024951, C451-2)의 D38999/46WD35PN·SN
# (#22D-37EA), 37콘텍트. 도면(A20024968, C406-2)의 D38999/46WE35PN
# (#22D-55EA), 55콘텍트. 둘 다 도면에서 벌집형(육각 조밀) 배치인 것은
# 확인했지만, 스캔 해상도가 낮고 숫자가 서로 겹쳐서 핀 번호 각각의 정확한
# 위치까지는 육안으로 확정하지 못했다. 그래서 이건 실측이 아니라 "N콘텍트
# 표준 벌집형 배치"의 수학적 근사(중심 1개 + 6개 링 + 12개 링 + 18개 링 +
# ... 6각 격자로 링마다 6*ring개)이다. 균등 원형(각도만 등분)보다는 실제
# 형태에 가깝지만, 번호별 정확한 매칭은 보장하지 않는다 - 정밀도가 필요해지면
# 고해상도 원본으로 재판독해야 한다.
def _build_honeycomb_grid(total_pins: int, max_ring: int) -> Dict[str, Tuple[float, float]]:
    import math
    grid: Dict[str, Tuple[float, float]] = {}
    grid["1"] = (0.0, 0.0)

    pin = 2
    for ring in range(1, max_ring + 1):
        count = ring * 6
        for i in range(count):
            angle = math.pi / 2 - (2 * math.pi * i / count)
            r = ring / max_ring
            grid[str(pin)] = (r * math.cos(angle), r * math.sin(angle))
            pin += 1
            if pin > total_pins:
                break
        if pin > total_pins:
            break
    return grid


HONEYCOMB_37PIN_GRID: Dict[str, Tuple[float, float]] = _build_honeycomb_grid(37, max_ring=3)
HONEYCOMB_55PIN_GRID: Dict[str, Tuple[float, float]] = _build_honeycomb_grid(55, max_ring=4)


# 출처: 도면(A20024982, C405-2)의 D38999/46WH55PN, D38999/26WH55SA
# (#20-55EA), 55콘텍트. 앞의 숫자(1~55) 라벨 55핀 벌집형과 핀 개수·기하학적
# 형태(벌집형)는 같지만, 이 규격은 라벨이 숫자가 아니라 A~Z, a~z, AA, BB...
# 순서의 알파벳이다. 좌표 계산 로직은 55핀 벌집형과 동일하게 재사용하고,
# 라벨만 알파벳 시퀀스로 바꿔 매핑한다.
def _alpha_pin_labels(n: int) -> List[str]:
    labels = []
    for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        labels.append(c)
    for c in "abcdefghijklmnopqrstuvwxyz":
        labels.append(c)
    i = 0
    while len(labels) < n:
        labels.append(chr(ord('A') + i) * 2)  # AA, BB, CC, ...
        i += 1
    return labels[:n]


def _build_honeycomb_grid_alpha(total_pins: int, max_ring: int) -> Dict[str, Tuple[float, float]]:
    numeric = _build_honeycomb_grid(total_pins, max_ring)
    labels = _alpha_pin_labels(total_pins)
    # numeric의 키는 "1".."total_pins" 문자열 - 순서대로 알파벳 라벨에 매핑
    return {labels[int(k) - 1]: v for k, v in numeric.items()}


HONEYCOMB_55PIN_ALPHA_GRID: Dict[str, Tuple[float, float]] = _build_honeycomb_grid_alpha(55, max_ring=4)


# 출처: 도면(A50035007-6)의 P1 커넥터 "연결기 후면 투영 상세도",
# MS3475W14-19P (19핀). 5행 배치:
#         B   A   M
#       C   P   N   L
#     D   R   V   U   K
#       E   S   T   J
#         F   G   H
HEX_19PIN_GRID: Dict[str, Tuple[float, float]] = {
    "B": (2, -0.5), "A": (2, 0.5), "M": (2, 1.5),
    "C": (1, -1.5), "P": (1, -0.5), "N": (1, 0.5), "L": (1, 1.5),
    "D": (0, -2.0), "R": (0, -1.0), "V": (0, 0.0), "U": (0, 1.0), "K": (0, 2.0),
    "E": (-1, -1.5), "S": (-1, -0.5), "T": (-1, 0.5), "J": (-1, 1.5),
    "F": (-2, -0.5), "G": (-2, 0.5), "H": (-2, 1.5),
}

# 같은 도면의 P2 커넥터, MS3475W12-10P (10핀). 3행 배치:
#     B   A   H
#   C   J   K   G
#     D   E   F
# 참고: 앞서 등록한 MS3475W12-10SW(3-4-3, H/A/B - G/K/J/C - F/E/D)와 핀 개수는
# 같지만 좌우가 반전돼 있다 - 커넥터는 Pin(P)/Socket(S) 버전에 따라 후면
# 투영이 거울상이 되는 게 정상이므로 별개 항목으로 둔다.
HEX_10PIN_P_GRID: Dict[str, Tuple[float, float]] = {
    "B": (1, -1.0), "A": (1, 0.0), "H": (1, 1.0),
    "C": (0, -1.5), "J": (0, -0.5), "K": (0, 0.5), "G": (0, 1.5),
    "D": (-1, -1.0), "E": (-1, 0.0), "F": (-1, 1.0),
}


# 출처: 도면(A60024600, C331-...)의 P2 커넥터 "연결자 후면 투영도",
# D38999/46FF11PN (#16-11EA), 11핀. 4행 배치:
#     A       J
#   B   K       H
#   C   L       G
#     D   E   F
HEX_11PIN_GRID: Dict[str, Tuple[float, float]] = {
    "A": (1.5, -0.5), "J": (1.5, 1.5),
    "B": (0.5, -1.5), "K": (0.5, -0.5), "H": (0.5, 1.5),
    "C": (-0.5, -1.5), "L": (-0.5, -0.5), "G": (-0.5, 1.5),
    "D": (-1.5, -1.0), "E": (-1.5, 0.0), "F": (-1.5, 1.0),
}


def get_known_connector_layout(connector_type: str) -> Optional[Dict[str, Tuple[float, float]]]:
    """알려진 커넥터 규격명에 대해 (u, v) 상대 좌표 dict를 반환한다.

    반환값의 u, v는 [-1, 1] 근방의 상대 좌표이며, 호출자가 원하는 반경으로
    곱해서 쓴다. 등록되지 않은 규격이면 None을 반환한다 (호출자는 이 경우
    connector_layout.layout_pins_uniform_circle 같은 근사치로 fallback해야 한다).
    """
    key = connector_type.strip().upper().replace(" ", "")

    hex_18pin_names = (
        "MS3475W14-18P", "MS3475-W14-18P", "MS3475W1418P",
        "D38999/46WD18PN", "D38999-46WD18PN", "D3899946WD18PN",
        "D38999/46WD18SN", "D38999-46WD18SN", "D3899946WD18SN",
        "HEX18PIN", "18PIN-HEX",
    )
    if key in hex_18pin_names:
        return {
            pin: (col * _GRID_TO_UV_SCALE, row * _GRID_TO_UV_SCALE)
            for pin, (row, col) in HEX_18PIN_GRID.items()
        }
    if key in ("DIAMOND4PIN", "4PIN-DIAMOND", "4PINDIAMOND"):
        return dict(DIAMOND_4PIN_GRID)
    if key in ("MS3475W12-10SW", "MS3475-W12-10SW", "MS3475W1210SW", "10PIN-HEX", "HEX10PIN"):
        return {
            pin: (col * _DIAMOND_10PIN_SCALE, row * _DIAMOND_10PIN_SCALE)
            for pin, (row, col) in DIAMOND_10PIN_GRID.items()
        }
    if key in ("D38999/46WC35PN", "D38999-46WC35PN", "D3899946WC35PN",
               "D38999/46WC35SN", "D38999-46WC35SN", "D3899946WC35SN",
               "22PIN-CONCENTRIC", "CONCENTRIC22PIN"):
        return dict(CONCENTRIC_22PIN_GRID)
    if key in ("D38999/46WD35PN", "D38999-46WD35PN", "D3899946WD35PN",
               "D38999/46WD35SN", "D38999-46WD35SN", "D3899946WD35SN",
               "37PIN-HONEYCOMB", "HONEYCOMB37PIN"):
        return dict(HONEYCOMB_37PIN_GRID)
    if key in ("D38999/46WE35PN", "D38999-46WE35PN", "D3899946WE35PN",
               "D38999/46WE35SN", "D38999-46WE35SN", "D3899946WE35SN",
               "55PIN-HONEYCOMB", "HONEYCOMB55PIN"):
        return dict(HONEYCOMB_55PIN_GRID)
    if key in ("D38999/46WH55PN", "D38999-46WH55PN", "D3899946WH55PN",
               "D38999/26WH55SA", "D38999-26WH55SA", "D3899926WH55SA",
               "55PIN-HONEYCOMB-ALPHA", "HONEYCOMB55PINALPHA"):
        return dict(HONEYCOMB_55PIN_ALPHA_GRID)
    if key in ("MS3475W14-19P", "MS3475-W14-19P", "MS3475W1419P", "19PIN-HEX", "HEX19PIN"):
        return {
            pin: (col * _GRID_TO_UV_SCALE, row * _GRID_TO_UV_SCALE)
            for pin, (row, col) in HEX_19PIN_GRID.items()
        }
    if key in ("MS3475W12-10P", "MS3475-W12-10P", "MS3475W1210P", "10PIN-HEX-P", "HEX10PINP"):
        return {
            pin: (col * _DIAMOND_10PIN_SCALE, row * _DIAMOND_10PIN_SCALE)
            for pin, (row, col) in HEX_10PIN_P_GRID.items()
        }
    if key in ("D38999/46FF11PN", "D38999-46FF11PN", "D3899946FF11PN",
               "D38999/46FF11SA", "D38999-46FF11SA", "D3899946FF11SA",
               "11PIN-HEX", "HEX11PIN"):
        return {
            pin: (col * _GRID_TO_UV_SCALE, row * _GRID_TO_UV_SCALE)
            for pin, (row, col) in HEX_11PIN_GRID.items()
        }
    return None


if __name__ == "__main__":
    for name in ("MS3475 W14-18P", "D38999/46WD18PN", "4pin diamond"):
        layout = get_known_connector_layout(name)
        print(f"-- {name} --")
        if layout is None:
            print("  (not found)")
            continue
        for pin, (u, v) in sorted(layout.items()):
            print(f"  {pin}: ({u:+.3f}, {v:+.3f})")
