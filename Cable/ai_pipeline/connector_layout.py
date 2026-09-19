"""
커넥터 핀 UV(원형 배치) 좌표 생성.

우선순위:
  1. connector_specs에 등록된 실제 규격(예: MS3475 W14-18P)이면 그 도면 기반
     좌표를 그대로 쓴다 - 임의의 근사가 아니라 실물 배치다.
  2. 등록되지 않은 규격이거나 규격명을 모르면, 균등 원형 배치로 근사한다
     (placeholder). 이 경우 실제 각도와는 다를 수 있음을 호출자가 인지해야 한다.

나머지 파이프라인은 이 모듈이 반환하는 {pin_name: (u, v)} dict에만 의존하므로,
새 규격을 connector_specs에 추가하기만 하면 자동으로 정확한 배치가 반영된다.
"""
from typing import List, Dict, Tuple, Optional
import numpy as np

from connector_specs import get_known_connector_layout


def layout_pins_uniform_circle(pin_names: List[str], radius: float = 1.0) -> Dict[str, Tuple[float, float]]:
    """핀 이름 리스트를 받아 원 둘레에 균등 분포시킨 (u, v) 좌표를 반환한다.

    placeholder 근사: 실제 커넥터 규격 각도가 아니라 등각 분할이다.
    """
    n = len(pin_names)
    layout: Dict[str, Tuple[float, float]] = {}
    if n == 0:
        return layout

    for i, name in enumerate(pin_names):
        angle = (i / n) * 2 * np.pi
        u = radius * np.cos(angle)
        v = radius * np.sin(angle)
        layout[name] = (u, v)

    return layout


def layout_pins(pin_names: List[str], radius: float = 1.0, connector_type: Optional[str] = None) -> Dict[str, Tuple[float, float]]:
    """핀 배치를 결정한다: connector_type이 알려진 규격이면 실제 도면 배치를,
    아니면 균등 원형 근사를 사용한다.

    connector_type이 알려진 규격이더라도 pin_names에 그 규격에 없는 핀
    (예: TCL에만 있는 'Braid')이 섞여 있으면, 알려진 핀은 정확한 위치에,
    나머지는 원 바깥 여백에 배치해 데이터 손실 없이 모두 그린다.
    """
    if connector_type:
        known = get_known_connector_layout(connector_type)
        if known:
            layout: Dict[str, Tuple[float, float]] = {}
            unresolved = []
            for name in pin_names:
                if name in known:
                    u, v = known[name]
                    layout[name] = (u * radius, v * radius)
                else:
                    unresolved.append(name)
            # 알려진 규격에 없는 핀(예: Braid/실드)은 원 가장자리 바깥에 균등 배치
            for i, name in enumerate(unresolved):
                angle = (i / max(1, len(unresolved))) * 2 * np.pi
                layout[name] = (radius * 1.15 * np.cos(angle), radius * 1.15 * np.sin(angle))
            return layout

    return layout_pins_uniform_circle(pin_names, radius=radius)


if __name__ == "__main__":
    pins = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'Braid']

    print("-- Unknown connector (uniform circle fallback) --")
    for name, (u, v) in layout_pins_uniform_circle(pins).items():
        print(f"  {name}: ({u:.3f}, {v:.3f})")

    print("-- MS3475 W14-18P (real drawing-based layout) --")
    for name, (u, v) in layout_pins(pins, connector_type="MS3475 W14-18P").items():
        print(f"  {name}: ({u:.3f}, {v:.3f})")
