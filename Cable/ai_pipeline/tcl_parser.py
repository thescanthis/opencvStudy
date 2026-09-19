"""
CableEye .tcl 파일 파서.

포맷 (@로 구분된 필드):
  p@Connector@PinName@Bank:Index      - 핀 정의 (커넥터의 어느 핀이 어느 뱅크 채널에 매핑되는지)
  o@From@To@FromBank@ToBank@Width     - Open(단선) 검사 대상 = 실제 점대점 배선.
                                         이것이 "그려야 할 전선"이다.
  s@(...)@(...)@(...)@(...)          - Short(합선) 검사용 핀 조합. 배선이 아니라
                                         "이 두 지점이 붙어있으면 불량"이라는 검증
                                         페어이므로 시각화 대상이 아니다.
  i@(...)@(...)@(...)@(...)          - Isolation(절연) 검사용 페어. s@ 와 동일하게
                                         시각화 대상이 아니다.

From/To는 "Connector-PinName" 형태 (예: "J14-A").
"""
from dataclasses import dataclass
from typing import List, Dict, Optional
import os


@dataclass
class PinDef:
    connector: str   # 예: "J14"
    pin_name: str     # 예: "A", "Braid"
    bank: str         # 예: "B5"
    index: int        # 예: 1

    @property
    def full_name(self) -> str:
        return f"{self.connector}-{self.pin_name}"


@dataclass
class WireConnection:
    """o@ 라인 하나 = 실제로 존재하는 점대점 배선 한 가닥."""
    from_pin: str   # "J14-A"
    to_pin: str     # "J26-A"
    from_bank_ch: str  # "B5:1"
    to_bank_ch: str    # "B5:33"


@dataclass
class CableTopology:
    pins: List[PinDef]
    wires: List[WireConnection]

    @property
    def connectors(self) -> List[str]:
        """등장 순서를 보존한 커넥터 이름 목록 (예: ["J14", "J26"])."""
        seen = []
        for p in self.pins:
            if p.connector not in seen:
                seen.append(p.connector)
        return seen

    def pins_of(self, connector: str) -> List[PinDef]:
        return [p for p in self.pins if p.connector == connector]

    def connector_of_full_pin(self, full_pin_name: str) -> Optional[str]:
        """"P1-S" 같은 "Connector-PinName" 문자열에서 커넥터 이름을 찾는다.

        핀 이름 자체에 '-'가 섞여 있을 수 있으므로(예: "J14-Braid" 는 안전하지만
        혹시 모를 케이스 대비), 단순 split(1)이 아니라 알려진 커넥터 이름 중
        가장 긴 접두사와 매치되는 것을 찾는다.
        """
        best = None
        for c in self.connectors:
            prefix_hyphen = c + "-"
            prefix_space = c + " "
            if full_pin_name.startswith(prefix_hyphen) or full_pin_name.startswith(prefix_space) or full_pin_name == c:
                if best is None or len(c) > len(best):
                    best = c
        return best


def parse_tcl(tcl_path: str) -> CableTopology:
    """cable.tcl을 읽어 핀 정의와 실제 배선(o@) 목록을 반환한다.

    s@ / i@ 라인은 검사용 조합일 뿐 배선이 아니므로 의도적으로 무시한다.
    """
    pins: List[PinDef] = []
    wires: List[WireConnection] = []

    if not tcl_path or not os.path.exists(tcl_path):
        return CableTopology(pins=pins, wires=wires)

    with open(tcl_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            fields = line.split("@")
            tag = fields[0]

            if tag == "p" and len(fields) >= 4:
                connector, pin_name, bank_idx = fields[1], fields[2], fields[3]
                bank, _, idx_str = bank_idx.partition(":")
                try:
                    index = int(idx_str)
                except ValueError:
                    index = -1
                pins.append(PinDef(connector=connector, pin_name=pin_name, bank=bank, index=index))

            elif tag == "o" and len(fields) >= 5:
                from_pin, to_pin, from_bank_ch, to_bank_ch = fields[1], fields[2], fields[3], fields[4]
                wires.append(WireConnection(
                    from_pin=from_pin, to_pin=to_pin,
                    from_bank_ch=from_bank_ch, to_bank_ch=to_bank_ch,
                ))

            # tag == "s" / "i": Short/Isolation 검사 조합. 배선이 아니므로 스킵.

    return CableTopology(pins=pins, wires=wires)


if __name__ == "__main__":
    test_path = r"C:\AIND_Library\Git_PredicTest\x64_Cami_Lib\PSCD1500\PSCD1500\Cable\TestData\50073721_5995-37-501-9359\cable.tcl"
    topo = parse_tcl(test_path)
    print(f"Connectors: {topo.connectors}")
    for c in topo.connectors:
        pins = topo.pins_of(c)
        print(f"  {c}: {len(pins)} pins -> {[p.pin_name for p in pins]}")
    print(f"Wires (actual point-to-point connections): {len(topo.wires)}")
    for w in topo.wires[:5]:
        print(f"  {w.from_pin} -> {w.to_pin}")
