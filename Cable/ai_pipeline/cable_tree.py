"""
TCL의 배선 목록에서 케이블의 "물리적 트리 구조"를 추론한다.

케이블은 결국 하나의 몸통(트렁크)에서 갈라져 나가는 트리다. TCL에는 그
트리가 직접 적혀있지 않고 점대점 배선(o@)만 있으므로, 커넥터끼리 몇 가닥이
연결돼 있는지를 세어 트리를 복원한다.

핵심 아이디어:
  - 커넥터를 노드, "두 커넥터 사이에 배선이 몇 가닥 있는가"를 간선 가중치로
    보는 그래프를 만든다.
  - 가장 가닥 수가 많은(= 다른 커넥터들과 가장 많이 이어진) 커넥터를 루트
    (트렁크 시작점)로 잡는다.
  - 루트에서 시작해 가중치가 큰 간선을 우선으로 방문하는 최대 신장 트리를
    만든다. 배선이 많이 지나가는 경로가 물리적으로 굵은 몸통이라는 뜻이므로,
    이렇게 하면 실제 케이블의 몸통-가지 구조와 대체로 일치한다.

예) A60025763: P1-P2가 3가닥, P2-P3와 P2-P4가 각 1가닥
    -> 트리는 P1 -> P2 -> {P3, P4}, 즉 P2에서 다시 갈라지는 2중 분기.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import Counter

from tcl_parser import CableTopology


@dataclass
class CableNode:
    """트리의 한 노드(커넥터). children이 있으면 여기서 갈라진다."""
    connector: str
    children: List["CableNode"] = field(default_factory=list)
    parent: Optional[str] = None
    # 이 노드와 부모를 잇는 구간을 지나는 배선 가닥 수(굵기 참고용)
    wire_count: int = 0

    def walk(self):
        """자기 자신부터 시작해 모든 자손 노드를 순회한다."""
        yield self
        for child in self.children:
            yield from child.walk()

    def depth_of(self, connector: str, _depth: int = 0) -> Optional[int]:
        if self.connector == connector:
            return _depth
        for child in self.children:
            found = child.depth_of(connector, _depth + 1)
            if found is not None:
                return found
        return None


def connector_link_counts(topo: CableTopology) -> Counter:
    """커넥터 쌍별로 그 사이를 잇는 배선이 몇 가닥인지 센다."""
    pairs: Counter = Counter()
    for wire in topo.wires:
        a = topo.connector_of_full_pin(wire.from_pin)
        b = topo.connector_of_full_pin(wire.to_pin)
        if a is None or b is None or a == b:
            continue
        pairs[tuple(sorted([a, b]))] += 1
    return pairs


def pick_root(topo: CableTopology, pairs: Counter) -> str:
    """트렁크 시작점을 고른다.

    "다른 커넥터들과 이어진 총 가닥 수"가 가장 많은 커넥터를 루트로 본다.
    동률이면 이어진 상대 커넥터 수가 많은 쪽, 그래도 동률이면 핀 수가 많은
    쪽을 택해 결과가 실행할 때마다 흔들리지 않게 한다.
    """
    total_wires: Counter = Counter()
    neighbors: Dict[str, set] = {c: set() for c in topo.connectors}
    for (a, b), count in pairs.items():
        total_wires[a] += count
        total_wires[b] += count
        neighbors[a].add(b)
        neighbors[b].add(a)

    def score(connector: str):
        return (
            total_wires.get(connector, 0),
            len(neighbors.get(connector, ())),
            len(topo.pins_of(connector)),
            # 이름은 마지막 tie-breaker (역순: P1이 P2보다 우선)
            -topo.connectors.index(connector),
        )

    return max(topo.connectors, key=score)


def build_cable_tree(topo: CableTopology, root: Optional[str] = None) -> Optional[CableNode]:
    """커넥터 연결 관계에서 케이블의 트리 구조를 복원한다.

    가중치가 큰 간선을 우선 채택하는 최대 신장 트리(Prim 방식)를 만든다.
    배선이 많이 지나가는 경로가 물리적으로 몸통에 해당하므로, 이렇게 하면
    "몸통에서 가지가 뻗고, 그 가지에서 또 갈라지는" 실제 구조에 가깝게 나온다.
    """
    if not topo.connectors:
        return None

    pairs = connector_link_counts(topo)
    if not pairs:
        # 배선이 전혀 없거나 전부 같은 커넥터 안에서 끝나는 경우
        return CableNode(connector=topo.connectors[0])

    root_name = root or pick_root(topo, pairs)

    adjacency: Dict[str, List[Tuple[str, int]]] = {c: [] for c in topo.connectors}
    for (a, b), count in pairs.items():
        adjacency[a].append((b, count))
        adjacency[b].append((a, count))

    nodes: Dict[str, CableNode] = {root_name: CableNode(connector=root_name)}
    visited = {root_name}
    # (가닥 수, 부모, 자식) - 가닥 수가 많은 간선부터 채택
    frontier: List[Tuple[int, str, str]] = [
        (count, root_name, other) for other, count in adjacency[root_name]
    ]

    while frontier:
        frontier.sort(key=lambda item: (-item[0], item[2]))
        count, parent_name, child_name = frontier.pop(0)
        if child_name in visited:
            continue
        visited.add(child_name)
        child = CableNode(connector=child_name, parent=parent_name, wire_count=count)
        nodes[child_name] = child
        nodes[parent_name].children.append(child)
        for other, other_count in adjacency[child_name]:
            if other not in visited:
                frontier.append((other_count, child_name, other))

    # 어떤 커넥터가 배선으로 이어져 있지 않으면(고아) 루트 바로 아래 붙인다.
    for connector in topo.connectors:
        if connector not in visited:
            orphan = CableNode(connector=connector, parent=root_name, wire_count=0)
            nodes[connector] = orphan
            nodes[root_name].children.append(orphan)

    return nodes[root_name]


def describe_tree(node: CableNode, indent: int = 0) -> str:
    """트리를 사람이 읽을 수 있게 문자열로 만든다(디버그/확인용)."""
    pad = "  " * indent
    suffix = f" ({node.wire_count}가닥)" if node.parent else " [트렁크 시작]"
    lines = [f"{pad}{node.connector}{suffix}"]
    for child in node.children:
        lines.append(describe_tree(child, indent + 1))
    return "\n".join(lines)


if __name__ == "__main__":
    import os
    from tcl_parser import parse_tcl

    base = r"C:\AIND_Library\Git_PredicTest\x64_Cami_Lib\PSCD1500\PSCD1500\Cable\TestData"
    for name in [
        "A60025772_6150-37-520-1801",
        "A60025765_6150-37-520-1795",
        "A60025763_6150-37-520-1796",
        "A20016148_6145-37-520-5297",
        "A60025764_6150-37-520-1794",
    ]:
        topo = parse_tcl(os.path.join(base, name, "cable.tcl"))
        tree = build_cable_tree(topo)
        print(f"=== {name}")
        print(describe_tree(tree))
        print()
