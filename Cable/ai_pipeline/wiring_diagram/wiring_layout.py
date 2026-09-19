"""
회로도/배선도 렌더링 공통 레이아웃 엔진.

지금까지 A20016147 작업에서 겪은 문제: 각 핀/라벨/실드링의 y좌표를
전부 손으로 숫자를 정해서 박아넣다 보니, 다른 항목과 조금만 구조가
바뀌어도 겹침이 생기고 매번 좌표를 다시 조정해야 했다(비효율적이고
계속 반복되는 버그의 원인).

이 모듈은 "핀 이름과 순서"만 선언하면 y좌표를 자동으로 계산해서 겹치지
않는 간격을 보장하는 최소한의 레이아웃 엔진을 제공한다:
  - Column: 세로로 나열되는 핀 그룹 하나(커넥터 하나에 대응). 핀 개수만
    주면 각 핀의 y좌표를 row_h 간격으로 자동 계산한다.
  - 핀 y좌표가 이미 다른 목적으로 쓰인 경우(예: 목적지 커넥터의 순서를
    소스 커넥터에 맞추고 싶을 때) constrain_y()로 특정 핀의 y를
    강제로 맞추되, 그 결과 다른 핀과 간격이 row_h 미만으로 좁아지면
    자동으로 나머지 핀들을 밀어서 겹치지 않게 재배치한다.
"""


class Column:
    def __init__(self, name, pins, x, y0=0, row_h=60):
        self.name = name
        self.pins = list(pins)
        self.x = x
        self.row_h = row_h
        self._y = {pin: y0 + i * row_h for i, pin in enumerate(self.pins)}

    def y(self, pin):
        return self._y[pin]

    def pos(self, pin):
        return (self.x, self._y[pin])

    def set_y(self, pin, y):
        """특정 핀의 y를 직접 지정한 뒤, 순서를 지키면서 다른 핀들과
        row_h 미만으로 겹치지 않도록 전체를 재배치한다."""
        self._y[pin] = y
        self._resolve_overlaps()

    def _resolve_overlaps(self):
        # 핀을 현재 y값 기준으로 정렬해서 순서를 확인하고, 인접 간격이
        # row_h 미만이면 뒤쪽 핀들을 아래로 밀어낸다.
        ordered = sorted(self.pins, key=lambda p: self._y[p])
        for i in range(1, len(ordered)):
            prev, cur = ordered[i - 1], ordered[i]
            min_y = self._y[prev] + self.row_h
            if self._y[cur] < min_y:
                self._y[cur] = min_y

    def y_range(self):
        ys = list(self._y.values())
        return min(ys), max(ys)

    def height(self):
        lo, hi = self.y_range()
        return hi - lo


def place_after(prev_column, gap=90):
    """prev_column 바로 아래에 이어붙일 다음 컬럼의 y0 를 계산."""
    _, hi = prev_column.y_range()
    return hi + gap
