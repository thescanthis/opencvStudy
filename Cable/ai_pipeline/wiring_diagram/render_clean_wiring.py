"""
60309822 도면의 배선도(결선도)를, 원본과 동일한 레이아웃(P1 좌측 세로,
P2 상단 가로, P3/P4/P5 우측)으로 노이즈 없이 깨끗하게 다시 그린다.

연결관계는 원본 고해상도 이미지(out/_wiring_full2.png 등)를 직접 육안
대조해 확정한 값이다 - OCR/템플릿 매칭 결과가 아니라 사람이 도면을 보고
검증한 최종 정답. 이 스크립트는 "그 정답을 깨끗한 폰트/직선으로 그리는"
용도로만 쓰인다(글자 복구와 동일한 원칙: 구조를 이미 알고 있으므로
노이즈 낀 원본을 억지로 복원하지 않고 새로 그린다).
"""
import os
import cv2
import numpy as np

HERE = os.path.dirname(__file__)

FONT = cv2.FONT_HERSHEY_SIMPLEX
BLACK = (0, 0, 0)
RED = (0, 0, 200)
GRAY = (150, 150, 150)
GREEN = (0, 120, 0)

P1_PINS = ["S", "Q", "P", "N", "M", "L", "K", "J", "H", "G",
           "F", "E", "C", "D", "B", "A", "T", "U", "R", "V"]
P1_WIRES = ["K-16", "K-15", "K-14", "K-13", "K-12", "K-11", "K-10", "K-9",
            "K-8", "K-7", "K-6", "K-5", "K-4", "K-3", "K-2", "K-1",
            "K-17", "K-18", "K-19", "K-20"]

P2_TOP_PINS = ["N", "I", "G", "J", "H", "F", "E", "D", "C", "B", "A", "K", "L", "M"]
# 각 K-n 이 도착하는 P2 상단 핀 (원본 고해상도 이미지 직접 확대 대조로 확정.
# K,L,M 세 수직 리드선을 직접 픽셀 단위로 추적한 결과 K-21->M, K-22->L,
# K-23->K 이다 - 처음 육안 추정 때는 이게 반대로 잘못 읽혔었음) ----
P2_WIRE_TO_PIN = {
    "K-16": "J", "K-15": "H", "K-14": "F", "K-13": "E", "K-12": "D",
    "K-11": "C", "K-10": "B", "K-9": "A", "K-21": "M", "K-22": "L", "K-23": "K",
}

P3_PINS = ["H", "F", "E", "D", "G", "A", "C", "B"]
P3_WIRE_TO_PIN = {
    "K-21": "H", "K-22": "F", "K-23": "E", "K-8": "D",
    "K-7": "G", "K-6": "A", "K-5": "C", "K-4": "B",
}
P4_PINS = ["A", "B", "C"]
P4_WIRE_TO_PIN = {"K-18": "A", "K-19": "B", "K-20": "C"}
P5_PINS = ["A", "C", "B"]
P5_WIRE_TO_PIN = {"K-2": "A", "K-1": "C", "K-17": "B"}  # K-2 = GND


def box(img, x, y, w, h, text, font_scale=0.55):
    cv2.rectangle(img, (x, y), (x + w, y + h), BLACK, 1)
    (tw, th), _ = cv2.getTextSize(text, FONT, font_scale, 1)
    cv2.putText(img, text, (x + (w - tw) // 2, y + (h + th) // 2),
                FONT, font_scale, BLACK, 1, cv2.LINE_AA)


def render(out_path="out/60309822_clean_wiring.png"):
    W, H = 2000, 1300
    img = np.full((H, W, 3), 255, dtype=np.uint8)
    cv2.putText(img, "60309822 - Wiring Diagram (clean re-render, verified against source)",
                (30, 25), FONT, 0.6, BLACK, 1, cv2.LINE_AA)

    # ---- P1 좌측 세로열 ----
    p1_x = 60
    p1_y0 = 260
    row_h = 42
    box_w, box_h = 40, 30
    p1_positions = {}
    for i, pin in enumerate(P1_PINS):
        y = p1_y0 + i * row_h
        box(img, p1_x, y, box_w, box_h, pin)
        p1_positions[pin] = (p1_x + box_w, y + box_h // 2)

    # ---- P2 상단 가로열: 원본처럼 P1보다 훨씬 위쪽에 배치하고, 핀 아래로
    # 고정 길이 리드선을 내린 지점(p2_lead_positions)을 배선의 실제 도착점으로
    # 쓴다 - 그래야 "짧은 수직 리드선 + 긴 대각선"이라는 원본 구조가 재현됨 ----
    p2_x0 = 700
    p2_y = 40
    p2_cell_w = 44
    p2_lead_len = 55
    p2_used_pins = set(P2_WIRE_TO_PIN.values())  # 실제 배선이 연결된 핀만
    p2_positions = {}
    p2_lead_positions = {}
    for i, pin in enumerate(P2_TOP_PINS):
        x = p2_x0 + i * p2_cell_w
        box(img, x, p2_y, p2_cell_w - 4, 40, pin)
        cx = x + (p2_cell_w - 4) // 2
        p2_positions[pin] = (cx, p2_y + 40)
        p2_lead_positions[pin] = (cx, p2_y + 40 + p2_lead_len)
        if pin in p2_used_pins:
            cv2.line(img, p2_positions[pin], p2_lead_positions[pin], GRAY, 1)

    # ---- P3 우측열: 원본처럼 P1의 H,G,F,E,C,D,B 줄과 거의 같은 높이에 배치
    # (K-8~K-4 는 H~C 줄과 나란히 수평, K-21~23 만 P2 에서 내려옴) ----
    p3_x = 1750
    p3_anchor_pins = ["H", "G", "F", "E", "C", "D", "B"]  # P1 기준 K-8..K-4 의 y
    p3_positions = {}
    p3_row_pins = ["H", "F", "E", "D", "G", "A", "C", "B"]
    # P3 D,G,A,C,B (=K-8,7,6,5,4) 는 P1 H,G,F,E,C 줄과 같은 y
    same_row_map = {"D": "H", "G": "G", "A": "F", "C": "E", "B": "C"}
    for pin in ("D", "G", "A", "C", "B"):
        p1_pin = same_row_map[pin]
        _, y1 = p1_positions[p1_pin]
        p3_positions[pin] = (p3_x, y1)
    # P3 H,F,E (=K-21,22,23) 는 그 위쪽에 별도 배치
    top_y0 = p3_positions["D"][1] - 3 * row_h
    for i, pin in enumerate(("H", "F", "E")):
        p3_positions[pin] = (p3_x, top_y0 + i * row_h)

    for pin in P3_PINS:
        _, y = p3_positions[pin]
        box(img, p3_x, y - box_h // 2, box_w, box_h, pin)

    # P4 A,B,C(=K-18,19,20) 는 P1 U,R,V 줄과 같은 y (원본처럼 수평선)
    p4_x = 1750
    p4_same_row = {"A": "U", "B": "R", "C": "V"}
    p4_positions = {}
    for pin, p1_pin in p4_same_row.items():
        _, y1 = p1_positions[p1_pin]
        p4_positions[pin] = (p4_x, y1)
        box(img, p4_x, y1 - box_h // 2, box_w, box_h, pin)

    # P5 A,C,B(=K-2,1,17=GND) 는 P1 B,A,T 줄과 같은 y
    p5_x = 1750
    p5_same_row = {"A": "B", "C": "A", "B": "T"}
    p5_positions = {}
    for pin, p1_pin in p5_same_row.items():
        _, y1 = p1_positions[p1_pin]
        p5_positions[pin] = (p5_x, y1)
        box(img, p5_x, y1 - box_h // 2, box_w, box_h, pin)
    p4_y0 = p4_positions["A"][1]
    p5_y0 = p5_positions["A"][1]

    cv2.putText(img, "P1 28-16P", (p1_x, p1_y0 + len(P1_PINS) * row_h + 25),
                FONT, 0.55, BLACK, 1, cv2.LINE_AA)
    cv2.putText(img, "P2 20-27P", (p2_x0, p2_y - 15), FONT, 0.55, BLACK, 1, cv2.LINE_AA)
    cv2.putText(img, "P3 20-7P", (p3_x + 50, p3_positions["H"][1] - 15), FONT, 0.55, BLACK, 1, cv2.LINE_AA)
    cv2.putText(img, "P4 16S-5S", (p4_x + 50, p4_y0 - 15), FONT, 0.55, BLACK, 1, cv2.LINE_AA)
    cv2.putText(img, "P5 10SL-3SN", (p5_x + 50, p5_y0 - 15), FONT, 0.55, BLACK, 1, cv2.LINE_AA)

    # ---- 배선 그리기: P1 -> (P2 위쪽 분기 | P3/P4/P5 직선) ----
    mid_x = 620  # P1 리드선이 모이는 중간 x
    for pin, wire in zip(P1_PINS, P1_WIRES):
        x1, y1 = p1_positions[pin]
        cv2.line(img, (x1, y1), (x1 + 40, y1), GRAY, 1)
        cv2.putText(img, wire, (x1 + 45, y1 + 4), FONT, 0.45, RED, 1, cv2.LINE_AA)

        if wire in P2_WIRE_TO_PIN:
            # 원본 확대 이미지로 직접 확인한 실제 형태: 거의 전 구간 수평,
            # 목적지(P2 핀의 수직 리드선 x좌표) 바로 앞에서만 짧게 수직으로
            # 꺾여 올라간다 - L자형(직각)이지 사선이 아니다.
            dest_pin = P2_WIRE_TO_PIN[wire]
            dx, dy = p2_lead_positions[dest_pin]
            cv2.line(img, (x1 + 40, y1), (dx, y1), GRAY, 1)
            cv2.line(img, (dx, y1), (dx, dy), GRAY, 1)
        else:
            for dests, positions in ((P3_WIRE_TO_PIN, p3_positions),
                                      (P4_WIRE_TO_PIN, p4_positions),
                                      (P5_WIRE_TO_PIN, p5_positions)):
                if wire in dests:
                    dx, dy = positions[dests[wire]]
                    cv2.line(img, (x1 + 40, y1), (dx, dy), GRAY, 1)
                    cv2.putText(img, wire, (dx - 90, dy + 4), FONT, 0.45, RED, 1, cv2.LINE_AA)
                    break

    # ---- P2 -> P3 (K-21,K-22,K-23) 분기선: P2 K,L,M 핀에서 수직으로 내려와
    # P3 H,F,E 높이에서 수평으로 꺾여 도착 (원본과 동일한 L자형) ----
    p2_to_p3 = {"K-21": ("M", "H"), "K-22": ("L", "F"), "K-23": ("K", "E")}
    for wire, (p2pin, p3pin) in p2_to_p3.items():
        sx, sy = p2_positions[p2pin]
        dx, dy = p3_positions[p3pin]
        cv2.line(img, (sx, sy), (sx, dy), GRAY, 1)
        cv2.line(img, (sx, dy), (dx, dy), GRAY, 1)
        cv2.putText(img, wire, (sx + 5, min(sy, dy) + 15), FONT, 0.42, RED, 1, cv2.LINE_AA)

    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    cv2.imwrite(os.path.join(HERE, out_path), img)
    print(f"-> {out_path}")


if __name__ == "__main__":
    render()
