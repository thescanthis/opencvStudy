"""
50073721 도면의 배선도를 깨끗하게 재구성한다. 이 도면은 60309822보다
훨씬 단순한 구조: J14 커넥터와 J26 커넥터가 같은 핀 이름끼리 그대로
1:1 직결된다(A-A, B-B, ... U-U). 원본 확대 이미지를 직접 확인해 확정.
"""
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = os.path.dirname(__file__)

PINS = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K",
        "L", "M", "N", "P", "R", "S", "T", "U"]

BLACK = "black"
GRAY = "0.55"


def render_png(out_path="out/50073721_clean_wiring.png"):
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    W, H = 900, 900
    img = np.full((H, W, 3), 255, dtype=np.uint8)
    cv2.putText(img, "50073721 - Wiring Diagram (J14 <-> J26, direct 1:1)",
                (20, 25), FONT, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    row_h = 40
    y0 = 70
    box_w, box_h = 40, 28
    left_x, right_x = 60, 800

    cv2.putText(img, "J14", (left_x, y0 - 15), FONT, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, "J26", (right_x, y0 - 15), FONT, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    for i, pin in enumerate(PINS):
        y = y0 + i * row_h
        cv2.rectangle(img, (left_x, y), (left_x + box_w, y + box_h), (0, 0, 0), 1)
        (tw, th), _ = cv2.getTextSize(pin, FONT, 0.5, 1)
        cv2.putText(img, pin, (left_x + (box_w - tw) // 2, y + (box_h + th) // 2),
                    FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.rectangle(img, (right_x, y), (right_x + box_w, y + box_h), (0, 0, 0), 1)
        cv2.putText(img, pin, (right_x + (box_w - tw) // 2, y + (box_h + th) // 2),
                    FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cy = y + box_h // 2
        cv2.line(img, (left_x + box_w, cy), (right_x, cy), (150, 150, 150), 1)

    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    cv2.imwrite(os.path.join(HERE, out_path), img)
    print(f"-> {out_path}")


def render_pdf(out_path="out/50073721_clean_wiring.pdf"):
    W, H = 900, 900
    fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
    ax.set_xlim(0, W)
    ax.set_ylim(-H, 0)
    ax.axis("off")
    ax.text(20, -20, "50073721 - Wiring Diagram (J14 <-> J26, direct 1:1)", fontsize=9, va="center")

    row_h = 40
    y0 = 70
    box_w, box_h = 40, 28
    left_x, right_x = 60, 800

    ax.text(left_x, -(y0 - 15), "J14", fontsize=9)
    ax.text(right_x, -(y0 - 15), "J26", fontsize=9)

    for i, pin in enumerate(PINS):
        y = y0 + i * row_h
        ax.add_patch(Rectangle((left_x, -(y + box_h)), box_w, box_h, fill=False, edgecolor=BLACK, linewidth=0.8))
        ax.text(left_x + box_w / 2, -(y + box_h / 2), pin, ha="center", va="center", fontsize=8)
        ax.add_patch(Rectangle((right_x, -(y + box_h)), box_w, box_h, fill=False, edgecolor=BLACK, linewidth=0.8))
        ax.text(right_x + box_w / 2, -(y + box_h / 2), pin, ha="center", va="center", fontsize=8)
        cy = y + box_h / 2
        ax.plot([left_x + box_w, right_x], [-cy, -cy], color=GRAY, linewidth=0.7)

    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    full_path = os.path.join(HERE, out_path)
    fig.savefig(full_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"-> {full_path}")


if __name__ == "__main__":
    render_png()
    render_pdf()
