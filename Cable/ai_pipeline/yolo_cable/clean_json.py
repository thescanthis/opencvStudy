"""
X-AnyLabeling json 정리:
  1) 완전 중복 박스 제거 (IoU > 0.9)
  2) (수동 지정) 잘못 친 shape 인덱스 제거

사용: python clean_json.py <name> [삭제할 인덱스들...]
  예: python clean_json.py 50073721 26 27 28 29 30 31 32 33
  (인덱스는 list_json.py 로 확인. 중복 제거는 자동)
"""
import os
import sys
import json

HERE = os.path.dirname(__file__)
IMG_DIR = os.path.join(HERE, "images", "all")


def bbox(shape):
    xs = [p[0] for p in shape["points"]]
    ys = [p[1] for p in shape["points"]]
    return min(xs), min(ys), max(xs), max(ys)


def iou(a, b):
    ax1, ay1, ax2, ay2 = bbox(a)
    bx1, by1, bx2, by2 = bbox(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua


def main():
    name = sys.argv[1]
    drop = set(int(x) for x in sys.argv[2:])
    p = os.path.join(IMG_DIR, f"{name}.json")
    d = json.load(open(p, encoding="utf-8"))
    shapes = d["shapes"]

    # 1) 수동 지정 인덱스 제거
    kept = [s for i, s in enumerate(shapes) if i not in drop]
    removed_manual = len(shapes) - len(kept)

    # 2) 중복(IoU>0.9, 같은 라벨) 제거 - 뒤에 오는 것 삭제
    final = []
    for s in kept:
        dup = any(s["label"] == t["label"] and iou(s, t) > 0.9 for t in final)
        if not dup:
            final.append(s)
    removed_dup = len(kept) - len(final)

    d["shapes"] = final
    json.dump(d, open(p, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    from collections import Counter
    c = Counter(s["label"] for s in final)
    print(f"{name}: {len(shapes)} -> {len(final)} shapes "
          f"(수동삭제 {removed_manual}, 중복삭제 {removed_dup})")
    for k, v in sorted(c.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
