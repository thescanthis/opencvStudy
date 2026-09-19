"""
X-AnyLabeling / labelme 형식 .json -> YOLO .txt 변환.

X-AnyLabeling 은 <name>.json (labelme 형식)으로 저장한다.
이 스크립트는 JSON_DIRS(images/all, labels/Json) 의 *.json 을 읽어
labels/all/*.txt (YOLO 정규화 박스)로 변환한다. rectangle shape 만 처리
(polygon 은 bbox 로 변환).

사용:
    python labelme2yolo.py
"""
import os
import json
import glob

HERE = os.path.dirname(__file__)
# X-AnyLabeling 이 만든 .json 을 찾을 위치들 (사용자 폴더 운영에 따라 여러 곳).
JSON_DIRS = [
    os.path.join(HERE, "images", "all"),
    os.path.join(HERE, "labels", "Json"),
]
LBL_DIR = os.path.join(HERE, "labels", "all")

CLASSES = ["connector", "branch_sleeve", "branch_bare", "elbow", "dimension", "balloon"]


def convert_one(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    W = data["imageWidth"]
    H = data["imageHeight"]
    lines = []
    for sh in data.get("shapes", []):
        label = sh["label"]
        if label not in CLASSES:
            print(f"  [skip] unknown label '{label}' in {os.path.basename(json_path)}")
            continue
        ci = CLASSES.index(label)
        pts = sh["points"]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        cx = (x1 + x2) / 2 / W
        cy = (y1 + y2) / 2 / H
        w = (x2 - x1) / W
        h = (y2 - y1) / H
        lines.append(f"{ci} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    name = os.path.splitext(os.path.basename(json_path))[0]
    out = os.path.join(LBL_DIR, f"{name}.txt")
    with open(out, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    return name, len(lines)


def main():
    os.makedirs(LBL_DIR, exist_ok=True)
    jsons = []
    for d in JSON_DIRS:
        jsons.extend(glob.glob(os.path.join(d, "*.json")))
    jsons = sorted(set(jsons))
    if not jsons:
        print(f"no .json in {JSON_DIRS} (X-AnyLabeling 으로 라벨링 먼저)")
        return
    total = 0
    for j in jsons:
        name, n = convert_one(j)
        total += n
        print(f"{name}: {n} boxes -> labels/all/{name}.txt")
    print(f"\n{len(jsons)} files, {total} boxes total")


if __name__ == "__main__":
    main()
