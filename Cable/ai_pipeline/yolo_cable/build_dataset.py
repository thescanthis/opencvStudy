"""
labels/Success/*.json (완료된 라벨) + 같은 폴더의 원본 png 를 이용해
dataset/images/{train,val}, dataset/labels/{train,val} 을 구성한다.

- json -> YOLO txt 변환은 labelme2yolo.py 의 convert_one() 로직을 그대로 쓴다.
- val 은 마지막 1장(또는 지정 개수)만 떼어내고 나머지는 전부 train.
  (7장 안팎의 소량 데이터라 val 을 너무 많이 떼면 학습 데이터가 부족해진다.)
"""
import os
import glob
import json
import shutil

HERE = os.path.dirname(__file__)
SUCCESS_DIR = os.path.join(HERE, "labels", "Success")
DATASET_DIR = os.path.join(HERE, "dataset")

CLASSES = ["connector", "branch_sleeve", "branch_bare", "elbow", "dimension", "balloon"]

VAL_COUNT = 1  # 소량 데이터라 val 은 최소로


def convert_to_yolo_lines(data):
    W = data["imageWidth"]
    H = data["imageHeight"]
    lines = []
    for sh in data.get("shapes", []):
        label = sh["label"]
        if label not in CLASSES:
            print(f"  [skip] unknown label '{label}'")
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
    return lines


def main():
    json_paths = sorted(glob.glob(os.path.join(SUCCESS_DIR, "*.json")))
    names = [os.path.splitext(os.path.basename(p))[0] for p in json_paths]
    if not names:
        print(f"no json in {SUCCESS_DIR}")
        return

    val_names = set(names[-VAL_COUNT:]) if len(names) > VAL_COUNT else set()
    train_names = [n for n in names if n not in val_names]

    for split in ("train", "val"):
        os.makedirs(os.path.join(DATASET_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(DATASET_DIR, "labels", split), exist_ok=True)

    def place(name, split):
        img_src = os.path.join(SUCCESS_DIR, f"{name}.png")
        if not os.path.isfile(img_src):
            print(f"  [skip] {name}: png 없음 ({img_src})")
            return False
        json_src = os.path.join(SUCCESS_DIR, f"{name}.json")
        with open(json_src, "r", encoding="utf-8") as f:
            data = json.load(f)
        lines = convert_to_yolo_lines(data)

        img_dst = os.path.join(DATASET_DIR, "images", split, f"{name}.png")
        lbl_dst = os.path.join(DATASET_DIR, "labels", split, f"{name}.txt")
        shutil.copyfile(img_src, img_dst)
        with open(lbl_dst, "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        print(f"  [{split}] {name}: {len(lines)} boxes")
        return True

    print(f"train {len(train_names)}장, val {len(val_names)}장")
    ok = 0
    for n in train_names:
        ok += place(n, "train")
    for n in val_names:
        ok += place(n, "val")
    print(f"\n완료: {ok}/{len(names)}")


if __name__ == "__main__":
    main()
