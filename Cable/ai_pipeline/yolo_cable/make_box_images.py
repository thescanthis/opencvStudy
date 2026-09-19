"""
라벨링 완료한 도면들의 박스 시각화 이미지를 일괄 생성한다.

labels/Json/*.json 과 labels/Success/*.json 을 모두 스캔해서(중복 이름은
한 번만), 짝이 되는 원본 이미지를 찾아 클래스별 색상 박스를 그린 뒤
labels/BoxCableImg/<name>_boxed.png 로 저장한다.

원본 이미지는 다음 순서로 찾는다(있는 곳 사용):
  1. labels/Success/<name>.png   (완성본 원본이 여기 있는 경우)
  2. images/all/<name>.png       (render_pages.py 로 렌더된 원본)

사용법:
    python make_box_images.py            # 전체 재생성
    python make_box_images.py 50073721   # 특정 도면만
"""
import os
import sys
import glob
import json
import cv2

HERE = os.path.dirname(__file__)
JSON_DIRS = [
    os.path.join(HERE, "labels", "Json"),
    os.path.join(HERE, "labels", "Success"),
    os.path.join(HERE, "images", "all"),
]
IMG_CANDIDATE_DIRS = [
    os.path.join(HERE, "labels", "Success"),
    os.path.join(HERE, "images", "all"),
]
OUT_DIR = os.path.join(HERE, "labels", "BoxCableImg")

CLASS_COLORS = {
    "connector": (0, 0, 255),       # 빨강
    "branch_sleeve": (255, 0, 0),   # 파랑
    "branch_bare": (255, 128, 0),   # 주황
    "elbow": (0, 128, 255),         # 하늘
    "dimension": (0, 200, 0),       # 초록
    "balloon": (200, 0, 200),       # 자홍
}


def find_source_image(name):
    for d in IMG_CANDIDATE_DIRS:
        p = os.path.join(d, f"{name}.png")
        if os.path.isfile(p):
            return p
    return None


def make_one(name):
    json_path = None
    for d in JSON_DIRS:
        p = os.path.join(d, f"{name}.json")
        if os.path.isfile(p):
            json_path = p
            break
    if json_path is None:
        print(f"  [skip] {name}: json 없음")
        return False

    img_path = find_source_image(name)
    if img_path is None:
        print(f"  [skip] {name}: 원본 이미지 없음 (labels/Success, images/all 확인)")
        return False

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img = cv2.imread(img_path)
    if img is None:
        print(f"  [skip] {name}: 이미지 로드 실패 ({img_path})")
        return False

    for sh in data.get("shapes", []):
        pts = [tuple(map(int, p)) for p in sh["points"]]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        color = CLASS_COLORS.get(sh["label"], (0, 0, 0))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{name}_boxed.png")
    cv2.imwrite(out_path, img)
    print(f"  -> {out_path}")
    return True


def main():
    args = sys.argv[1:]
    if args:
        names = args
    else:
        names = set()
        for d in JSON_DIRS:
            for p in glob.glob(os.path.join(d, "*.json")):
                names.add(os.path.splitext(os.path.basename(p))[0])
        names = sorted(names)

    if not names:
        print("json 없음 (labels/Json, labels/Success 확인)")
        return

    print(f"{len(names)}개 도면 처리:")
    ok = 0
    for name in names:
        if make_one(name):
            ok += 1
    print(f"\n완료: {ok}/{len(names)}")


if __name__ == "__main__":
    main()
