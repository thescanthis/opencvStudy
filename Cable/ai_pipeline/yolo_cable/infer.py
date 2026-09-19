"""
학습된 YOLO 로 도면에서 점(커넥터/분기점) 검출 -> JSON.
케이블 선 자체는 이 좌표들을 그리기 로직이 연결해서 그린다.

출력 JSON:
{
  "image": "A20016147.png", "w": 2000, "h": 1398,
  "detections": [
    {"cls": "connector", "conf": 0.91, "box": [x1,y1,x2,y2], "center": [cx,cy]},
    ...
  ]
}
"""
import os
import sys
import json
import cv2
from ultralytics import YOLO

HERE = os.path.dirname(__file__)
WEIGHTS = os.path.join(HERE, "runs", "cable_v1", "weights", "best.pt")
NAMES = ["connector", "branch_sleeve", "branch_bare", "elbow", "dimension", "balloon"]


def infer(image_path, weights=WEIGHTS, conf=0.25, save_vis=True):
    m = YOLO(weights)
    r = m.predict(image_path, conf=conf, imgsz=960, verbose=False)[0]
    img = cv2.imread(image_path)
    H, W = img.shape[:2]

    dets = []
    for b in r.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        cls = NAMES[int(b.cls[0])]
        dets.append({
            "cls": cls,
            "conf": round(float(b.conf[0]), 3),
            "box": [round(x1), round(y1), round(x2), round(y2)],
            "center": [round((x1 + x2) / 2), round((y1 + y2) / 2)],
        })

    out = {
        "image": os.path.basename(image_path),
        "w": W, "h": H,
        "detections": dets,
    }
    name = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(os.path.join(HERE, "infer_out"), exist_ok=True)
    with open(os.path.join(HERE, "infer_out", f"{name}.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    if save_vis:
        vis = img.copy()
        col = {"connector": (0, 180, 0), "branch_sleeve": (0, 0, 220),
               "branch_bare": (200, 0, 200), "elbow": (220, 140, 0),
               "dimension": (150, 150, 150), "balloon": (0, 200, 220)}
        for d in dets:
            x1, y1, x2, y2 = d["box"]
            c = col[d["cls"]]
            cv2.rectangle(vis, (x1, y1), (x2, y2), c, 2)
            cv2.putText(vis, f"{d['cls']} {d['conf']}", (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(HERE, "infer_out", f"{name}_vis.png"), vis)

    print(f"{name}: {len(dets)} detections -> infer_out/{name}.json")
    return out


if __name__ == "__main__":
    for p in sys.argv[1:]:
        if not os.path.isabs(p):
            p = os.path.join(HERE, "images", "all", p if p.endswith(".png") else p + ".png")
        infer(p)
