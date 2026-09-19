"""
YOLOv8 학습. 1차 검증은 데이터 3장이라 과적합이 정상 - "학습이 도는지"만 본다.

도면은 크고 객체(커넥터)는 작으므로 imgsz 를 크게(1280) 쓴다.
스캔 도면이라 색/노이즈 증강은 켜고, 좌우반전은 배선 방향이 의미 있으니 끈다.
"""
import os
from ultralytics import YOLO

HERE = os.path.dirname(__file__)


def main(epochs=60, imgsz=1280, model="yolov8n.pt", **over):
    m = YOLO(model)
    cfg = dict(
        data=os.path.join(HERE, "data.yaml"),
        epochs=epochs,
        imgsz=imgsz,
        batch=4,
        device=0,
        workers=0,
        project=os.path.join(HERE, "runs"),
        name="cable_v1",
        exist_ok=True,
        plots=False,         # seg-fault 회피 (plotting 단계)
        val=True,
        amp=False,           # RTX 5080 최신 아키텍처 - AMP 끔
        cache=False,
        # 증강
        fliplr=0.0,          # 좌우반전 끔 (배선 방향 보존)
        flipud=0.0,
        degrees=3.0,         # 살짝 회전 (스캔 기울어짐 모사)
        translate=0.05,
        scale=0.2,
        hsv_v=0.4,           # 밝기 (스캔 품질 편차)
        hsv_s=0.0,           # 도면은 흑백이라 채도 증강 무의미
        mosaic=0.0,          # 데이터 적을 때 mosaic 은 오히려 방해
    )
    cfg.update(over)
    m.train(**cfg)
    print("\n학습 완료. 결과: runs/cable_v1/")
    print("best weights: runs/cable_v1/weights/best.pt")


if __name__ == "__main__":
    main()
