# 케이블 도면 형상 추출 YOLO — 인수인계 문서

이 문서 하나로 처음부터 끝까지 재현 가능하도록 작성했다. 순서대로 따라가면 된다.

---

## 0. 이 작업이 무엇인가

**목표**: 케이블 도면(`Cable.pdf`, 스캔 이미지)에서 케이블 배선 **형상**(몸통·가지의
정점 좌표)을 뽑아, 2D 벡터 이미지로 재현한다.

**왜 YOLO인가**: 도면에는 치수선, 풍선번호, 부품표 등 케이블이 아닌 요소가 많다.
규칙 기반(OCR + 고정 크롭)은 도면마다 레이아웃이 달라 실패한다. YOLO 객체 검출로
"커넥터", "분기점" 같은 **점**을 찾고, 케이블 선은 코드가 그 점들을 연결해서 그린다.

**YOLO가 하는 일**: 이미지 1장 → `[클래스, 박스좌표(x,y,w,h), 신뢰도]` 리스트.
YOLO는 "선"을 못 그린다. 점만 준다. 선 연결은 후처리 코드 몫.

**지도학습이다. 강화학습 아니다.** 사람이 박스(정답)를 주면 모델이 따라 배운다.

---

## 1. 환경

- OS: Windows 11
- Python 3.12 (Anaconda: `C:\ProgramData\anaconda3`)
- GPU: RTX 5080 (CUDA 12.8, torch 2.11.0+cu128)
- 핵심 패키지: `ultralytics` 8.4.146 이상 (YOLOv8/v11)
  - **8.0.122 는 torch 2.11 과 충돌**(weights_only). 반드시 `pip install -U ultralytics`
- PDF 렌더: `pymupdf` (import 이름 `fitz`)
- 기타: `opencv-python`, `numpy`

설치 확인:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import ultralytics; print(ultralytics.__version__)"
python -c "import fitz, cv2, numpy; print('ok')"
```

---

## 2. 폴더 구조

작업 루트: `C:\AIND_Library\Git_PredicTest\x64_Cami_Lib\PSCD1500\PSCD1500\Cable\ai_pipeline\yolo_cable\`

```
yolo_cable/
├── README.md              # 개요 + 클래스/라벨링 규칙
├── HANDOFF.md             # 이 문서
├── render_pages.py        # ① PDF -> PNG 렌더
├── grid_view.py           # ② 라벨 좌표 읽기용 그리드 오버레이
├── label_tools.py         # ③ 픽셀박스 -> YOLO txt + 미리보기
├── make_labels_batch1.py  # ③ 실제 라벨 정의(3장 예시). 여기에 도면 추가
├── data.yaml              # ④ 학습 설정
├── train.py               # ④ 학습
├── infer.py               # ⑤ 추론 -> 검출 JSON
├── images/all/            # 렌더된 PNG (라벨링 대상, 장변 2000px)
├── labels/all/            # YOLO txt 라벨 (make_labels_batch1.py 가 생성)
├── preview/               # 라벨 미리보기 PNG (검수용)
├── dataset/               # 학습용 train/val 분할본
│   ├── images/{train,val}/
│   └── labels/{train,val}/
├── runs/cable_v1/         # 학습 결과 (weights/best.pt 등)
└── infer_out/             # 추론 결과 JSON + 시각화
```

---

## 3. 클래스 정의 (6개)

`data.yaml` 과 `label_tools.py` 의 `CLASSES` 가 **정확히 이 순서**여야 한다.

| id | 이름 | 무엇 | 박스 치는 법 |
|----|------|------|--------------|
| 0 | `connector` | 케이블 끝의 커넥터/플러그(원형·엘보·사각). 나사선/링 있는 원통. | 커넥터 몸통 외곽에 딱 맞게. 케이블 선은 넣지 않음. 부트(고무커버)까지 포함. |
| 1 | `branch_sleeve` | 분기점의 **굵은 커플러 부품**(몸통보다 뚜렷하게 굵어진 마디). | 굵어진 마디만. 들어오고 나가는 가는 선은 최소만. |
| 2 | `branch_bare` | 부품 **없이** 선들이 한 점에서 갈라지는 분기점. 부채꼴 밑동. | 선들이 모이는 교차점 주변 작은 사각형(몸통두께 2~3배). |
| 3 | `elbow` | 케이블 **중간**의 90도 꺾임 부품(엘보 어댑터). | 꺾이는 부품 외곽. 끝에 있는 엘보 플러그는 connector. |
| 4 | `dimension` | 치수 숫자 + 치수선(화살표). 제외 대상이지만 일부러 학습 → 오인식 방지. | 숫자 + 양끝 화살표 포함. 얇은 치수보조선은 제외. |
| 5 | `balloon` | 풍선번호(숫자 든 원) + 지시선. 제외 대상, 오인식 방지용. | 원 하나당 박스 하나. |

`branch_bare` 는 시각 특징이 약해 정확도 낮음. 운영 시 대안: "분기점인데 sleeve
검출 안 됨 → bare 로 간주". 1차 데이터엔 넣되 기대치 낮게.

---

## 4. 라벨링 규칙 (정확도의 대부분을 좌우 — 반드시 일관되게)

1. **한 도면을 라벨링하면 그 도면의 모든 해당 객체를 빠짐없이.** 커넥터 8개면 8개
   전부. 누락하면 모델에 "이건 커넥터 아님"을 가르치는 셈 → 최악.
2. **박스는 객체에 딱 맞게(tight).** 여백 크게 두지 않는다.
3. **애매하면 뺀다.** 50% 미만 잘림 / 흐려서 판별 불가 → 라벨 안 함. 이 "50% 규칙"을
   모든 도면에 동일 적용.
4. **하단 배선도(핀맵 다이어그램), 부품표, 타이틀블록, 수정이력 영역은 통째로 무시.**
   그 안의 커넥터 그림·원·숫자도 라벨 안 함. 케이블 배선 그림 영역만 대상.
5. **커넥터 vs 엘보**: 케이블 "끝"이면 connector, "중간"에서 꺾는 부품이면 elbow.
6. 워터마크 무시.
7. **어려운 도면(부채꼴/다중분기/커넥터 다수)을 전체의 30~40%** 넣는다. 쉬운
   직선 케이블만 학습하면 어려운 걸 못 잡는다.

---

## 5. 실행 순서

### STEP 1 — PDF를 PNG로 렌더

`render_pages.py` 의 `CABLES` 리스트에 학습에 쓸 도면 폴더명을 넣는다.
폴더는 `C:\AIND_Library\Git_PredicTest\x64_Cami_Lib\PSCD1500\PSCD1500\Cable\TestData\` 아래.

```bash
cd C:\AIND_Library\Git_PredicTest\x64_Cami_Lib\PSCD1500\PSCD1500\Cable\ai_pipeline\yolo_cable
python render_pages.py
```

결과: `images/all/<도면명>.png` (장변 2000px). 이 해상도가 라벨 좌표의 기준이다.
**추론 때도 반드시 같은 방식(장변 2000px)으로 렌더해야** 좌표가 맞는다.

### STEP 2 — 라벨링

두 가지 방법. **B 권장**(정확).

#### 방법 A — 코드로 좌표 입력 (지금까지 Claude가 한 방식, 임시)

1. 좌표를 읽기 위해 그리드 오버레이 이미지 생성:
   ```bash
   # 전체
   python grid_view.py A20016147
   # 배선 영역만 확대 (x0,y0,x1,y1 = 원본 2000px 이미지 좌표, scale = 확대배율)
   python grid_view.py A20016147 x0=120 y0=380 x1=1950 y1=780 scale=1.5
   ```
   → `preview/_grid_<name>.png` 에 빨간 그리드(100px 간격 + 좌표숫자)가 그려진다.
   이 이미지를 보고 각 객체의 박스 픽셀 좌표(좌상단 x1,y1 / 우하단 x2,y2)를 읽는다.
   **그리드 숫자는 항상 원본 2000px 이미지 좌표**(확대해도 숫자는 원본 기준).

2. `make_labels_batch1.py` 에 도면별 박스 리스트를 추가:
   ```python
   A20016147 = [
       ("connector", 268, 425, 400, 505),      # (클래스, x1, y1, x2, y2) 픽셀
       ("branch_sleeve", 858, 443, 917, 495),
       ("dimension", 300, 508, 560, 535),
       ("balloon", 182, 362, 208, 388),
       # ...
   ]
   # 파일 맨 아래 __main__ 에:
   save_label("A20016147", A20016147)
   ```

3. 실행 → YOLO txt + 미리보기 생성:
   ```bash
   python make_labels_batch1.py
   ```
   → `labels/all/<name>.txt` (YOLO 정규화 좌표), `preview/<name>_boxes.png` (박스 그린 것)

4. `preview/<name>_boxes.png` 를 열어 박스가 실제 객체에 맞는지 확인. 어긋나면
   2번 좌표 수정 후 재실행.

#### 방법 B — 라벨링 GUI 툴 (권장)

> 사람이 직접 마우스로 박스 치는 상세 절차는 **`LABELING_GUIDE.md`** 참고
> (labelImg 설치·설정·단축키, 클래스별 "무엇을 어떻게 박스 치나" 그림 설명,
> train/val 분할·학습·추론까지 전 과정, Roboflow 대안, 자주 하는 실수).

**옵션 B-1: labelImg (로컬, 간단)**
```bash
pip install labelImg
labelImg
```
1. 좌상단 "Open Dir" → `yolo_cable/images/all` 선택
2. "Change Save Dir" → `yolo_cable/labels/all` 선택
3. 좌하단 포맷을 **YOLO** 로 전환 (PascalVOC 아님)
4. `yolo_cable/labels/all/classes.txt` 파일을 아래 내용으로 미리 만들어 둔다:
   ```
   connector
   branch_sleeve
   branch_bare
   elbow
   dimension
   balloon
   ```
5. 단축키: `w` = 박스 그리기 시작, 드래그, 클래스 선택. `d` = 다음 이미지,
   `a` = 이전. `Ctrl+S` = 저장.
6. 4번 규칙대로 도면당 모든 객체를 친다.
7. 저장되면 `labels/all/<name>.txt` 가 YOLO 형식으로 생긴다.

**옵션 B-2: Roboflow (웹, 협업/증강 편함)**
1. roboflow.com 무료 계정 → New Project → Object Detection
2. 클래스 6개 등록 (위 순서대로)
3. `images/all/*.png` 업로드
4. 웹에서 박스 치기 (Annotate)
5. Generate → Export → **YOLOv8** 포맷 → zip 다운로드
6. zip 안의 `train/`, `valid/` 를 `yolo_cable/dataset/` 로 옮긴다
   (이 경우 STEP 3 분할은 건너뜀, Roboflow가 이미 나눔)

### STEP 3 — train/val 분할

라벨링한 도면을 8:2 로 나눈다. 어려운 도면이 train/val 양쪽에 골고루 들어가게.

```bash
cd yolo_cable
# 예: 라벨링 완료 도면이 A,B,C,...,Z 라면 약 80%를 train, 20%를 val
mkdir -p dataset/images/train dataset/images/val dataset/labels/train dataset/labels/val

# train (예시 - 실제 도면명으로)
for n in A20016147 A60025764 A60025763 A60025765 A60026321 A20016151 A60025776 A60023110; do
  cp images/all/$n.png dataset/images/train/
  cp labels/all/$n.txt dataset/labels/train/
done
# val
for n in A60023111 A20016148; do
  cp images/all/$n.png dataset/images/val/
  cp labels/all/$n.txt dataset/labels/val/
done
```

`data.yaml` 확인 (경로가 절대경로로 맞는지):
```yaml
path: C:/AIND_Library/Git_PredicTest/x64_Cami_Lib/PSCD1500/PSCD1500/Cable/ai_pipeline/yolo_cable/dataset
train: images/train
val: images/val
names:
  0: connector
  1: branch_sleeve
  2: branch_bare
  3: elbow
  4: dimension
  5: balloon
```

### STEP 4 — 학습

```bash
cd yolo_cable
python train.py
```

`train.py` 주요 설정 (RTX 5080 에서 seg-fault 회피 위해 필수):
- `plots=False` — plotting 단계에서 죽는 문제 회피
- `amp=False` — 최신 아키텍처 AMP 이슈 회피
- `workers=0` — 윈도우 dataloader 안정
- `imgsz=960` (데이터 적을 때) ~ `1280` (많을 때). 도면은 크고 커넥터는 작아서
  640 은 부족.
- `fliplr=0.0` — 배선 방향이 의미 있으므로 좌우반전 금지
- `mosaic=0.0` — 데이터 적을 때 방해
- `hsv_v=0.4` — 스캔 밝기 편차 대응
- `degrees=3.0` — 스캔 기울어짐 모사

에폭: 1차엔 40~60. 데이터 30장+면 100~150.

결과: `runs/cable_v1/`
- `weights/best.pt` — 검증 성능 최고 모델
- `weights/last.pt` — 마지막 에폭
- `results.csv` — 에폭별 loss/mAP

성능 지표: `mAP50` (0.5 IoU 기준 평균정밀도). 0.6 이상이면 쓸만, 0.8+면 좋음.
클래스별 성능도 나옴 — `connector` 는 잘 나오고 `branch_bare` 는 낮을 것.

`train.py` 를 파라미터 바꿔 재호출:
```bash
python -c "from train import main; main(epochs=100, imgsz=1280)"
```

### STEP 5 — 추론

```bash
cd yolo_cable
python infer.py A20016147          # images/all/A20016147.png 사용
python infer.py C:\path\to\any.png # 절대경로도 가능
```

결과: `infer_out/<name>.json`
```json
{
  "image": "A20016147.png", "w": 2000, "h": 1398,
  "detections": [
    {"cls": "connector", "conf": 0.91, "box": [268,425,400,505], "center": [334,465]},
    {"cls": "branch_sleeve", "conf": 0.87, "box": [858,443,917,495], "center": [887,469]}
  ]
}
```
+ `infer_out/<name>_vis.png` (박스 시각화)

`conf` 임계값은 `infer.py` 의 `conf=0.25`. 검출이 너무 많으면 올리고, 놓치면 낮춘다.

---

## 6. 추론 결과를 형상으로 (다음 단계 — 아직 미구현)

`infer.py` 가 주는 건 **점들**이다. 이걸 케이블 형상으로 만들려면:

1. **토폴로지 추론**: 어느 점이 어느 점과 연결되는가.
   - 커넥터/분기점들을 거리·각도로 연결 (최근접, 몸통 라인 추정)
   - 또는 OpenCV 선 검출로 점 사이 경로 확인
2. **슬리브 판단**: `branch_sleeve` 검출된 분기점 = 슬리브 그림.
   `branch_bare` 또는 sleeve 미검출 = 선만 갈라지게.
3. **그리기**: `../test_5cables_2d.py` 의 `double_line` / `fused_sleeve` /
   `draw_fan` 재사용. 좌표는 검출된 center 를 쓴다.
   - 확정 규칙: 메인 슬리브 폭 SLEEVE_W=270, 가지 슬리브 = 절반(135),
     shapely union, 스무딩 없음. (`../a20016147_2d_shape.py`)

이 부분은 검출 정확도가 어느 정도 나온 뒤에 붙인다.

---

## 7. 반복 개선 사이클

```
라벨 30장 → 학습 → 추론 → 틀린 도면 확인
   → 그 도면(오검출 유형)을 라벨링해서 데이터에 추가 → 재학습
```

- 특정 클래스가 안 잡히면: 그 클래스가 많은 도면을 더 넣는다.
- 특정 도면 유형(부채꼴 등)에서 실패하면: 비슷한 도면을 더 넣는다.
- `dimension`/`balloon` 오검출이 케이블을 방해하면: 그 영역 라벨링을 더 꼼꼼히.

---

## 8. 지금까지 완료된 것 (2026-09-10 기준)

- 파이프라인 전 단계 스크립트 작성 및 동작 확인 (렌더→라벨→학습→추론)
- 도면 36장 렌더 완료 (`images/all/`)
- X-AnyLabeling 설치 (전용 venv + PyQt6 6.7.1, `run_labeling.bat`). config 에
  클래스 6개 등록.
- RTX 5080 seg-fault 원인 파악 및 회피 설정 적용 (`train.py`)
- **정확한 라벨 1장**: `labels/Json/50073721.json` — 사용자가 X-AnyLabeling GUI
  로 직접 + Claude 가 중복/오류 정리 (connector 2, dimension 1, balloon 18)

### 폐기된 것 (교훈)

- Claude/Gemini 가 **코드로 좌표를 직접 입력**해서 만든 라벨 ~12장 전부 폐기.
  박스가 실제 객체에서 수십~수백 px 어긋남. **AI 는 도면 이미지에서 정확한
  픽셀 좌표를 못 찍는다** (grid_view 로 확대해도 개선만 될 뿐). OpenCV Hough
  Circle 자동검출도 스캔 노이즈로 실패.
  → **정확한 라벨은 X-AnyLabeling GUI 로 사람이 직접 그려야 한다.**

## 9. 남은 것 (반자동 전략)

1. **시드 15~20장 정확히 라벨링** (X-AnyLabeling GUI). 쉬운것+어려운것 섞어서.
   AI 좌표는 못 쓰므로 사람이 직접. ← 최우선, 여기서 막혀 있음.
2. 15~20장으로 YOLO 1차 학습 (epochs 100, imgsz 1280)
3. 1차 모델로 나머지 도면 자동 검출 (모델 예측은 이미지 기반이라 AI 추정보다
   정확) → X-AnyLabeling 으로 보정
4. 전체로 최종 학습, mAP 확인
5. 검출 JSON → 형상 그리기 로직 연결 (섹션 6, `../test_5cables_2d.py` 재사용)

## 10. 주의사항 / 함정

- **AI 는 좌표를 못 찍는다**: 코드로 박스 좌표를 만들지 말 것. GUI 로 직접.
- **ultralytics 버전**: 8.0.x 는 torch 2.11 과 충돌. 8.4+ 필수.
- **X-AnyLabeling 실행**: Anaconda 기본환경은 PyQt6 DLL 충돌. `run_labeling.bat`
  (전용 venv `C:\Users\Administrator\anylabel_env`, PyQt6 6.7.1) 로만 실행.
- **렌더 해상도 일관성**: 라벨링한 PNG 와 추론할 PNG 의 렌더 배율이 같아야 함
  (`render_pages.py` 의 `LONG_SIDE_PX=2000` 고정 유지).
- **클래스 순서**: `data.yaml`, `label_tools.py CLASSES`, `classes.txt` 가 전부
  같은 순서(0~5)여야 함. 어긋나면 라벨이 뒤섞임.
- **seg-fault**: `plots=False`, `amp=False`, `workers=0` 유지. 이거 빼면 RTX 5080
  에서 첫 에폭 검증 단계에서 죽는다.
- **배선도/부품표 영역**: 절대 라벨링하지 않는다. 케이블 배선 그림만.
- **라벨 누락 금지**: 도면 하나 손대면 그 도면의 모든 해당 객체를 친다.
- **이미지/dataset/runs 는 git 에 안 올림** (재생성 가능, 대용량).
  `render_pages.py` 로 언제든 이미지 복원.
