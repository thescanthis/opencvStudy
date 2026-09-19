# 수동 라벨링 가이드 — X-AnyLabeling

이 문서는 **사람이 직접 박스를 치는** 절차다. 툴은 **X-AnyLabeling**
(labelImg 후속, AI 자동 라벨 내장, 로컬 실행 = 도면 유출 없음).

---

## A. 설치 (완료됨)

**Anaconda 기본 환경에 설치하면 PyQt6 DLL 충돌로 실행이 안 됐다.**
그래서 **전용 가상환경**(`C:\Users\Administrator\anylabel_env`)에 설치했고,
PyQt6 를 6.11 → **6.7.1** 로 낮춰서 해결했다.

### 실행 방법

**방법 1 (권장): 배치 파일 더블클릭**
```
yolo_cable\run_labeling.bat  ← 더블클릭
```

**방법 2: 명령창에서**
```bash
"C:\Users\Administrator\anylabel_env\Scripts\python.exe" -c "from anylabeling.app import main; main()"
```

→ 창이 뜨면 성공. (`.xanylabelingrc` 설정 파일이 홈 폴더에 자동 생성됨)

> **주의**: 그냥 `xanylabeling` 만 치면 Anaconda 기본환경 것이 실행돼서 **DLL
> 에러**가 난다. 반드시 위 두 방법(anylabel_env)으로 실행할 것.

### 재설치가 필요하면 (환경 날아갔을 때)
```bash
python -m venv C:\Users\Administrator\anylabel_env
C:\Users\Administrator\anylabel_env\Scripts\pip install x-anylabeling-cvhub
C:\Users\Administrator\anylabel_env\Scripts\pip install "PyQt6==6.7.1" "PyQt6-Qt6==6.7.3" "PyQt6-WebEngine==6.7.0" "PyQt6-WebEngine-Qt6==6.7.3"
```

---

## B. 프로젝트 파일

- 도면 이미지: `yolo_cable/images/all/*.png` (36장, 장변 2000px, 렌더 완료)
- 클래스 목록: `yolo_cable/classes.txt` 및 `yolo_cable/images/all/classes.txt`
  ```
  connector
  branch_sleeve
  branch_bare
  elbow
  dimension
  balloon
  ```
  **이 순서 절대 변경 금지** (YOLO 클래스 번호가 이 순서)

---

## C. 첫 실행 설정

1. `run_labeling.bat` 더블클릭 (또는 섹션 A 방법 2)
2. 상단 메뉴 또는 좌측 툴바에서 **Open Dir** (폴더 열기)
   → `yolo_cable/images/all` 선택
   → 왼쪽/오른쪽에 이미지 목록이 뜬다
3. **저장 포맷 확인**: X-AnyLabeling 은 기본으로 자체 `.json`(labelme 형식)으로
   저장한다. 라벨링을 다 한 뒤 **한 번에 YOLO 로 변환**한다(섹션 G).
   - 또는 상단 메뉴 `Tools → Save Format → YOLO` 가 있으면 그걸로 바꿔서
     `.txt` 직접 저장도 가능 (버전에 따라 다름)
4. **자동 저장 켜기**: 메뉴 `File → Auto Save` (또는 우측 상단 아이콘)
   → 이미지 넘길 때마다 자동 저장

---

## D. 화면 조작

| 조작 | 방법 |
|------|------|
| 박스 그리기 | 좌측 툴바 **Rectangle** 아이콘 클릭 (단축키 보통 `R` 또는 `W`) → 드래그 |
| 다음/이전 이미지 | `D` / `A` (또는 `→` / `←`) |
| 저장 | `Ctrl + S` (자동저장 켰으면 불필요) |
| 확대/축소 | `Ctrl + 마우스휠`, 또는 `+` / `-` |
| 화면 이동 | 마우스 가운데 버튼 드래그, 또는 스크롤바 |
| 화면 맞춤 | `Ctrl + F` (fit window) |
| 박스 삭제 | 박스 선택 → `Delete` |
| 박스 클래스 변경 | 우측 라벨 리스트에서 더블클릭 |

도면이 크고 커넥터가 작으니 **확대해서** 작업. 배선 그림 부분만 화면에 키워놓고
좌→우로 훑으며 박스 친다.

---

## E. AI 자동 라벨링 (선택, 부담 줄이기)

X-AnyLabeling 의 핵심 기능. 완벽하진 않지만 1차 박스를 자동 생성 → 사람이 보정.

1. 좌측 툴바 **Brain(AI)** 아이콘 → 모델 선택
   - **Segment Anything (SAM / SAM2)**: 클릭 한 번으로 그 위치 객체를 자동으로
     박스/마스크. 커넥터처럼 뚜렷한 부품에 잘 먹힘.
   - **YOLO 계열**: 우리가 학습한 `best.pt` 를 로드하면(모델이 어느정도
     성능 나온 뒤) 자동 예측 → 보정. **반복 개선 사이클에서 유용**.
2. SAM 사용법:
   - AI 모드 진입 → 커넥터 중앙을 **클릭** → 자동으로 영역 잡힘 → 클래스 지정
   - 잘못 잡으면 `-` 클릭(빼기 포인트)으로 조정
3. 자동 라벨은 **반드시 사람이 검수**. 규칙(섹션 F) 어긋나면 수정/삭제.

**처음 20장은 AI 없이 수동으로** 하는 게 감 잡기 좋다. AI 보조는 데이터가
어느정도 쌓이고 우리 YOLO 모델이 나온 뒤 쓰는 게 효과적.

---

## F. 무엇을 박스 칠 것인가 (클래스별)

### 0. connector — 케이블 "끝"의 플러그

**찾기**: 케이블 선 양 끝. 나사선(빗금) 원통, 커플링 링, 엘보(ㄱ자) 플러그.
도면의 P1, P2, P3... 지점.

**박스**: 플러그 몸통 + 커플링 링 + 백셸(케이블 들어가는 고무 커버)까지 딱 맞게.
- 커넥터에서 나오는 **케이블 선은 박스에 넣지 않는다**
- 커넥터 옆의 **원형 핀배치 상세도(동그라미+점들)는 제외** — 실물 아니라 설명도

```
   ┌──────────┐   ← 이 사각형만
 ══╡ ▨▨▨▨▨▨▨ ╞═════  ← 케이블 선 제외
   └──────────┘
```

### 1. branch_sleeve — 부품 있는 분기점

**찾기**: 몸통이 갈라지는 곳에서 몸통보다 **뚜렷하게 굵어진 마디**(원통형 커플러).
A20016147 의 Y 분기점.

**박스**: 굵어진 마디만. 가는 선은 최소만.

```
         │ ← 가지
 ════▉▉▉▉════  ← 굵은 마디에 박스
```

### 2. branch_bare — 부품 없는 분기점 (선만 갈라짐)

**찾기**: 가는 선들이 **부품 없이** 갈라지는 지점. 부채꼴 밑동, 빗살 각 지점.
A60023110, A60023111.

**박스**: 선들이 모이는 교차점 주변 작은 사각형 (몸통 두께 2~3배).

```
 ═══╤╤╤╤╤═══   ← 교차점에 작은 박스
    ││││
```

**sleeve vs bare 헷갈리면**: 확대해서 "몸통보다 굵은 부품 덩어리가 있나?"
→ 있으면 sleeve, 없으면 bare.

### 3. elbow — 케이블 "중간"의 꺾임 부품

**찾기**: 선 도중 90도 꺾이는 곳의 어댑터. **끝**의 엘보 플러그는 connector.

**박스**: 꺾임 부품 외곽.

### 4. dimension — 치수 (제외 대상, 일부러 라벨)

**찾기**: 숫자(`1800.0` 등) + 양끝 **화살표 달린 치수선**.

**박스**: **숫자 텍스트만 타이트하게.** 양끝 화살표·치수선·얇은 세로
보조선은 전부 제외(박스가 길어지면 다른 객체와 겹쳐서 학습에 방해됨).

```
  │←──── 1800.0 ────→│   ← 숫자에만 박스, 화살표/선은 제외
              [ ]
```

**왜**: 치수선을 케이블로 오인하는 걸 막으려면 "이건 dimension" 이라고 명시 학습.

### 5. balloon — 풍선번호 (제외 대상)

**찾기**: 숫자 든 작은 원. 보통 도면 위에 줄지어 있고 지시선이 붙음.

**박스**: **원 하나당 박스 하나**. 지시선 꼬리는 원 근처만.

---

## G. 라벨링 후: YOLO 형식으로 변환

X-AnyLabeling 은 기본 `.json`(labelme) 저장. YOLO `.txt` 로 변환한다.

### 방법 1: 툴 내장 export

메뉴 `Tools → Export → YOLO` (또는 `Export Annotations`)
→ 출력 폴더 지정 (`yolo_cable/labels/all`)
→ 클래스 순서가 `classes.txt` 와 맞는지 확인
→ 이미지와 같은 이름의 `.txt` 생성됨

### 방법 2: 변환 스크립트 (툴 export 가 애매하면)

`yolo_cable/labelme2yolo.py` 를 만들어 두면:
```bash
python labelme2yolo.py
```
→ `images/all/*.json` 을 읽어 `labels/all/*.txt` 로 변환

(이 스크립트는 필요 시 별도 작성)

---

## H. 라벨링 규칙 (꼭 지킬 것)

1. **한 도면 시작하면 그 도면의 해당 객체를 전부.** 커넥터 8개면 8개 다.
   누락하면 모델이 "커넥터인데 라벨 없음 = 커넥터 아님" 으로 잘못 배운다.
2. **박스는 딱 맞게(tight).** 여백 크게 두면 "커넥터 = 빈 영역" 으로 오해.
3. **애매하면 뺀다.** 50% 이상 잘림/흐림 → 라벨 안 함. 모든 도면 동일 기준.
4. **배선도(하단 핀맵), 부품표, 도면 테두리, 타이틀블록, 수정이력 표 무시.**
   그 안의 그림/원/숫자도 라벨 안 함. **케이블 배선 그림 영역만.**
5. **커넥터 vs 엘보**: 끝 = connector, 중간 꺾임 = elbow.
6. **워터마크 무시.**
7. 같은 종류는 항상 같은 방식으로 (백셸 포함 규칙 등 일관되게).

---

## I. 작업 분량 / 순서

### 어떤 도면부터

`images/all/` 36장. 비율:
- **쉬운 것** (직선/분기 1~2개): 절반 — `A60025772`, `A50035007-6`, `A60025765`, `A60024600` 등
- **어려운 것** (부채꼴/다중분기/커넥터 다수): 절반 — `A20016151`, `A60025776`, `A60023110`, `A60023111`, `A60026426-1` 등

**목표 20~30장.** 도면당 커넥터 5~15 + 분기점 1~5 + 치수/풍선 몇 개.
도면당 5~15분.

### 진행 체크

10장쯤 되면 한 번 학습(섹션 아래) → 감 확인 → 계속.

---

## J. 라벨링 완료 후: 학습

### J-1. train/val 나누기 (8:2)

`yolo_cable/` 에서:
```bash
mkdir -p dataset/images/train dataset/images/val dataset/labels/train dataset/labels/val
rm -f dataset/images/train/* dataset/images/val/* dataset/labels/train/* dataset/labels/val/*

# train (라벨링한 도면명으로)
for n in A20016147 A60025764 A60025763 A60025765 A60026321 \
         A20016151 A60025776 A60023110 A60025772 A50035007-6 \
         A60024600 A20016148 A20016149 A60023103 A60023104 A60023105; do
  cp images/all/$n.png dataset/images/train/
  cp labels/all/$n.txt dataset/labels/train/
done

# val (train 에 없는 것, 어려운 것 1~2개 포함)
for n in A60023111 A60025766 A60023106 A60026322; do
  cp images/all/$n.png dataset/images/val/
  cp labels/all/$n.txt dataset/labels/val/
done

# classes.txt 가 섞였으면 제거
rm -f dataset/labels/train/classes.txt dataset/labels/val/classes.txt
```

### J-2. 학습

```bash
cd C:\AIND_Library\Git_PredicTest\x64_Cami_Lib\PSCD1500\PSCD1500\Cable\ai_pipeline\yolo_cable
python -c "from train import main; main(epochs=100, imgsz=1280)"
```

- box_loss / cls_loss 가 줄어들면 정상
- 끝나면 `runs/cable_v1/weights/best.pt`
- **RTX 5080 에서 죽으면**: `train.py` 에 `plots=False, amp=False, workers=0`
  들어있는지 확인 (기본 포함). 그래도 죽으면 `imgsz=960`.

### J-3. 성능 확인

```bash
type runs\cable_v1\results.csv
```
마지막 줄 `metrics/mAP50(B)`:
- < 0.3 : 데이터 부족 or 라벨 문제
- 0.5~0.7 : 쓸 만함
- 0.8+ : 좋음

### J-4. 추론 테스트

```bash
python infer.py A60023110
```
→ `infer_out/A60023110.json` + `infer_out/A60023110_vis.png` (박스 그림)
→ `_vis.png` 열어서 눈으로 확인

---

## K. 반복 개선

```
20장 라벨 → 학습 → 추론 결과 확인
  → 못 잡는 유형 발견
  → 그런 도면 5장 더 라벨링 (이때 AI 자동 라벨 = 우리 best.pt 로드해서 보조)
  → 재학습
```

목표 mAP50 0.7 넘으면 다음 단계(검출 → 형상 그리기 연결).

---

## L. 문제 해결

### xanylabeling 실행 안 됨 / 창 안 뜸

1. **PyQt6 충돌**: 다른 프로젝트가 PyQt5 를 쓰면 충돌 가능.
   → 전용 가상환경 권장:
   ```bash
   python -m venv C:\anylabel_env
   C:\anylabel_env\Scripts\activate
   pip install x-anylabeling-cvhub
   xanylabeling
   ```
2. **prebuilt exe 사용**: https://github.com/CVHub520/X-AnyLabeling/releases
   에서 Windows `.exe` 다운로드 → 더블클릭 (pip 설치 불필요, 가장 확실)
3. **에러 메시지 확인**: 명령창에서 실행하면 에러가 뜬다. 그대로 캡처.

### opencv headless 충돌

`x-anylabeling-cvhub` 설치 시 `opencv-contrib-python-headless` 로 바뀌었다.
우리 파이프라인 스크립트는 정상 작동 확인됨. 만약 다른 곳에서 GUI 관련
`cv2.imshow` 등이 필요하면:
```bash
pip install opencv-python  # headless 를 일반 버전으로 되돌림
```
(X-AnyLabeling 은 exe 로 쓰면 이 충돌 자체가 없음)

### 자주 하는 실수

| 실수 | 결과 | 방지 |
|------|------|------|
| 클래스 순서 바꿈 | 라벨 클래스 전부 뒤섞임 | classes.txt 절대 안 건드림 |
| 도면 일부 객체만 라벨 | 모델 혼란, 성능 하락 | 시작한 도면은 전부 |
| 배선도(핀맵)까지 라벨 | 오검출 증가 | 배선 그림만 |
| 박스에 케이블 선까지 크게 | 위치 부정확 | 부품 외곽에 딱 |
| train/val 에 같은 도면 | 성능 부풀려짐 | 겹치지 않게 |
| 렌더 배율 바꿈 | 라벨 좌표 안 맞음 | LONG_SIDE_PX=2000 유지 |
| json 만 저장하고 YOLO 변환 안 함 | 학습 못 함 | 섹션 G 로 변환 |
