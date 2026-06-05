# 🏢 레거시 2D 아파트 도면 디지털화를 위한 AI 베이스라인 모델 구축 및 도메인 갭 극복 실험

> **숭실대학교 소프트웨어학과 — 딥러닝분석 파이널 프로젝트**

## 주제 (Project Goal)

본 프로젝트는 **"과거의 종이 도면(레거시 2D 평면도)을 구조화된 디지털 데이터(JSON)로 자동 변환하기 위한 AI 베이스라인 모델 구축"**입니다. 
YOLOv8 기반의 객체 탐지 모델을 중심으로, **① 최적 데이터 양 탐색(Data Scaling Ablation)**, **② 도메인 맞춤형 증강의 실전 효과 증명(Simulated Legacy Test Set)**, **③ 전이학습(Transfer Learning) 유효성 검증**, **④ JSON 좌표 후처리(Topological Post-processing)**까지의 공학적 파이프라인을 설계·검증합니다.

---

## 데이터셋 및 도메인 한계 극복 전략

AI Hub에서 제공하는 아파트 도면 데이터를 활용합니다.

| 과제 | 사용 데이터 | 내용 | 활용 Phase |
|------|------|------|-----------| 
| **가구/설비** (Object Layout) | `object_layout` & `structural_elements` | 주요 가구/설비 및 구조물 7종 (변기, 세면대, 싱크대, 욕조, 가스레인지, 문, 창호) BBox | Phase 1 & 2 |
| **텍스트** (OCR) | `ocr` | 도면 내 텍스트(방 이름, 치수 등) BBox | Phase 3 |

> **도메인 갭(Domain Gap) 이슈와 극복 방안**
> 학습 데이터셋은 깔끔한 CAD 기반 도면이지만, 실제 타겟은 **'옛날 도면(노이즈, 변색, 낮은 해상도)'**입니다. 이 갭을 극복하기 위해:
> 1. **도메인 맞춤형 데이터 증강** (황변, 스캔 노이즈, 회전 등)
> 2. **Simulated Legacy Test Set** 구축으로 증강 효과를 실전 기준으로 증명
> 3. **JSON 좌표 후처리** (직교화 + 스냅)로 출력 품질 향상

---

## 프로젝트 워크플로우

### Phase 1: 최적 데이터 양 탐색 (Data Scaling Ablation)
- **목표**: 100~1000장까지 데이터 양을 늘려가며 성능 수렴점(Saturation Point)을 탐색.
- **설정**: 각 구간 **20 에폭** 학습, mAP@50 기준 비교.
- **의의**: "무조건 많은 데이터가 정답이 아니다"를 공학적으로 증명.

### Phase 2: 도메인 맞춤형 증강 효과 검증 (Augmentation Ablation)
- **실험 2-A**: Baseline(순정) vs Augmented(증강) 모델을 **깨끗한 val 셋**에서 비교.
- **실험 2-B** ⭐: **Simulated Legacy Test Set**(인위적 열화 테스트셋)을 구축하여 재평가.
  - 기존 test 셋 50장에 황변, 노이즈, 해상도 저하를 적용한 가상 레거시 도면 생성.
  - **핵심 논증**: "Clean val에서는 Baseline이 이기지만, Legacy Test에서는 Augmented가 압도적 승리"

### Phase 3: 도면 도메인 전이학습 효과 분석 (Transfer vs Scratch)
- **목표**: 가구/설비 학습으로 익힌 도메인 특징(Feature)이 OCR 태스크에 전이되는지 검증.
- **실험 A (Scratch)**: `yolov8n.pt` → OCR 데이터 **30 에폭** 학습.
- **실험 B (Transfer)**: `best.pt` (Phase 2 결과) → OCR 데이터 **30 에폭** 학습.
- **의의**: 충분한 에폭(30)으로 Scratch도 수렴한 상태에서 Transfer의 우위를 공정하게 입증.

### Phase 4: 레거시 도면 실전 추론 및 JSON 구조화
- **목표**: 학습 완료된 최적 모델로 진짜 옛날 도면을 분석, JSON 결과물을 생성.
- **전처리**: CLAHE, Unsharp Masking 기반 화질 개선.
- **추론**: SAHI 슬라이싱으로 고해상도 도면의 미탐지 방지.
- **후처리** ⭐: **Topological Post-processing** (좌표 직교화 + 스냅) 적용.
  - 비뚤어진 bbox를 수평/수직으로 스냅, 가까운 좌표를 통일하여 JSON 품질 향상.
- **한계 분석**: 과거 건축 기호(Symbol) 차이로 인한 미탐지(False Negative) 현상 기록.
- **Future Work**: 
  1. **실제 축척(Scale)을 반영한 Real-world JSON 변환** ⭐: 현재 YOLO가 출력하는 JSON은 도면 이미지 내 상대적 픽셀 좌표 기준입니다. 진정한 디지털화를 위해서는 OCR을 통해 도면 하단의 '축척(ex. 1:100)' 텍스트를 인식하고, 이를 기반으로 픽셀 좌표를 실제 물리적 치수(mm/cm)로 역산하는 스케일링 로직 추가가 필수적입니다.
  2. 레거시 도면 소량 라벨링 후 Fine-tuning.
  3. End-to-End Vectorization(RoomFormer 등) 모델 도입 제안.

---

## 실행 환경

### Google Colab (메인 런타임)
1. 구글 드라이브에 데이터셋(`architectural_drawing_data.tar.zst`) 업로드.
2. `notebooks/floorplan_training.ipynb` 열기.
3. 런타임 → **L4 GPU** 설정.
4. 셀 순서대로 실행.

### 로컬 IDE (코드 뷰어 및 편집용)
```bash
source .venv/bin/activate
pip install -r requirements.txt
```
> 작업은 주로 `.py` 스크립트를 편집하며, Jupytext를 통해 Colab과 연동되는 `.ipynb`로 자동 변환/동기화 시킵니다.

---

## 🛠️ 트러블슈팅 (Troubleshooting)

### YOLO 추론 시 한글 라벨이 물음표(`?????`)로 깨지는 현상
YOLOv8의 기본 폰트(Arial)가 한글을 지원하지 않아 발생하는 문제입니다. 다음과 같이 한글 폰트(나눔고딕)를 다운로드하여 YOLO 내부 폰트를 덮어씌우면 해결됩니다.

**Colab 해결 방법 (새 셀을 열고 아래 코드 실행):**
```bash
# 1. 나눔고딕 폰트 다운로드
!wget -q -O NanumGothic.ttf https://github.com/naver/nanumfont/raw/master/NanumFont/NanumGothic.ttf

# 2. YOLO 폰트 디렉토리 생성 및 덮어쓰기
!mkdir -p /root/.config/Ultralytics
!mv NanumGothic.ttf /root/.config/Ultralytics/Arial.ttf
```
실행 후 다시 시각화/추론 코드를 돌리면 한글이 정상 출력됩니다.
