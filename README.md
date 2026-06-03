# 🏢 2D 아파트 도면 AI 분석 — 객체 검출부터 구조 인식까지

> **숭실대학교 소프트웨어학과 — 딥러닝분석 프로젝트**

## 주제

2D 건축 도면(아파트 평면도) 이미지를 딥러닝으로 **디지털 구조화**하는 AI 모델 구축.

단순 객체 탐지를 넘어, 도면 속 **가구 → 텍스트 → 공간 → 구조**를 단계적으로 인식하는 멀티태스크 파이프라인 설계.

---

## 데이터셋

AI Hub 제공 아파트 도면 샘플. 4개 세부 과제별 라벨링 제공:

| 과제 | 폴더 | 내용 | 활용 Phase |
|------|------|------|-----------|
| **OBJ** (Object) | `OBJ/` | 변기·세면대·싱크대·욕조·가스레인지 바운딩 박스 | Phase 1 |
| **OCR** (Text) | `OCR/` | 도면 내 텍스트(방 이름, 치수 등) | Phase 2 |
| **SPA** (Space) | `SPA/` | 거실·안방·욕실 등 공간 영역 | Phase 2 |
| **STR** (Structure) | `STR/` | 벽체·기둥·창문·문 구조선 | Phase 2 |

> 상세 전처리·클래스 매핑은 [DATASET_GUIDE.md](data/DATASET_GUIDE.md) 참조.

---

## 워크플로우

### Phase 1: 가구/설비 객체 탐지 (OBJ) — 현재 단계

> **목표**: 도면 PNG에서 5종 가구/설비를 YOLOv8로 검출하는 베이스라인 확보.

```
도면 이미지(PNG)
  ↓
JSON 라벨 → YOLO 포맷(.txt) 변환 (COCO bbox → 정규화 좌표)
  ↓
Train/Val 스플릿 (8:2)
  ↓
데이터 증강 (오래된 도면 도메인 갭 방어)
  · 회전(±15°), 원근 왜곡, 블러, 모자이크
  ↓
YOLOv8n 학습 (Colab T4 GPU, 50 epochs)
  ↓
평가: mAP50 / mAP50-95 / Confusion Matrix
  ↓
추론 시각화: Ground Truth vs Prediction 비교
```

**핵심 포인트**:
- 오래된 스캔/팩스 도면 대응 위해 **강한 데이터 증강** 적용
- 학습 데이터 20장(소량) → 증강으로 보완, 베이스라인 성능 우선 확보

---

### Phase 2: 멀티태스크 확장 (OCR / SPA / STR)

> **목표**: Phase 1 모델을 기반으로 도면의 다른 요소(텍스트, 공간, 구조)까지 인식 범위 확장.

```
Phase 1 학습 완료 모델 (가구 검출 베이스라인)
  ↓
┌────────────────────────────────────────────┐
│  실험 비교: 전이학습 vs 처음부터 학습           │
│  · Transfer Learning: Phase 1 가중치 재활용  │
│  · From Scratch: 각 과제별 독립 학습         │
│  → 어느 쪽이 소량 데이터에서 유리한지 검증     │
└────────────────────────────────────────────┘
  ↓
OCR: 도면 내 텍스트 인식 (방 이름, 치수)
SPA: 공간 영역 분할 (거실, 안방, 욕실 등)
STR: 구조 요소 검출 (벽체, 창문, 문)
```

**연구 질문**:
- Phase 1에서 학습한 도면 도메인 feature가 OCR/SPA/STR에 **전이(transfer)**되는가?
- 소량 데이터(과제당 20장) 환경에서 전이학습이 유의미한 성능 차이를 만드는가?

---

## 실행 환경

### Google Colab (메인)
1. 구글 드라이브에 데이터셋 업로드
2. `notebooks/floorplan_training.ipynb` 열기
3. 런타임 → T4 GPU 설정
4. 셀 순서대로 실행 (전처리 → 학습 → 평가 원스톱)

### 로컬 IDE (코드 편집용)
```bash
uv venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt
python tools/jupytext_watch.py   # .py ↔ .ipynb 실시간 동기화
```
> `.py` 파일만 편집하면 Jupytext가 `.ipynb` 자동 갱신.

---

## 향후 확장

Phase 1~2 완료 후, 검출 결과(클래스, 좌표, 크기)를 JSON 추출 → Three.js/WebGL 기반 3D 공간에 가구 자동 배치. **2D-to-3D Auto Extrusion Pipeline** 구상.
