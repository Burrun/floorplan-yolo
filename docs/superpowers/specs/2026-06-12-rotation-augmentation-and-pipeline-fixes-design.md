# 90°/180°/270° 회전 증강 및 파이프라인 필수 수정 설계 문서

## 1. 목적

`floorplan_training.py` 파이프라인에 다음 3가지 작업을 수행한다:
1. **오프라인 90°/180°/270° 회전 데이터 증강** — 도면의 스캔 방향 다양성을 반영하여 학습 데이터를 4배 확장
2. **Critical 버그 6건 수정** — `pipeline_review.md`에서 지적된 필수 수정사항
3. **가성비 높은 개선 2건** — 적은 코드 변경으로 논증 강도를 높이는 개선

모든 변경은 `floorplan_training.py` 단일 파일 내에서 수행한다.

---

## 2. 회전 증강 전처리 (신규 섹션)

### 2.1. 위치
Phase 1(Data Scaling Ablation)과 Phase 2(Baseline vs Augmented) 사이에 **"Section 4.5: Offline Rotation Augmentation"** 으로 삽입.

### 2.2. 대상
- `master_dataset/images/train` (1600장 .webp)
- `master_dataset/labels/train` (1600개 .txt)

### 2.3. 출력
- `master_dataset/images/train_aug/` — 원본 1600장 복사 + 회전 4800장 = 6400장
- `master_dataset/labels/train_aug/` — 대응 라벨 6400개
- `master_dataset/dataset_augmented.yaml` — train 경로를 `images/train_aug`로 지정

### 2.4. YOLO 라벨 회전 변환 수학

YOLO 포맷: `class_id cx cy w h` (0~1 정규화)

| 회전 각도 | cx' | cy' | w' | h' |
|---|---|---|---|---|
| 90° (반시계) | cy | 1 - cx | h | w |
| 180° | 1 - cx | 1 - cy | w | h |
| 270° (반시계) | 1 - cy | cx | h | w |

### 2.5. 파일 네이밍
- 원본: `master_train_0001.webp` → 그대로 복사
- 90°: `master_train_0001_rot90.webp`
- 180°: `master_train_0001_rot180.webp`
- 270°: `master_train_0001_rot270.webp`

### 2.6. 이미 생성 여부 확인
`train_aug` 폴더가 존재하고 파일 수가 6400 이상이면 생성을 스킵한다.

### 2.7. Phase 2 연동
- Phase 2 Augmented 학습에서 `dataset_augmented.yaml` 경로 사용
- Baseline은 기존 `dataset.yaml` (원본 1600장) 그대로 사용
- Phase 3 OCR 전이학습은 Augmented 모델의 `best.pt`를 기반으로 하므로 자동 혜택

---

## 3. 필수 버그 수정 (6건)

### B1. Phase 2-B: Baseline Legacy 평가 코드 추가
- **위치**: L528 부근 (Augmented 평가 직전)
- **변경**: Baseline 모델(`train_baseline/weights/best.pt`)도 `legacy_yaml_path`로 `model.val()` 수행
- **시각화**: Baseline vs Augmented Legacy mAP 비교 바 차트 추가

### B2. Phase 3: 시각화 제목 수정
- **위치**: L935
- **변경**: `"Scratch / 8-Class baseline Model"` → `"Scratch (Random Init) Model"`
- `"(Misses, False Positives, Feature Collision)"` → `"(Slower Convergence, Lower Accuracy)"`

### B3. 재현성: random.seed 추가
- **위치**: 스크립트 최상단 import 직후
- **변경**: `random.seed(42)`, `np.random.seed(42)`, `torch.manual_seed(42)` 추가

### B4. Phase 1: 샘플링 편향 수정
- **위치**: L326
- **변경**: `train_images[:size]` 앞에 `random.shuffle(train_images)` 삽입 (seed 이미 설정됨)

### B5. EDA dead code 정리
- **위치**: L269-271
- **변경**: `_fixed` fallback 로직 후 `random.choice`로 덮어쓰는 L271 제거

### B6. CLASS_COLOR_MAP 중복 제거
- **위치**: L978 부근
- **변경**: Phase 4 셀의 `CLASS_COLOR_MAP` 재정의 삭제. 상단(L215) 정의만 사용.

---

## 4. 가성비 높은 개선 (2건)

### I1. Phase 3 수렴 속도 지표 추가
- **위치**: Phase 3 시각화 코드 말미
- **변경**: mAP≥0.9 도달 에폭 수를 Scratch vs Transfer 각각 계산하여 print 출력
- 예: `"Transfer: 8 에폭 만에 mAP≥0.9 도달 / Scratch: 35 에폭 소요 (4.4x 빠름)"`

### I2. Augmented hsv 파라미터 주석 보강
- **위치**: L465-466
- **변경**: `hsv_s=0.2`, `hsv_v=0.2`가 YOLO 기본값(0.7, 0.4)보다 의도적으로 약한 이유를 주석에 명시
- 이유: 도면은 흑백/저채도 기반이므로 과도한 색상 변형은 비현실적

---

## 5. 변경하지 않는 것

- Phase 1 로직 (Nano 모델 사용 유지 — 트렌드 탐색 목적)
- Phase 4 추론 파이프라인 (SAHI + 앙상블 구조 유지)
- val/test 셋 (증강 대상이 아님)
- OCR 데이터셋 (별도 증강 불필요)
- 문서(README.md, REPORT.md, TODO.md) — 코드 수정 후 별도 동기화

---

## 6. 검증 계획

1. 회전 증강: `train_aug` 폴더의 파일 수 확인 (6400장), 샘플 1장의 라벨 좌표가 시각적으로 올바른지 matplotlib 확인
2. Baseline Legacy 평가: `val()` 결과 출력 확인
3. 시각화 제목: 문자열 변경 확인
4. random.seed: 동일 시드로 2회 실행 시 Phase 1 결과 동일 여부 확인
5. dead code: L271 제거 후 EDA 셀 정상 실행 확인
