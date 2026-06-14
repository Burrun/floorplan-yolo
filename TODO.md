# 🎯 목표 (Goal)

## 1. 목적
- Clean val/test 셋. 노이즈/증강 시 성능 하락.
- 타겟: 레거시(Legacy) 도면 디지털화.

## 2. 파이프라인 (최종 확정)

### Phase 1: Data Scaling Ablation (`master_dataset`)
- [x] 300~1500장, 30 에폭, YOLOv8n 고정.
- [x] 점진적 상승 곡선 확보 → 최적 데이터 크기 결정.

### Phase 2: 전처리 파이프라인 (Pre-processing)
- [x] Adaptive Gaussian Thresholding 채택.
- [x] 레거시 망점, 변색, 불균일 음영 제거.
- [x] `apply_adaptive_gaussian_preprocessing()` 함수화 완료.

### Phase 3: 도메인 맞춤형 하이퍼 파라미터 탐색 (`master_dataset`)
- [x] 3-A: Baseline vs Domain-Tuned (150 에폭, patience=20).
- [x] 3-B ⭐: Simulated Legacy 시각적 강건성 검증 (Visual Inspection).
  - 무작위 5장에 황변/노이즈/퇴색/블러 모사 적용.
  - 논증: 실전 레거시 환경에서 Domain-Tuned 모델의 시각적 안정성 증명.

### Phase 4: OCR 전이학습 (`ocr_dataset`)
- [x] 2-Stage 분리 전이학습: 마스터 모델(`train_domain_tuned`) → OCR 전용 모델.
- [x] Scratch vs Transfer, 50 에폭, patience=10.
- [x] mAP 수렴 곡선, Confusion Matrix, PR-Curve, F1-Curve 비교 완료.
- [x] 도면 도메인 OCR 전이 효과 증명 (전이학습 선택의 당위성).

### Phase 5: 최종 파이프라인 실전 추론 (Overall Pipeline)
- [x] SAHI 슬라이싱 (512×512, overlap 20%) + Domain-Tuned 모델(`best.pt`).
- [x] 앙상블 (Master 7cls + OCR 1cls 융합) ⭐:
  - 개별 모델 Confidence 파싱 및 MIN_CONFIDENCE=0.3 중재 로직 적용.
- [x] 픽셀 기반 BBox JSON 구조화 (`structures`/`furnitures`/`ocr`) 및 Export 완료 ⭐.

## 3. Future Work
- [중장기] 3D 렌더링용 고도화 JSON 후처리 (Phase 6).
  - 1단계: IoA > 0.8 부모-자식 편입.
  - 2단계: Topological Post-processing: 0,90,180,270도 직교화(Orthogonalize) + 가상 벽체 스냅(Snap).
- End-to-End Vectorization (RoomFormer, FloorplanVLM).
- GNN Topological Consistency.
- Pix2Pix/Diffusion 도면 복원.
- 레거시 소량 라벨링 → Fine-tuning.
