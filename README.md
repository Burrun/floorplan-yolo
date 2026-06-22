# 🏢 레거시 2D 아파트 도면 디지털화를 위한 AI 베이스라인 모델 구축 및 도메인 갭 극복 실험

> **숭실대학교 소프트웨어학과 — 딥러닝분석 파이널 프로젝트**

## 주제 (Project Goal)

종이 도면(레거시 2D 평면도)을 디지털 JSON으로 변환하는 AI 베이스라인 구축.
YOLOv8 기반 파이프라인: ① Data Scaling Ablation ② 도메인 증강 실전 효과 (Simulated Legacy Test Set) ③ Transfer Learning 검증 ④ JSON 좌표 후처리 (Topological Post-processing).

---

## 데이터셋 및 도메인 한계 극복 전략

AI Hub 아파트 도면 데이터 활용.

*   **데이터 구축 (Scrambled Hash Matching)**: 난독화 파일들을 Size+CRC32 해시 대조로 병합. 약 5,000장 마스터 데이터셋 구축 (가구+문/창호 포함).
*   **데이터 분할**: Train:Val:Test = **8:1:1**

| 과제 | 사용 데이터 | 내용 | 활용 Phase |
|------|------|------|-----------| 
| **가구/설비/구조물** | `object_layout` & `structural_elements` | 가구/설비/구조물 7종 통합 모델 | Phase 1 & 2 |
| **텍스트(OCR) 탐지** | `ocr` | 도면 내 텍스트 위치(BBox) 탐지 | Phase 3 |

> **경량화 아키텍처: 왜 Text Recognition 모델을 붙이지 않는가?**
> 핵심 목표는 "레이아웃+좌표 JSON 구조화". 무거운 인식 엔진(PaddleOCR) 제외, **YOLO로 위치(BBox)만 탐지해 JSON 기록하는 경량 앙상블**. 속도/복잡도 최적화.

> **도메인 갭(Domain Gap) 이슈와 극복 방안**
> 학습 데이터는 깨끗함, 타겟은 **'옛날 도면(노이즈/변색/저해상도)'**. 극복:
> 1. **도메인 맞춤형 증강** (황변, 노이즈, 회전)
> 2. **Simulated Legacy Test Set** 증강 효과 검증
> 3. **JSON 좌표 후처리** (직교화 + 스냅) 출력 품질 향상

---

## 프로젝트 워크플로우

### Phase 1: 최적 데이터 양 탐색 (Data Scaling Ablation)
- **목표**: 300~1500장 늘리며 성능 수렴점 탐색.
- **설정**: 구간당 **30 에폭**, mAP@50 비교.
- **의의**: "많은 데이터가 무조건 정답은 아님" 증명.

### Phase 2: 도메인 맞춤 전처리 파이프라인 (Pre-processing)
- **실험**: 레거시 스캔본 고유의 망점(우글우글한 점), 변색, 불균일 음영 등을 제거하기 위해 다양한 이진화 및 모폴로지 연산 평가.
- **결과**: **적응형 가우시안 이진화(Adaptive Gaussian)** 채택. 얇은 선의 디테일을 유지하면서 조명 변화와 배경 노이즈를 완벽하게 제거하여 입력 이미지 품질 극대화.

### Phase 3: 도메인 맞춤형 하이퍼 파라미터 튜닝 (Domain-Tuned vs Baseline)
- **실험 3-A**: Baseline 모델 vs Domain-Tuned(회전 2.0도, 상하반전 50%, 흑백화 20% 등 스캔 특성 반영) 모델 비교.
- **실험 3-B**: **Simulated Legacy Test Set** 환경에서의 증강 효과 시각적 검증.
- **핵심 논증**: "깨끗한 도면에서는 Baseline이 우세할지라도, 왜곡이 존재하는 실제 타겟(Legacy) 환경에서는 Domain-Tuned 모델의 강건성이 압도함."

### Phase 4: 아키텍처 분리 전략 (8-Class 동시학습 vs 2-Stage 전이학습)
- **문제점 (Feature Interference)**: 구조물(7클래스)과 텍스트 박스(1클래스)를 통합한 8클래스 동시 학습 시, 기하학적 형태와 박스 비율의 이질성 때문에 상호 mAP 성능을 깎아먹는 특징 충돌 발생.
- **해결 (2-Stage 분리 및 Transfer Learning)**: 구조물 전용 마스터 모델(Stage 1)과 OCR 텍스트 전용 모델(Stage 2)로 분리.
- **의의**: 텍스트 모델을 밑바닥(Scratch)부터 학습하는 대신, Phase 3에서 완성된 마스터 모델의 '노이즈 저항성' 가중치를 넘겨받아 전이학습(Transfer)을 수행함으로써, 심각한 노이즈 속에서도 글자를 완벽히 찾아내는 압도적인 정확도와 수렴 속도 증명.

### Phase 5: 최종 전체 파이프라인 (Overall Pipeline) 및 JSON 구조화
- **파이프라인 흐름**: 
  1. (Phase 2) 원본 도면 이미지를 Adaptive Gaussian 이진화로 전처리.
  2. (Phase 3) 구조물 탐지 마스터 모델 적용하여 7클래스 객체 추출.
  3. (Phase 4) 마스터 지식을 전이받은 텍스트 전용 모델이 OCR 영역 추출.
  4. 앙상블(Ensemble) 병합 및 SAHI 슬라이싱 적용으로 탐지 성능 극대화.
- **JSON 파이프라인** ⭐: 
  - 최종 앙상블 BBox 좌표(x_min, y_min, x_max, y_max)를 추출, 객체를 `structures`, `furnitures`, `ocr` 카테고리별로 분리하여 JSON 저장.
- **Future Work**: 
  1. **고도화 후처리**: 위상 수학 기반 직교화(Orthogonalize), 스냅, 부모-자식 계층화 적용.
  2. **Real-world JSON 치수 역산**: 도면 하단 '축척' 텍스트 인식, 픽셀 좌표를 물리적 치수(mm/cm)로 변환하는 알고리즘 고도화.

---

## 실행 환경

### Google Colab (메인 런타임)
1. 드라이브에 `master_dataset.tar.zst` 업로드.
2. `notebooks/floorplan_training.ipynb` 열기.
3. 런타임 → **L4 GPU** 설정.
4. 셀 실행.

### Modal 클라우드 GPU (A10G)
```bash
pip install modal
modal setup          # 최초 1회 인증
modal run modal_train.py
```
> 학습 완료 후 `runs/` 폴더가 자동으로 로컬에 다운로드됩니다.
