# 🏢 레거시 2D 아파트 도면의 완벽한 디지털화(Digitization) 파이프라인 구축

> **숭실대학교 소프트웨어학과 — 딥러닝분석 파이널 프로젝트**

## 주제 (Project Goal)

본 프로젝트의 최종 목적은 **"과거의 종이 도면이나 단순 이미지 파일(레거시 2D 평면도)을 완벽히 구조화된 디지털 데이터(JSON)로 자동 변환하는 AI 파이프라인 구축"**입니다. 
도면 내의 가구/설비(Object), 공간 구획(Space), 텍스트(OCR), 구조선(Structure) 등을 통합적으로 탐지하여, 직방 등 프롭테크(Prop-Tech) 산업이나 2D-to-3D 변환의 기반이 되는 자동화 솔루션을 목표로 합니다.

이러한 범용적인 '마스터 파이프라인'을 만들기 위한 핵심 과정으로서, **① 베이스라인 모델 구축**, **②극한의 데이터 증강을 포함한 하이퍼파라미터 튜닝(Ablation Study)**, **③전이학습(Transfer Learning)**의 유효성을 공학적으로 검증합니다.

---

## 데이터셋 및 도메인 한계 극복 전략

AI Hub에서 제공하는 아파트 도면 데이터(모든 과제가 동일한 1~23 카테고리 ID 시스템 공유)를 활용합니다. 

| 과제 | 사용 데이터 | 내용 | 활용 Phase |
|------|------|------|-----------|
| **가구/설비** (Object Layout) | `object_layout` | 변기, 세면대, 싱크대, 욕조, 가스레인지 BBox | Phase 1 & 2 |
| **텍스트** (OCR) | `ocr` | 도면 내 텍스트(방 이름, 치수 등) BBox | Phase 3 |
| **마스터 통합 (옵션)** | 전체 통합 | 가구 + 공간(방) + 벽체 + 텍스트 동시 탐지 | Phase 4 (최종) |

> **도메인 갭(Domain Gap) 이슈와 극복 방안**
> 학습 데이터셋은 비교적 깔끔한 CAD 기반의 획일화된 도면입니다. 이를 실제 '옛날 도면(노이즈, 변색, 낮은 해상도)'에 적용할 때 발생하는 한계를 극복하기 위해 **극단적인 데이터 증강(Augmentation: 블러, 명암 조절, 노이즈 추가 등)** 기법을 도입하여 모델의 실전 범용성(Generalization)을 끌어올립니다.

---

## 프로젝트 워크플로우

### Phase 1: 베이스라인 모델 구축 (Baseline)
- **목표**: `YOLOv8n` (가장 가벼운 Nano 모델)을 사용하여 `object_layout` 데이터셋에 대한 1차 베이스라인 학습 진행.
- **의의**: 이후 진행될 하이퍼파라미터 튜닝과 범용성 테스트의 기준점(Anchor) 확보.

### Phase 2: 실제 현장 적용을 위한 하이퍼파라미터 튜닝 (Ablation Study)
- **목표**: 레거시 도면의 스케일 차이와 스캔 노이즈를 극복하기 위한 파라미터를 도출하고 성능 개선 효과를 증명.
- **실험 변수**:
  - `imgsz` (해상도): 640 vs 1024. 도면 내 가구/설비의 스케일 차이에 따른 탐지율 변화 분석.
  - **데이터 증강 (Augmentation)**: 낡은 스캔본을 시뮬레이션하기 위한 증강 기법이 실제 범용 성능에 미치는 영향 비교.
  - `lr0` (초기 학습률): 기본 학습률(0.01)과 조정된 학습률 간의 수렴 안정성 비교.

### Phase 3: 도면 도메인 전이학습 효과 분석 (Transfer vs Scratch)
- **목표**: '도면'이라는 특수한 도메인의 특징(Feature)이 다른 태스크(예: OCR 탐지)에 긍정적인 전이가 일어나는지 검증.
- **실험 A (Scratch)**: `YOLOv8n` 원본 가중치로 `ocr` 데이터셋 훈련.
- **실험 B (Transfer)**: Phase 1/2를 거쳐 도면 도메인에 완전히 적응한 가중치(`best.pt`)로 `ocr` 데이터셋 훈련.

### Phase 4: 구형 도면 실전 추론 및 한계 분석 (Limitations & Future Work)
- **목표**: 학습 완료된 최적의 모델로 구글 등에서 수집한 '진짜 옛날 아파트 평면도'를 분석하고, JSON 결과물을 덤프(Export)하여 최종 실용성을 검증.
- **한계점 도출**: 노이즈와 해상도 저하는 증강 기법으로 극복했으나, **과거의 건축 기호(Symbol) 자체가 최신 기호와 완전히 다른 경우** 발생하는 미탐지(False Negative) 한계 현상 분석.
- **Future Work**: 추후 레거시 도면 소량 데이터만 라벨링하여 Fine-tuning 하면 즉각 상용화가 가능함을 제안하며 공학적 결론 도출.

---

## 실행 환경

### Google Colab (메인 런타임)
1. 구글 드라이브에 데이터셋(`architectural_drawing_data.7z`) 업로드.
2. `notebooks/floorplan_training.ipynb` 열기.
3. 런타임 → T4 GPU 설정.
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
