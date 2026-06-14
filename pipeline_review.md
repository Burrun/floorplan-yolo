# 🔬 프로젝트 다면적 평가: 2D 아파트 도면 디지털화 파이프라인

> 평가 대상: [README.md](file:///c:/Users/jack3/Desktop/pj/README.md), [REPORT.md](file:///c:/Users/jack3/Desktop/pj/REPORT.md), [TODO.md](file:///c:/Users/jack3/Desktop/pj/TODO.md), [floorplan_training.py](file:///c:/Users/jack3/Desktop/pj/floorplan-yolo/notebooks/floorplan_training.py)

---

## 1. 전체 파이프라인 구조 평가

### ✅ 잘 된 점
- **Phase 1 → 2 → 3 → 4** 순차적 의존 구조가 논리적으로 깔끔함. 각 단계의 산출물이 다음 단계의 입력이 되는 **파이프라인 체이닝**이 잘 설계됨.
- **"데이터 양 → 증강 → 전이학습 → 추론"** 순서는 ML 실험 설계의 정석에 가까움.
- **Hardware-Aware Dual Profile** (PRO/NORMAL 모드)은 실용적 엔지니어링 센스가 돋보임.
- **SAHI 슬라이싱 + 앙상블 + JSON 구조화**까지 이어지는 End-to-End 파이프라인이 완성되어 있음.

### ⚠️ 구조적 리스크
- Phase 1에서 결정한 최적 데이터 양이 **Nano 모델 기준**인데, Phase 2부터는 **Medium 모델**을 사용. 모델 용량이 다르면 최적 데이터 포화점도 달라질 수 있으므로, Phase 1의 결론이 Phase 2에 그대로 이전된다는 보장이 없음.

---

## 2. 실험 설계 및 논리적 엄밀성

### Phase 1: Data Scaling Ablation

> [!WARNING]
> **샘플링 편향 (Critical)**
> [floorplan_training.py:326](file:///c:/Users/jack3/Desktop/pj/floorplan-yolo/notebooks/floorplan_training.py#L326)에서 `train_images[:size]`로 **앞에서부터 순서대로** 잘라서 사용. 이는 랜덤 샘플링이 아님. 파일 시스템 정렬 순서(파일명 알파벳)에 의존하므로, 특정 출처나 스타일의 도면이 앞쪽에 몰려 있다면 **체계적 편향(Systematic Bias)**이 발생함.
>
> **수정 제안**: `random.seed(42); random.shuffle(train_images)` 후 슬라이싱.

| 항목 | 문서 (README/TODO) | 코드 (`.py`) | 불일치 |
|---|---|---|---|
| 데이터 범위 | 100~1000장 | 300~1500장 | ⚠️ |
| 에폭 수 | 20 에폭 | 30 에폭 | ⚠️ |
| 모델 | 명시 안 함 | `yolov8n.pt` 고정 | - |

- **긍정적**: 데이터 양에 따른 mAP 추세선을 그리는 것 자체는 학술적으로 매우 유효한 ablation 방법론.
- **부정적**: 단일 실행(single run)이므로 에러바(error bar)가 없음. 랜덤 시드에 따른 변동을 구분할 수 없어, 0.510 → 0.570의 상승이 통계적으로 유의미한지 판단 불가.

---

### Phase 2: Baseline vs Augmented

> [!IMPORTANT]
> **"순정(Baseline)" 정의의 모호성**
> YOLO의 기본 학습에도 이미 `hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, mosaic=1.0` 등 **디폴트 증강**이 포함되어 있음. 따라서 코드상의 "Baseline"은 **증강 없음**이 아니라 **YOLO 기본 증강**이고, "Augmented"는 **기본 증강 + 커스텀 증강**임. 문서에서 이 점을 명확히 하지 않으면 "증강의 효과"라는 논증이 오해를 살 수 있음.

**증강 파라미터 분석** ([floorplan_training.py:464-469](file:///c:/Users/jack3/Desktop/pj/floorplan-yolo/notebooks/floorplan_training.py#L464-L469)):

| 파라미터 | 설정값 | YOLO 기본값 | 평가 |
|---|---|---|---|
| `degrees` | 2.0 | 0.0 | ✅ 보수적이고 합리적 |
| `hsv_s` | 0.2 | 0.7 | ⬇️ 오히려 기본보다 **약함** |
| `hsv_v` | 0.2 | 0.4 | ⬇️ 오히려 기본보다 **약함** |
| `perspective` | 0.0005 | 0.0 | ✅ 미세 왜곡, 합리적 |
| `scale` | 0.5 | 0.5 | ➡️ 동일 |
| `mosaic` | 1.0 | 1.0 | ➡️ 동일 (기본값) |

> [!CAUTION]
> `hsv_s`와 `hsv_v`가 기본값보다 오히려 **약하게** 설정됨. 즉, Augmented 모델은 색상 변형 측면에서는 Baseline보다 **덜 강한 증강**을 받고 있을 가능성이 있음. "도메인 갭 극복을 위한 강력한 증강"이라는 서사와 실제 코드 사이에 괴리가 있음.

---

### Phase 2-B: Simulated Legacy Test Set

> [!WARNING]
> **비교 실험 누락 (Critical)**
> 코드([floorplan_training.py:528-532](file:///c:/Users/jack3/Desktop/pj/floorplan-yolo/notebooks/floorplan_training.py#L528-L532))에서 **Augmented 모델만** Legacy Test Set에 평가하고, **Baseline 모델의 Legacy Test 평가가 없음**.
>
> 발표 스크립트에는 "순정 0.281 vs 증강 0.463"이라고 적혀 있으나, 이 비교를 생성하는 코드가 `.py`에 존재하지 않음. Baseline도 동일 Legacy Set에서 `model.val()` 해야 공정한 비교.

- **긍정적**: Simulated Legacy Test Set이라는 아이디어 자체는 매우 영리함. 실제 타겟 도메인을 모사하여 평가한다는 접근은 학술적으로 강력한 논증 수단.
- `simulate_legacy_degradation()` 함수의 황변+퇴색+노이즈+블러 조합은 실제 스캔 도면의 열화를 잘 모사.

---

### Phase 3: Transfer Learning

**논리적 문제점들:**

1. **"Scratch" 라벨링 오류**: [floorplan_training.py:935](file:///c:/Users/jack3/Desktop/pj/floorplan-yolo/notebooks/floorplan_training.py#L935)에서 시각화 제목이 `"Scratch / 8-Class baseline Model"`인데, 실제로 이 모델은 **1-class OCR 모델**(yolov8m.pt → OCR만 학습). 8-class 동시 학습 실험은 코드에 존재하지 않음. **실행하지 않은 실험을 결과에 등치시키는 것은 문제**.

2. **Transfer의 우위는 당연한 결과**: Transfer 모델은 `ImageNet pretrained → 도면 7cls 150에폭 학습 → OCR 50에폭` 총 200에폭 이상의 학습량. Scratch는 `ImageNet pretrained → OCR 50에폭`만. 더 많이 학습한 모델이 더 좋은 건 **예상 가능한(trivial) 결과**이지, 전이학습의 효과를 증명하는 강력한 근거는 아님.

3. **공정한 비교를 위한 대안**: Transfer의 진짜 가치를 보이려면:
   - 같은 총 에폭 수에서 비교 (예: Scratch 200에폭 vs Transfer 50에폭)
   - 또는 **수렴 속도** 차이를 핵심 논증으로 가져가야 함 (이미 발표 스크립트에서 "8에폭 만에 수렴"이라고 했는데, 이 부분을 코드+그래프로 더 강조할 것)

| 항목 | 문서 (README/TODO) | 코드 (`.py`) | 불일치 |
|---|---|---|---|
| 에폭 수 | 30 에폭 | 50 에폭 | ⚠️ |
| Scratch 모델 | `yolov8n.pt` | `BASE_WEIGHT` (=`yolov8m.pt`) | ⚠️ README만 |

---

### Phase 4: 앙상블 추론 + JSON

- **긍정적**: SAHI + 2-model 앙상블 + JSON Export까지 완성한 것은 프로젝트의 완결성 면에서 우수.
- **긍정적**: `MIN_CONFIDENCE = 0.3` 으로 Confidence 필터링, `structures`/`furnitures`/`ocr` 카테고리 분리 — 실무에서 바로 쓸 수 있는 구조.

> [!NOTE]
> **TODO/future_work.md에 언급된 IoA 부모-자식 계층화, Confidence-Aware Arbitration, Topological Snap 등은 코드에 미구현 상태.** 현재 코드는 단순 Confidence 필터링 + 카테고리 분리 수준. "후처리 파이프라인 완성"이라고 하기엔 아직 갈 길이 있음.

---

## 3. 코드 품질 평가

| 이슈 | 위치 | 심각도 |
|---|---|---|
| `CLASS_COLOR_MAP` 중복 정의 | [L215](file:///c:/Users/jack3/Desktop/pj/floorplan-yolo/notebooks/floorplan_training.py#L215) & [L978](file:///c:/Users/jack3/Desktop/pj/floorplan-yolo/notebooks/floorplan_training.py#L978) | 🟡 Low |
| EDA 샘플 `_fixed` 선택 직후 `random.choice`로 덮어씀 (dead code) | [L269-271](file:///c:/Users/jack3/Desktop/pj/floorplan-yolo/notebooks/floorplan_training.py#L269-L271) | 🟡 Low |
| Phase 2-B에서 Baseline Legacy 평가 코드 누락 | [L528-532](file:///c:/Users/jack3/Desktop/pj/floorplan-yolo/notebooks/floorplan_training.py#L528-L532) | 🔴 High |
| Phase 3 시각화 제목이 실제 실험과 불일치 ("8-Class") | [L935](file:///c:/Users/jack3/Desktop/pj/floorplan-yolo/notebooks/floorplan_training.py#L935) | 🔴 High |
| `random.seed()` 미설정 → 재현 불가능 | 전체 | 🟠 Medium |

---

## 4. 문서-코드 일관성 평가

총 **6건의 수치 불일치** 발견:

```diff
  Phase 1:
-  문서: 100~1000장, 20 에폭
+  코드: 300~1500장, 30 에폭

  Phase 2:
-  TODO: 50 에폭
+  코드: 150 에폭

  Phase 3:
-  README: yolov8n.pt, 30 에폭
+  코드: yolov8m.pt (BASE_WEIGHT), 50 에폭
```

> [!CAUTION]
> 제출물(노트북+보고서)에서 이 불일치가 그대로 노출되면 **"코드를 직접 작성/실행하지 않았다"는 인상**을 줄 수 있음. 문서 OR 코드 중 하나로 통일 필수.

---

## 5. 학술적 논증 강도 평가 (5점 만점)

| 논증 | 점수 | 평가 |
|---|---|---|
| "데이터 양과 성능은 단조증가하지만 수렴한다" | ⭐⭐⭐⭐ | 그래프 증거 있음. 다만 에러바 없음 |
| "도메인 증강이 레거시 도면에 효과적이다" | ⭐⭐⭐ | 아이디어 우수하나, Baseline Legacy 평가 코드 누락 |
| "8클래스 동시학습은 특징 충돌을 일으킨다" | ⭐⭐ | **코드에 8클래스 실험 없음.** 주장만 있고 실험적 근거 부재 |
| "전이학습이 Scratch보다 압도적이다" | ⭐⭐⭐ | 결과는 맞지만 총 학습량 차이를 통제하지 않아 trivial한 결론 |
| "JSON 구조화 파이프라인이 완성됐다" | ⭐⭐⭐⭐ | 기본 구조 잘 됨. 고급 후처리는 Future Work |

---

## 6. 종합 의견

### 🏆 이 프로젝트의 강점
1. **스토리라인이 매우 강력함** — "도메인 갭"이라는 명확한 문제 정의 → 단계별 극복이라는 서사가 대학 프로젝트 수준을 넘어섬
2. **데이터 전처리(Hash Matching)가 독창적** — 이 부분은 진짜 엔지니어링 역량이 드러남
3. **Simulated Legacy Test Set 아이디어** — 평가 방법론 자체가 창의적
4. **End-to-End 완결성** — 학습부터 JSON Export까지 전체 파이프라인이 실제로 돌아감

### 🚨 반드시 수정해야 할 3가지
1. **Phase 2-B**: Baseline 모델도 Legacy Test Set에서 `val()` 실행하는 코드 추가
2. **Phase 3**: 시각화 제목에서 "8-Class" 제거. 또는 진짜 8-class 실험을 추가
3. **문서-코드 수치 통일**: 에폭/데이터 수 등 6건 불일치 해소

### 💡 가성비 높은 개선 제안 (적은 노력, 큰 효과)
1. `random.seed(42)` 추가 → 재현성 확보
2. Phase 1의 `train_images[:size]` → 셔플 후 슬라이싱
3. Phase 3에서 **"수렴 에폭 수"** 비교를 핵심 지표로 전환 (mAP 0.9 도달까지 걸린 에폭 비교)
4. Augmented 모델의 hsv 파라미터가 기본값보다 약한 점을 인지하고, 의도적이라면 문서에 설명 추가
