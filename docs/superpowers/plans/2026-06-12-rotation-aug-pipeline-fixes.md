# Rotation Augmentation & Pipeline Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 90°/180°/270° offline rotation augmentation to master_dataset train set, fix 6 critical pipeline bugs, and apply 2 high-ROI improvements — all within `floorplan_training.py`.

**Architecture:** Single-file edit. New augmentation section inserted between Phase 1 and Phase 2. Bug fixes are surgical edits across existing phases. No new dependencies.

**Tech Stack:** Python, OpenCV, NumPy, PyTorch, Ultralytics YOLO, matplotlib

---

## File Map

- Modify: `floorplan-yolo/notebooks/floorplan_training.py`
  - Lines ~33-47: Add random seed setup
  - Lines ~269-271: Fix EDA dead code
  - Lines ~322-326: Fix Phase 1 sampling bias
  - Lines ~410-470: Insert new rotation augmentation section + modify Phase 2 Augmented data path
  - Lines ~465-466: Improve hsv comments
  - Lines ~528-532: Add Baseline Legacy evaluation
  - Lines ~935: Fix Phase 3 viz title
  - Lines ~970-987: Remove duplicate CLASS_COLOR_MAP
  - Lines ~700 area: Add convergence speed metric

---

### Task 1: Add Random Seed + Fix EDA Dead Code (B3, B5)

**Files:**
- Modify: `floorplan-yolo/notebooks/floorplan_training.py:33-47` (seed)
- Modify: `floorplan-yolo/notebooks/floorplan_training.py:269-271` (dead code)

- [ ] **Step 1: Add random seed block after imports (L47)**

After line 47 (`import cv2`), insert:

```python
# ──────────────────────────────────────────────
# 🎯 재현성 확보 (Reproducibility)
# ──────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
```

- [ ] **Step 2: Fix EDA dead code (L269-271)**

Replace lines 269-271:
```python
        _fixed = train_img_dir / "master_train_0204.webp"
        sample_img = _fixed if _fixed.exists() else img_files[0]
        sample_img = random.choice(img_files)
```

With:
```python
        sample_img = random.choice(img_files)
```

- [ ] **Step 3: Commit**

```bash
git add floorplan-yolo/notebooks/floorplan_training.py
git commit -m "fix: add random seed for reproducibility, remove EDA dead code"
```

---

### Task 2: Fix Phase 1 Sampling Bias (B4)

**Files:**
- Modify: `floorplan-yolo/notebooks/floorplan_training.py:317-326`

- [ ] **Step 1: Add shuffle before slicing**

At line 318, after `train_images = list(...)`, insert:

```python
random.shuffle(train_images)  # Prevent systematic bias from filesystem ordering
```

The existing `train_images[:size]` on L326 now slices from a shuffled list. Seed already set in Task 1.

- [ ] **Step 2: Commit**

```bash
git add floorplan-yolo/notebooks/floorplan_training.py
git commit -m "fix: shuffle train images before Phase 1 size ablation"
```

---

### Task 3: Insert Rotation Augmentation Section (New Section 4.5)

**Files:**
- Modify: `floorplan-yolo/notebooks/floorplan_training.py` — insert between Phase 1 (ends ~L408) and Phase 2 (starts ~L411)

- [ ] **Step 1: Insert new markdown + code cell after Phase 1 plot**

After the Phase 1 plot block (after L408 `if scaling_results:` block), insert:

```python
# %% [markdown]
# ## 4.5 [Augmentation] 90°/180°/270° Offline Rotation Augmentation
# - 도면은 스캐너 방향에 따라 0°/90°/180°/270° 네 방향이 모두 자연스럽습니다.
# - 원본 train 1600장 × 4방향 = 6400장으로 학습 데이터를 확장합니다.
# - YOLO 라벨(cx, cy, w, h)도 수학적으로 동기 변환합니다.

# %%
print("=" * 60)
print("🔄 [Augmentation] 90°/180°/270° Offline Rotation")
print("=" * 60)

AUG_IMG_DIR = MASTER_DATASET_DIR / "images" / "train_aug"
AUG_LBL_DIR = MASTER_DATASET_DIR / "labels" / "train_aug"


def rotate_yolo_label(cx, cy, w, h, angle):
    """Rotate YOLO normalized coords by 90/180/270 degrees (counterclockwise)."""
    if angle == 90:
        return cy, 1.0 - cx, h, w
    elif angle == 180:
        return 1.0 - cx, 1.0 - cy, w, h
    elif angle == 270:
        return 1.0 - cy, cx, h, w
    else:
        return cx, cy, w, h


def augment_with_rotations(src_img_dir, src_lbl_dir, dst_img_dir, dst_lbl_dir):
    """Copy originals + generate 90/180/270° rotated images and labels."""
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    img_files = sorted(src_img_dir.glob("*.webp"))
    angles = [90, 180, 270]
    # OpenCV rotation codes
    rot_codes = {
        90: cv2.ROTATE_90_COUNTERCLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_CLOCKWISE,
    }

    total = 0
    for img_path in img_files:
        stem = img_path.stem
        lbl_path = src_lbl_dir / (stem + ".txt")

        # 1) Copy original
        shutil.copy2(str(img_path), str(dst_img_dir / img_path.name))
        if lbl_path.exists():
            shutil.copy2(str(lbl_path), str(dst_lbl_dir / lbl_path.name))
        total += 1

        # 2) Generate rotated versions
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Read label lines once
        label_lines = []
        if lbl_path.exists():
            with open(lbl_path, "r") as f:
                label_lines = f.readlines()

        for angle in angles:
            # Rotate image
            rotated = cv2.rotate(img, rot_codes[angle])
            rot_name = f"{stem}_rot{angle}.webp"
            cv2.imwrite(str(dst_img_dir / rot_name), rotated)

            # Rotate labels
            rot_lbl_name = f"{stem}_rot{angle}.txt"
            with open(dst_lbl_dir / rot_lbl_name, "w") as f:
                for line in label_lines:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = parts[0]
                    cx, cy, w, h = map(float, parts[1:5])
                    ncx, ncy, nw, nh = rotate_yolo_label(cx, cy, w, h, angle)
                    f.write(f"{cls_id} {ncx:.6f} {ncy:.6f} {nw:.6f} {nh:.6f}\n")
            total += 1

    return total


# Skip if already generated
existing_count = len(list(AUG_IMG_DIR.glob("*.webp"))) if AUG_IMG_DIR.exists() else 0
EXPECTED_AUG_COUNT = 6400  # 1600 originals × 4 orientations

if existing_count >= EXPECTED_AUG_COUNT:
    print(f"✅ Augmented train set already exists ({existing_count} images). Skipping.")
else:
    print("⏳ Generating rotated augmentation data...")
    count = augment_with_rotations(
        MASTER_DATASET_DIR / "images" / "train",
        MASTER_DATASET_DIR / "labels" / "train",
        AUG_IMG_DIR,
        AUG_LBL_DIR,
    )
    print(f"✅ Rotation augmentation complete: {count} images in train_aug/")

# Generate augmented dataset.yaml
aug_yaml_path = MASTER_DATASET_DIR / "dataset_augmented.yaml"
with open(aug_yaml_path, "w", encoding="utf-8") as f:
    yaml.dump(
        {
            "path": str(MASTER_DATASET_DIR.resolve()),
            "train": "images/train_aug",
            "val": "images/val",
            "test": "images/test",
            "names": class_names,
        },
        f,
        allow_unicode=True,
        default_flow_style=False,
    )
print(f"📄 dataset_augmented.yaml saved at {aug_yaml_path}")
```

- [ ] **Step 2: Update Phase 2 Augmented to use augmented dataset**

In Phase 2 Augmented training (~L452), change:
```python
        data=str(MASTER_DATASET_DIR / "dataset.yaml"),
```
to:
```python
        data=str(MASTER_DATASET_DIR / "dataset_augmented.yaml"),
```

- [ ] **Step 3: Commit**

```bash
git add floorplan-yolo/notebooks/floorplan_training.py
git commit -m "feat: add 90/180/270 rotation augmentation, use augmented dataset for Phase 2"
```

---

### Task 4: Fix Phase 2-B Baseline Legacy Evaluation (B1)

**Files:**
- Modify: `floorplan-yolo/notebooks/floorplan_training.py:528-532`

- [ ] **Step 1: Add Baseline Legacy eval before Augmented eval**

Before `print("\n📊 Augmented 모델 → Legacy Test Set 평가")` (L528), insert:

```python
print("\n📊 Baseline 모델 → Legacy Test Set 평가")
baseline_weight = PROJECT_ROOT / "runs/detect/train_baseline/weights/best.pt"
legacy_baseline_metrics = None
if baseline_weight.exists():
    model_base_eval = YOLO(str(baseline_weight))
    legacy_baseline_metrics = model_base_eval.val(data=str(legacy_yaml_path))
else:
    print("⚠️ Baseline 가중치 없음 — Legacy 비교 스킵")
```

- [ ] **Step 2: Add comparison bar chart after both evals**

After the Augmented eval block, insert:

```python
# ── Baseline vs Augmented on Legacy Test Set comparison ──
if legacy_baseline_metrics is not None and aug_weight.exists():
    base_map50 = legacy_baseline_metrics.results_dict.get("metrics/mAP50(B)", 0)
    aug_map50 = legacy_aug_metrics.results_dict.get("metrics/mAP50(B)", 0)

    fig, ax = plt.subplots(figsize=(8, 5))
    models = ["Baseline\n(Default Aug)", "Augmented\n(Domain Aug + Rotation)"]
    scores = [base_map50, aug_map50]
    colors = ["#FF6B6B", "#4ECDC4"]
    bars = ax.bar(models, scores, color=colors, width=0.5, edgecolor="white", linewidth=2)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{score:.3f}", ha="center", va="bottom", fontsize=14, fontweight="bold")
    ax.set_ylabel("mAP@50", fontsize=13)
    ax.set_title("Legacy Test Set: Baseline vs Augmented", fontsize=15, fontweight="bold")
    ax.set_ylim(0, max(scores) * 1.2 if max(scores) > 0 else 1.0)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    legacy_comp_path = PROJECT_ROOT / "runs" / "legacy_baseline_vs_augmented.png"
    plt.savefig(legacy_comp_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"✅ Legacy 비교 저장: {legacy_comp_path}")
```

- [ ] **Step 3: Commit**

```bash
git add floorplan-yolo/notebooks/floorplan_training.py
git commit -m "fix: add baseline model Legacy Test evaluation for fair comparison"
```

---

### Task 5: Fix Phase 3 Viz Title + Add Convergence Metric (B2, I1)

**Files:**
- Modify: `floorplan-yolo/notebooks/floorplan_training.py:935` (title)
- Modify: `floorplan-yolo/notebooks/floorplan_training.py` ~L790 area (convergence metric)

- [ ] **Step 1: Fix viz title (L935)**

Replace:
```python
            "Scratch / 8-Class baseline Model\n(Misses, False Positives, Feature Collision)",
```
With:
```python
            "Scratch (Random Init) Model\n(Slower Convergence, Lower Accuracy)",
```

- [ ] **Step 2: Add convergence speed comparison after mAP plot**

After the metrics comparison plot saving (~L792), insert:

```python
        # ── Convergence Speed Comparison ──
        MAP_THRESHOLD = 0.9
        def find_convergence_epoch(df, col, threshold):
            """Find first epoch where metric >= threshold."""
            above = df[df[col] >= threshold]
            return int(above["epoch"].iloc[0]) if len(above) > 0 else None

        if col_map in df_scratch.columns and col_map in df_transfer.columns:
            scratch_conv = find_convergence_epoch(df_scratch, col_map, MAP_THRESHOLD)
            transfer_conv = find_convergence_epoch(df_transfer, col_map, MAP_THRESHOLD)
            print(f"\n⚡ Convergence Speed (mAP≥{MAP_THRESHOLD}):")
            if transfer_conv is not None:
                print(f"   Transfer: {transfer_conv} epochs")
            else:
                print(f"   Transfer: did not reach {MAP_THRESHOLD}")
            if scratch_conv is not None:
                print(f"   Scratch:  {scratch_conv} epochs")
            else:
                print(f"   Scratch:  did not reach {MAP_THRESHOLD}")
            if transfer_conv and scratch_conv:
                speedup = scratch_conv / transfer_conv
                print(f"   → Transfer is {speedup:.1f}x faster convergence")
```

- [ ] **Step 3: Commit**

```bash
git add floorplan-yolo/notebooks/floorplan_training.py
git commit -m "fix: correct Phase 3 viz title, add convergence speed metric"
```

---

### Task 6: Remove Duplicate CLASS_COLOR_MAP + Improve HSV Comments (B6, I2)

**Files:**
- Modify: `floorplan-yolo/notebooks/floorplan_training.py:965-987` (remove dup)
- Modify: `floorplan-yolo/notebooks/floorplan_training.py:465-466` (hsv comments)

- [ ] **Step 1: Remove duplicate CLASS_COLOR_MAP at L975-987**

Delete lines 975-987 (the entire second `CLASS_COLOR_MAP` definition in Phase 4):
```python
# ──────────────────────────────────────────────
# 🎨 전역 색상 팔레트 (8클래스: 7 마스터 + 1 OCR)
# ──────────────────────────────────────────────
CLASS_COLOR_MAP = {
    "toilet": (255, 80, 80),  # 빨강
    "washbasin": (80, 180, 255),  # 하늘
    "sink": (80, 220, 120),  # 초록
    "bathtub": (255, 180, 50),  # 주황
    "gas_stove": (180, 80, 255),  # 보라
    "door": (255, 220, 50),  # 노랑
    "window": (50, 220, 220),  # 시안(청록)
    "text": (255, 120, 200),  # 핑크
}
```

- [ ] **Step 2: Improve HSV parameter comments (L465-466)**

Replace:
```python
        hsv_s=0.2,  # 채도 변형 비율 (20%): 오래된 도면의 색 빠짐 모사
        hsv_v=0.2,  # 명도 변형 비율 (20%): 도면의 황변 현상 및 퇴색 모사
```
With:
```python
        hsv_s=0.2,  # 채도 변형 (20%) — YOLO 기본 0.7 대비 의도적 약화: 도면은 흑백/저채도 기반이므로 과도한 색상 변형은 비현실적
        hsv_v=0.2,  # 명도 변형 (20%) — YOLO 기본 0.4 대비 의도적 약화: 도면 특성상 명도 분포가 좁아 과도한 변형은 노이즈 유발
```

- [ ] **Step 3: Commit**

```bash
git add floorplan-yolo/notebooks/floorplan_training.py
git commit -m "fix: remove duplicate CLASS_COLOR_MAP, document hsv parameter rationale"
```

---

## Verification Checklist

After all tasks complete:

- [ ] `train_aug/` dir exists with 6400 .webp files
- [ ] `dataset_augmented.yaml` points to `images/train_aug`
- [ ] Phase 2 Augmented references `dataset_augmented.yaml`
- [ ] Phase 2-B has both Baseline + Augmented Legacy eval
- [ ] Phase 3 viz title says "Scratch (Random Init)" not "8-Class"
- [ ] No duplicate `CLASS_COLOR_MAP`
- [ ] `random.seed(42)` at script top
- [ ] Phase 1 uses shuffled train_images
- [ ] EDA has no dead `_fixed` code
- [ ] HSV comments explain intentional reduction
- [ ] Convergence speed metric prints in Phase 3
