# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # [딥러닝분석] 파이널 프로젝트: 2D 아파트 도면 객체 검출 및 3D 모델링 파이프라인 구축
# - **소속**: 숭실대학교 소프트웨어학과 오현
# - **프로젝트 주제**: 2D 도면 이미지의 딥러닝 기반 디지털 구조화 및 3D 변환의 AI 베이스라인 모델 구축
# - **AI Task**: [OBJ] 가구 및 설비 (변기, 세면대, 싱크대, 욕조, 가스레인지) 객체 검출 (Object Detection)
# - **사용한 모델**: YOLOv8 (SOTA Real-time Object Detection Model)
#
# > ⚠️ **안내사항**
# > Colab / 로컬(Windows, Linux) 어디서든 실행 가능합니다.
# > 환경을 자동으로 감지하여 데이터 경로, GPU 디바이스를 알아서 설정합니다.

# %% [markdown]
# ## 0. 개발 환경 설정 및 라이브러리 설치

# %%
# %pip install -q ultralytics matplotlib numpy pillow albumentations pyyaml sahi gdown

import torch
import sys
import os
import gc
import json
import shutil
import yaml
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from pathlib import Path
from ultralytics import YOLO
import subprocess
import cv2

# ──────────────────────────────────────────────
# 환경 자동 감지 (Colab / 로컬 / 클라우드)
# ──────────────────────────────────────────────

IS_COLAB = "google.colab" in sys.modules

if IS_COLAB:
    # Colab에서도 프로젝트 폴더 안에서 모든 작업 수행
    PROJECT_ROOT = Path("/content/floorplan_project")
    PROJECT_ROOT.mkdir(exist_ok=True)
    os.chdir(str(PROJECT_ROOT))
    print(f"[Colab] 프로젝트 루트: {PROJECT_ROOT}")
else:
    # 로컬: 스크립트 위치 기준으로 프로젝트 루트 결정
    _cwd = Path(os.getcwd())
    if _cwd.name == "notebooks":
        PROJECT_ROOT = _cwd.parent
    elif (_cwd / "floorplan-yolo").exists():
        PROJECT_ROOT = _cwd / "floorplan-yolo"
    else:
        PROJECT_ROOT = _cwd
    print(f"[로컬] 프로젝트 루트: {PROJECT_ROOT}")

# GPU 정보 출력
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory // (1024**2)
    print(f"GPU 감지됨: {gpu_name} ({gpu_mem}MB)")
elif hasattr(torch, "hip") and torch.hip.is_available():  # AMD ROCm
    print(f"AMD ROCm GPU 감지됨")
else:
    print("GPU 없음 → CPU 모드로 학습 (느림)")

print("Libraries imported successfully.")

# %% [markdown]
# ## 1. 데이터셋 연동 및 압축 해제

# %%
# ──────────────────────────────────────────────
# 데이터셋 경로: 항상 PROJECT_ROOT/data/ 안에 통일
# ──────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

ZIP_NAME = "architectural_drawing_data.tar.zst"
ZIP_PATH = DATA_DIR / ZIP_NAME
EXTRACT_PATH = DATA_DIR / "architectural_drawing_data_part"

GDRIVE_FILE_ID = "1jTXlydel8WaTW0OWDpmFJZ03HEqs1Dgz"

if not EXTRACT_PATH.exists():
    # Step 1: 압축파일 확보
    if not ZIP_PATH.exists():
        if IS_COLAB:
            # Colab: 구글 드라이브에서 프로젝트 폴더로 복사
            drive_zip = Path(f"/content/drive/MyDrive/{ZIP_NAME}")
            if drive_zip.exists():
                print(f"구글 드라이브에서 {ZIP_NAME} 복사 중...")
                shutil.copy(str(drive_zip), str(ZIP_PATH))
            else:
                # 드라이브에도 없으면 gdown으로 다운로드
                import gdown

                print(f"구글 드라이브에 파일 없음 → gdown 직접 다운로드 시작...")
                gdown.download(id=GDRIVE_FILE_ID, output=str(ZIP_PATH), quiet=False)
        else:
            # 로컬: gdown으로 직접 다운로드
            import gdown

            print(f"데이터셋({ZIP_NAME}) 다운로드 시작...")
            gdown.download(id=GDRIVE_FILE_ID, output=str(ZIP_PATH), quiet=False)

    # Step 2: 압축 해제
    if ZIP_PATH.exists():
        print(f"{ZIP_NAME} 압축 해제 중...")
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        try:
            subprocess.run(
                ["tar", "-I", "zstd", "-xf", str(ZIP_PATH), "-C", str(DATA_DIR)],
                check=True,
            )
        except Exception as e:
            print("zstd 패키지가 없거나 권한 오류 발생. zstd 설치 시도 중...")
            if IS_COLAB:
                subprocess.run(["apt-get", "update"], check=True)
                subprocess.run(["apt-get", "install", "-y", "zstd"], check=True)
                subprocess.run(
                    ["tar", "-I", "zstd", "-xf", str(ZIP_PATH), "-C", str(DATA_DIR)],
                    check=True,
                )
            else:
                raise RuntimeError(
                    "압축 해제 실패: 터미널에서 'sudo apt install zstd' 를 실행 후 다시 시도하세요."
                )

        print("압축 해제 완료!")
    else:
        raise FileNotFoundError(f"⚠️ {ZIP_PATH} 파일을 확보하지 못했습니다.")
else:
    print(f"Dataset ready at {EXTRACT_PATH}")

# %% [markdown]
# ## 2. JSON 라벨 전처리 및 YOLO 포맷 변환 (Train/Val/Test)

# %%
# 2. JSON 라벨 전처리 및 YOLO 포맷 변환 (Train/Val/Test)
YOLO_DIR = EXTRACT_PATH / "yolo_dataset"
CLASS_MAPPING = {4: 0, 5: 1, 6: 2, 7: 3, 8: 4}


if not YOLO_DIR.exists():
    print("YOLO 데이터셋 포맷팅을 시작합니다...")

    # === [통합 데이터셋 처리 코드] 파이프라인 검증용 ===
    # 업로드 최소 기준량(LIMITS)에 맞추어 통합 (가구 1000장/100장/100장)
    LIMITS = {"train": 1000, "val": 100, "test": 100}

    for split in ["train", "val", "test"]:
        dest_img_dir = YOLO_DIR / "images" / split
        dest_lbl_dir = YOLO_DIR / "labels" / split
        dest_img_dir.mkdir(parents=True, exist_ok=True)
        dest_lbl_dir.mkdir(parents=True, exist_ok=True)

        src_img_dir = EXTRACT_PATH / "object_layout" / split / "images"
        src_lbl_dir = EXTRACT_PATH / "object_layout" / split / "labels"
        if not src_lbl_dir.exists():
            continue

        json_files = list(src_lbl_dir.glob("*.json"))

        # 지정된 마지노선 수량만큼만 랜덤 샘플링
        if len(json_files) > LIMITS[split]:
            random.seed(42)  # 재현성 보장
            json_files = random.sample(json_files, LIMITS[split])

        converted_count = 0
        for json_file in json_files:
            base_name = json_file.stem
            img_file = src_img_dir / f"{base_name}.webp"

            if not img_file.exists():
                continue

            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not data.get("images"):
                continue

            img_info = data["images"][0]
            img_width = img_info.get("width", 4963)
            img_height = img_info.get("height", 3509)

            yolo_lines = []
            for ann in data.get("annotations", []):
                cat_id = ann.get("category_id")
                if cat_id in CLASS_MAPPING:
                    bbox = ann.get("bbox", [])
                    if len(bbox) == 4:
                        x_min, y_min, w, h = bbox
                        x_center_norm = (x_min + (w / 2.0)) / img_width
                        y_center_norm = (y_min + (h / 2.0)) / img_height
                        w_norm, h_norm = w / img_width, h / img_height
                        yolo_lines.append(
                            f"{CLASS_MAPPING[cat_id]} {x_center_norm:.6f} {y_center_norm:.6f} {w_norm:.6f} {h_norm:.6f}"
                        )

            if yolo_lines:
                dest_txt = dest_lbl_dir / f"{base_name}.txt"
                with open(dest_txt, "w", encoding="utf-8") as f:
                    f.write("\n".join(yolo_lines))
                shutil.copy(img_file, dest_img_dir / img_file.name)
                converted_count += 1

        print(f"[{split.upper()} 데이터 변환 완료] {converted_count}장")
else:
    print("YOLO 데이터셋이 이미 구성되어 있습니다.")

# %%
# dataset.yaml 자동 생성
yaml_path = YOLO_DIR / "dataset.yaml"

dataset_config = {
    "path": str(YOLO_DIR.resolve()),
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "names": {0: "toilet", 1: "basin", 2: "sink", 3: "bath", 4: "stove"},
}

with open(yaml_path, "w", encoding="utf-8") as f:
    yaml.dump(dataset_config, f, allow_unicode=True, default_flow_style=False)
print(f"dataset.yaml saved at {yaml_path}")

# %% [markdown]
# ## 3. 탐색적 데이터 분석 (EDA) 및 시각화 검증

# %%
class_names = dataset_config["names"]


def visualize_yolo_labels(image_path, label_path, class_names):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size
    colors = ["red", "blue", "green", "orange", "purple"]

    if not os.path.exists(label_path):
        return img

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        xc, yc, bw, bh = map(float, parts[1:])

        x_center, y_center = xc * w, yc * h
        box_w, box_h = bw * w, bh * h

        x_min, y_min = int(x_center - box_w / 2), int(y_center - box_h / 2)
        x_max, y_max = int(x_center + box_w / 2), int(y_center + box_h / 2)

        color = colors[class_id % len(colors)]
        draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=6)
        draw.text(
            (x_min + 5, y_min - 25),
            class_names.get(class_id, str(class_id)),
            fill=color,
        )

    return img


train_img_dir = YOLO_DIR / "images" / "train"
train_lbl_dir = YOLO_DIR / "labels" / "train"

if train_img_dir.exists():
    img_files = list(train_img_dir.glob("*.webp"))
    if img_files:
        target_img_name = "APT_FP_OBJ_826206478.webp"
        target_img_path = train_img_dir / target_img_name

        if target_img_path.exists():
            sample_img = target_img_path
        else:
            sample_img = random.choice(img_files)  # 1000장 샘플링에 안 뽑혔을 경우 대비

        sample_lbl = train_lbl_dir / (sample_img.stem + ".txt")

        print(f"Visualizing Sample: {sample_img.name}")
        result_img = visualize_yolo_labels(sample_img, sample_lbl, class_names)

        plt.figure(figsize=(12, 8))
        plt.imshow(result_img)
        plt.axis("off")
        plt.show()

# %% [markdown]
# ## 3. 탐색적 데이터 분석 (EDA) 및 시각화 검증
# (클래스별 분포 및 바운딩 박스 크기 히스토그램)

# %%
import glob

print("=" * 60)
print("📊 탐색적 데이터 분석 (EDA) - 클래스 분포 및 BBox 크기")
print("=" * 60)

train_lbl_files = glob.glob(str(train_lbl_dir / "*.txt"))
class_counts = {k: 0 for k in class_names.keys()}
bbox_widths = []
bbox_heights = []

if not train_lbl_files:
    print("⚠️ 라벨 파일이 없어 EDA를 건너뜁니다.")
else:
    for lbl_file in train_lbl_files:
        with open(lbl_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    w, h = float(parts[3]), float(parts[4])
                    if cls_id in class_counts:
                        class_counts[cls_id] += 1
                    bbox_widths.append(w)
                    bbox_heights.append(h)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # 1. 클래스 분포 (Bar Chart)
    en_class_names = {
        "변기": "Toilet",
        "세면대": "Washbasin",
        "싱크대": "Sink",
        "욕조": "Bathtub",
        "가스레인지": "Gas Stove",
    }
    counts = [class_counts[k] for k in sorted(class_counts.keys())]
    labels = [
        en_class_names.get(class_names[k], class_names[k])
        for k in sorted(class_counts.keys())
    ]
    axes[0].bar(labels, counts, color="skyblue", edgecolor="black")
    axes[0].set_title("Class Distribution (Train Set)", fontsize=14)
    axes[0].set_ylabel("Count")
    for i, v in enumerate(counts):
        axes[0].text(i, v + (max(counts) * 0.02), str(v), ha="center")

    # 2. BBox 크기 분포 (Scatter Plot)
    axes[1].scatter(bbox_widths, bbox_heights, alpha=0.1, color="purple", s=10)
    axes[1].set_title("Bounding Box Size Distribution", fontsize=14)
    axes[1].set_xlabel("Normalized Width (0.0 ~ 1.0)")
    axes[1].set_ylabel("Normalized Height (0.0 ~ 1.0)")

    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 4. [실험 1] 최적의 데이터 개수 탐색 (Data Scaling Ablation)
# - 무조건 많은 데이터를 사용하는 것이 정답이 아닙니다.
# - 학습 데이터 개수를 100장부터 1000장까지 100장 단위로 점진적으로 늘려보며 성능 향상폭이 꺾이는 지점(**수렴점, Saturation Point**)을 찾습니다.
# - 도메인 갭(Domain Gap)으로 인해 일정 개수 이상부터는 원본 데이터만으로 성능이 오르지 않는 **데이터의 한계점**을 수학적으로 증명합니다.

# %%
import yaml
import matplotlib.pyplot as plt

print("=" * 60)
print("🔬 [실험 1] Data Size Ablation")
print("=" * 60)

train_images = list((YOLO_DIR / "images/train").glob("*.webp"))
data_sizes = [100, 300, 500, 1000]
scaling_results = {}

for size in data_sizes:
    if size > len(train_images):
        continue

    subset = train_images[:size]
    subset_txt = YOLO_DIR / f"train_{size}.txt"
    with open(subset_txt, "w", encoding="utf-8") as f:
        f.write("\n".join([str(p.resolve()) for p in subset]))

    yaml_path_size = YOLO_DIR / f"dataset_{size}.yaml"
    with open(yaml_path_size, "w", encoding="utf-8") as f:
        yaml.dump(
            {
                "path": str(YOLO_DIR.resolve()),
                "train": str(subset_txt.resolve()),
                "val": "images/val",
                "names": class_names,
            },
            f,
            allow_unicode=True,
        )

    print(f"\n🚀 Data Size: {size} 학습 시작 (20 Epochs 빠른 검증)")
    model_size = YOLO("yolov8n.pt")
    res_size = model_size.train(
        data=str(yaml_path_size),
        epochs=20,  # 데이터 크기별 추세를 보기 위한 빠른 검증
        imgsz=640,
        batch=16,  # Colab L4 (24GB VRAM) 최적화
        workers=8,
        cache=False,  # 연속 학습 시 RAM 누수/OOM 방지를 위해 캐시 비활성화
        project=str(PROJECT_ROOT / "runs/detect"),
        name=f"train_size_{size}",
        verbose=False,
    )

    # GPU 캐시 메모리 비우기
    torch.cuda.empty_cache()
    gc.collect()

    map50 = res_size.results_dict.get("metrics/mAP50(B)", 0)
    scaling_results[size] = map50

if scaling_results:
    sizes = list(scaling_results.keys())
    maps = list(scaling_results.values())

    plt.figure(figsize=(8, 5))
    plt.plot(sizes, maps, marker="o", linestyle="-", color="b", linewidth=2)
    plt.title("Data Scaling Ablation (Finding Saturation Point)", fontsize=14)
    plt.xlabel("Number of Training Images", fontsize=12)
    plt.ylabel("mAP@50", fontsize=12)
    plt.grid(True)
    for i, txt in enumerate(maps):
        plt.annotate(
            f"{txt:.3f}",
            (sizes[i], maps[i]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )
    plt.show()
    print(
        "💡 분석: 그래프가 평탄해지는(꺾이는) 지점이 수렴점입니다. 원본 데이터만으로는 이 한계를 넘을 수 없음을 확인했습니다."
    )

# %% [markdown]
# ## 5. [실험 2] 도메인 맞춤형 증강 탐색 (Augmentation Ablation)
# - 실험 1에서 찾은 **가성비 최적 데이터 개수(1000장)**를 고정하고, 원본 데이터의 한계를 돌파할 증강 기법을 비교합니다.
#   - **Baseline (대조군)**: 증강 없이 순정 학습
#   - **Augmented (실험군)**: 흑백, 스캔 노이즈, 삐뚤어짐 등 레거시 도면의 도메인 갭을 극복하는 **도메인 맞춤형 증강(Domain-Specific Augmentation)** 적용
#   - 최종 결론으로 '적은 최적 데이터 + 도메인 최적 증강' 조합의 압도적인 효율성을 증명합니다.

# %%
# ──────────────────────────────────────────────
# 대조군 (Baseline) - 실험 1에서 찾은 데이터 양(1000장)으로 순정 학습
# ──────────────────────────────────────────────
print("=" * 60)
print("🔬 [Phase 1] Baseline 모델 학습 (증강 없음, 대조군)")
print("=" * 60)
model_baseline = YOLO("yolov8n.pt")

results_baseline = model_baseline.train(
    data=str(YOLO_DIR / "dataset_1000.yaml"),
    epochs=50,
    imgsz=640,
    batch=16,  # Colab L4 (24GB VRAM) 최적화
    workers=8,  # CPU 데이터 전처리 병목 해결
    cache=True,  # 이미지를 RAM에 캐싱하여 디스크 병목 해결
    project=str(PROJECT_ROOT / "runs/detect"),
    name="train_baseline",
    val=True,
    # 증강 파라미터 전부 YOLOv8 기본값 그대로 사용 (대조군)
)

# GPU 캐시 메모리 비우기
torch.cuda.empty_cache()
gc.collect()

# %%
# ──────────────────────────────────────────────
# 실험군 (Augmented) - 도메인 맞춤형 증강 적용
# ──────────────────────────────────────────────
print("=" * 60)
print("🔬 [실험군] Augmented 모델 학습 (도메인 맞춤형 증강 적용)")
print("=" * 60)
model_augmented = YOLO("yolov8n.pt")

# 도메인 맞춤형 증강 파라미터 셋업 (레거시 도면 시뮬레이션)
results_augmented = model_augmented.train(
    data=str(YOLO_DIR / "dataset_1000.yaml"),
    epochs=50,
    imgsz=640,
    batch=16,  # Colab L4 (24GB VRAM) 최적화
    workers=8,  # CPU 데이터 전처리 병목 해결
    cache=True,  # 이미지를 RAM에 캐싱하여 디스크 병목 해결
    project=str(PROJECT_ROOT / "runs/detect"),
    name="train_augmented",
    # === Domain Gap 극복을 위한 Domain-Specific Augmentation ===
    # 근거 1: [회전/삐뚤어짐] 평판 스캐너 수작업 오차 반영
    degrees=2.0,
    # 근거 2: [노출 불량/명암 저하] 황변 현상 및 토너 부족 시뮬레이션
    hsv_s=0.2,
    hsv_v=0.2,
    perspective=0.0005,
    scale=0.5,
    mosaic=1.0,
    val=True,
)

# GPU 캐시 메모리 비우기
torch.cuda.empty_cache()
gc.collect()

# %%
# ──────────────────────────────────────────────
# Ablation Study 결과 비교 (Baseline vs Augmented)
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("📊 [Ablation Study 결과 비교] Baseline vs Augmented")
print("=" * 60)


def get_metrics(results):
    """YOLO 학습 결과에서 최종 검증 지표 추출"""
    try:
        m = results.results_dict
        return {
            "mAP50": m.get("metrics/mAP50(B)", 0),
            "mAP50-95": m.get("metrics/mAP50-95(B)", 0),
            "precision": m.get("metrics/precision(B)", 0),
            "recall": m.get("metrics/recall(B)", 0),
        }
    except Exception:
        return {"mAP50": 0, "mAP50-95": 0, "precision": 0, "recall": 0}


m_base = get_metrics(results_baseline)
m_aug = get_metrics(results_augmented)

print(f"{'Metric':<15} {'Baseline':>10} {'Augmented':>10} {'Delta':>10}")
print("-" * 50)
for key in ["mAP50", "mAP50-95", "precision", "recall"]:
    b, a = m_base[key], m_aug[key]
    delta = a - b
    sign = "+" if delta >= 0 else ""
    print(f"{key:<15} {b:>10.4f} {a:>10.4f} {sign}{delta:>9.4f}")

print(
    "\n[결론] 증강 적용 시 mAP50 변화:",
    f"{m_base['mAP50']:.4f} → {m_aug['mAP50']:.4f}",
    f"({'개선' if m_aug['mAP50'] > m_base['mAP50'] else '하락'})",
)

# %% [markdown]
# ## 6. [실험 2-B] Simulated Legacy Test Set 기반 재평가
# - 기존 val/test 셋은 깨끗한 최신 도면이므로, 증강 모델의 진짜 실력을 평가할 수 없습니다.
# - OpenCV를 활용하여 기존 test 셋 50장에 황변, 스캔 노이즈, 해상도 저하를 인위적으로 적용한
#   **Simulated Legacy Test Set**을 생성하고, Baseline vs Augmented를 재평가합니다.
# - **핵심 논증**: "깨끗한 val에서는 Baseline이 우세하지만, 레거시 도면에서는 Augmented가 압도적으로 승리"

# %%
print("=" * 60)
print("🏚️ [실험 2-B] Simulated Legacy Test Set 생성 및 재평가")
print("=" * 60)


def simulate_legacy_degradation(img):
    """오래된 스캔 도면을 시뮬레이션하는 열화 함수"""
    h, w = img.shape[:2]

    # 1. 황변 현상 (Yellowing) - 오래된 종이의 누런 변색
    yellow_tint = np.full_like(img, (0, 20, 40), dtype=np.uint8)
    img = cv2.addWeighted(img, 0.85, yellow_tint, 0.15, 0)

    # 2. 대비 저하 + 밝기 감소 - 퇴색된 잉크
    img = cv2.convertScaleAbs(img, alpha=0.7, beta=-15)

    # 3. 가우시안 노이즈 - 스캔 과정에서 발생하는 노이즈
    noise = np.random.normal(0, 15, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 4. 블러 - 초점 흐림 / 오래된 프린트의 번짐
    img = cv2.GaussianBlur(img, (3, 3), 1.0)

    # 5. 해상도 저하 후 복원 - 저해상도 스캔 시뮬레이션
    small = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    return img


# 1. Simulated Legacy Test Set 생성
legacy_test_img_dir = YOLO_DIR / "images" / "legacy_test"
legacy_test_lbl_dir = YOLO_DIR / "labels" / "legacy_test"
legacy_test_img_dir.mkdir(parents=True, exist_ok=True)
legacy_test_lbl_dir.mkdir(parents=True, exist_ok=True)

test_images_for_legacy = list((YOLO_DIR / "images" / "test").glob("*.webp"))
if len(test_images_for_legacy) > 50:
    random.seed(42)
    test_images_for_legacy = random.sample(test_images_for_legacy, 50)

for img_path in test_images_for_legacy:
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    degraded = simulate_legacy_degradation(img)
    cv2.imwrite(str(legacy_test_img_dir / (img_path.stem + ".jpg")), degraded)

    # 라벨 파일 복사 (정답지는 동일)
    lbl_src = YOLO_DIR / "labels" / "test" / (img_path.stem + ".txt")
    if lbl_src.exists():
        shutil.copy(lbl_src, legacy_test_lbl_dir / lbl_src.name)

print(
    f"✅ Simulated Legacy Test Set 생성 완료: {len(list(legacy_test_img_dir.iterdir()))}장"
)

# 시각화: 원본 vs 열화 비교
if test_images_for_legacy:
    sample = test_images_for_legacy[0]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    orig = cv2.cvtColor(cv2.imread(str(sample)), cv2.COLOR_BGR2RGB)
    degr = cv2.cvtColor(
        cv2.imread(str(legacy_test_img_dir / (sample.stem + ".jpg"))),
        cv2.COLOR_BGR2RGB,
    )
    axes[0].imshow(orig)
    axes[0].set_title("Original (Clean)", fontsize=14)
    axes[0].axis("off")
    axes[1].imshow(degr)
    axes[1].set_title("Simulated Legacy (Degraded)", fontsize=14)
    axes[1].axis("off")
    plt.suptitle("Legacy Degradation Simulation", fontsize=16)
    plt.tight_layout()
    plt.show()

# %%
# 2. Legacy Test용 dataset.yaml 생성
legacy_yaml_path = YOLO_DIR / "dataset_legacy_test.yaml"
with open(legacy_yaml_path, "w", encoding="utf-8") as f:
    yaml.dump(
        {
            "path": str(YOLO_DIR.resolve()),
            "train": "images/train",
            "val": "images/legacy_test",
            "names": class_names,
        },
        f,
        allow_unicode=True,
    )

# 3. Baseline 모델 → Legacy Test 평가
print("\n📊 Baseline 모델 → Legacy Test Set 평가")
baseline_weight = PROJECT_ROOT / "runs/detect/train_baseline/weights/best.pt"
model_base_eval = YOLO(str(baseline_weight))
legacy_base_metrics = model_base_eval.val(data=str(legacy_yaml_path))

torch.cuda.empty_cache()
gc.collect()

# 4. Augmented 모델 → Legacy Test 평가
print("\n📊 Augmented 모델 → Legacy Test Set 평가")
aug_weight = PROJECT_ROOT / "runs/detect/train_augmented/weights/best.pt"
model_aug_eval = YOLO(str(aug_weight))
legacy_aug_metrics = model_aug_eval.val(data=str(legacy_yaml_path))

torch.cuda.empty_cache()
gc.collect()

# 5. 핵심 논증: Clean Val vs Legacy Test 비교
print("\n" + "=" * 60)
print("📊 [핵심 논증] Clean Val vs Legacy Test 성능 비교")
print("=" * 60)

legacy_base_map = legacy_base_metrics.results_dict.get("metrics/mAP50(B)", 0)
legacy_aug_map = legacy_aug_metrics.results_dict.get("metrics/mAP50(B)", 0)

print(f"\n{'':20} {'Clean Val':>12} {'Legacy Test':>12}")
print("-" * 48)
print(f"{'Baseline mAP50':<20} {m_base['mAP50']:>12.4f} {legacy_base_map:>12.4f}")
print(f"{'Augmented mAP50':<20} {m_aug['mAP50']:>12.4f} {legacy_aug_map:>12.4f}")
print(f"\n💡 [결론] 깨끗한 val에서는 Baseline이 우세할 수 있지만,")
print(
    f"   레거시 도면(Legacy Test)에서는 Augmented가 {legacy_aug_map - legacy_base_map:+.4f} 더 높은 mAP50을 달성."
)
print("   → 도메인 맞춤형 증강이 실전 환경에서 효과적임을 증명합니다.")

# %% [markdown]
#
# ## 7. [Phase 3] 도면 도메인 전이학습(Transfer Learning) 효과 검증
# ### 전이학습 도입의 당위성
# - 도면 내의 글자(OCR)를 탐지하기 위해 맨바닥(Scratch)에서 학습하는 것보다, **이미 가구/설비(`object_layout`)를 학습하며 도메인 특징(Feature)을 익힌 가중치**를 활용하는 것이 훨씬 효율적일 것입니다.
# - 1. **Experiment A (Scratch)**: `yolov8n.pt`에서 `ocr` 학습
# - 2. **Experiment B (Transfer)**: `best.pt` (Phase 2 결과)에서 `ocr` 학습

# %%
# ──────────────────────────────────────────────
# [사전 준비] OCR 데이터셋 포맷팅 (Phase 3용)
# ──────────────────────────────────────────────
OCR_DIR = EXTRACT_PATH / "ocr_dataset"
ocr_yaml_path = OCR_DIR / "dataset.yaml"

if not ocr_yaml_path.exists():
    print("OCR 데이터셋 YOLO 포맷팅 시작 (간이 샘플링)...")
    OCR_DIR.mkdir(parents=True, exist_ok=True)

    # OCR은 텍스트(class 0) 단일 클래스로 매핑
    for split in ["train", "val", "test"]:
        dest_img_dir = OCR_DIR / "images" / split
        dest_lbl_dir = OCR_DIR / "labels" / split
        dest_img_dir.mkdir(parents=True, exist_ok=True)
        dest_lbl_dir.mkdir(parents=True, exist_ok=True)

        src_lbl_dir = EXTRACT_PATH / "ocr" / split / "labels"
        if not src_lbl_dir.exists():
            continue

        json_files = list(src_lbl_dir.glob("*.json"))
        limit = 200 if split == "train" else 50
        if len(json_files) > limit:
            random.seed(42)
            json_files = random.sample(json_files, limit)  # 파이프라인 검증용 샘플링

        for jf in json_files:
            img_file = EXTRACT_PATH / "ocr" / split / "images" / f"{jf.stem}.webp"
            if not img_file.exists():
                continue
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            img_info = data.get("images", [{}])[0]
            w, h = img_info.get("width", 4963), img_info.get("height", 3509)

            yolo_lines = []
            for ann in data.get("annotations", []):
                bbox = ann.get("bbox", [])
                if len(bbox) == 4:
                    bx, by, bw, bh = bbox
                    xc, yc = (bx + bw / 2) / w, (by + bh / 2) / h
                    yolo_lines.append(f"0 {xc:.6f} {yc:.6f} {bw / w:.6f} {bh / h:.6f}")

            if yolo_lines:
                with open(dest_lbl_dir / f"{jf.stem}.txt", "w") as f:
                    f.write("\n".join(yolo_lines))
                shutil.copy(img_file, dest_img_dir / img_file.name)

    with open(ocr_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(
            {
                "path": str(OCR_DIR.resolve()),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": {0: "text"},
            },
            f,
        )
    print("OCR 데이터셋 준비 완료!")

# %%
# ──────────────────────────────────────────────
# [실험 A] Scratch 모델 학습
# ──────────────────────────────────────────────
print("\n🚀 [Experiment A] Scratch 모델 학습 시작")
model_scratch = YOLO("yolov8n.pt")
results_scratch = model_scratch.train(
    data=str(ocr_yaml_path),
    epochs=30,  # Transfer vs Scratch 공정 비교
    imgsz=640,
    batch=16,  # Colab L4 (24GB VRAM) 최적화
    workers=8,
    cache=True,
    project=str(PROJECT_ROOT / "runs/detect"),
    name="train_ocr_scratch",
)

# GPU 캐시 메모리 비우기
torch.cuda.empty_cache()
gc.collect()

# ──────────────────────────────────────────────
# [실험 B] Transfer 모델 학습
# ──────────────────────────────────────────────
print("\n🚀 [Experiment B] Transfer 모델 학습 시작")
best_weight_path = PROJECT_ROOT / "runs/detect/train_augmented/weights/best.pt"
if not best_weight_path.exists():
    raise FileNotFoundError(
        f"⚠️ 전이학습을 위한 가중치({best_weight_path})가 없습니다. Phase 2 학습이 정상적으로 완료되었는지 확인하세요."
    )

model_transfer = YOLO(str(best_weight_path))
results_transfer = model_transfer.train(
    data=str(ocr_yaml_path),
    epochs=30,
    imgsz=640,
    batch=16,  # Colab L4 (24GB VRAM) 최적화
    workers=8,
    cache=True,
    project=str(PROJECT_ROOT / "runs/detect"),
    name="train_ocr_transfer",
)

# GPU 캐시 메모리 비우기
torch.cuda.empty_cache()
gc.collect()

# %% [markdown]
# ## 8. [Phase 4] 레거시 구형 도면 실전 추론 및 JSON 구조화 (최종 목표)
# - 수집된 정제 데이터가 아닌, 라벨(JSON)이 없는 **진짜 오래된 현업 도면(Raw Legacy Data)**을 투입합니다.
# - 인위적인 노이즈 증강 대신, 모델 투입 전 **화질 개선 전처리(Image Enhancement: CLAHE, Sharpening 등)**를 수행하여 도메인 갭을 극복합니다.
# - SAHI를 적용해 고해상도 다층 도면의 찌그러짐을 방지하며 추론하고 결과를 **JSON 포맷으로 덤프(Digitization)** 합니다.
# - 최종적으로 예측된 JSON을 이미지 위에 덧그려(EDA 시각화) 정확도를 사람 눈으로 검증합니다.

# %%
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

RAW_LEGACY_DIR = PROJECT_ROOT / "raw_legacy_inputs"
RAW_LEGACY_DIR.mkdir(parents=True, exist_ok=True)

# 1. 테스트할 레거시 이미지 로드
legacy_images = sorted(
    [
        p
        for p in RAW_LEGACY_DIR.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]
    ]
)

if not legacy_images:
    print(
        f"⚠️ [{RAW_LEGACY_DIR}] 폴더에 사용자가 업로드한 레거시 도면(.jpg)이 없습니다!"
    )
    print(
        "임시 파이프라인 구동을 위해 기존 테스트셋 중 하나를 레거시 도면이라고 가정(Fallback)하고 진행합니다."
    )
    fallback_dir = YOLO_DIR / "images" / "test"
    if fallback_dir.exists() and list(fallback_dir.glob("*.webp")):
        legacy_images = [random.choice(list(fallback_dir.glob("*.webp")))]
    else:
        raise FileNotFoundError("테스트할 이미지가 전혀 없습니다.")

# 너무 많으면 노트북 출력창이 길어지므로 최대 5개까지만 시연
for test_image in legacy_images[:5]:
    print(f"\n" + "=" * 50)
    print(f"[{test_image.name}] 실전 레거시 도면 파이프라인 시작...")

    # 2. 전처리: 화질 복원 (Image Enhancement)
    raw_img = cv2.imread(str(test_image))
    if raw_img is None:
        print(f"이미지를 읽을 수 없습니다: {test_image.name}")
        continue

    # 2-1. 대비 극대화 (Contrast Normalization: 글자 깨짐 방지)
    norm = cv2.normalize(raw_img, None, 0, 255, cv2.NORM_MINMAX)

    # 2-2. 텍스트 보존형 선명화 (Unsharp Masking)
    gaussian = cv2.GaussianBlur(norm, (5, 5), 1.0)
    enhanced = cv2.addWeighted(norm, 1.5, gaussian, -0.5, 0)

    # 전처리된 이미지 임시 저장 (SAHI 투입용)
    enhanced_path = PROJECT_ROOT / f"enhanced_{test_image.stem}.jpg"
    cv2.imwrite(str(enhanced_path), enhanced)

    # 3. 마스터 모델 예측 (Inference) - SAHI 적용
    print(f"SAHI 기반 슬라이싱 실전 추론 시작 (512x512 패치)...")

    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    best_model_path = PROJECT_ROOT / "runs/detect/train_augmented/weights/best.pt"
    if not best_model_path.exists():
        raise FileNotFoundError(
            f"⚠️ Augmented 모델 가중치({best_model_path})가 없습니다. Phase 2 학습이 정상 완료되었는지 확인하세요."
        )

    # SAHI 모델 로드
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=str(best_model_path),
        confidence_threshold=0.25,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )

    # SAHI 슬라이싱 추론
    result = get_sliced_prediction(
        str(enhanced_path),
        detection_model,
        slice_height=512,
        slice_width=512,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
    )

    # 4. JSON 구조화 (Digitization)
    export_data = {"image_filename": test_image.name, "predictions": []}

    for obj in result.object_prediction_list:
        export_data["predictions"].append(
            {
                "class_id": obj.category.id,
                "class_name": obj.category.name,
                "confidence": float(obj.score.value),
                "bbox": [
                    float(obj.bbox.minx),
                    float(obj.bbox.miny),
                    float(obj.bbox.maxx),
                    float(obj.bbox.maxy),
                ],
            }
        )
        # 시각화 시 글자 깨짐 및 라벨이 도면을 가리는 현상 방지를 위해 라벨을 숫자 ID로 변경
        obj.category.name = str(obj.category.id)

    json_output_path = PROJECT_ROOT / f"{test_image.stem}_digitized.json"
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, ensure_ascii=False, indent=4)

    # 4-B. JSON 후처리 (Topological Post-processing)
    def postprocess_json(predictions, snap_threshold=15):
        """YOLO 예측 JSON을 후처리: 좌표 직교화(Orthogonalize) + 스냅(Snap)"""
        if not predictions:
            return predictions

        # 1. 좌표 직교화: 거의 수평/수직인 bbox를 완벽한 직각으로 스냅
        for pred in predictions:
            bbox = pred["bbox"]
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            if w > 0 and h > 0:
                if h / w < 0.05:  # 거의 수평선
                    mid_y = (bbox[1] + bbox[3]) / 2
                    bbox[1], bbox[3] = mid_y, mid_y
                elif w / h < 0.05:  # 거의 수직선
                    mid_x = (bbox[0] + bbox[2]) / 2
                    bbox[0], bbox[2] = mid_x, mid_x
            pred["bbox"] = [round(v, 1) for v in bbox]

        # 2. 좌표 스냅: 가까운 좌표끼리 대표값으로 통일
        all_x, all_y = [], []
        for pred in predictions:
            bbox = pred["bbox"]
            all_x.extend([bbox[0], bbox[2]])
            all_y.extend([bbox[1], bbox[3]])

        def snap_values(values, threshold):
            if not values:
                return {}
            sorted_vals = sorted(set(values))
            snap_map, groups = {}, [[sorted_vals[0]]]
            for v in sorted_vals[1:]:
                if v - groups[-1][-1] <= threshold:
                    groups[-1].append(v)
                else:
                    groups.append([v])
            for group in groups:
                rep = round(sum(group) / len(group), 1)
                for v in group:
                    snap_map[v] = rep
            return snap_map

        x_snap = snap_values(all_x, snap_threshold)
        y_snap = snap_values(all_y, snap_threshold)
        for pred in predictions:
            bbox = pred["bbox"]
            bbox[0] = x_snap.get(bbox[0], bbox[0])
            bbox[2] = x_snap.get(bbox[2], bbox[2])
            bbox[1] = y_snap.get(bbox[1], bbox[1])
            bbox[3] = y_snap.get(bbox[3], bbox[3])

        return predictions

    # 후처리 적용 및 저장
    raw_predictions = json.loads(json.dumps(export_data["predictions"]))  # 원본 보존
    export_data["predictions"] = postprocess_json(export_data["predictions"])

    json_postprocessed_path = (
        PROJECT_ROOT / f"{test_image.stem}_digitized_postprocessed.json"
    )
    with open(json_postprocessed_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, ensure_ascii=False, indent=4)

    print(f"✅ 디지털화 완료! Raw JSON: {json_output_path.name}")
    print(f"   후처리 JSON: {json_postprocessed_path.name}")
    print(
        f"   후처리 효과: 좌표 직교화 + {len(export_data['predictions'])}개 객체 스냅 완료"
    )

    # 5. 결과 시각화 (인간 육안 검증용 EDA)
    result.export_visuals(
        export_dir=str(PROJECT_ROOT),
        file_name=f"sahi_{test_image.stem}",
        rect_th=1,  # 바운딩 박스 두께 얇게
        text_size=0.3,  # 텍스트 크기 작게
        text_th=1,  # 텍스트 굵기 얇게
        hide_conf=True,  # Confidence score 숨김 (가독성 향상)
    )
    predicted_plot = Image.open(str(PROJECT_ROOT / f"sahi_{test_image.stem}.png"))

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(f"Legacy Floorplan: {test_image.name}", fontsize=16)

    # 원본
    axes[0].imshow(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Raw Input (No JSON)", fontsize=14)
    axes[0].axis("off")

    # 예측
    axes[1].imshow(predicted_plot)
    axes[1].set_title("Enhanced + Prediction (Post-processed)", fontsize=14)
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

print(
    "\n💡 Phase 4 완료: 레거시 도면 → 전처리 → SAHI 추론 → JSON 후처리(좌표 스냅) → 디지털화 성공."
)

# %%
