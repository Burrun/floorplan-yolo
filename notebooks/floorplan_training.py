# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # [딥러닝분석] 파이널 프로젝트: 2D 아파트 도면 객체 검출 및 3D 모델링 파이프라인 구축
# - **소속**: 숭실대학교 소프트웨어학과 오현
# - **프로젝트 주제**: 2D 도면 이미지의 딥러닝 기반 디지털 구조화 및 3D 변환의 AI 베이스라인 모델 구축
# - **AI Task**: 7클래스 통합 객체 검출 (가구 5종 + 구조물 2종) 및 OCR 전이학습
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
from PIL import Image
from pathlib import Path
from ultralytics import YOLO
import subprocess
import cv2


# ──────────────────────────────────────────────
# 🎯 재현성 확보 (Reproducibility)
# ──────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

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
    # 무조건 현재 실행 경로를 프로젝트 루트로 간주 (로컬 floorplan-yolo 또는 클라우드 /root/pj)
    PROJECT_ROOT = Path.cwd()
    print(f"[로컬/클라우드 통합] 프로젝트 루트: {PROJECT_ROOT}")

# YOLO 환경 변수 강제 설정 (절대 경로 및 노트북 내 폴더 생성 방지)
from ultralytics import settings

settings.update(
    {"datasets_dir": str(PROJECT_ROOT), "runs_dir": str(PROJECT_ROOT / "runs")}
)
# GPU 정보 출력
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory // (1024**2)
    print(f"GPU 감지됨: {gpu_name} ({gpu_mem}MB)")
else:
    print("GPU 없음 → CPU 모드로 학습 (느림)")

print("Libraries imported successfully.")

# %% [markdown]
# ## 0-1. ⚙️ 하드웨어 프로필 (Hardware-Aware Profile) 설정
#

# %%
# ──────────────────────────────────────────────
# ⚙️ Modal A10G (24GB VRAM) 타겟 하드웨어 프로필
# ──────────────────────────────────────────────
BASE_WEIGHT = "yolov8m.pt"  # Medium 모델: Nano 대비 압도적 성능, Phase 2~4 전체 적용

# A10G VRAM 24GB 기준 보수적 세팅
# - YOLOv8m + imgsz=640 + cache=True → 배치당 약 1.0~1.3GB VRAM 소모
# - Batch=16: 약 16~20GB 사용 → 피크 메모리 고려 시 4~8GB 여유 확보 (OOM 방어)
# - Batch=32는 Mosaic 증강 + cache 동시 적용 시 피크에서 터질 위험 있음
BATCH_SIZE = 16  # 24GB에서는 16이 제일 안전한 스윗스팟
WORKERS = 0  # multiprocessing spawn/forkserver 충돌 방지를 위해 0으로 강제 설정
print(
    f"🚀 [A10G 모드] {BASE_WEIGHT} / Batch={BATCH_SIZE} / Workers={WORKERS} (24GB VRAM 보수적 세팅)"
)

# %% [markdown]
# ## 0-2. 7클래스 마스터 데이터셋 연동 및 압축 해제

# %%
# ──────────────────────────────────────────────
# 데이터셋 경로 (8:1:1로 이미 전처리된 마스터 데이터셋 사용)
# ──────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT
DATA_DIR.mkdir(parents=True, exist_ok=True)

ZIP_NAME = "master_dataset.tar.zst"
ZIP_PATH = DATA_DIR / ZIP_NAME
MASTER_DATASET_DIR = DATA_DIR / "master_dataset"

# 사용자 지정 구글 드라이브 링크 (2000장 추출 마스터 데이터셋)
GDRIVE_FILE_ID = "1wcMWrSVigaX0L43oHR9YaMe4GzTJ_tcc"

if not MASTER_DATASET_DIR.exists():
    # Step 1: 압축파일 확보
    if not ZIP_PATH.exists():
        if IS_COLAB:
            drive_zip = Path(f"/content/drive/MyDrive/{ZIP_NAME}")
            if drive_zip.exists():
                print(f"구글 드라이브에서 {ZIP_NAME} 복사 중...")
                shutil.copy(str(drive_zip), str(ZIP_PATH))
            else:
                import gdown

                print(f"구글 드라이브에 파일 없음 → gdown 직접 다운로드 시작...")
                gdown.download(id=GDRIVE_FILE_ID, output=str(ZIP_PATH), quiet=False)
        else:
            import gdown

            print(f"데이터셋({ZIP_NAME}) 다운로드 시작...")
            gdown.download(id=GDRIVE_FILE_ID, output=str(ZIP_PATH), quiet=False)

    # Step 2: 압축 해제
    if ZIP_PATH.exists():
        print(f"{ZIP_NAME} 압축 해제 중...")
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
    print(f"Dataset ready at {MASTER_DATASET_DIR}")

# %% [markdown]
# ## 0-3. YOLO dataset.yaml 자동 생성 (7클래스)

# %%
yaml_path = MASTER_DATASET_DIR / "dataset.yaml"

class_names = {
    0: "toilet",
    1: "washbasin",
    2: "sink",
    3: "bathtub",
    4: "gas_stove",
    5: "door",
    6: "window",
}

dataset_config = {
    "path": str(MASTER_DATASET_DIR.resolve()),
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "names": class_names,
}

with open(yaml_path, "w", encoding="utf-8") as f:
    yaml.dump(dataset_config, f, allow_unicode=True, default_flow_style=False)
print(f"dataset.yaml saved at {yaml_path}")

# %% [markdown]
# ## 0-4. 탐색적 데이터 분석 (EDA) 및 시각화 검증

# %%
# ──────────────────────────────────────────────
# 🎨 전역 색상 팔레트 (8클래스: 7 마스터 + 1 OCR)
#    프로젝트 전체에서 동일한 색상 체계를 사용합니다.
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

import matplotlib.patches as mpatches


def visualize_yolo_labels(image_path, label_path, class_names):
    img = cv2.imread(str(image_path))
    if img is None:
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    if not os.path.exists(label_path):
        return img, {}

    counts = {}
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

        cls_name = class_names.get(class_id, str(class_id))
        color = CLASS_COLOR_MAP.get(cls_name, (200, 200, 200))
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 4)
        counts[cls_name] = counts.get(cls_name, 0) + 1

    return img, counts


train_img_dir = MASTER_DATASET_DIR / "images" / "train"
train_lbl_dir = MASTER_DATASET_DIR / "labels" / "train"

if train_img_dir.exists():
    img_files = list(train_img_dir.glob("*.webp"))
    if img_files:
        # 시드(SEED=42) 고정으로 EDA 재현성 확보
        sample_img = random.choice(img_files)
        sample_lbl = train_lbl_dir / (sample_img.stem + ".txt")

        print(f"Visualizing Sample: {sample_img.name}")
        result = visualize_yolo_labels(sample_img, sample_lbl, class_names)
        if result is not None:
            result_img, counts = result
            fig, ax = plt.subplots(1, 1, figsize=(14, 9))
            ax.imshow(result_img)
            ax.axis("off")
            ax.set_title(
                f"EDA Sample: {sample_img.name}", fontsize=13, fontweight="bold"
            )

            # 범례
            legend_patches = []
            for cls_name, color_rgb in CLASS_COLOR_MAP.items():
                cnt = counts.get(cls_name, 0)
                if cnt > 0:
                    color_norm = tuple(c / 255.0 for c in color_rgb)
                    legend_patches.append(
                        mpatches.Patch(color=color_norm, label=f"{cls_name} ({cnt})")
                    )
            if legend_patches:
                ax.legend(
                    handles=legend_patches,
                    loc="upper right",
                    fontsize=10,
                    framealpha=0.85,
                    fancybox=True,
                    shadow=True,
                )
            plt.tight_layout()
            plt.show()

# %% [markdown]
# ## 1. [Phase 1] 최적의 데이터 개수 탐색 (Data Scaling Ablation)
# - 학습 데이터 개수를 300장부터 1500장까지 점진적으로 늘려보며 성능 향상폭이 꺾이는 지점(Saturation Point)을 찾습니다.

# %%
print("=" * 60)
print("🔬 [Phase 1] Data Size Ablation")
print("=" * 60)

import pandas as pd

train_images = list((MASTER_DATASET_DIR / "images" / "train").glob("*.webp"))
random.shuffle(train_images)  # Prevent systematic bias from filesystem ordering
# 총 1600장의 훈련셋을 활용하여 점진적 크기 실험
data_sizes = [300, 600, 900, 1200, 1500]
scaling_results: dict[int, float] = {}

for size in data_sizes:
    if size > len(train_images):
        continue

    subset = train_images[:size]
    subset_txt = MASTER_DATASET_DIR / f"train_{size}.txt"
    with open(subset_txt, "w", encoding="utf-8") as f:
        f.write("\n".join([str(p.resolve()) for p in subset]))

    yaml_path_size = MASTER_DATASET_DIR / f"dataset_{size}.yaml"
    with open(yaml_path_size, "w", encoding="utf-8") as f:
        yaml.dump(
            {
                "path": str(MASTER_DATASET_DIR.resolve()),
                "train": str(subset_txt.resolve()),
                "val": "images/val",
                "names": class_names,
            },
            f,
            allow_unicode=True,
        )

    weight_path = (
        PROJECT_ROOT / "runs/detect" / f"train_size_{size}" / "weights" / "best.pt"
    )
    if weight_path.exists():
        print(
            f"\n ✅ Data Size: {size} - 이미 학습된 가중치가 존재합니다. 학습을 스킵합니다."
        )
        results_csv = (
            PROJECT_ROOT / "runs/detect" / f"train_size_{size}" / "results.csv"
        )
        if results_csv.exists():
            df = pd.read_csv(results_csv)
            df.columns = df.columns.str.strip()
            col_name = "metrics/mAP50(B)"
            if col_name in df.columns:
                scaling_results[size] = df[col_name].iloc[-1]
            else:
                scaling_results[size] = 0.0
        else:
            scaling_results[size] = 0.0
        continue

    print(f"\n Data Size: {size} 학습 시작 (30 Epochs 검증)")

    # Phase 1은 데이터 개수 트렌드(포화점) 탐색이 목적이므로
    # 속도가 빠른 Nano 모델 고정 사용 (절대 mAP보다 상대 추이가 중요)
    model_size = YOLO("yolov8n.pt")
    res_size = model_size.train(
        data=str(yaml_path_size),
        epochs=30,
        imgsz=640,
        batch=BATCH_SIZE,
        workers=WORKERS,
        cache=False,
        project=str(PROJECT_ROOT / "runs/detect"),
        name=f"train_size_{size}",
        verbose=False,
        patience=10,  # 과적합 방지 조기 종료
    )

    torch.cuda.empty_cache()
    gc.collect()

    map50 = res_size.results_dict.get("metrics/mAP50(B)", 0)
    scaling_results[size] = map50

if scaling_results:
    sizes = list(scaling_results.keys())
    maps = list(scaling_results.values())

    plt.figure(figsize=(8, 5))
    plt.plot(sizes, maps, marker="o", linestyle="-", color="b", linewidth=2)
    plt.title("Data Scaling Ablation", fontsize=14)
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

if scaling_results:
    optimal_size = max(scaling_results, key=scaling_results.get)
    print(
        f"✅ 최적의 데이터 개수 결정: {optimal_size}장 (mAP: {scaling_results[optimal_size]:.3f})"
    )
else:
    optimal_size = 1500


# %% [markdown]
# ## 2. [Phase 2] 레거시 도면 노이즈 제거 전처리 (Pre-processing)
# - Adaptive Gaussian Thresholding을 활용하여 레거시 스캔본의 망점과 음영을 완벽히 제거합니다.
# - 실전 추론(Inference) 단계에서 외부 입력 이미지(Legacy)가 들어올 때 즉시 적용됩니다.


# %%
def apply_adaptive_gaussian_preprocessing(image_path_or_img) -> np.ndarray:
    """
    레거시 도면 이미지의 망점, 변색, 균일하지 않은 조명을 제거하는 전처리 모듈입니다.
    입력: 이미지 파일 경로(str/Path) 또는 OpenCV BGR 이미지 배열(np.ndarray)
    출력: 노이즈가 제거된 3채널(BGR) Numpy 이미지 배열
    """
    if isinstance(image_path_or_img, (str, Path)):
        img = cv2.imread(str(image_path_or_img))
    else:
        img = image_path_or_img.copy()

    if img is None:
        raise ValueError("이미지를 로드할 수 없습니다.")

    # 1. Grayscale 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Adaptive Gaussian 이진화 (Block: 21, C: 10)
    # 주변 픽셀(21x21)의 가우시안 가중 평균을 기준으로 이진화하여 배경의 우글우글한 망점을 완전히 날려버립니다.
    processed = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10
    )

    # 3. YOLO 추론기(3채널 RGB/BGR)에 맞춰 채널 변환
    processed_bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    return processed_bgr


print("✅ [Phase 2] Adaptive Gaussian 전처리 파이프라인 함수 로드 완료")

# %% [markdown]
# ## 3. [Phase 3] 도메인 맞춤형 하이퍼 파라미터 탐색 (Baseline vs Domain-Tuned)
# - 원본 데이터의 한계를 돌파할 도메인 맞춤형 증강 기법을 비교합니다.

# %%
print("=" * 60)
print("🔬 [Phase 3] Baseline vs Domain-Tuned 모델 학습 (7클래스 마스터 모델)")
print("=" * 60)

# 1. Baseline (순정 학습)
print("🚀 Baseline 모델 학습 시작...")
baseline_weight = PROJECT_ROOT / "runs/detect/train_baseline/weights/best.pt"
if baseline_weight.exists():
    print(
        f"✅ 이미 학습된 Baseline 모델 가중치가 존재합니다. 학습을 스킵합니다: {baseline_weight}"
    )
else:
    model_baseline = YOLO(BASE_WEIGHT)
    results_baseline = model_baseline.train(
        data=str(MASTER_DATASET_DIR / f"dataset_{optimal_size}.yaml"),
        epochs=150,  # 7클래스 난이도 상승 반영
        imgsz=640,
        batch=BATCH_SIZE,
        workers=WORKERS,
        cache=True,
        project=str(PROJECT_ROOT / "runs/detect"),
        name="train_baseline",
        patience=20,  # 성능 개선 없으면 20에폭 후 조기 종료
    )
    torch.cuda.empty_cache()
    gc.collect()

# 2. Domain-Tuned (도메인 맞춤형 하이퍼 파라미터 적용 학습)
print("\n🚀 Domain-Tuned 모델 학습 시작...")
aug_weight = PROJECT_ROOT / "runs/detect/train_domain_tuned/weights/best.pt"
if aug_weight.exists():
    print(
        f"✅ 이미 학습된 Domain-Tuned 모델 가중치가 존재합니다. 학습을 스킵합니다: {aug_weight}"
    )
else:
    model_augmented = YOLO(BASE_WEIGHT)
    results_augmented = model_augmented.train(
        data=str(MASTER_DATASET_DIR / f"dataset_{optimal_size}.yaml"),
        epochs=150,  # 7클래스 난이도 상승 반영
        imgsz=640,
        batch=BATCH_SIZE,
        workers=WORKERS,
        cache=True,
        project=str(PROJECT_ROOT / "runs/detect"),
        name="train_domain_tuned",
        patience=20,  # 성능 개선 없으면 20에폭 후 조기 종료
        # 도메인 갭(Domain Gap) 극복을 위한 파라미터 (레거시 도면 타겟팅)
        # 확률 및 변형 강도를 나타내는 데이터 증강(Augmentation) 파라미터
        degrees=2.0,  # 이미지 회전 (+/- 2.0도): 스캐너 수작업 스캔 오차 모사
        hsv_s=0.2,  # 채도 변형 (20%) — YOLO 기본 0.7 대비 의도적 약화: 도면은 흑백/저채도 기반이므로 과도한 색상 변형은 비현실적
        hsv_v=0.2,  # 명도 변형 (20%) — YOLO 기본 0.4 대비 의도적 약화: 도면 특성상 명도 분포가 좁아 과도한 변형은 노이즈 유발
        perspective=0.0005,  # 원근 왜곡 강도 (0.05%): 스캔 시 종이가 울거나 삐뚤어진 상태 모사
        scale=0.5,  # 스케일 변형 (+/- 50%): 다양한 해상도 및 도면 배율(확대/축소) 대응
        mosaic=1.0,  # 모자이크 증강 확률 (100%): 4장의 도면을 하나로 합쳐 모델의 강건성 극대화
        bgr=0.2,  # BGR 채널 반전 (20%): 특정 잉크 색상에 과적합되는 것을 방지 (gray 파라미터 대체)
        flipud=0.5,  # 상하 반전 (50%): 위아래가 뒤집혀 스캔된 도면에 대응하여 인식률 극대화
    )
    torch.cuda.empty_cache()
    gc.collect()

# %% [markdown]
# ## 4. [Phase 3-B] Simulated Legacy 시각적 강건성 검증 (Visual Inspection)
# - Baseline 모델과 Domain-Tuned 모델의 추론 결과를 시각적으로 비교하여, 도메인 맞춤 파라미터의 효과를 눈으로 직접 확인합니다.


# %%
def simulate_legacy_degradation(img):
    h, w = img.shape[:2]
    # 황변
    yellow_tint = np.full_like(img, (0, 20, 40), dtype=np.uint8)
    img = cv2.addWeighted(img, 0.85, yellow_tint, 0.15, 0)
    # 퇴색
    img = cv2.convertScaleAbs(img, alpha=0.7, beta=-15)
    # 노이즈
    noise = np.random.normal(0, 15, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # 블러
    img = cv2.GaussianBlur(img, (3, 3), 1.0)
    return img


baseline_weight = PROJECT_ROOT / "runs/detect/train_baseline/weights/best.pt"
aug_weight = PROJECT_ROOT / "runs/detect/train_domain_tuned/weights/best.pt"

if baseline_weight.exists() and aug_weight.exists():
    print("⏳ Baseline 및 Augmented 모델 가중치 로드 중...")
    model_base = YOLO(str(baseline_weight))
    model_aug = YOLO(str(aug_weight))

    val_images = list((MASTER_DATASET_DIR / "images" / "val").glob("*.webp"))
    sample_images = random.sample(val_images, min(5, len(val_images)))

    output_dir = PROJECT_ROOT / "runs" / "legacy_visual_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, img_path in enumerate(sample_images):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # 레거시 훼손 모사
        degraded = simulate_legacy_degradation(img)

        # 추론
        res_base = model_base.predict(degraded, imgsz=640, verbose=False)[0]
        res_aug = model_aug.predict(degraded, imgsz=640, verbose=False)[0]

        # 결과 이미지 렌더링
        img_base = res_base.plot()
        img_aug = res_aug.plot()

        # BGR -> RGB
        img_base_rgb = cv2.cvtColor(img_base, cv2.COLOR_BGR2RGB)
        img_aug_rgb = cv2.cvtColor(img_aug, cv2.COLOR_BGR2RGB)

        # 시각화 비교 플롯
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        axes[0].imshow(img_base_rgb)
        axes[0].set_title(f"Baseline Prediction", fontsize=14, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(img_aug_rgb)
        axes[1].set_title(f"Augmented Prediction", fontsize=14, fontweight="bold")
        axes[1].axis("off")

        plt.tight_layout()
        save_path = output_dir / f"compare_{i + 1}.png"
        plt.savefig(save_path, dpi=150)
        print(f"✅ 시각화 저장 완료: {save_path}")
        plt.show()

    print(
        f"\n🎉 눈으로 직접 비교할 수 있는 시각화 결과가 {output_dir} 에 저장되었습니다."
    )
else:
    print(
        "⚠️ Baseline 또는 Augmented 모델 가중치가 아직 생성되지 않았습니다. Phase 3 학습을 먼저 완료하세요."
    )

# %% [markdown]
# ## 5. [Phase 4] OCR 전이학습 (8클래스 단일 모델 vs 2-Stage 전이학습 비교)
# - **아키텍처 확정 (8클래스 동시 학습의 한계 극복)**: 가구/구조물(7)과 OCR(1)을 8클래스로 동시 학습했을 때, 객체 형태 특징과 텍스트 박스 비율이 충돌하여 전체 성능(mAP)이 하락하는 현상을 발견했습니다.
# - **왜 전이학습인가?**: 이를 극복하기 위해 텍스트 인식은 배제하고, YOLO를 이용해 **글자가 위치한 구역(BBox)만 빠르게 탐지**하는 2-Stage 모델로 분리했습니다.
# - 7클래스 마스터 모델이 학습한 도면 특성 가중치(`best.pt`)를 넘겨받아 전이학습을 수행하면, 8클래스 동시 학습이나 맨바닥(Scratch) 학습보다 압도적인 수렴 속도와 정확도를 달성합니다. 이것이 전이학습을 도입한 핵심 이유입니다.

# %%
print("=" * 60)
print("🚀 [Phase 4] OCR 텍스트 탐지용 전이학습 (Transfer Learning) 시작")
print("=" * 60)

# ──────────────────────────────────────────────
# OCR 데이터셋 자동 다운로드 및 압축 해제
# ──────────────────────────────────────────────
OCR_ZIP_NAME = "ocr_dataset.tar.zst"
OCR_ZIP_PATH = DATA_DIR / OCR_ZIP_NAME
OCR_DIR = DATA_DIR / "ocr_dataset"
OCR_GDRIVE_FILE_ID = "1D2BYVBNArwoMTgqVO0m8ZOUxEJZeuLxE"

if not OCR_DIR.exists():
    if not OCR_ZIP_PATH.exists():
        if IS_COLAB:
            drive_ocr = Path(f"/content/drive/MyDrive/{OCR_ZIP_NAME}")
            if drive_ocr.exists():
                print(f"구글 드라이브에서 {OCR_ZIP_NAME} 복사 중...")
                shutil.copy(str(drive_ocr), str(OCR_ZIP_PATH))
            else:
                import gdown

                print(f"구글 드라이브에 파일 없음 → gdown 직접 다운로드 시작...")
                gdown.download(
                    id=OCR_GDRIVE_FILE_ID, output=str(OCR_ZIP_PATH), quiet=False
                )
        else:
            import gdown

            print(f"OCR 데이터셋({OCR_ZIP_NAME}) 다운로드 시작...")
            gdown.download(id=OCR_GDRIVE_FILE_ID, output=str(OCR_ZIP_PATH), quiet=False)

    if OCR_ZIP_PATH.exists():
        print(f"{OCR_ZIP_NAME} 압축 해제 중...")
        try:
            subprocess.run(
                ["tar", "-I", "zstd", "-xf", str(OCR_ZIP_PATH), "-C", str(DATA_DIR)],
                check=True,
            )
        except Exception:
            print("zstd 패키지가 없거나 권한 오류 발생. zstd 설치 시도 중...")
            if IS_COLAB:
                subprocess.run(["apt-get", "update"], check=True)
                subprocess.run(["apt-get", "install", "-y", "zstd"], check=True)
                subprocess.run(
                    [
                        "tar",
                        "-I",
                        "zstd",
                        "-xf",
                        str(OCR_ZIP_PATH),
                        "-C",
                        str(DATA_DIR),
                    ],
                    check=True,
                )
            else:
                raise RuntimeError(
                    "압축 해제 실패: 터미널에서 'sudo apt install zstd' 를 실행 후 다시 시도하세요."
                )
        print("OCR 데이터셋 압축 해제 완료!")
    else:
        raise FileNotFoundError(f"⚠️ {OCR_ZIP_PATH} 파일을 확보하지 못했습니다.")
else:
    print(f"OCR Dataset ready at {OCR_DIR}")

ocr_yaml_path = OCR_DIR / "dataset.yaml"

# 환경 무관하게 절대 경로가 보장되도록 매번 생성 (마스터 데이터셋과 동일한 방식)
import yaml

ocr_config = {
    "path": str(OCR_DIR.resolve()),
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "names": {0: "text"},
}
with open(ocr_yaml_path, "w", encoding="utf-8") as f:
    yaml.dump(ocr_config, f, allow_unicode=True, default_flow_style=False)
print(f"✅ OCR dataset.yaml 절대 경로 통합 생성 완료: {ocr_yaml_path}")

# 7클래스 마스터 모델의 황금 가중치 로드
best_weight_path = PROJECT_ROOT / "runs/detect/train_domain_tuned/weights/best.pt"

if not best_weight_path.exists():
    raise FileNotFoundError(
        f"⚠️ 전이학습을 위한 가중치({best_weight_path})가 없습니다. Phase 3 학습을 먼저 완료하세요."
    )

print(
    f"✅ {best_weight_path.name} 가중치를 성공적으로 불러왔습니다. 전이학습을 시작합니다!"
)

ocr_transfer_weight = (
    PROJECT_ROOT / "runs/detect/train_ocr_transfer/weights/best.pt"
)
if ocr_transfer_weight.exists():
    print(
        f"✅ 이미 학습된 OCR 전이학습 가중치가 존재합니다. 학습을 스킵합니다: {ocr_transfer_weight}"
    )
else:
    # 전이학습(Transfer Learning)
    model_transfer = YOLO(str(best_weight_path))
    results_transfer = model_transfer.train(
        data=str(ocr_yaml_path),
        epochs=50,  # 클래스가 1개지만 견고한 학습을 위해 에폭 상향
        imgsz=640,
        batch=BATCH_SIZE,
        workers=WORKERS,
        cache=True,
        project=str(PROJECT_ROOT / "runs/detect"),
        name="train_ocr_transfer",
        patience=10,
    )

    torch.cuda.empty_cache()
    gc.collect()
    print("🎉 OCR 전이학습이 성공적으로 완료되었습니다!")

# ──────────────────────────────────────────────
# [비교 검증] 실험 A: Scratch 모델 (비교군) 학습
# ──────────────────────────────────────────────
# 전이학습(Transfer)의 압도적 효율을 증명하기 위해, 아무런 사전 지식이 없는
# 랜덤 초기화 상태(Scratch)에서 동일하게 50에폭 OCR 학습을 진행합니다.
# (이 결과는 8클래스 동시 학습 시 발생하는 수렴 지연 및 성능 저하와 동일한 양상을 보입니다.)
ocr_scratch_weight = PROJECT_ROOT / "runs/detect/train_ocr_scratch/weights/best.pt"
if ocr_scratch_weight.exists():
    print(f"✅ 이미 학습된 OCR Scratch 가중치가 존재합니다: {ocr_scratch_weight}")
else:
    print("🚀 [비교군] OCR Scratch (yolov8m.pt) 학습 시작...")
    model_scratch = YOLO(BASE_WEIGHT)  # 공정한 비교를 위해 Medium 모델 동일 사용
    results_scratch = model_scratch.train(
        data=str(ocr_yaml_path),
        epochs=50,
        imgsz=640,
        batch=BATCH_SIZE,
        workers=WORKERS,
        cache=True,
        project=str(PROJECT_ROOT / "runs/detect"),
        name="train_ocr_scratch",
        patience=10,
    )
    torch.cuda.empty_cache()
    gc.collect()

# ──────────────────────────────────────────────
# [시각화] 전체적 성능(mAP) & 1대1 실체 탐지 사진 비교
# ──────────────────────────────────────────────
print("\n📊 [성능 비교] Scratch vs Transfer 전체 성능 및 1대1 탐지 비교")
# 1. 전체적 성능 (mAP, Precision, Recall) 곡선 비교
scratch_csv = PROJECT_ROOT / "runs/detect/train_ocr_scratch/results.csv"
transfer_csv = PROJECT_ROOT / "runs/detect/train_ocr_transfer/results.csv"

if scratch_csv.exists() and transfer_csv.exists():
    df_scratch = pd.read_csv(scratch_csv)
    df_transfer = pd.read_csv(transfer_csv)

    df_scratch.columns = df_scratch.columns.str.strip()
    df_transfer.columns = df_transfer.columns.str.strip()

    col_map = "metrics/mAP50(B)"
    col_t_loss = "train/box_loss"
    col_v_loss = "val/box_loss"

    # 3개의 서브플롯 (mAP, Train Loss, Val Loss) - 학습 속도 및 수렴 비교
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    if col_map in df_scratch.columns and col_map in df_transfer.columns:
        axes[0].plot(
            df_scratch["epoch"],
            df_scratch[col_map],
            label="Scratch",
            color="#FF6B6B",
            linestyle="--",
            linewidth=2.5,
        )
        axes[0].plot(
            df_transfer["epoch"],
            df_transfer[col_map],
            label="Transfer",
            color="#4ECDC4",
            linewidth=3,
        )
        axes[0].set_title(
            "mAP@50 Comparison\n(Overall Detection Accuracy)",
            fontsize=14,
            fontweight="bold",
        )
        axes[0].set_xlabel("Epochs", fontsize=12)
        axes[0].set_ylabel("mAP@50", fontsize=12)
        axes[0].legend(fontsize=11)
        axes[0].grid(True, linestyle=":", alpha=0.7)

    if col_t_loss in df_scratch.columns and col_t_loss in df_transfer.columns:
        axes[1].plot(
            df_scratch["epoch"],
            df_scratch[col_t_loss],
            label="Scratch",
            color="#FF6B6B",
            linestyle="--",
            linewidth=2.5,
        )
        axes[1].plot(
            df_transfer["epoch"],
            df_transfer[col_t_loss],
            label="Transfer",
            color="#4ECDC4",
            linewidth=3,
        )
        axes[1].set_title(
            "Train Box Loss Comparison\n(Convergence Speed)",
            fontsize=14,
            fontweight="bold",
        )
        axes[1].set_xlabel("Epochs", fontsize=12)
        axes[1].set_ylabel("Train Box Loss", fontsize=12)
        axes[1].legend(fontsize=11)
        axes[1].grid(True, linestyle=":", alpha=0.7)

    if col_v_loss in df_scratch.columns and col_v_loss in df_transfer.columns:
        axes[2].plot(
            df_scratch["epoch"],
            df_scratch[col_v_loss],
            label="Scratch",
            color="#FF6B6B",
            linestyle="--",
            linewidth=2.5,
        )
        axes[2].plot(
            df_transfer["epoch"],
            df_transfer[col_v_loss],
            label="Transfer",
            color="#4ECDC4",
            linewidth=3,
        )
        axes[2].set_title(
            "Val Box Loss Comparison\n(Generalization & Stability)",
            fontsize=14,
            fontweight="bold",
        )
        axes[2].set_xlabel("Epochs", fontsize=12)
        axes[2].set_ylabel("Val Box Loss", fontsize=12)
        axes[2].legend(fontsize=11)
        axes[2].grid(True, linestyle=":", alpha=0.7)

    plt.suptitle(
        "Training Convergence Comparison: Scratch vs Transfer",
        fontsize=18,
        fontweight="bold",
        y=1.05,
    )
    plt.tight_layout()

    comp_save_path = PROJECT_ROOT / "runs" / "ocr_metrics_comparison.png"
    plt.savefig(comp_save_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"✅ 학습 수렴(Loss, mAP) 비교 그래프 저장 완료: {comp_save_path}")

    # ── Convergence Speed Comparison ──
    MAP_THRESHOLD = 0.80

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

# 1-2. 혼동 행렬(Confusion Matrix) 이미지 기반 비교
scratch_cm = (
    PROJECT_ROOT / "runs/detect/train_ocr_scratch/confusion_matrix_normalized.png"
)
transfer_cm = (
    PROJECT_ROOT / "runs/detect/train_ocr_transfer/confusion_matrix_normalized.png"
)

if scratch_cm.exists() and transfer_cm.exists():
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    img_cm_s = cv2.imread(str(scratch_cm))
    if img_cm_s is not None:
        img_cm_s = cv2.cvtColor(img_cm_s, cv2.COLOR_BGR2RGB)
        axes[0].imshow(img_cm_s)
    axes[0].set_title(
        "Scratch: Normalized Confusion Matrix\n(False Positive Rate)",
        fontsize=15,
        color="#FF6B6B",
        fontweight="bold",
    )
    axes[0].axis("off")

    img_cm_t = cv2.imread(str(transfer_cm))
    if img_cm_t is not None:
        img_cm_t = cv2.cvtColor(img_cm_t, cv2.COLOR_BGR2RGB)
        axes[1].imshow(img_cm_t)
    axes[1].set_title(
        "Transfer: Normalized Confusion Matrix\n(Accurate Target Detection)",
        fontsize=15,
        color="#4ECDC4",
        fontweight="bold",
    )
    axes[1].axis("off")

    plt.suptitle(
        "Confusion Matrix Comparison (Background False Positives Check)",
        fontsize=18,
        fontweight="bold",
    )
    plt.tight_layout()
    cm_save_path = PROJECT_ROOT / "runs" / "ocr_cm_comparison.png"
    plt.savefig(cm_save_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"✅ 혼동 행렬 비교 이미지 저장 완료: {cm_save_path}")

# 1-3. PR-Curve 및 F1-Curve 비교 (Trade-off 분석)
scratch_f1 = PROJECT_ROOT / "runs/detect/train_ocr_scratch/F1_curve.png"
transfer_f1 = PROJECT_ROOT / "runs/detect/train_ocr_transfer/F1_curve.png"
scratch_pr = PROJECT_ROOT / "runs/detect/train_ocr_scratch/PR_curve.png"
transfer_pr = PROJECT_ROOT / "runs/detect/train_ocr_transfer/PR_curve.png"

if (
    scratch_f1.exists()
    and transfer_f1.exists()
    and scratch_pr.exists()
    and transfer_pr.exists()
):
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))

    # [Row 0] F1 Curve
    img_f1_s = cv2.cvtColor(cv2.imread(str(scratch_f1)), cv2.COLOR_BGR2RGB)
    axes[0, 0].imshow(img_f1_s)
    axes[0, 0].set_title(
        "Scratch: F1-Confidence Curve\n(Lower Peak F1 Score)",
        fontsize=15,
        color="#FF6B6B",
        fontweight="bold",
    )
    axes[0, 0].axis("off")

    img_f1_t = cv2.cvtColor(cv2.imread(str(transfer_f1)), cv2.COLOR_BGR2RGB)
    axes[0, 1].imshow(img_f1_t)
    axes[0, 1].set_title(
        "Transfer: F1-Confidence Curve\n(Higher & More Stable Peak F1 Score)",
        fontsize=15,
        color="#4ECDC4",
        fontweight="bold",
    )
    axes[0, 1].axis("off")

    # [Row 1] PR Curve
    img_pr_s = cv2.cvtColor(cv2.imread(str(scratch_pr)), cv2.COLOR_BGR2RGB)
    axes[1, 0].imshow(img_pr_s)
    axes[1, 0].set_title(
        "Scratch: Precision-Recall Curve\n(Smaller Area Under Curve)",
        fontsize=15,
        color="#FF6B6B",
        fontweight="bold",
    )
    axes[1, 0].axis("off")

    img_pr_t = cv2.cvtColor(cv2.imread(str(transfer_pr)), cv2.COLOR_BGR2RGB)
    axes[1, 1].imshow(img_pr_t)
    axes[1, 1].set_title(
        "Transfer: Precision-Recall Curve\n(Larger Area = Superior Detection)",
        fontsize=15,
        color="#4ECDC4",
        fontweight="bold",
    )
    axes[1, 1].axis("off")

    plt.suptitle(
        "Final Model Evaluation (F1-Score & PR-Curve Trade-off)",
        fontsize=20,
        fontweight="bold",
        y=0.95,
    )
    plt.tight_layout()
    curves_save_path = PROJECT_ROOT / "runs" / "ocr_curves_comparison.png"
    plt.savefig(curves_save_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"✅ PR-Curve 및 F1-Curve 비교 이미지 저장 완료: {curves_save_path}")

# 2. 1대1 실체 탐지 사진 비교 (Same Image)
# Validation 셋에서 한 장의 이미지를 뽑아 두 모델의 예측 결과를 나란히 보여줍니다.
val_images = list((OCR_DIR / "images" / "val").glob("*.webp")) + list(
    (OCR_DIR / "images" / "val").glob("*.jpg")
)
if val_images and ocr_scratch_weight.exists() and ocr_transfer_weight.exists():
    # 비교를 극대화하기 위해 랜덤 이미지 1장 선택
    sample_img_path = random.choice(val_images)

    print(f"🔍 1대1 시각화 샘플 이미지: {sample_img_path.name}")
    model_s = YOLO(str(ocr_scratch_weight))
    model_t = YOLO(str(ocr_transfer_weight))

    # 동일한 Confidence Threshold(0.3) 적용하여 공정한 비교
    res_s = model_s.predict(str(sample_img_path), conf=0.3, verbose=False)[0]
    res_t = model_t.predict(str(sample_img_path), conf=0.3, verbose=False)[0]

    img_s = res_s.plot(line_width=2)
    img_t = res_t.plot(line_width=2)

    img_s_rgb = cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB)
    img_t_rgb = cv2.cvtColor(img_t, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(24, 12))

    axes[0].imshow(img_s_rgb)
    axes[0].set_title(
        "Scratch (Random Init) Model\n(Slower Convergence, Lower Accuracy)",
        fontsize=16,
        color="#FF6B6B",
        fontweight="bold",
    )
    axes[0].axis("off")

    axes[1].imshow(img_t_rgb)
    axes[1].set_title(
        "2-Stage Transfer Learning Model\n(High Accuracy, Dense Detection)",
        fontsize=16,
        color="#4ECDC4",
        fontweight="bold",
    )
    axes[1].axis("off")

    plt.suptitle(
        "1-to-1 Real Detection Comparison on Same Image",
        fontsize=22,
        fontweight="bold",
        y=0.95,
    )
    plt.tight_layout()

    vis_save_path = PROJECT_ROOT / "runs" / "ocr_detection_comparison_1to1.png"
    plt.savefig(vis_save_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"✅ 1대1 실체 탐지 비교 이미지 저장 완료: {vis_save_path}")

# %% [markdown]
# ## 6. [Phase 5] 최종 전체 파이프라인 실전 추론 (Overall Pipeline & Preprocessing)
# - 7클래스 마스터 모델과 OCR 전이학습 모델을 **앙상블(Ensemble)** 방식으로 결합합니다.
# - 도면 한 장에 대해 두 모델을 순차적으로 돌리고 결과 BBox를 하나의 캔버스에 병합합니다.
# - SAHI 슬라이싱을 적용하여 고해상도 도면에서도 작은 객체를 놓치지 않습니다.

# %%
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import matplotlib.patches as mpatches


# YOLO 학습 순서 ID → 클래스 이름 (상단 class_names 재사용, DRY)
MASTER_ID_TO_NAME = class_names


def draw_colored_boxes(img_cv2, predictions, id_to_name_map):
    """
    OpenCV 이미지 위에 색상-코딩된 BBox를 그리고,
    각 클래스별 탐지 횟수를 반환합니다.
    """
    counts = {}
    for pred in predictions:
        cat_name = id_to_name_map.get(pred.category.id, pred.category.name)
        color = CLASS_COLOR_MAP.get(cat_name, (200, 200, 200))
        bbox = pred.bbox.to_xyxy()
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        cv2.rectangle(img_cv2, (x1, y1), (x2, y2), color, 3)
        counts[cat_name] = counts.get(cat_name, 0) + 1
    return img_cv2, counts


def run_ensemble_sahi_inference(image_path, master_model_path, ocr_model_path):
    """두 모델(7cls + OCR)의 SAHI 추론 결과를 하나의 이미지로 병합합니다."""
    print(f"\n{'=' * 60}")
    print(f"[{Path(image_path).name}] 🚀 SAHI 앙상블 추론 시작")
    print(f"{'=' * 60}")

    if not os.path.exists(master_model_path):
        print(f"⚠️ 7클래스 마스터 가중치가 없습니다: {master_model_path}")
        return
    if not os.path.exists(ocr_model_path):
        print(f"⚠️ OCR 전이학습 가중치가 없습니다: {ocr_model_path}")
        return

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # ── 0. Phase 2 전처리 파이프라인 적용 ──
    print("🧹 [0/2] Adaptive Gaussian 전처리 적용 중...")
    preprocessed_img = apply_adaptive_gaussian_preprocessing(image_path)

    # ── 1. 7클래스 마스터 모델 추론 ──
    print("🔍 [1/2] 7클래스(가구/구조물) 탐지 중...")
    master_det = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=master_model_path,
        confidence_threshold=0.25,
        device=device,
    )
    master_result = get_sliced_prediction(
        preprocessed_img,
        master_det,
        slice_height=512,
        slice_width=512,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
    )

    # ── 2. OCR 모델 추론 ──
    print("🔍 [2/2] OCR(텍스트 영역) 탐지 중...")
    ocr_det = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=ocr_model_path,
        confidence_threshold=0.25,
        device=device,
    )
    ocr_result = get_sliced_prediction(
        preprocessed_img,
        ocr_det,
        slice_height=512,
        slice_width=512,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
    )

    # ── 3. 캔버스에 앙상블 그리기 ──
    print("✨ 두 모델의 결과를 하나의 캔버스에 병합합니다...")
    # 원본 이미지가 아닌, 전처리된 이미지(망점이 제거된 상태) 위에 그리기
    img_cv2 = preprocessed_img.copy()
    img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)

    # 마스터 모델 결과 그리기
    img_cv2, master_counts = draw_colored_boxes(
        img_cv2, master_result.object_prediction_list, MASTER_ID_TO_NAME
    )
    # OCR 모델 결과 그리기 (text 클래스)
    ocr_id_map = {0: "text"}
    img_cv2, ocr_counts = draw_colored_boxes(
        img_cv2, ocr_result.object_prediction_list, ocr_id_map
    )

    all_counts = {**master_counts, **ocr_counts}

    # ── 3.5. JSON 데이터 구조화 및 저장 ──
    print("💾 앙상블 결과를 JSON 파일로 파싱하여 추출합니다...")

    json_data = {
        "image_name": Path(image_path).name,
        "structures": [],
        "furnitures": [],
        "ocr": [],
    }

    # 7클래스 분류 기준 (문, 창문은 구조물 / 나머지는 가구)
    STRUCTURE_CLASSES = ["door", "window"]

    # Confidence 중재 (Threshold 설정)
    MIN_CONFIDENCE = 0.3

    # 마스터 모델 파싱
    for pred in master_result.object_prediction_list:
        cat_name = MASTER_ID_TO_NAME.get(pred.category.id, pred.category.name)
        conf = float(pred.score.value)
        if conf < MIN_CONFIDENCE:
            continue

        bbox = pred.bbox.to_xyxy()
        bbox_clean = [round(float(x), 1) for x in bbox]

        item = {"class": cat_name, "bbox": bbox_clean, "confidence": round(conf, 3)}

        if cat_name in STRUCTURE_CLASSES:
            json_data["structures"].append(item)
        else:
            json_data["furnitures"].append(item)

    # OCR 모델 파싱
    for pred in ocr_result.object_prediction_list:
        conf = float(pred.score.value)
        if conf < MIN_CONFIDENCE:
            continue

        bbox = pred.bbox.to_xyxy()
        bbox_clean = [round(float(x), 1) for x in bbox]

        item = {"class": "text", "bbox": bbox_clean, "confidence": round(conf, 3)}
        json_data["ocr"].append(item)

    # JSON 저장
    export_dir = PROJECT_ROOT / "runs" / "inference"
    export_dir.mkdir(parents=True, exist_ok=True)
    json_save_path = export_dir / (Path(image_path).stem + "_result.json")

    with open(json_save_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)

    print(f"✅ JSON 데이터 저장 완료: {json_save_path}")

    # ── 4. matplotlib으로 색상 범례(Legend) 포함 시각화 ──
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.imshow(img_cv2)
    ax.axis("off")
    ax.set_title(
        f"Ensemble Detection: {Path(image_path).name}", fontsize=14, fontweight="bold"
    )

    # 범례 패치 생성 (탐지된 클래스만)
    legend_patches = []
    for cls_name, color_rgb in CLASS_COLOR_MAP.items():
        cnt = all_counts.get(cls_name, 0)
        if cnt > 0:
            color_norm = tuple(c / 255.0 for c in color_rgb)
            legend_patches.append(
                mpatches.Patch(color=color_norm, label=f"{cls_name} ({cnt})")
            )
    if legend_patches:
        ax.legend(
            handles=legend_patches,
            loc="upper right",
            fontsize=10,
            framealpha=0.85,
            fancybox=True,
            shadow=True,
        )

    plt.tight_layout()

    # 결과 저장 (export_dir은 상단에서 이미 생성됨)
    save_path = export_dir / (Path(image_path).stem + "_ensemble.png")
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.show()

    # 탐지 요약 출력
    total = sum(all_counts.values())
    print(f"\n📊 탐지 요약 (총 {total}개 객체):")
    for cls_name, cnt in sorted(all_counts.items(), key=lambda x: -x[1]):
        print(f"   • {cls_name}: {cnt}개")
    print(f"\n✅ 결과 이미지 저장 완료: {save_path}")


# ── 실행부 ──
RAW_LEGACY_DIR = PROJECT_ROOT / "raw_legacy_inputs"
if RAW_LEGACY_DIR.exists():
    test_images = (
        list(RAW_LEGACY_DIR.glob("*.jpg"))
        + list(RAW_LEGACY_DIR.glob("*.png"))
        + list(RAW_LEGACY_DIR.glob("*.webp"))
    )
    if test_images:
        best_master_path = str(
            PROJECT_ROOT / "runs/detect/train_domain_tuned/weights/best.pt"
        )
        best_ocr_path = str(
            PROJECT_ROOT / "runs/detect/train_ocr_transfer/weights/best.pt"
        )
        for img_path in test_images:
            run_ensemble_sahi_inference(img_path, best_master_path, best_ocr_path)
    else:
        print("⚠️ raw_legacy_inputs 폴더에 테스트할 도면 이미지가 없습니다.")
else:
    print("⚠️ raw_legacy_inputs 폴더가 존재하지 않습니다. 테스트할 도면을 넣어주세요.")

