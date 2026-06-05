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
else:
    print("GPU 없음 → CPU 모드로 학습 (느림)")

print("Libraries imported successfully.")

# %% [markdown]
# ## 0-1. ⚙️ 하드웨어 프로필 (Hardware-Aware Profile) 설정
# - RTX 5060 등 개인용 GPU와 L40S Pro 등 서버용 고성능 GPU의 스펙 차이가 큽니다.
# - 고성능 GPU에서 VRAM 병목(점유율 60% 등)을 해결하고 100% 성능을 끌어내기 위해 **모드 변경 단 한 줄**로 모델 크기와 배치 사이즈를 자동 튜닝합니다.

# %%
# "consumer" : RTX 3060/4060/5060 등 8GB~12GB VRAM 용 (안정성 위주)
# "pro"      : L40S, A100 등 24GB~48GB VRAM 용 (최고 속도 및 성능 펌핑)

HARDWARE_PROFILE = "pro"  # <--- 여기서 모드만 바꾸세요!

if HARDWARE_PROFILE == "pro":
    BASE_WEIGHT = "yolov8m.pt" # (Medium 모델) Nano 대비 압도적 성능, 페이즈 2~4 전체 적용
    # VRAM 50GB / RAM 120GB 한도를 고려한 '절대 안 터지는(Anti-OOM)' 최고 효율 세팅
    BATCH_SIZE = 32            # 64는 간혹 피크치에서 튈 수 있으므로 32로 안정성 확보 (VRAM 약 12~15GB 소모 추정)
    WORKERS = 8                # RAM(공유 메모리) 폭발을 막기 위해 16 대신 8로 타협
    print(f"🚀 [PRO 모드 활성화] {BASE_WEIGHT} 가중치 / Batch={BATCH_SIZE} / Workers={WORKERS} (안정성+고속 세팅 완료!)")
else:
    BASE_WEIGHT = "yolov8n.pt" # (Nano 모델) VRAM 절약 및 빠른 학습
    BATCH_SIZE = 16
    WORKERS = 4
    print(f"💻 [CONSUMER 모드 활성화] {BASE_WEIGHT} 가중치 및 Batch={BATCH_SIZE}로 안정성 세팅 완료!")

# %% [markdown]
# ## 1. 7클래스 마스터 데이터셋 연동 및 압축 해제

# %%
# ──────────────────────────────────────────────
# 데이터셋 경로 (8:1:1로 이미 전처리된 마스터 데이터셋 사용)
# ──────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

ZIP_NAME = "master_dataset.tar.zst"
ZIP_PATH = DATA_DIR / ZIP_NAME
MASTER_DATASET_DIR = DATA_DIR / "master_dataset"

# 사용자 지정 구글 드라이브 링크 (1500장 추출 마스터 데이터셋)
GDRIVE_FILE_ID = "1lr_ELPmHK-00T1KhVUhjPULcHV4TL8Ms"

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
            subprocess.run(["tar", "-I", "zstd", "-xf", str(ZIP_PATH), "-C", str(DATA_DIR)], check=True)
        except Exception as e:
            print("zstd 패키지가 없거나 권한 오류 발생. zstd 설치 시도 중...")
            if IS_COLAB:
                subprocess.run(["apt-get", "update"], check=True)
                subprocess.run(["apt-get", "install", "-y", "zstd"], check=True)
                subprocess.run(["tar", "-I", "zstd", "-xf", str(ZIP_PATH), "-C", str(DATA_DIR)], check=True)
            else:
                raise RuntimeError("압축 해제 실패: 터미널에서 'sudo apt install zstd' 를 실행 후 다시 시도하세요.")
        print("압축 해제 완료!")
    else:
        raise FileNotFoundError(f"⚠️ {ZIP_PATH} 파일을 확보하지 못했습니다.")
else:
    print(f"Dataset ready at {MASTER_DATASET_DIR}")

# %% [markdown]
# ## 2. YOLO dataset.yaml 자동 생성 (7클래스)

# %%
yaml_path = MASTER_DATASET_DIR / "dataset.yaml"

class_names = {
    0: "toilet",
    1: "washbasin",
    2: "sink",
    3: "bathtub",
    4: "gas_stove",
    5: "door",
    6: "window"
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
# ## 3. 탐색적 데이터 분석 (EDA) 및 시각화 검증

# %%
def visualize_yolo_labels(image_path, label_path, class_names):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size
    colors = ["red", "blue", "green", "orange", "purple", "cyan", "magenta"]

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
        draw.text((x_min + 5, y_min - 25), class_names.get(class_id, str(class_id)), fill=color)

    return img

train_img_dir = MASTER_DATASET_DIR / "images" / "train"
train_lbl_dir = MASTER_DATASET_DIR / "labels" / "train"

if train_img_dir.exists():
    img_files = list(train_img_dir.glob("*.webp"))
    if img_files:
        sample_img = random.choice(img_files)
        sample_lbl = train_lbl_dir / (sample_img.stem + ".txt")

        print(f"Visualizing Sample: {sample_img.name}")
        result_img = visualize_yolo_labels(sample_img, sample_lbl, class_names)

        plt.figure(figsize=(12, 8))
        plt.imshow(result_img)
        plt.axis("off")
        plt.show()

# %% [markdown]
# ## 4. [Phase 1] 최적의 데이터 개수 탐색 (Data Scaling Ablation)
# - 학습 데이터 개수를 300장부터 1200장(최대치)까지 점진적으로 늘려보며 성능 향상폭이 꺾이는 지점(Saturation Point)을 찾습니다.

# %%
print("=" * 60)
print("🔬 [Phase 1] Data Size Ablation")
print("=" * 60)

train_images = list((MASTER_DATASET_DIR / "images" / "train").glob("*.webp"))
# 총 1200장의 훈련셋을 활용하여 점진적 크기 실험
data_sizes = [300, 600, 900, 1200]
scaling_results = {}

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

    print(f"\n🚀 Data Size: {size} 학습 시작 (30 Epochs 검증)")
    model_size = YOLO(BASE_WEIGHT)
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
        patience=10, # 과적합 방지 조기 종료
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
        plt.annotate(f"{txt:.3f}", (sizes[i], maps[i]), textcoords="offset points", xytext=(0, 10), ha="center")
    plt.show()

# %% [markdown]
# ## 5. [Phase 2] 도메인 맞춤형 증강 탐색 (Baseline vs Augmented)
# - 최적의 데이터 개수(최대치 1200장)를 고정하고, 원본 데이터의 한계를 돌파할 도메인 맞춤형 증강 기법을 비교합니다.

# %%
print("=" * 60)
print("🔬 [Phase 2] Baseline vs Augmented 모델 학습 (7클래스 마스터 모델)")
print("=" * 60)

# 1. Baseline (순정 학습)
print("🚀 Baseline 모델 학습 시작...")
model_baseline = YOLO(BASE_WEIGHT)
results_baseline = model_baseline.train(
    data=str(MASTER_DATASET_DIR / "dataset.yaml"),
    epochs=150, # 7클래스 난이도 상승 반영
    imgsz=640,
    batch=BATCH_SIZE,
    workers=WORKERS,
    cache=True,
    project=str(PROJECT_ROOT / "runs/detect"),
    name="train_baseline",
    patience=20, # 성능 개선 없으면 20에폭 후 조기 종료
)
torch.cuda.empty_cache()
gc.collect()

# 2. Augmented (도메인 맞춤형 증강 학습)
print("\n🚀 Augmented 모델 학습 시작...")
model_augmented = YOLO(BASE_WEIGHT)
results_augmented = model_augmented.train(
    data=str(MASTER_DATASET_DIR / "dataset.yaml"),
    epochs=150, # 7클래스 난이도 상승 반영
    imgsz=640,
    batch=BATCH_SIZE,
    workers=WORKERS,
    cache=True,
    project=str(PROJECT_ROOT / "runs/detect"),
    name="train_augmented",
    patience=20, # 성능 개선 없으면 20에폭 후 조기 종료
    # 도메인 갭(Domain Gap) 극복을 위한 파라미터 (레거시 도면 타겟팅)
    degrees=2.0,       # 스캐너 수작업 오차
    hsv_s=0.2, hsv_v=0.2, # 황변 현상 및 퇴색
    perspective=0.0005,
    scale=0.5,
    mosaic=1.0,
)
torch.cuda.empty_cache()
gc.collect()

# %% [markdown]
# ## 6. [Phase 2-B] Simulated Legacy Test Set 기반 재평가
# - 기존 val/test 셋은 깨끗한 도면이므로, 인위적 노이즈를 추가해 레거시 환경을 모사한 후 Augmented 모델의 실전 우위를 증명합니다.

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

legacy_test_img_dir = MASTER_DATASET_DIR / "images" / "legacy_test"
legacy_test_lbl_dir = MASTER_DATASET_DIR / "labels" / "legacy_test"
legacy_test_img_dir.mkdir(parents=True, exist_ok=True)
legacy_test_lbl_dir.mkdir(parents=True, exist_ok=True)

test_images = list((MASTER_DATASET_DIR / "images" / "test").glob("*.webp"))
for img_path in test_images:
    img = cv2.imread(str(img_path))
    if img is None: continue
    degraded = simulate_legacy_degradation(img)
    cv2.imwrite(str(legacy_test_img_dir / (img_path.stem + ".jpg")), degraded)
    lbl_src = MASTER_DATASET_DIR / "labels" / "test" / (img_path.stem + ".txt")
    if lbl_src.exists():
        shutil.copy(lbl_src, legacy_test_lbl_dir / lbl_src.name)

legacy_yaml_path = MASTER_DATASET_DIR / "dataset_legacy_test.yaml"
with open(legacy_yaml_path, "w", encoding="utf-8") as f:
    yaml.dump({
        "path": str(MASTER_DATASET_DIR.resolve()),
        "train": "images/train",
        "val": "images/legacy_test",
        "names": class_names,
    }, f, allow_unicode=True)

print("\n📊 Augmented 모델 → Legacy Test Set 평가")
aug_weight = PROJECT_ROOT / "runs/detect/train_augmented/weights/best.pt"
if aug_weight.exists():
    model_aug_eval = YOLO(str(aug_weight))
    legacy_aug_metrics = model_aug_eval.val(data=str(legacy_yaml_path))

# %% [markdown]
# ## 7. [Phase 3] OCR 전이학습 (Transfer Learning)
# - **아키텍처 확정**: 텍스트 인식은 배제하고, YOLO를 이용해 **글자가 위치한 구역(BBox)만 빠르게 탐지**하는 모델을 추가합니다.
# - 7클래스 마스터 모델이 학습한 도면 특성 가중치(`best.pt`)를 넘겨받아 전이학습을 수행하여 극강의 수렴 속도와 정확도를 달성합니다.

# %%
print("=" * 60)
print("🚀 [Phase 3] OCR 텍스트 탐지용 전이학습 (Transfer Learning) 시작")
print("=" * 60)

# (주의: 사용자가 별도의 OCR 데이터셋을 ocr_dataset 폴더에 준비해두었다고 가정합니다.)
# ocr_dataset.yaml은 클래스 0 (text) 단일 구조를 가져야 합니다.
OCR_DIR = DATA_DIR / "ocr_dataset"
ocr_yaml_path = OCR_DIR / "dataset.yaml"

if not ocr_yaml_path.exists():
    print("⚠️ OCR 데이터셋 폴더(ocr_dataset)를 찾을 수 없습니다.")
    print("AI 허브의 TL_OCR 데이터를 다운받아 YOLO 형식(class: 0 text)으로 변환 후 재시도해주세요.")
else:
    # 7클래스 마스터 모델의 황금 가중치 로드
    best_weight_path = PROJECT_ROOT / "runs/detect/train_augmented/weights/best.pt"
    
    if not best_weight_path.exists():
        raise FileNotFoundError(f"⚠️ 전이학습을 위한 가중치({best_weight_path})가 없습니다. Phase 2 학습을 먼저 완료하세요.")

    print(f"✅ {best_weight_path.name} 가중치를 성공적으로 불러왔습니다. 전이학습을 시작합니다!")
    
    # 전이학습(Transfer Learning)
    model_transfer = YOLO(str(best_weight_path))
    results_transfer = model_transfer.train(
        data=str(ocr_yaml_path),
        epochs=50,      # 클래스가 1개지만 견고한 학습을 위해 에폭 상향
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

# %% [markdown]
# ## 8. [Phase 4] 실전 도면 추론 파이프라인 (Inference)
# - SAHI 슬라이싱을 이용해 구형 레거시 도면에서 놓치는 객체 없이 꼼꼼하게 검출합니다.

# %%
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

def run_sahi_inference(image_path, model_path):
    print(f"[{Path(image_path).name}] SAHI 추론 시작...")
    
    if not os.path.exists(model_path):
        print(f"⚠️ 모델 가중치가 없습니다: {model_path}")
        return
        
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=model_path,
        confidence_threshold=0.25,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    
    # 카테고리 매핑 강제 변환 (한글/깨짐 방지)
    detection_model.category_mapping = {str(k): str(k) for k in detection_model.category_mapping.keys()}
    
    result = get_sliced_prediction(
        str(image_path),
        detection_model,
        slice_height=512,
        slice_width=512,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
    )
    
    # 시각화
    plt.figure(figsize=(10, 10))
    plt.imshow(result.image)
    plt.axis("off")
    plt.show()

# 사용자가 raw_legacy_inputs에 넣은 실제 도면 중 하나로 테스트
RAW_LEGACY_DIR = PROJECT_ROOT / "raw_legacy_inputs"
if RAW_LEGACY_DIR.exists():
    test_images = list(RAW_LEGACY_DIR.glob("*.jpg")) + list(RAW_LEGACY_DIR.glob("*.png"))
    if test_images:
        best_model_path = str(PROJECT_ROOT / "runs/detect/train_augmented/weights/best.pt")
        run_sahi_inference(test_images[0], best_model_path)
