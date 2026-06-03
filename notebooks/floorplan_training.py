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
# > 현재 보고 계시는 `.py` 파일은 로컬 IDE에서의 편집 및 Jupytext 동기화를 위한 샘플(Sample) 스크립트입니다. 
# > 실제 전체 워크플로우는 **Google Colab의 `.ipynb` 환경에서 원스톱(One-go)으로 한 번에 다 돌아가도록 완벽하게 통합**되어 있습니다.

# %% [markdown]
# ## 0. 개발 환경 설정 및 라이브러리 설치

# %%
# YOLOv8 및 의존성 라이브러리 설치
# %pip install -q ultralytics matplotlib numpy pillow albumentations pyyaml

# %%
import sys
import os
import json
import shutil
import yaml
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from pathlib import Path
from ultralytics import YOLO
import zipfile
import subprocess

try:
    from google.colab import drive  # type: ignore
except ImportError:
    pass

# 환경 감지 및 프로젝트 루트 정의
current_dir = Path(os.getcwd())
project_root = current_dir.parent if current_dir.name == 'notebooks' else current_dir
IS_COLAB = 'google.colab' in sys.modules or os.path.exists('/content')

# 구글 드라이브 마운트 실행
if IS_COLAB:
    try:
        drive.mount('/content/drive')
    except Exception as e:
        raise RuntimeError(f"Google Drive 마운트 실패. 워크플로우를 중단합니다: {e}")
else:
    raise EnvironmentError("본 코드는 Google Colab 환경 전용입니다. 로컬 환경 실행을 차단하고 전체 중단합니다.")

print("Libraries imported successfully.")

# %% [markdown]
# ## 1. 데이터셋 연동 및 압축 해제

# %%
# 데이터셋 경로 정의
if not IS_COLAB:
    raise EnvironmentError("Colab 환경이 아닙니다. 실행을 중단합니다.")

ZIP_PATH = Path("/content/drive/MyDrive/architectural_drawing_data.7z")
EXTRACT_PATH = Path("/content/architectural_drawing_data")

# 데이터셋 압축 해제 (7z 방식, 드라이브 직접 마운트)
if not EXTRACT_PATH.exists():
    if ZIP_PATH.exists():
        print(f"Extracting {ZIP_PATH.name} from Google Drive...")
        try:
            subprocess.run(["7z", "x", str(ZIP_PATH), f"-o{EXTRACT_PATH.parent}", "-y"], check=True)
            print("Extraction complete.")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"압축 해제 실패. 프로세스를 중단합니다: {e}")
    else:
        raise FileNotFoundError(f"⚠️ 에러: {ZIP_PATH} 파일을 찾을 수 없습니다. 구글 드라이브 마운트 상태 및 파일 존재 여부를 확인하세요.")
else:
    print(f"Dataset ready at {EXTRACT_PATH}")

# %% [markdown]
# ## 2. JSON 라벨 전처리 및 YOLO 포맷 변환 (Train/Val/Test)

# %%
YOLO_DIR = EXTRACT_PATH / "yolo_dataset"
CLASS_MAPPING = {4: 0, 5: 1, 6: 2, 7: 3, 8: 4}

if not YOLO_DIR.exists():
    print("YOLO 데이터셋 포맷팅을 시작합니다...")
    
    # === [기존 원본 데이터셋 처리 코드] (추후 전체 데이터 사용 시 아래 주석 해제 후 간이 코드 주석 처리) ===
    """
    for split in ['train', 'val', 'test']:
        dest_img_dir = YOLO_DIR / "images" / split
        dest_lbl_dir = YOLO_DIR / "labels" / split
        dest_img_dir.mkdir(parents=True, exist_ok=True)
        dest_lbl_dir.mkdir(parents=True, exist_ok=True)
        
        src_img_dir = EXTRACT_PATH / "object_layout" / split / "images"
        src_lbl_dir = EXTRACT_PATH / "object_layout" / split / "labels"
            
        json_files = list(src_lbl_dir.glob("*.json"))
        converted_count = 0
        
        for json_file in json_files:
            base_name = json_file.stem
            img_file = src_img_dir / f"{base_name}.webp"
            
            if not img_file.exists():
                continue
                
            with open(json_file, 'r', encoding='utf-8') as f:
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
                        yolo_lines.append(f"{CLASS_MAPPING[cat_id]} {x_center_norm:.6f} {y_center_norm:.6f} {w_norm:.6f} {h_norm:.6f}")
            
            if yolo_lines:
                dest_txt = dest_lbl_dir / f"{base_name}.txt"
                with open(dest_txt, 'w', encoding='utf-8') as f:
                    f.write("\n".join(yolo_lines))
                shutil.copy(img_file, dest_img_dir / img_file.name)
                converted_count += 1
                
        print(f"[{split.upper()} 데이터 변환 완료] {converted_count}장")
    """

    # === [간이 데이터셋 처리 코드] 병목 방지를 위해 val 세트 중 일부만 사용 ===
    for split in ['train', 'val', 'test']:
        dest_img_dir = YOLO_DIR / "images" / split
        dest_lbl_dir = YOLO_DIR / "labels" / split
        dest_img_dir.mkdir(parents=True, exist_ok=True)
        dest_lbl_dir.mkdir(parents=True, exist_ok=True)
        
        # train, val, test 모두 'val' 원본 폴더에서 데이터를 끌어다 씀 (간소화)
        src_img_dir = EXTRACT_PATH / "object_layout" / "val" / "images"
        src_lbl_dir = EXTRACT_PATH / "object_layout" / "val" / "labels"
            
        # val 세트 전체 사용 (전체 데이터의 약 1% 수준)
        json_files = list(src_lbl_dir.glob("*.json"))
        converted_count = 0
        
        for json_file in json_files:
            base_name = json_file.stem
            img_file = src_img_dir / f"{base_name}.webp"
            
            if not img_file.exists():
                continue
                
            with open(json_file, 'r', encoding='utf-8') as f:
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
                        yolo_lines.append(f"{CLASS_MAPPING[cat_id]} {x_center_norm:.6f} {y_center_norm:.6f} {w_norm:.6f} {h_norm:.6f}")
            
            if yolo_lines:
                dest_txt = dest_lbl_dir / f"{base_name}.txt"
                with open(dest_txt, 'w', encoding='utf-8') as f:
                    f.write("\n".join(yolo_lines))
                shutil.copy(img_file, dest_img_dir / img_file.name)
                converted_count += 1
                
        print(f"[{split.upper()} 간이 데이터 변환 완료] {converted_count}장 (val 데이터 기반)")
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
    "names": {
        0: '변기',
        1: '세면대',
        2: '싱크대',
        3: '욕조',
        4: '가스레인지'
    }
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
        draw.text((x_min + 5, y_min - 25), class_names.get(class_id, str(class_id)), fill=color)
        
    return img

train_img_dir = YOLO_DIR / "images" / "train"
train_lbl_dir = YOLO_DIR / "labels" / "train"

if train_img_dir.exists():
    img_files = list(train_img_dir.glob("*.webp"))
    if img_files:
        sample_img = random.choice(img_files)
        sample_lbl = train_lbl_dir / (sample_img.stem + ".txt")
        
        print(f"Visualizing Sample: {sample_img.name}")
        result_img = visualize_yolo_labels(sample_img, sample_lbl, class_names)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(result_img)
        plt.axis('off')
        plt.show()

# %% [markdown]
# ## 4. 데이터 증강 및 학습 진행

# %%
model = YOLO("yolov8n.pt")

results = model.train(
    data=str(yaml_path),
    epochs=50,
    imgsz=640,
    batch=8,
    device=0,
    workers=2,
    degrees=15.0,
    perspective=0.0005,
    scale=0.5,
    blur=0.01,
    mosaic=1.0,
    val=True
)

# %% [markdown]
# ## 5. 정량적 모델 성능 평가 (Evaluation)

# %%
train_results_dir = Path("runs/detect/train")

if train_results_dir.exists():
    results_png = train_results_dir / "results.png"
    if results_png.exists():
        plt.figure(figsize=(15, 10))
        plt.imshow(Image.open(results_png))
        plt.axis('off')
        plt.title("YOLOv8 Training Metrics History", fontsize=16)
        plt.show()
        
    confusion_matrix_png = train_results_dir / "confusion_matrix.png"
    if confusion_matrix_png.exists():
        plt.figure(figsize=(12, 10))
        plt.imshow(Image.open(confusion_matrix_png))
        plt.axis('off')
        plt.title("Confusion Matrix", fontsize=16)
        plt.show()

# %% [markdown]
# ## 6. 실전 추론 및 예측 결과 검증 (Test Set Inference)

# %%
# 테스트(Test) 도면 무작위 선정 및 모델 추론
test_img_dir = YOLO_DIR / "images" / "test"
test_lbl_dir = YOLO_DIR / "labels" / "test"

if test_img_dir.exists():
    test_images = list(test_img_dir.glob("*.webp"))
    if test_images:
        test_image = random.choice(test_images)
        
        # 1. 모델 예측
        predict_results = model.predict(source=test_image, conf=0.25, save=False)
        predicted_plot = predict_results[0].plot()
        
        # 2. 정답 라벨 시각화
        gt_lbl_path = test_lbl_dir / (test_image.stem + ".txt")
        gt_image = visualize_yolo_labels(test_image, gt_lbl_path, class_names)
        
        # 3. 비교 시각화
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        axes[0].imshow(gt_image)
        axes[0].set_title("Ground Truth (Test Set)", fontsize=15)
        axes[0].axis('off')
        
        axes[1].imshow(predicted_plot)
        axes[1].set_title("AI Prediction", fontsize=15)
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
