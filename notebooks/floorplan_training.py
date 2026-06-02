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
# ---

# %% [markdown]
# ## 0. 개발 환경 설정 및 라이브러리 설치
# 구글 코랩 환경에서 컴퓨터 비전 라이브러리 `ultralytics` 및 필요 라이브러리 설치/임포트.

# %%
# YOLOv8 및 의존성 라이브러리 설치
# %pip install -q ultralytics matplotlib numpy pillow albumentations pyyaml

# %%
import sys
import os
import yaml
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from pathlib import Path
from ultralytics import YOLO

# 커널 실행 위치에 상관없이 프로젝트 루트 디렉토리를 모듈 검색 경로에 추가
current_dir = Path(os.getcwd())
project_root = current_dir.parent if current_dir.name == 'notebooks' else current_dir
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.core import yolo_core

print("Libraries imported successfully.")

# %% [markdown]
# ## 1. 데이터셋 연동 및 압축 해제
# 구글 드라이브 업로드 `floorplan_dataset_yolo.zip` 코랩 로컬 런타임 압축 해제.
#
# > **팁**: 구글 드라이브 연동 시 폴더 아이콘 -> 드라이브 마운트 활용.

# %%
# 구글 드라이브 마운트 실행 (선택 사항)
# from google.colab import drive
# drive.mount('/content/drive')

# 코랩 로컬 드라이브 직접 업로드 기준 압축 해제.
# 구글드라이브 업로드 시 경로 수정 요망 (/content/drive/MyDrive/... 등)
ZIP_PATH = "/content/floorplan_dataset_yolo.zip"
EXTRACT_PATH = "/content/yolo_dataset"

if os.path.exists(ZIP_PATH):
    # !unzip -q {ZIP_PATH} -d {EXTRACT_PATH}
    print("Dataset unzipped successfully at /content/yolo_dataset")
else:
    print(f"Error: Zip file not found at {ZIP_PATH}. Please upload the zip file to the Colab files section.")

# %% [markdown]
# ## 2. 탐색적 데이터 분석 (EDA) 및 시각화 검증
# 학습 전, 전처리 데이터 바운딩 박스 좌표 시각적 검증.
# YOLO 포맷 라벨 파일(`[class_id, x_center, y_center, width, height]` 정규화 값) 분석 및 이미지 위 바운딩 박스 투영.

# %%
# 클래스 이름 매핑 정보 로드
yaml_path = Path("/content/yolo_dataset/dataset.yaml")
if yaml_path.exists():
    with open(yaml_path, "r", encoding="utf-8") as f:
        dataset_info = yaml.safe_load(f)
    class_names = dataset_info.get("names", {})
    print("Loaded Class Names:", class_names)
else:
    class_names = {0: '변기', 1: '세면대', 2: '싱크대', 3: '욕조', 4: '가스레인지'}
    print("Using default class mapping.")


# %%
def visualize_yolo_labels(image_path, label_path, class_names):
    """YOLO 이미지 및 라벨 시각화 함수"""
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    w, h = img.size
    
    # 바운딩 박스 색상 정의 (클래스별 다른 색상)
    colors = ["red", "blue", "green", "orange", "purple"]
    
    if not os.path.exists(label_path):
        print(f"No labels found for {image_path.name}")
        return img
        
    with open(label_path, "r") as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        xc, yc, bw, bh = map(float, parts[1:])
        
        # 정규화 좌표를 픽셀 좌표로 복원
        x_center = xc * w
        y_center = yc * h
        box_w = bw * w
        box_h = bh * h
        
        x_min = int(x_center - box_w / 2)
        y_min = int(y_center - box_h / 2)
        x_max = int(x_center + box_w / 2)
        y_max = int(y_center + box_h / 2)
        
        # 바운딩 박스 그리기
        color = colors[class_id % len(colors)]
        draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=6)
        
        # 텍스트 라벨 추가
        label_text = class_names.get(class_id, str(class_id))
        draw.text((x_min + 5, y_min - 25), label_text, fill=color)
        
    return img

# 무작위 하나 학습 이미지 선정, 라벨링 검증 시각화
train_img_dir = Path("/content/yolo_dataset/images/train")
train_lbl_dir = Path("/content/yolo_dataset/labels/train")

if train_img_dir.exists():
    img_files = list(train_img_dir.glob("*.PNG")) + list(train_img_dir.glob("*.png"))
    if img_files:
        sample_img = random.choice(img_files)
        sample_lbl = train_lbl_dir / (sample_img.stem + ".txt")
        
        print(f"Visualizing Sample: {sample_img.name}")
        result_img = visualize_yolo_labels(sample_img, sample_lbl, class_names)
        
        # 적절한 크기 축소 시각화
        plt.figure(figsize=(12, 8))
        plt.imshow(result_img)
        plt.axis('off')
        plt.show()
    else:
        print("No images found. Please check dataset extraction.")

# %% [markdown]
# ## 3. 데이터 증강(Data Augmentation) 및 학습 설계
#
# ### **도메인 갭(Domain Gap) 극복 전략**
# 학습용 도면은 깨끗함. 최종 목표인 **'오래된 도면'**은 스캔 왜곡, 번짐, 각도 비뚤어짐 등 다양한 노이즈 존재.
# YOLOv8 모델 하이퍼파라미터 조정 통한 강력한 **실시간 데이터 증강(On-the-fly Data Augmentation)** 주입. 강건한(Robust) 일반화 모델 설계.
#
# - `degrees=15.0`: 무작위 회전 주입. 비뚤어진 스캔 방지.
# - `perspective=0.0005`: 투영 변환. 비스듬한 스캔 방어.
# - `blur=0.01`: 블러 필터 적용. 흐린 해상도 도면 대응.
# - `scale=0.5`: 스케일 변화 유연성 확보.
# - `mosaic=1.0`: 모자이크 합성(4분할). 가구 탐지 공간 왜곡 학습 극대화.

# %%
# 베이스라인 사전학습 가중치(yolov8n.pt - Nano) 다운로드 및 로드.
# 학습 속도 빠름. 코랩 가벼운 리소스 환경 최적.
model = YOLO("yolov8n.pt")

# %%
# 데이터 증강 규칙 주입 및 학습 개시
results = model.train(
    data="/content/yolo_dataset/dataset.yaml",
    epochs=50,                  # 에포크 횟수
    imgsz=640,                  # 이미지 훈련 사이즈
    batch=8,                    # 배치 사이즈
    device=0,                   # 코랩 런타임 T4 GPU 지정
    workers=2,
    # --- 데이터 증강 (오래된 도면 도메인 갭 방어용) ---
    degrees=15.0,               # 무작위 회전 (+/- 15도)
    perspective=0.0005,         # 원근 변환
    scale=0.5,                  # 스케일 확대/축소
    blur=0.01,                  # 가우시안 블러 주입
    mosaic=1.0,                 # 모자이크 기법 (도면 합성)
    val=True                    # 훈련 중 실시간 검증 성능 측정
)

# %% [markdown]
# ## 4. 정량적 모델 성능 평가 (Evaluation)
# YOLO 자체 평가 성능 지표 기반 분석.
# **mAP50, mAP50-95** 및 숭실대 분석 가이드 요구 **혼동행렬(Confusion Matrix)** 시각화 자료 추출.

# %%
# 훈련 성능 히스토리 그래프 확인
train_results_dir = Path("runs/detect/train")

if train_results_dir.exists():
    # 1. 학습 로스 및 성능 지표 추이 차트 시각화
    results_png = train_results_dir / "results.png"
    if results_png.exists():
        plt.figure(figsize=(15, 10))
        plt.imshow(Image.open(results_png))
        plt.axis('off')
        plt.title("YOLOv8 Training Metrics History", fontsize=16)
        plt.show()
        
    # 2. 숭실대 요구 평가 기준 핵심: 혼동행렬(Confusion Matrix) 플롯
    confusion_matrix_png = train_results_dir / "confusion_matrix.png"
    if confusion_matrix_png.exists():
        plt.figure(figsize=(12, 10))
        plt.imshow(Image.open(confusion_matrix_png))
        plt.axis('off')
        plt.title("Confusion Matrix - Evaluation on Val Split", fontsize=16)
        plt.show()
else:
    print("Train directory not found. Please verify training execution successfully finished.")

# %% [markdown]
# ## 5. 실전 추론 및 예측 결과 검증 (Inference)
# 모델 추론 시각적 테스트 진행.
# 검증용 데이터셋(`val/images`) 무작위 선택, 바운딩 박스 출력.
# 예측 결과(Prediction)와 정답 라벨(Ground Truth) 대조 분석.

# %%
# 검증(Val) 도면 무작위 선정, 모델 추론 테스트
val_img_dir = Path("/content/yolo_dataset/images/val")
val_lbl_dir = Path("/content/yolo_dataset/labels/val")

if val_img_dir.exists():
    val_images = list(val_img_dir.glob("*.PNG")) + list(val_img_dir.glob("*.png"))
    if val_images:
        test_image = random.choice(val_images)
        
        # 1. 모델 예측 (conf=0.25 임계값)
        predict_results = model.predict(source=test_image, conf=0.25, save=False)
        predicted_plot = predict_results[0].plot() # YOLO 기본 시각화
        
        # 2. 정답 라벨(Ground Truth) 시각화
        gt_lbl_path = val_lbl_dir / (test_image.stem + ".txt")
        gt_image = visualize_yolo_labels(test_image, gt_lbl_path, class_names)
        
        # 3. 비교 시각화
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        # Ground Truth
        axes[0].imshow(gt_image)
        axes[0].set_title("Ground Truth", fontsize=15)
        axes[0].axis('off')
        
        # Prediction
        axes[1].imshow(predicted_plot)
        axes[1].set_title("AI Model Prediction", fontsize=15)
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
    else:
        print("No validation images found.")

# %% [markdown]
# ## 6. 결론 및 향후 개선 과제 (BIM/3D Modeling 연계)
#
# ### **분석 결과 정리**
# 1. **베이스라인 성능 확보**: YOLOv8 모델 활용 도면 객체 탐지 성공.
# 2. **일반화 성능 확보**: Data Augmentation (회전, 블러 등) 통한 노후 도면 환경 모델 강건성 증대.
#
# ### **한계점 및 향후 연구 방향 (3D 자동 모델링)**
# - **3D Extrusion 연계**: 1차 객체 검출(OBJ) 기술에 **벽체/구조 분석(STR)** 데이터셋 통합 연동 예정.
# - **3D 공간 Auto-Spawn**: 검출 객체의 `class`, `center`, 너비 크기, 회전 각도 수치 데이터 JSON 추출. WebGL/Three.js 기반 3D 환경 내 `gltf` 객체 자동 배치. **2D-to-3D Auto Extrusion Pipeline** 파이프라인 구성.
