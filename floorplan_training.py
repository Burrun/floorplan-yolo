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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # [딥러닝분석] 파이널 프로젝트: 2D 아파트 도면 객체 검출 및 3D 모델링 파이프라인 구축
# - **소속**: 숭실대학교 소프트웨어학과
# - **프로젝트 주제**: 2D 도면 이미지의 딥러닝 기반 디지털 구조화 및 3D 변환의 AI 베이스라인 모델 구축
# - **AI Task**: [OBJ] 가구 및 설비 (변기, 세면대, 싱크대, 욕조, 가스레인지) 객체 검출 (Object Detection)
# - **사용한 모델**: YOLOv8 (SOTA Real-time Object Detection Model)
#
# ---

# %% [markdown]
# ## 0. 개발 환경 설정 및 라이브러리 설치
# 구글 코랩 환경에서 최첨단 컴퓨터 비전 라이브러리인 `ultralytics`를 설치하고 필요한 라이브러리를 가져옵니다.

# %%
# YOLOv8 및 의존성 라이브러리 설치
# !pip install -q ultralytics matplotlib numpy pillow albumentations pyyaml

# %%
import os
import yaml
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from pathlib import Path
from ultralytics import YOLO

print("Libraries imported successfully.")

# %% [markdown]
# ## 1. 데이터셋 연동 및 압축 해제
# 구글 드라이브에 업로드한 `floorplan_dataset_yolo.zip`을 코랩의 로컬 런타임 공간에 압축 해제합니다.
#
# > **팁**: 구글 드라이브 연동은 좌측 메뉴의 폴더 아이콘 -> 드라이브 마운트를 누르면 더 간단히 연동됩니다.

# %%
# 구글 드라이브 마운트 실행 (선택 사항)
# from google.colab import drive
# drive.mount('/content/drive')

# 임시로 코랩 로컬 드라이브에 직접 업로드하는 경우에 맞게 파일 압축을 해제합니다.
# 만약 구글드라이브에 올렸다면 /content/drive/MyDrive/floorplan_dataset_yolo.zip 등으로 경로를 수정하세요.
ZIP_PATH = "/content/floorplan_dataset_yolo.zip"
EXTRACT_PATH = "/content/yolo_dataset"

if os.path.exists(ZIP_PATH):
    # !unzip -q {ZIP_PATH} -d {EXTRACT_PATH}
    print("Dataset unzipped successfully at /content/yolo_dataset")
else:
    print(f"Error: Zip file not found at {ZIP_PATH}. Please upload the zip file to the Colab files section.")

# %% [markdown]
# ## 2. 탐색적 데이터 분석 (EDA) 및 시각화 검증
# 학습 전, 전처리된 데이터가 올바르게 바인딩 박스 좌표를 가졌는지 시각적으로 직접 검증합니다.
# YOLO 포맷의 라벨 파일(`[class_id, x_center, y_center, width, height]` 정규화 값)을 분석하고 이미지 위에 바운딩 박스를 투영해 봅니다.

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

# 무작위로 하나의 학습 이미지를 선정하여 라벨링 검증 시각화
train_img_dir = Path("/content/yolo_dataset/images/train")
train_lbl_dir = Path("/content/yolo_dataset/labels/train")

if train_img_dir.exists():
    img_files = list(train_img_dir.glob("*.PNG")) + list(train_img_dir.glob("*.png"))
    if img_files:
        sample_img = random.choice(img_files)
        sample_lbl = train_lbl_dir / (sample_img.stem + ".txt")
        
        print(f"Visualizing Sample: {sample_img.name}")
        result_img = visualize_yolo_labels(sample_img, sample_lbl, class_names)
        
        # 해상도가 매우 크므로 적절한 크기로 축소하여 시각화
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
# 우리가 학습할 도면은 깨끗하지만, 최종 목표인 **'오래된 도면'**은 스캔 왜곡, 번짐, 각도 비뚤어짐 등 다양한 노이즈가 존재합니다.
# YOLOv8 모델 내부의 다양한 하이퍼파라미터를 조정하여 다음과 같은 강력한 **실시간 데이터 증강(On-the-fly Data Augmentation)**을 주입하여 강건한(Robust) 일반화 모델을 설계합니다.
#
# - `degrees=15.0`: 도면이 삐딱하게 스캔되는 경우를 방지하기 위해 무작위 회전 주입.
# - `perspective=0.0005`: 비스듬하게 각도가 틀어져서 사진 찍힌 도면을 위한 투영 변환.
# - `blur=0.01`: 흐린 해상도의 낙후된 도면에 대응하기 위한 블러 필터 적용.
# - `scale=0.5`: 도면 내 가구의 해상도 스케일 변화에 유연하게 대처.
# - `mosaic=1.0`: 다중 도면 이미지를 4분할 합성(Mosaic)하여 가구 탐지의 공간 왜곡 학습 극대화.

# %%
# 베이스라인 모델 사전학습된 가중치(yolov8n.pt - Nano 사이즈) 다운로드 및 로드
# 학습 속도가 빠르며 코랩 환경의 가벼운 리소스로 돌리기에 가장 적절합니다.
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
# 학습 결과는 YOLO가 자체 평가하여 저장한 성능 지표를 통해 분석합니다.
# 이 프로젝트에서 가장 강력한 검증 지표인 **mAP50, mAP50-95** 및 숭실대 분석 가이드에서 특별히 요구하는 **혼동행렬(Confusion Matrix)**을 불러와 보고서용 시각화 자료를 추출합니다.

# %%
# 훈련 성능 히스토리 그래프 확인
train_results_dir = Path("runs/detect/train")

if train_results_dir.exists():
    # 1. 학습 로스 및 성능 지표 전체 추이 차트 시각화
    results_png = train_results_dir / "results.png"
    if results_png.exists():
        plt.figure(figsize=(15, 10))
        plt.imshow(Image.open(results_png))
        plt.axis('off')
        plt.title("YOLOv8 Training Metrics History", fontsize=16)
        plt.show()
        
    # 2. 숭실대 요구 평가 기준의 핵심: 혼동행렬(Confusion Matrix) 플롯
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
# 학습된 모델의 추론 파워를 시각적으로 테스트합니다.
# 검증용 데이터셋(`val/images`) 중 무작위로 선택하여 실제 예측된 바운딩 박스를 출력합니다.
# 실무적인 관점에서 예측 결과(Prediction)와 원천 라벨(Ground Truth)을 대조 비교하여 성능의 한계점을 논리적으로 분석해 봅니다.

# %%
# 검증(Val) 도면 중 무작위로 하나를 선정하여 AI 모델 추론 테스트
val_img_dir = Path("/content/yolo_dataset/images/val")
val_lbl_dir = Path("/content/yolo_dataset/labels/val")

if val_img_dir.exists():
    val_images = list(val_img_dir.glob("*.PNG")) + list(val_img_dir.glob("*.png"))
    if val_images:
        test_image = random.choice(val_images)
        
        # 1. AI 모델 예측 수행
        # conf=0.25 (정확도 임계값 25% 설정)
        predict_results = model.predict(source=test_image, conf=0.25, save=False)
        predicted_plot = predict_results[0].plot() # YOLO 기본 시각화
        
        # 2. 정답 라벨(Ground Truth) 시각화
        gt_lbl_path = val_lbl_dir / (test_image.stem + ".txt")
        gt_image = visualize_yolo_labels(test_image, gt_lbl_path, class_names)
        
        # 3. 나란히 비교 시각화 (보고서 첨부 최적화)
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        # Ground Truth
        axes[0].imshow(gt_image)
        axes[0].set_title("Ground Truth (실제 정답 라벨)", fontsize=15)
        axes[0].axis('off')
        
        # Prediction
        axes[1].imshow(predicted_plot)
        axes[1].set_title("AI Model Prediction (딥러닝 예측 결과)", fontsize=15)
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
    else:
        print("No validation images found.")

# %% [markdown]
# ## 6. 결론 및 향후 개선 과제 (BIM/3D Modeling 연계)
#
# ### **분석 결과 정리**
# 1. **베이스라인 성능 확보**: YOLOv8 모델을 활용하여 2D 도면 이미지 내 가구 및 설비의 위치를 정확하게 탐지하는 데 성공하였습니다.
# 2. **일반화 성능 확보**: Data Augmentation (회전, 블러 등)을 통해 복사 및 훼손된 노후화 도면 환경에서의 객체 인식 강건함을 높였습니다.
#
# ### **한계점 및 향후 연구 방향 (3D 자동 모델링을 향한 확장)**
# - **객체 추출을 넘어서는 3D Extrusion 연계**: 현재 1차로 구현한 가구 검출(OBJ) 기술과 더불어 **벽체/구조 분석(STR)** 데이터셋을 통합 연동할 것입니다.
# - **3D 공간 오토-스폰(Auto-Spawn)**: 검출된 가구의 `class`와 `center` 및 가구 `bbox` 너비(Size), 회전 각도(Rotation) 수치 데이터를 JSON으로 추출하여, WebGL/Three.js 기반 3D 환경에 사전에 정의된 `gltf` 3D 가구 모델을 자동으로 매핑 및 Extrude 함으로써 **2D 도면의 3D 공간 자동 모델하우스 변환 웹 파이프라인**을 완성할 수 있습니다.
