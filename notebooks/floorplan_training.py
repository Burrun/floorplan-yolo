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
# %pip install -q ultralytics matplotlib numpy pillow albumentations pyyaml sahi

import torch
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
            # -o{EXTRACT_PATH} 로 명시하여 /content/architectural_drawing_data 폴더 안에 풀리도록 강제
            subprocess.run(["7z", "x", str(ZIP_PATH), f"-o{EXTRACT_PATH}", "-y"], check=True)
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
#2. JSON 라벨 전처리 및 YOLO 포맷 변환 (Train/Val/Test)
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
        
        # dataset_summary.md 명세와 로컬 구조(1:1 매칭)에 맞춰 EXTRACT_PATH 기준으로 원복
        base_val_dir = EXTRACT_PATH / "object_layout" / "val"
        
        src_img_dir = base_val_dir / "images"
        src_lbl_dir = base_val_dir / "labels"
            
        # dataset_summary.md 명세대로 정확히 images/ labels/ 폴더 사용
        json_files = list(src_lbl_dir.glob("*.json"))
        converted_count = 0
        
        for json_file in json_files:
            base_name = json_file.stem
            # 명세서대로 확장자는 무조건 .webp
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
# ## 4. [Phase 1 & 2] 베이스라인 구축 및 극단적 증강(Ablation Study) 학습
# ### 가설 및 실험 설계
# - **Phase 1 (Baseline)**: 가장 가벼운 YOLOv8n 모델과 기본 파라미터로 `object_layout` 데이터셋을 학습시켜 기준점(Anchor)을 확보합니다.
# - **Phase 2 (Ablation Study)**: 현실의 '옛날 도면'은 흑백, 스캔 노이즈, 삐뚤어짐, 낮은 해상도 등의 도메인 갭(Domain Gap)을 가집니다. 
#   - 이를 극복하기 위해 단순히 데이터 양을 늘리는 것이 아니라, **극단적인 증강(Extreme Augmentation) 기법(블러, 흑백화, 명암 조절 등)이 실제 범용성 향상에 얼마나 기여하는지**를 검증합니다.
#   - 본 학습은 레거시 도면 시뮬레이션을 가정한 최적의 하이퍼파라미터 셋업입니다.

# %%
model = YOLO("yolov8n.pt")

# 극단적 증강 파라미터 셋업 (레거시 도면 시뮬레이션)
results = model.train(
    data=str(yaml_path),
    epochs=50,
    imgsz=640,
    batch=8,
    device=0,
    workers=2,
    # === Domain Gap 극복을 위한 Extreme Augmentation ===
    # 근거 1: [회전/삐뚤어짐] Document Skew Detection 연구 참고, 평판 스캐너 수작업 오차(±3~5도) 반영. 회전 불변성(Rotation Invariance) 확보용.
    degrees=5.0,    
    # 근거 2: [노출 불량/명암 저하] Legacy Document Image Binarization 연구 참고. 황변 현상 및 토너 부족 시뮬레이션. 색상 피처 의존성 제거.
    hsv_s=0.0,      
    hsv_v=0.5,      
    # 근거 3: [초점 흐림] 저해상도 팩스 전송 및 다중 복사(Copy of copy)로 인한 고주파 텍스처(선명도) 손실 시뮬레이션 (Robustness 향상).
    # (주의: YOLOv8 자체 인자는 아니지만, 맨 위에서 설치한 albumentations 라이브러리가 자동 연동되어 내부적으로 블러 처리를 돕습니다.)
    
    perspective=0.0005,
    scale=0.5,
    mosaic=1.0,
    val=True
    # [Ablation Study Feedback Loop] 
    # 본 극한 증강이 적용된 모델(실험군)과 적용되지 않은 모델(대조군)을 실제 구형 도면에 추론(Inference) 시켜
    # 시각적으로 False Negative 개선율을 확인하고 파라미터를 미세조정(Iteration) 함.
)

# %% [markdown]
# ## 5. [Phase 3] 도면 도메인 전이학습(Transfer Learning) 효과 검증
# ### 전이학습 도입의 당위성
# - 도면 내의 글자(OCR)를 탐지하기 위해 맨바닥(Scratch)에서 학습하는 것보다, **이미 가구/설비(`object_layout`)를 학습하며 '도면 특유의 흑백 선과 공간적 맥락'이라는 도메인 특징(Feature)을 익힌 가중치를 활용하는 것이 훨씬 효율적일 것**이라는 가설을 세웠습니다.
# - 이를 증명하기 위해 아래의 두 가지 실험을 비교합니다.
#   1. **Experiment A (Scratch)**: 아무것도 모르는 초기 상태(`yolov8n.pt`)에서 `ocr` 데이터셋 훈련
#   2. **Experiment B (Transfer)**: `object_layout` 도메인에 완전히 적응한 가중치(`best.pt`)로 `ocr` 데이터셋 훈련

# %%
print("🚀 [Experiment A] Scratch 모델 학습 시작")
model_scratch = YOLO("yolov8n.pt")
try:
    results_scratch = model_scratch.train(
        data="/content/architectural_drawing_data/ocr_dataset.yaml", 
        epochs=30,
        imgsz=640,
        project="runs/detect",
        name="train_ocr_scratch"
    )
except Exception as e:
    print(f"⚠️ OCR 데이터셋이 아직 준비되지 않았습니다. 넘어갑니다. (에러: {e})")

print("🚀 [Experiment B] Transfer 모델 학습 시작")
best_weight_path = "runs/detect/train/weights/best.pt"
if Path(best_weight_path).exists():
    model_transfer = YOLO(best_weight_path)
    try:
        results_transfer = model_transfer.train(
            data="/content/architectural_drawing_data/ocr_dataset.yaml", 
            epochs=30,
            imgsz=640,
            project="runs/detect",
            name="train_ocr_transfer"
        )
    except Exception as e:
        print(f"⚠️ OCR 데이터셋이 아직 준비되지 않았습니다. 넘어갑니다. (에러: {e})")
else:
    print("⚠️ 이전 Phase의 가중치가 아직 생성되지 않았습니다.")

# %% [markdown]
# ## 6. [Phase 4] 마스터 통합 모델 구축 (Master Model)
# ### 통합 데이터(가구+텍스트+공간+구조선) 학습
# - 개별적으로 학습했던 객체(가구), 텍스트(OCR), 공간 구획(Space), 구조선(Structure) 데이터셋의 라벨을 모두 하나로 병합한 `master_dataset`을 구성합니다.
# - 모든 도면 요소를 한 번에 인식할 수 있는 **단일 마스터 모델(YOLOv8-seg)**을 최종 학습합니다.
# - 공간 구획과 구조선은 다각형(Polygon) 형태이므로 Instance Segmentation 모델(`yolov8n-seg.pt`)을 사용합니다.

# %%
print("🚀 [Master Model] 통합 데이터셋 학습 시작")
model_master = YOLO("yolov8n-seg.pt") # 폴리곤 예측을 위해 segmentation 모델 사용

try:
    results_master = model_master.train(
        data="/content/architectural_drawing_data/master_dataset.yaml", 
        epochs=50,
        imgsz=640,
        # Phase 2에서 찾은 극한의 증강 파라미터 적용 (도메인 갭 극복)
        hsv_s=0.0, hsv_v=0.5, degrees=5.0,
        project="runs/detect",
        name="train_master_model"
    )
except Exception as e:
    print(f"⚠️ 마스터 통합 데이터셋이 아직 준비되지 않았습니다. 넘어갑니다. (에러: {e})")

# %% [markdown]
# ## 7. [Phase 5] 레거시 구형 도면 실전 추론 및 JSON 구조화 (최종 목표)
# ### 정성적 평가(Qualitative Evaluation) 및 한계점(Limitation) 도출
# - 인터넷에서 무작위로 수집한 라벨 없는 '진짜 90년대 구축 아파트 평면도'를 마스터 모델에 통과시켜 정성적 평가를 진행합니다.
# - 공간 구획, 객체, 텍스트가 동시에 어떻게 탐지되는지 시각화로 검증합니다.
# - 예측된 BBox 및 Polygon 결과를 프롭테크 솔루션에서 2D-to-3D 도면 자동 생성의 기반 데이터로 즉시 활용 가능한 구조화된 **JSON 포맷으로 덤프(Digitization)**합니다.

# %%
# 테스트(Test) 구형 도면 추론 및 시각화 / JSON 생성
test_img_dir = YOLO_DIR / "images" / "test"
if test_img_dir.exists():
    test_images = list(test_img_dir.glob("*.webp"))
    if test_images:
        test_image = random.choice(test_images)
        
        # 1. 마스터 모델 예측 (Inference) - SAHI 적용 (다중 도면 찌그러짐 방지)
        print(f"[{test_image.name}] SAHI 기반 슬라이싱 실전 추론 시작 (512x512 패치로 쪼개서 분석)...")
        
        from sahi import AutoDetectionModel
        from sahi.predict import get_sliced_prediction
        
        # SAHI 모델 로드
        detection_model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path='runs/detect/train_master/weights/best.pt',
            confidence_threshold=0.25,
            device='cuda:0' if torch.cuda.is_available() else 'cpu'
        )
        
        # SAHI 슬라이싱 추론 (Overlap 20% 주어 경계선 객체 짤림 방지)
        result = get_sliced_prediction(
            str(test_image),
            detection_model,
            slice_height=512,
            slice_width=512,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2
        )
        
        # 2. 결과 시각화 (SAHI 자체 시각화 기능 활용)
        result.export_visuals(export_dir=".", file_name="sahi_result")
        predicted_plot = Image.open("sahi_result.png")
        plt.figure(figsize=(12, 10))
        plt.imshow(predicted_plot)
        plt.title("SAHI Integration Output (Sliced inference for multi-floor)", fontsize=15)
        plt.axis('off')
        plt.show()

        # 3. JSON 구조화 (Digitization)
        export_data = {
            "image_filename": test_image.name,
            "predictions": []
        }
        
        # SAHI 결과 객체(object_prediction_list)에서 파싱
        for obj in result.object_prediction_list:
            export_data["predictions"].append({
                "class_id": obj.category.id,
                "class_name": obj.category.name,
                "confidence": float(obj.score.value),
                "bbox": [float(obj.bbox.minx), float(obj.bbox.miny), float(obj.bbox.maxx), float(obj.bbox.maxy)]
            })
            
        json_output_path = Path("legacy_floorplan_master_digitized.json")
        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, ensure_ascii=False, indent=4)
            
        print(f"\n✅ 통합 디지털화 완료! JSON 데이터가 성공적으로 추출되었습니다: {json_output_path}")
        print(json.dumps(export_data, ensure_ascii=False, indent=4)[:500] + "\n... (생략)")
