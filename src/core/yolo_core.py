import os
import yaml
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from pathlib import Path
from ultralytics import YOLO

def setup_dataset(zip_path="/content/floorplan_dataset_yolo.zip", extract_path="/content/yolo_dataset"):
    """데이터셋 압축 해제 (Colab 환경 전용)"""
    if os.path.exists(zip_path):
        os.system(f"unzip -q {zip_path} -d {extract_path}")
        print(f"Dataset unzipped successfully at {extract_path}")
    else:
        print(f"Error: Zip file not found at {zip_path}. Please upload it to Colab.")

def get_class_names(dataset_yaml_path="/content/yolo_dataset/dataset.yaml"):
    """클래스 매핑 정보 로드"""
    yaml_path = Path(dataset_yaml_path)
    if yaml_path.exists():
        with open(yaml_path, "r", encoding="utf-8") as f:
            dataset_info = yaml.safe_load(f)
        return dataset_info.get("names", {})
    else:
        print("Using default class mapping.")
        return {0: '변기', 1: '세면대', 2: '싱크대', 3: '욕조', 4: '가스레인지'}

def visualize_yolo_labels(image_path, label_path, class_names):
    """단일 이미지 및 YOLO 라벨 시각화"""
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    w, h = img.size
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
        
        x_center, y_center = xc * w, yc * h
        box_w, box_h = bw * w, bh * h
        
        x_min, y_min = int(x_center - box_w / 2), int(y_center - box_h / 2)
        x_max, y_max = int(x_center + box_w / 2), int(y_center + box_h / 2)
        
        color = colors[class_id % len(colors)]
        draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=6)
        
        label_text = class_names.get(class_id, str(class_id))
        draw.text((x_min + 5, y_min - 25), label_text, fill=color)
        
    return img

def train_yolo(data_yaml="/content/yolo_dataset/dataset.yaml", epochs=50):
    """YOLOv8 모델 훈련 (데이터 증강 적용)"""
    model = YOLO("yolov8n.pt")
    
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=640,
        batch=8,
        device=0, # Colab GPU
        workers=2,
        degrees=15.0,
        perspective=0.0005,
        scale=0.5,
        blur=0.01,
        mosaic=1.0,
        val=True
    )
    return model, results

def visualize_inference(model, val_img_dir="/content/yolo_dataset/images/val", val_lbl_dir="/content/yolo_dataset/labels/val", class_names=None):
    """무작위 검증 데이터 추론 및 Ground Truth 대조"""
    if class_names is None:
        class_names = get_class_names()
        
    img_dir = Path(val_img_dir)
    lbl_dir = Path(val_lbl_dir)
    
    if img_dir.exists():
        val_images = list(img_dir.glob("*.PNG")) + list(img_dir.glob("*.png"))
        if val_images:
            test_image = random.choice(val_images)
            
            # AI 예측
            predict_results = model.predict(source=test_image, conf=0.25, save=False)
            predicted_plot = predict_results[0].plot()
            
            # Ground Truth
            gt_lbl_path = lbl_dir / (test_image.stem + ".txt")
            gt_image = visualize_yolo_labels(test_image, gt_lbl_path, class_names)
            
            # 시각화 비교
            fig, axes = plt.subplots(1, 2, figsize=(20, 10))
            axes[0].imshow(gt_image)
            axes[0].set_title("Ground Truth", fontsize=15)
            axes[0].axis('off')
            
            axes[1].imshow(predicted_plot)
            axes[1].set_title("YOLO Prediction", fontsize=15)
            axes[1].axis('off')
            plt.tight_layout()
            plt.show()
        else:
            print("No images found in validation dir.")
