import re

with open("/home/ubuntu/project/notebooks/floorplan_training.py", "r") as f:
    content = f.read()

# Find the start of the broken section or Phase 3
idx = content.find("# %% [markdown]\n# )")
if idx == -1:
    idx = content.find("# %% [markdown]\n# ## 5. [Phase 3]")
if idx == -1:
    idx = content.find("# )\n#\n# 학습 완료 후")

new_content = """# %% [markdown]
# ## 5. [Phase 3] 도면 도메인 전이학습(Transfer Learning) 효과 검증
# ### 전이학습 도입의 당위성
# - 도면 내의 글자(OCR)를 탐지하기 위해 맨바닥(Scratch)에서 학습하는 것보다, **이미 가구/설비(`object_layout`)를 학습하며 '도면 특유의 흑백 선과 공간적 맥락'이라는 도메인 특징(Feature)을 익힌 가중치를 활용하는 것이 훨씬 효율적일 것**이라는 가설을 세웠습니다.
# - 이를 증명하기 위해 아래의 두 가지 실험을 비교합니다.
#   1. **Experiment A (Scratch)**: 아무것도 모르는 초기 상태(`yolov8n.pt`)에서 `ocr` 데이터셋 훈련
#   2. **Experiment B (Transfer)**: `object_layout` 도메인에 완전히 적응한 가중치(`best.pt`)로 `ocr` 데이터셋 훈련

# %%
print("🚀 [Experiment A] Scratch 모델 학습 시작")
model_scratch = YOLO("yolov8n.pt")
# (임시로 ocr_dataset.yaml 경로 지정. 파일이 없으면 에러가 날 수 있으니 예외 처리)
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
# Phase 1&2에서 도면의 선/형태 피처를 익힌 best.pt 가중치 로드
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
        hsv_s=0.0, hsv_v=0.5, blur=0.25, degrees=5.0,
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
        
        # 1. 마스터 모델 예측 (Inference) - 여기서는 model을 임시로 사용하나 실제로는 model_master 사용
        print(f"[{test_image.name}] 실전 추론 시작...")
        predict_results = model.predict(source=test_image, conf=0.25, save=False)[0]
        
        # 2. 결과 시각화 (공간 구획 및 객체 통합 검증)
        predicted_plot = predict_results.plot()
        plt.figure(figsize=(10, 8))
        plt.imshow(predicted_plot)
        plt.title("Master Integration Output (Space, OCR, Object, Structure)", fontsize=15)
        plt.axis('off')
        plt.show()

        # 3. JSON 구조화 (Digitization)
        export_data = {
            "image_filename": test_image.name,
            "predictions": []
        }
        
        for box in predict_results.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
            
            export_data["predictions"].append({
                "class_id": cls_id,
                "class_name": class_names.get(cls_id, f"Class_{cls_id}"),
                "confidence": conf,
                "bbox": [x1, y1, x2, y2]
            })
            
        json_output_path = Path("legacy_floorplan_master_digitized.json")
        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, ensure_ascii=False, indent=4)
            
        print(f"\\n✅ 통합 디지털화 완료! JSON 데이터가 성공적으로 추출되었습니다: {json_output_path}")
        print(json.dumps(export_data, ensure_ascii=False, indent=4)[:500] + "\\n... (생략)")
"""

if idx != -1:
    final_content = content[:idx] + new_content
    with open("/home/ubuntu/project/notebooks/floorplan_training.py", "w") as f:
        f.write(final_content)
    print("Fixed via python script!")
else:
    print("Could not find insertion index.")

