import json

with open("notebooks/floorplan_training.ipynb", "r") as f:
    nb = json.load(f)

new_source = """print("=" * 60)
print("🔬 [Phase 1] Data Size Ablation")
print("=" * 60)

import pandas as pd

train_images = list((MASTER_DATASET_DIR / "images" / "train").glob("*.webp"))
# 총 1600장의 훈련셋을 활용하여 점진적 크기 실험
data_sizes = [300, 600, 900, 1200, 1500]
scaling_results = {}

for size in data_sizes:
    if size > len(train_images):
        continue

    subset = train_images[:size]
    subset_txt = MASTER_DATASET_DIR / f"train_{size}.txt"
    with open(subset_txt, "w", encoding="utf-8") as f:
        f.write("\\n".join([str(p.resolve()) for p in subset]))

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

    weight_path = PROJECT_ROOT / "runs/detect" / f"train_size_{size}" / "weights" / "best.pt"
    if weight_path.exists():
        print(f"\\n✅ Data Size: {size} - 이미 학습된 가중치가 존재합니다. 학습을 스킵합니다.")
        results_csv = PROJECT_ROOT / "runs/detect" / f"train_size_{size}" / "results.csv"
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

    print(f"\\n🚀 Data Size: {size} 학습 시작 (30 Epochs 검증)")
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
"""

# Format into line list to match Jupyter Notebook format
source_lines = [line + "\\n" for line in new_source.split("\\n")]
source_lines[-1] = source_lines[-1].strip("\\n") # last line shouldn't have trailing newline

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source_str = "".join(cell["source"])
        if "Phase 1" in source_str and "model_size.train" in source_str:
            cell["source"] = source_lines
            print("Successfully updated Phase 1 cell")
            break

with open("notebooks/floorplan_training.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("Notebook saved.")
