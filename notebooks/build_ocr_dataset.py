import os
import json
import zipfile
import shutil
from pathlib import Path
from tqdm import tqdm

def main():
    print("🚀 [1/3] OCR 데이터셋 구축 시작...")
    
    # 경로 설정
    BASE_DIR = Path(r"C:\Users\jack3\Desktop\pj")
    MASTER_DIR = BASE_DIR / "master_dataset"
    OCR_ZIP_PATH = Path(r"C:\Users\jack3\Downloads\239.건축 도면 데이터\01-1.정식개방데이터\Training\02.라벨링데이터\TL_OCR.zip")
    OCR_DIR = BASE_DIR / "ocr_dataset"
    
    if not OCR_ZIP_PATH.exists():
        print(f"❌ OCR 원본 ZIP 파일을 찾을 수 없습니다: {OCR_ZIP_PATH}")
        return

    # OCR 데이터셋 폴더 구조 생성
    for split in ["train", "val", "test"]:
        (OCR_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OCR_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
        
    print("🚀 [2/3] TL_OCR.zip 분석 중...")
    ocr_lookup = {}
    with zipfile.ZipFile(OCR_ZIP_PATH, 'r') as z:
        for info in z.infolist():
            if info.filename.endswith('.json'):
                # /APT_FP_OCR_751650857.json -> APT_FP_OCR_751650857
                basename = Path(info.filename).stem
                # ID 추출 (e.g. 751650857)
                parts = basename.split('_')
                if len(parts) >= 4:
                    num_id = parts[-1]
                    ocr_lookup[num_id] = info.filename
    
    print(f"총 {len(ocr_lookup)}개의 OCR 라벨을 찾았습니다.")
    
    print("🚀 [3/3] 1,500장 이미지와 OCR 라벨 매칭 및 변환...")
    
    found_count = 0
    missing_count = 0
    
    with zipfile.ZipFile(OCR_ZIP_PATH, 'r') as z:
        for split in ["train", "val", "test"]:
            json_dir = MASTER_DIR / "json" / split
            if not json_dir.exists():
                continue
                
            for json_file in tqdm(list(json_dir.glob("*.json")), desc=f"Processing {split}"):
                base_idx = json_file.stem  # master_train_0001
                
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                if not data.get("images"):
                    continue
                    
                # 원본 파일명 추출 (e.g. APT_FP_OBJ_751650857.PNG)
                orig_filename = data["images"][0].get("file_name", "")
                parts = Path(orig_filename).stem.split('_')
                if len(parts) < 4:
                    missing_count += 1
                    continue
                    
                num_id = parts[-1]
                
                # 원본 이미지 사이즈
                img_w = data["images"][0].get("width", 4963)
                img_h = data["images"][0].get("height", 3509)
                
                # OCR 라벨 찾기
                if num_id in ocr_lookup:
                    ocr_json_path = ocr_lookup[num_id]
                    ocr_data = json.loads(z.read(ocr_json_path).decode('utf-8'))
                    
                    yolo_lines = []
                    for ann in ocr_data.get("annotations", []):
                        # OCR은 단일 클래스 (0: text)
                        bbox = ann.get("bbox")
                        if bbox and len(bbox) == 4:
                            x_min, y_min, w, h = bbox
                            x_center = (x_min + (w / 2.0)) / img_w
                            y_center = (y_min + (h / 2.0)) / img_h
                            w_norm = w / img_w
                            h_norm = h / img_h
                            yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
                    
                    # 라벨 저장
                    with open(OCR_DIR / "labels" / split / f"{base_idx}.txt", "w", encoding="utf-8") as f:
                        if yolo_lines:
                            f.write('\n'.join(yolo_lines))
                    
                    # 이미지 복사 (YOLO는 labels와 짝이 맞는 images 폴더를 요구함)
                    src_img = MASTER_DIR / "images" / split / f"{base_idx}.webp"
                    dst_img = OCR_DIR / "images" / split / f"{base_idx}.webp"
                    if src_img.exists() and not dst_img.exists():
                        shutil.copy(src_img, dst_img)
                        
                    found_count += 1
                else:
                    missing_count += 1

    print(f"\n✅ 완료! 총 {found_count}장의 OCR 매칭 성공 (누락: {missing_count}장)")
    
    # dataset.yaml 생성
    yaml_path = OCR_DIR / "dataset.yaml"
    yaml_content = f"""path: {str(OCR_DIR.resolve())}
train: images/train
val: images/val
test: images/test
names:
  0: text
"""
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f"✅ OCR dataset.yaml 생성 완료: {yaml_path}")

if __name__ == '__main__':
    main()
