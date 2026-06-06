import os
import io
import json
import zipfile
import tarfile
import random
import zstandard
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Constants
EXTRACT_PATH = Path(r'C:\Users\jack3\Downloads\239.건축 도면 데이터\01-1.정식개방데이터\Training')
OUTPUT_DIR = Path(r'C:\Users\jack3\Desktop\pj\ocr_dataset')
OUTPUT_TAR_ZST = Path(r'C:\Users\jack3\Desktop\pj\ocr_dataset.tar.zst')

IMG_DIR = EXTRACT_PATH / '01.원천데이터'
LBL_DIR = EXTRACT_PATH / '02.라벨링데이터'

TARGET_COUNT = 2000

def compress_tar_zst(src_dir, output_file):
    print(f"Creating uncompressed tar from {src_dir}...")
    tar_path = output_file.with_suffix('')
    with tarfile.open(tar_path, "w") as tar:
        tar.add(src_dir, arcname=src_dir.name)
    
    print(f"Compressing to zstd {output_file}...")
    cctx = zstandard.ZstdCompressor(level=3)
    with open(tar_path, "rb") as f_in, open(output_file, "wb") as f_out:
        cctx.copy_stream(f_in, f_out)
        
    os.remove(tar_path)
    print("Compression complete!")

def main():
    print("🚀 [1/4] OCR 원본 파일 스캔 시작...")
    
    # 1. OCR JSON 리스트 확보
    ocr_zip_path = LBL_DIR / 'TL_OCR.zip'
    if not ocr_zip_path.exists():
        print(f"❌ OCR 라벨 ZIP을 찾을 수 없습니다: {ocr_zip_path}")
        return

    ocr_jsons = {}
    with zipfile.ZipFile(ocr_zip_path, 'r') as z:
        for name in z.namelist():
            if name.endswith('.json'):
                num_id = Path(name).stem.split('_')[-1]
                ocr_jsons[num_id] = name
                
    print(f"총 {len(ocr_jsons)}개의 OCR 정답지를 찾았습니다.")
    
    # 2. OCR 이미지 스캔
    img_sigs = {}
    for zip_path in IMG_DIR.glob('TS_OCR*.zip'):
        print(f"Scanning images in {zip_path.name}...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            for info in z.infolist():
                if info.filename.upper().endswith('.PNG'):
                    num_id = Path(info.filename).stem.split('_')[-1]
                    img_sigs[num_id] = (zip_path, info.filename)
                    
    print(f"총 {len(img_sigs)}개의 OCR 전용 이미지를 찾았습니다.")
    
    # 교집합 찾기
    matched_ids = list(set(ocr_jsons.keys()).intersection(set(img_sigs.keys())))
    print(f"완벽하게 짝이 맞는 OCR 데이터는 총 {len(matched_ids)}개 입니다.")
    
    # 셔플 및 추출
    random.seed(42)
    random.shuffle(matched_ids)
    selected_ids = matched_ids[:TARGET_COUNT]
    
    # 폴더 구조 생성
    for split in ['train', 'val', 'test']:
        (OUTPUT_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
    # 데이터 분할 비율
    train_count = int(TARGET_COUNT * 0.8)
    val_count = int(TARGET_COUNT * 0.1)
    
    print("🚀 [2/4] 이미지 리사이징 및 라벨 변환...")
    
    # 캐싱
    open_zips = {}
    def get_zip(path):
        if path not in open_zips:
            open_zips[path] = zipfile.ZipFile(path, 'r')
        return open_zips[path]

    for i, num_id in enumerate(tqdm(selected_ids)):
        if i < train_count:
            split = 'train'
        elif i < train_count + val_count:
            split = 'val'
        else:
            split = 'test'
            
        base_idx = f"ocr_{split}_{i+1:04d}"
        
        img_zip_path, img_filename = img_sigs[num_id]
        json_filename = ocr_jsons[num_id]
        
        # 1. Image Conversion
        z_img = get_zip(img_zip_path)
        img_data = z_img.read(img_filename)
        img = Image.open(io.BytesIO(img_data))
        
        img_width, img_height = img.size
        new_size = (img_width // 2, img_height // 2)
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        img.save(OUTPUT_DIR / 'images' / split / f"{base_idx}.webp", "WEBP", quality=80)
        
        # 2. JSON Parsing & YOLO Conversion
        z_lbl = get_zip(ocr_zip_path)
        ocr_data = json.loads(z_lbl.read(json_filename).decode('utf-8'))
        
        yolo_lines = []
        for ann in ocr_data.get('annotations', []):
            bbox = ann.get('bbox', [])
            if len(bbox) == 4:
                x_min, y_min, w, h = bbox
                x_center = (x_min + (w / 2.0)) / img_width
                y_center = (y_min + (h / 2.0)) / img_height
                w_norm = w / img_width
                h_norm = h / img_height
                yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
                
        with open(OUTPUT_DIR / 'labels' / split / f"{base_idx}.txt", "w", encoding="utf-8") as f:
            if yolo_lines:
                f.write('\n'.join(yolo_lines))

    # Close all
    for z in open_zips.values():
        z.close()
        
    print("🚀 [3/4] dataset.yaml 파일 생성...")
    yaml_content = f"""path: {str(OUTPUT_DIR.resolve())}
train: images/train
val: images/val
test: images/test
names:
  0: text
"""
    with open(OUTPUT_DIR / 'dataset.yaml', 'w', encoding='utf-8') as f:
        f.write(yaml_content)

    print("🚀 [4/4] 압축(tar.zst) 시작...")
    compress_tar_zst(OUTPUT_DIR, OUTPUT_TAR_ZST)
    print(f"✅ 완료! OCR 전용 데이터셋이 생성되었습니다: {OUTPUT_TAR_ZST}")

if __name__ == '__main__':
    main()
