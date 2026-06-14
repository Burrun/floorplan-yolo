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
OUTPUT_DIR = Path(r'C:\Users\jack3\Desktop\pj\master_dataset')
OUTPUT_TAR_ZST = Path(r'C:\Users\jack3\Desktop\pj\master_dataset.tar.zst')

IMG_DIR = EXTRACT_PATH / '01.원천데이터'
LBL_DIR = EXTRACT_PATH / '02.라벨링데이터'

CLASS_MAPPING = {4: 0, 5: 1, 6: 2, 7: 3, 8: 4, 9: 5, 10: 6}
TARGET_COUNT = 2000

def get_all_zip_signatures(folder_path, prefix):
    sigs = {}
    for zip_path in folder_path.glob(f"{prefix}*.zip"):
        print(f"Scanning {zip_path.name}...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            for info in z.infolist():
                if info.filename.upper().endswith('.PNG'):
                    # store: sig -> (path_to_zip, filename_in_zip)
                    sigs[(info.file_size, info.CRC)] = (zip_path, info.filename)
    return sigs

def build_all_json_lookup(folder_path, prefix):
    lookup = {}
    for zip_path in folder_path.glob(f"{prefix}*.zip"):
        print(f"Scanning labels in {zip_path.name}...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            for name in z.namelist():
                if name.endswith('.json'):
                    basename = Path(name).stem
                    lookup[basename] = (zip_path, name)
    return lookup

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
    print("Phase 1: Comprehensive ZIP Analysis...")
    obj_sigs = get_all_zip_signatures(IMG_DIR, 'TS_OBJ')
    str_sigs = get_all_zip_signatures(IMG_DIR, 'TS_STR')
    
    all_matches = []
    for sig, (str_zip, str_name) in str_sigs.items():
        if sig in obj_sigs:
            obj_zip, obj_name = obj_sigs[sig]
            all_matches.append((obj_zip, obj_name, str_zip, str_name))
            
    print(f"Found TOTAL {len(all_matches)} matching image pairs across all ZIPs!")
    if not all_matches:
        return
        
    # Shuffle and trim
    random.seed(42)
    random.shuffle(all_matches)
    matches = all_matches[:TARGET_COUNT]
    print(f"Selected {len(matches)} pairs for processing.")

    print("Phase 2: Building comprehensive JSON lookups...")
    obj_json_lookup = build_all_json_lookup(LBL_DIR, 'TL_OBJ')
    str_json_lookup = build_all_json_lookup(LBL_DIR, 'TL_STR')

    # Split sizes (8:1:1)
    train_count = int(len(matches) * 0.8)
    val_count = int(len(matches) * 0.1)
    # remaining goes to test

    # Output paths
    for split in ['train', 'val', 'test']:
        (OUTPUT_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / 'json' / split).mkdir(parents=True, exist_ok=True)

    print(f"Phase 3: Extracting, Merging, and Converting...")
    
    # Cache zipfile objects to avoid reopening
    open_zips = {}
    def get_zip(path):
        if path not in open_zips:
            open_zips[path] = zipfile.ZipFile(path, 'r')
        return open_zips[path]

    for i, (obj_zip_p, obj_png, str_zip_p, str_png) in enumerate(tqdm(matches)):
        if i < train_count:
            split = 'train'
        elif i < train_count + val_count:
            split = 'val'
        else:
            split = 'test'
            
        base_idx = f"master_{split}_{i+1:04d}"
        
        # 1. Image Conversion
        z_img = get_zip(obj_zip_p)
        img_data = z_img.read(obj_png)
        img = Image.open(io.BytesIO(img_data))
        
        # Downsize to 50%
        img_width, img_height = img.size
        new_size = (img_width // 2, img_height // 2)
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        img.save(OUTPUT_DIR / 'images' / split / f"{base_idx}.webp", "WEBP", quality=80)
        
        # 2. JSON Merging
        obj_basename = Path(obj_png).stem
        str_basename = Path(str_png).stem
        
        obj_j_info = obj_json_lookup.get(obj_basename)
        str_j_info = str_json_lookup.get(str_basename)
        
        merged_json = {}
        annotations = []
        
        if obj_j_info:
            z_obj_lbl = get_zip(obj_j_info[0])
            obj_data = json.loads(z_obj_lbl.read(obj_j_info[1]).decode('utf-8'))
            merged_json.update(obj_data)
            annotations.extend(obj_data.get('annotations', []))
            
        if str_j_info:
            z_str_lbl = get_zip(str_j_info[0])
            str_data = json.loads(z_str_lbl.read(str_j_info[1]).decode('utf-8'))
            if not merged_json:
                merged_json.update(str_data)
            annotations.extend(str_data.get('annotations', []))
            
        merged_json['annotations'] = annotations
        
        # Save merged JSON
        with open(OUTPUT_DIR / 'json' / split / f"{base_idx}.json", "w", encoding="utf-8") as f:
            json.dump(merged_json, f, ensure_ascii=False, indent=2)
            
        # 3. YOLO TXT Conversion
        yolo_lines = []
        for ann in annotations:
            cat_id = ann.get('category_id')
            if cat_id in CLASS_MAPPING:
                bbox = ann.get('bbox', [])
                if len(bbox) == 4:
                    x_min, y_min, w, h = bbox
                    x_center = (x_min + (w / 2.0)) / img_width
                    y_center = (y_min + (h / 2.0)) / img_height
                    w_norm = w / img_width
                    h_norm = h / img_height
                    yolo_lines.append(f"{CLASS_MAPPING[cat_id]} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
                    
        with open(OUTPUT_DIR / 'labels' / split / f"{base_idx}.txt", "w", encoding="utf-8") as f:
            if yolo_lines:
                f.write('\n'.join(yolo_lines))

    # Close all cached zips
    for z in open_zips.values():
        z.close()
        
    print("Phase 4: Compressing Master Dataset...")
    compress_tar_zst(OUTPUT_DIR, OUTPUT_TAR_ZST)
    print(f"All done! Master Dataset created at {OUTPUT_TAR_ZST}")

if __name__ == '__main__':
    main()
