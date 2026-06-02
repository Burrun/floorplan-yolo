# 🏢 2D 아파트 도면 객체 검출 (Object Detection) & 3D 파이프라인 베이스라인

> **숭실대학교 소프트웨어학과 - 딥러닝분석 프로젝트**  
> **주제**: 2D 도면 이미지의 딥러닝 기반 디지털 구조화 및 3D 변환을 위한 AI 베이스라인 모델 구축
> **AI Task**: 2D 아파트 도면에서 주요 가구 및 설비 5종(`변기`, `세면대`, `싱크대`, `욕조`, `가스레인지`) 객체 탐지 (YOLOv8 기반)

---

## 📂 1. 프로젝트 디렉토리 컴포넌트 구조

핵심 딥러닝 로직의 **모듈화(Modularization)**, 로컬 IDE-주피터 노트북 간의 **실시간 동기화(Jupytext)**, **Google Colab GPU 환경**에서의 효율적인 모델 훈련을 목표로 컴포넌트 단위 설계.

```text
floorplan-yolo/
├── 📁 .venv/                       # 로컬 가상환경 폴더
├── 📄 requirements.txt             # 프로젝트 의존성 라이브러리 목록
├── 🛠️ jupytext.toml                # Jupytext 동기화 설정 파일
│
├── 📁 data/                        
│   └── 📦 floorplan_dataset_yolo.zip   # 아파트 도면 데이터셋 압축 파일
│
├── 📁 src/
│   └── 📁 core/
│       └── 🧠 yolo_core.py             # 핵심 파이프라인 함수 모듈 (압축 해제, 훈련, 추론 시각화)
│
├── 📁 notebooks/
│   ├── 📓 floorplan_training.ipynb     # [전체 워크플로우] 분석 및 증강 훈련 평가 메인 노트북
│   └── 🐍 floorplan_training.py        # [전체 워크플로우] 파이썬 동기화 스크립트
│
└── 📁 tools/
    └── ⚙️ jupytext_watch.py            # .ipynb <-> .py 파일 실시간 재귀적 감시/동기화 툴
```

---

## 🔁 2. 로컬 IDE와 Jupyter 연동을 위한 Jupytext 환경

주피터 노트북의 고질적 단점인 **Git 버전 관리 충돌** 문제 해결. 에이전트 및 로컬 IDE에서의 코드 편집 편의성 극대화를 위해 Jupytext와 파일 시스템 감시(Watchdog) 도입.

### 설치 및 가상환경 세팅
```bash
# 1. 가상환경 생성 및 활성화
python3 -m venv .venv
source .venv/bin/activate

# 2. 필수 패키지 설치
pip install -r requirements.txt
```

### 🚀 실시간 감시 스크립 실행 (`tools/jupytext_watch.py`)
에디터(VS Code, PyCharm 등)에서 `notebooks/*.py` 코드 편집/저장 시 자동으로 `.ipynb` 파일 갱신(Sync)하는 유틸리티.

```bash
# 프로젝트 최상위 루트에서 실행
python tools/jupytext_watch.py
```
> **💡 작동 원리**:
> - 터미널 백그라운드에 스크립트 실행. IDE 코드 작성 시 JSON 깨짐 걱정 없이 `.py` 파일만 안전하게 편집 가능.
> - 저장 즉시 폴더 내 노트북 파일 자동 업데이트.

---

## ⚡ 3. 파이썬 스크립트 기반 핵심 로직 활용 (`src/core/yolo_core.py`)

기존 노트북 중심 스파게티 코드 탈피. 핵심 기능 `src/core/yolo_core.py`에 집약. 파이썬 배치 스크립트에서 손쉽게 사용 가능.

### 핵심 API 명세
*   `setup_dataset(zip_path, extract_path)`: ZIP 데이터셋 안전 압축 해제
*   `get_class_names(dataset_yaml_path)`: YAML에서 가구 클래스 추출
*   `train_yolo(data_yaml, epochs)`: 도메인 갭(오래된 스캔본) 방어용 데이터 증강(회전, 원근, 블러 등) 주입된 YOLO 모델 훈련 엔진
*   `visualize_inference(model, val_img_dir, val_lbl_dir)`: 임의 테스트 셋 대상 Ground Truth와 AI Prediction 결과 대조 플로팅

### 외부 파이썬 스크립트 연동 예제
```python
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..'))) # 프로젝트 루트 참조
from src.core import yolo_core

# 1. 모델 학습
model, results = yolo_core.train_yolo(epochs=30)

# 2. 결과 시각화
yolo_core.visualize_inference(model)
```

---

## ☁️ 4. Google Colab GPU 환경 실행 가이드

1. **데이터 마운트**: 구글 드라이브 내 `MyDrive/` 위치에 `floorplan_dataset_yolo.zip` 업로드
2. **노트북 열기**: 본 저장소를 Colab에서 열고 `notebooks/floorplan_training.ipynb` 선택
3. **런타임 세팅**: 'T4 GPU' 등 하드웨어 가속기 설정
4. 노트북 셀 차례로 실행. `src` 하위 코어 로직 자동 `import`하여 고속 학습 및 성능 평가 진행.

---

## 🛠️ 5. 향후 확장: BIM/3D 모델하우스 연동 계획
현재 구축된 베이스라인 모델로 도면 속 가구 탐지.
탐지된 가구의 **클래스(Class)**, **중심 좌표(Center XY)**, **너비/높이** 데이터를 JSON 스포닝(Spawning). Three.js 기반 웹 가상 공간에 3D 모델(gltf) 자동 배치. "2D-to-3D Auto Extrusion Pipeline"으로 고도화 예정.
