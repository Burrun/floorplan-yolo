# CLAUDE.md - Burrun/floorplan-yolo

## 1. 프로젝트 개요
- YOLO 기반 floorplan(도면) 오브젝트 감지 프로젝트.
- Python 환경 및 Jupytext를 이용한 Notebook 동기화 개발.

## 2. 기술 스택
- 언어: Python
- 모델: YOLO (Ultralytics 등)
- 환경 관리: venv (활성화: `source .venv/bin/activate`)
- 도구: Jupytext (Notebook-Script 동기화)

## 3. 핵심 개발 규칙
- **Jupyter 작업 워크플로우**: `.ipynb` 실시간 동기화 불필요. `.py` 파일 내 `# %%` (percent format) 기반으로 VSCode Interactive Window(Colab 원격 커널 연결)를 통해 셀 단위 실험 진행.
- **최종 출력 산출**: 모든 실험 및 코드 확정이 끝난 후 최종 제출/배포 시점에만 `jupytext --to ipynb notebooks/floorplan_training.py` 명령을 1회 실행하여 결과물 생성.
- **말투**: 항상 /caveman 단답형 사용. 미사여구 삭제, 사실만 전달.

## 4. 유용한 명령어
- 가상환경 활성화: `source .venv/bin/activate`
- 종속성 설치: `pip install -r requirements.txt`
- 최종 ipynb 내보내기: `jupytext --to ipynb notebooks/*.py`

## 5. 대용량 데이터셋 및 워크플로우 노하우 (에이전트 참고용)
- **Jupytext 동기화 폐지**: 실시간 양방향 동기화(`jupytext_watch.py`)는 충돌 우려가 있으므로 사용 안 함. 에이전트는 `.py` 소스코드 편집에만 집중할 것.
- **대용량(4GB+) 다운로드 방식**: `gdown`은 트래픽 초과 및 바이러스 스캔 제한으로 잦은 실패 발생함.
  - ➡️ Colab 환경에서는 **구글 드라이브 직접 마운트(`drive.mount`) + `subprocess`를 통한 `.7z` 직접 압축 해제** 방식을 최우선 적용할 것.
