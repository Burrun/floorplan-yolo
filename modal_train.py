import modal
import os
import subprocess

# 1. 앱 이름 지정
app = modal.App("floorplan-yolo-training")

# 2. 클라우드에 설치할 환경 설정
image = (
    modal.Image.debian_slim()
    .apt_install("zstd", "tar", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch",
        "torchvision",
        "ultralytics",
        "gdown",
        "pandas",
        "matplotlib",
        "sahi",
        "opencv-python-headless",
    )
)

# 3. 로컬 프로젝트 폴더를 클라우드로 마운트 (modal v1.5+ API)
image = image.add_local_dir(
    ".",
    remote_path="/root/pj",
    ignore=[
        "*.tar.gz",
        "temp_preprocessing_results",
        ".git",
        "__pycache__",
    ],
)


# 3.5 클라우드 볼륨(Volume) 생성 및 마운트
# (학습 도중 에러가 나더라도 클라우드 저장소에 결과물이 실시간 보존됩니다)
volume = modal.Volume.from_name("floorplan-runs-vol", create_if_missing=True)


# 4. 핵심 포인트: 클라우드 GPU(A10G) 할당
# - timeout=86400 (24시간)으로 설정하여 긴 학습 중 끊기지 않게 방어
@app.function(
    image=image,
    gpu="A10G",
    timeout=86400,
    volumes={"/vol/runs": volume},
)
def train_model():
    os.chdir("/root/pj")
    import shutil

    print("🔥 Modal 클라우드 GPU(A10G) 환경으로 진입했습니다! 🔥")

    # 1. 로컬에서 같이 업로드된 runs/ 폴더가 있다면 (심볼릭 링크가 아닌 실제 폴더)
    # 볼륨 마운트 경로(/vol/runs)로 덮어쓰지 않고 복사 후 원본 폴더 삭제
    if os.path.exists("runs") and not os.path.islink("runs"):
        print("📦 로컬에서 업로드된 runs/ 데이터를 클라우드 볼륨으로 이전 중...")
        os.system("cp -R -n runs/. /vol/runs/ 2>/dev/null || true")
        shutil.rmtree("runs")

    # 2. YOLO 코드가 그대로 /root/pj/runs 를 사용할 수 있게 심볼릭 링크 생성
    if not os.path.exists("runs"):
        os.symlink("/vol/runs", "runs")

    import torch

    print(f"✅ GPU 할당 확인: {torch.cuda.get_device_name(0)}")
    print(f"✅ 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB")

    # GUI 없는 환경에서 plt.show()가 멈추거나 에러나는 것을 방지 (백엔드를 Agg로 강제)
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"

    print("🚀 YOLO 모델 학습 파이프라인(floorplan_training.py) 실행 중...")
    try:
        subprocess.run(
            ["python", "notebooks/floorplan_training.py"], env=env, check=True
        )
    except subprocess.CalledProcessError as e:
        print(
            f"⚠️ 학습 중 에러가 발생했습니다 (코드: {e.returncode}). 지금까지 완료된 학습 결과만 로컬로 전송합니다."
        )
    except Exception as e:
        print(
            f"⚠️ 예기치 않은 에러 발생: {e}. 지금까지 완료된 학습 결과만 로컬로 전송합니다."
        )

    print("✅ 지금까지 완료된 학습 결과물(runs/)을 압축하여 로컬로 반환 준비 중...")
    if os.path.exists("runs"):
        subprocess.run(["tar", "-chzf", "runs_output.tar.gz", "runs/"])
    else:
        print("⚠️ runs/ 폴더가 생성되지 않았습니다. 빈 압축파일을 전송합니다.")
        with open("runs_output.tar.gz", "wb") as f:
            pass

    with open("runs_output.tar.gz", "rb") as f:
        return f.read()


# 5. 내 터미널에서 `modal run modal_train.py` 할 때 최초 진입점
@app.local_entrypoint()
def main():
    print("🚀 Modal 클라우드로 훈련 파이프라인과 로컬 코드를 전송합니다...")

    output_bytes = train_model.remote()

    print("⬇️ 클라우드 학습 완료! 결과물(runs_output.tar.gz) 수신 완료.")
    with open("runs_output.tar.gz", "wb") as f:
        f.write(output_bytes)

    print("📦 로컬 환경에 압축 해제 중... (기존 runs 폴더에 병합 또는 덮어쓰기)")
    import shutil
    try:
        shutil.unpack_archive("runs_output.tar.gz", ".")
    except Exception as e:
        print(f"⚠️ 압축 해제 중 오류 발생: {e}")

    print(
        "🎉 모든 클라우드 학습 과정이 성공적으로 끝났습니다! 로컬의 `runs/` 폴더를 확인하세요."
    )
