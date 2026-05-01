"""
DGX용 AI_CAR 서브샘플링 자동 실험 코드

목적:
- /data/AI_CAR/data/train_go
- /data/AI_CAR/data/train_left
- /data/AI_CAR/data/train_right

위 3개 폴더에서 폴더별 이미지 수를 다르게 샘플링하여 여러 실험을 자동 실행한다.
노트북/VSCode를 꺼도 nohup으로 계속 실행 가능하다.

실행 위치:
cd /data/AI_CAR/src

백그라운드 실행:
nohup python run_subsampling_experiments.py > /data/AI_CAR/output/subsampling_nohup.log 2>&1 &

로그 확인:
tail -f /data/AI_CAR/output/subsampling_nohup.log

결과 CSV:
/data/AI_CAR/output/subsampling_results.csv
"""

import os
import csv
import math
import random
import fnmatch
import datetime
import pickle

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


# =====================================================
# 1. 기본 경로
# =====================================================
BASE_DIR = "/data/AI_CAR"
DATA_DIR = os.path.join(BASE_DIR, "data", "video")
MODEL_DIR = os.path.join(BASE_DIR, "model")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
EXPERIMENT_DIR = os.path.join(OUTPUT_DIR, "subsampling_experiments")
RESULT_CSV = os.path.join(OUTPUT_DIR, "subsampling_results.csv")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EXPERIMENT_DIR, exist_ok=True)


# =====================================================
# 2. 학습 설정
# =====================================================
SEED = 42
BATCH_SIZE = 32          # 현재 DGX MIG 약 9.7GB 기준 안정값
EPOCHS = 10              # 빠른 비교용. 최종 학습은 20~30으로 증가 가능
TEST_SIZE = 0.2
LEARNING_RATE = 1e-3
USE_MIXED_PRECISION = True

# 실험 목록: 여기에 원하는 서브샘플링 조합을 계속 추가하면 됨
# None = 해당 폴더 전체 사용
EXPERIMENTS = [
    {
        "name": "exp01_go1000_left466_right558",
        "samples": {"train_go": 1000, "train_left": 466, "train_right": 558},
    },
    {
        "name": "exp02_go500_left466_right558",
        "samples": {"train_go": 500, "train_left": 466, "train_right": 558},
    },
    {
        "name": "exp03_go600_left466_right558",
        "samples": {"train_go": 600, "train_left": 466, "train_right": 558},
    },
    {
        "name": "exp04_go700_left466_right558",
        "samples": {"train_go": 700, "train_left": 466, "train_right": 558},
    },
    {
        "name": "exp05_go800_left466_right558",
        "samples": {"train_go": 800, "train_left": 466, "train_right": 558},
    },
    {
        "name": "exp06_go900_left466_right558",
        "samples": {"train_go": 900, "train_left": 466, "train_right": 558},
    },
    {
        "name": "exp07_go1200_left466_right558",
        "samples": {"train_go": 1200, "train_left": 466, "train_right": 558},
    },
    {
        "name": "exp08_go1500_left466_right558",
        "samples": {"train_go": 1500, "train_left": 466, "train_right": 558},
    },
    {
        "name": "exp09_go2000_left466_right558",
        "samples": {"train_go": 2000, "train_left": 466, "train_right": 558},
    },
    {
        "name": "exp10_all_data",
        "samples": {"train_go": None, "train_left": None, "train_right": None},
    },
]


# =====================================================
# 3. 환경 설정
# =====================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def setup_gpu():
    print("=" * 80)
    print("TensorFlow version:", tf.__version__)
    print("DATA_DIR:", DATA_DIR)
    print("MODEL_DIR:", MODEL_DIR)
    print("OUTPUT_DIR:", OUTPUT_DIR)
    print("BATCH_SIZE:", BATCH_SIZE)
    print("EPOCHS:", EPOCHS)

    gpus = tf.config.list_physical_devices("GPU")
    print("Detected GPUs:", gpus)

    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU memory_growth 설정 완료")
        except RuntimeError as e:
            print("GPU memory_growth 설정 실패:", e)
    else:
        print("[WARNING] GPU가 감지되지 않았습니다. CPU로 실행됩니다.")

    if USE_MIXED_PRECISION:
        mixed_precision.set_global_policy("mixed_float16")
        print("mixed precision 활성화: mixed_float16")

    print("=" * 80)


# =====================================================
# 4. 데이터 로딩 + 서브샘플링
# =====================================================
def parse_angle_from_filename(filename: str) -> int:
    """
    예시:
    train_go_00001_090.png -> 90
    train_left_00001_045.png -> 45
    train_right_00001_135.png -> 135
    """
    name_without_ext = os.path.splitext(filename)[0]
    angle_str = name_without_ext.split("_")[-1]
    return int(angle_str)


def load_data_with_subsampling(sample_config: dict, seed: int = 42) -> pd.DataFrame:
    image_paths = []
    steering_angles = []
    class_names = []

    for folder_name, max_count in sample_config.items():
        folder_path = os.path.join(DATA_DIR, folder_name)

        if not os.path.exists(folder_path):
            print(f"[WARNING] 폴더 없음: {folder_path}")
            continue

        files = [
            f for f in sorted(os.listdir(folder_path))
            if fnmatch.fnmatch(f.lower(), "*.png")
        ]

        rng = random.Random(seed)
        rng.shuffle(files)

        original_count = len(files)
        if max_count is not None:
            files = files[:min(max_count, len(files))]

        print(f"{folder_name}: 원본 {original_count}장 -> 사용 {len(files)}장")

        for filename in files:
            try:
                angle = parse_angle_from_filename(filename)
            except ValueError:
                print(f"[SKIP] 라벨 파싱 실패: {filename}")
                continue

            image_paths.append(os.path.join(folder_path, filename))
            steering_angles.append(angle)
            class_names.append(folder_name)

    if len(image_paths) == 0:
        raise RuntimeError("사용 가능한 PNG 이미지가 없습니다. /data/AI_CAR/data 하위 폴더를 확인하세요.")

    df = pd.DataFrame({
        "ImagePath": image_paths,
        "Angle": steering_angles,
        "Class": class_names,
    })

    print("총 사용 이미지 수:", len(df))
    print("클래스 분포:")
    print(df["Class"].value_counts())
    print("각도 분포:")
    print(df["Angle"].value_counts().sort_index())

    return df


# =====================================================
# 5. 전처리
# =====================================================
def my_imread(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {image_path}")

    # OpenCV는 BGR로 읽으므로 RGB로 변환
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def img_preprocess(image: np.ndarray) -> np.ndarray:
    """
    도로 인식용 전처리.
    - 하단 50%만 사용
    - RGB -> YUV 변환
    - blur
    - 200x66 resize
    - 0~1 normalization
    """
    height, _, _ = image.shape
    image = image[int(height / 2):, :, :]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = cv2.resize(image, (200, 66))
    image = image.astype(np.float32) / 255.0
    return image


# =====================================================
# 6. 데이터 생성기
# =====================================================
def image_data_generator(image_paths, steering_angles, batch_size):
    image_paths = np.asarray(image_paths)
    steering_angles = np.asarray(steering_angles)
    data_size = len(image_paths)

    while True:
        batch_images = []
        batch_angles = []

        indices = np.random.randint(0, data_size, batch_size)

        for idx in indices:
            image = my_imread(image_paths[idx])
            image = img_preprocess(image)
            angle = steering_angles[idx]

            batch_images.append(image)
            batch_angles.append(angle)

        yield np.asarray(batch_images, dtype=np.float32), np.asarray(batch_angles, dtype=np.float32)


# =====================================================
# 7. 모델
# =====================================================
def nvidia_model() -> tf.keras.Model:
    model = Sequential(name="Nvidia_CNN_AI_CAR")

    model.add(Conv2D(24, (5, 5), strides=(2, 2), input_shape=(66, 200, 3), activation="elu"))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="elu"))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="elu"))
    model.add(Conv2D(64, (3, 3), activation="elu"))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation="elu"))

    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(100, activation="elu"))
    model.add(Dense(50, activation="elu"))
    model.add(Dense(10, activation="elu"))

    # mixed precision 환경에서 회귀 출력은 float32 권장
    model.add(Dense(1, dtype="float32"))

    model.compile(
        loss="mse",
        optimizer=Adam(learning_rate=LEARNING_RATE),
        metrics=["mae"],
    )

    return model


# =====================================================
# 8. 결과 저장 유틸
# =====================================================
def save_angle_histogram(df: pd.DataFrame, exp_dir: str, exp_name: str):
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.hist(df["Angle"], bins=25, width=1)
    ax.set_title(f"Steering Angle Distribution - {exp_name}")
    ax.set_xlabel("Angle")
    ax.set_ylabel("Count")
    fig.tight_layout()

    path = os.path.join(exp_dir, "angle_distribution.png")
    fig.savefig(path)
    plt.close(fig)


def save_loss_plot(history, exp_dir: str):
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(history.history["loss"], label="train_loss")
    ax.plot(history.history["val_loss"], label="val_loss")
    ax.set_title("Training / Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    fig.tight_layout()

    path = os.path.join(exp_dir, "loss.png")
    fig.savefig(path)
    plt.close(fig)


def append_result_csv(row: dict):
    file_exists = os.path.exists(RESULT_CSV)

    with open(RESULT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def is_experiment_done(exp_dir: str) -> bool:
    return os.path.exists(os.path.join(exp_dir, "DONE.txt"))


def mark_experiment_done(exp_dir: str):
    with open(os.path.join(exp_dir, "DONE.txt"), "w", encoding="utf-8") as f:
        f.write("done\n")


# =====================================================
# 9. 단일 실험 실행
# =====================================================
def run_one_experiment(exp: dict):
    exp_name = exp["name"]
    sample_config = exp["samples"]
    exp_dir = os.path.join(EXPERIMENT_DIR, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    if is_experiment_done(exp_dir):
        print(f"[SKIP] 이미 완료된 실험: {exp_name}")
        return

    print("\n" + "=" * 80)
    print(f"실험 시작: {exp_name}")
    print("샘플링 설정:", sample_config)
    print("=" * 80)

    set_seed(SEED)
    df = load_data_with_subsampling(sample_config, seed=SEED)
    save_angle_histogram(df, exp_dir, exp_name)

    X = df["ImagePath"].values
    y = df["Angle"].values

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=SEED,
        shuffle=True,
        stratify=df["Angle"].values if len(np.unique(y)) > 1 else None,
    )

    steps_per_epoch = max(1, math.ceil(len(X_train) / BATCH_SIZE))
    validation_steps = max(1, math.ceil(len(X_valid) / BATCH_SIZE))

    print("Training data:", len(X_train))
    print("Validation data:", len(X_valid))
    print("steps_per_epoch:", steps_per_epoch)
    print("validation_steps:", validation_steps)

    train_generator = image_data_generator(X_train, y_train, BATCH_SIZE)
    valid_generator = image_data_generator(X_valid, y_valid, BATCH_SIZE)

    model = nvidia_model()

    checkpoint_path = os.path.join(exp_dir, "best_model.keras")
    final_model_path = os.path.join(exp_dir, "final_model.keras")
    h5_model_path = os.path.join(exp_dir, "final_model.h5")
    history_path = os.path.join(exp_dir, "history.pkl")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    start_time = datetime.datetime.now()

    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=valid_generator,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1,
    )

    end_time = datetime.datetime.now()
    elapsed_sec = (end_time - start_time).total_seconds()

    model.save(final_model_path)
    model.save(h5_model_path)

    with open(history_path, "wb") as f:
        pickle.dump(history.history, f)

    save_loss_plot(history, exp_dir)

    # 검증셋을 배치 단위로 예측해서 MSE / R2 / MAE 계산
    # (전체를 한 번에 메모리에 올리면 all_data 실험 등에서 OOM 발생 가능)
    y_pred_list = []
    for i in range(0, len(X_valid), BATCH_SIZE):
        batch_paths = X_valid[i:i + BATCH_SIZE]
        batch_imgs = np.asarray(
            [img_preprocess(my_imread(p)) for p in batch_paths],
            dtype=np.float32,
        )
        batch_pred = model.predict(batch_imgs, batch_size=BATCH_SIZE, verbose=0).reshape(-1)
        y_pred_list.append(batch_pred)
    y_pred = np.concatenate(y_pred_list)

    mse = mean_squared_error(y_valid, y_pred)
    mae = mean_absolute_error(y_valid, y_pred)
    r2 = r2_score(y_valid, y_pred)

    class_counts = df["Class"].value_counts().to_dict()
    angle_counts = df["Angle"].value_counts().to_dict()

    result = {
        "experiment": exp_name,
        "go_count": class_counts.get("train_go", 0),
        "left_count": class_counts.get("train_left", 0),
        "right_count": class_counts.get("train_right", 0),
        "angle_45_count": angle_counts.get(45, 0),
        "angle_90_count": angle_counts.get(90, 0),
        "angle_135_count": angle_counts.get(135, 0),
        "total_count": len(df),
        "batch_size": BATCH_SIZE,
        "epochs_requested": EPOCHS,
        "epochs_ran": len(history.history["loss"]),
        "best_val_loss": float(min(history.history["val_loss"])),
        "final_train_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
        "valid_mse": float(mse),
        "valid_mae": float(mae),
        "valid_r2": float(r2),
        "elapsed_sec": float(elapsed_sec),
        "best_model_path": checkpoint_path,
        "final_model_path": final_model_path,
    }

    append_result_csv(result)
    mark_experiment_done(exp_dir)

    print(f"실험 완료: {exp_name}")
    print("MSE:", mse)
    print("MAE:", mae)
    print("R2:", r2)
    print("소요 시간(초):", elapsed_sec)

    # 다음 실험 전 GPU 메모리 정리
    tf.keras.backend.clear_session()


# =====================================================
# 10. 전체 실험 실행
# =====================================================
def main():
    setup_gpu()

    print("총 실험 수:", len(EXPERIMENTS))
    for exp in EXPERIMENTS:
        run_one_experiment(exp)

    print("\n모든 서브샘플링 실험 완료")
    print("결과 CSV:", RESULT_CSV)


if __name__ == "__main__":
    main()
