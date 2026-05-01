"""
DGX용 AI_CAR Teacher 모델 정식 학습 코드

목적:
- 서브샘플링 실험으로 결정된 최적 비율을 사용해 Teacher 모델을 정식 학습한다.
- 학습된 모델은 model/lane_navigation_teacher.h5 로 저장되며,
  이후 지식증류(Step 2) → 가지치기(Step 3) → INT8 양자화(Step 4) 단계의 입력이 된다.

서브샘플링 실험과의 차이:
- EXPERIMENTS 리스트 없이 단일 비율 고정 (TEACHER_SAMPLES)
- EPOCHS 30 (정식 학습. 서브샘플링은 비교용 10이었음)
- EarlyStopping patience 10
- ReduceLROnPlateau 적용 (학습 정체 시 학습률 자동 감소)
- 데이터 증강(좌우반전 + 밝기) 적용
- 좌우반전 시 angle = 180 - angle 로 라벨도 함께 반전 (45 ↔ 135)
- stratify=Class 로 클래스 균형 유지하며 train/valid 분할
- 검증셋 배치 단위 predict (OOM 방지)

실행:
cd /data/AI_CAR/src
nohup python train_teacher.py > /data/AI_CAR/output/teacher.log 2>&1 &

산출물:
- /data/AI_CAR/model/lane_navigation_teacher.h5    ← 다음 단계의 입력 (Step 2 지식증류)
- /data/AI_CAR/model/lane_navigation_teacher.keras
- /data/AI_CAR/output/teacher_result.csv           ← 메트릭 누적 기록
- /data/AI_CAR/output/teacher/angle_distribution.png
- /data/AI_CAR/output/teacher/loss.png
- /data/AI_CAR/output/teacher/best_model.keras     ← 학습 중 best val_loss 시점
- /data/AI_CAR/output/teacher/history.pkl
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
# 1. 경로
# =====================================================
BASE_DIR = "/data/AI_CAR"
DATA_DIR = os.path.join(BASE_DIR, "data", "video")
MODEL_DIR = os.path.join(BASE_DIR, "model")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
TEACHER_DIR = os.path.join(OUTPUT_DIR, "teacher")
RESULT_CSV = os.path.join(OUTPUT_DIR, "teacher_result.csv")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEACHER_DIR, exist_ok=True)


# =====================================================
# 2. 학습 설정
# =====================================================
SEED = 42
BATCH_SIZE = 32
EPOCHS = 30                  # 정식 학습 (서브샘플링은 10이었음)
TEST_SIZE = 0.2
LEARNING_RATE = 1e-3
USE_MIXED_PRECISION = True
USE_AUGMENTATION = True      # 좌우반전 + 밝기 증강 ON

# 서브샘플링 결과로 결정된 비율 — 여기만 바꾸면 됨.
# None = 해당 폴더 전체 사용.
TEACHER_SAMPLES = {
    "train_go": 3000,
    "train_left": 3000,
    "train_right": 3000,
}

TEACHER_MODEL_NAME = "lane_navigation_teacher"


# =====================================================
# 3. 환경 설정
# =====================================================
def set_seed(seed: int = 42):
    """random / numpy / tensorflow 시드 고정으로 재현성 확보."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def setup_gpu():
    """GPU memory_growth + mixed precision 설정. MIG 9.7GB 환경 대응."""
    print("=" * 80)
    print("TensorFlow version:", tf.__version__)
    print("DATA_DIR:", DATA_DIR)
    print("MODEL_DIR:", MODEL_DIR)
    print("BATCH_SIZE:", BATCH_SIZE, " EPOCHS:", EPOCHS)
    print("TEACHER_SAMPLES:", TEACHER_SAMPLES)
    print("USE_AUGMENTATION:", USE_AUGMENTATION)

    gpus = tf.config.list_physical_devices("GPU")
    print("Detected GPUs:", gpus)
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory_growth 설정 완료")
    else:
        print("[WARNING] GPU 없음, CPU 실행")

    if USE_MIXED_PRECISION:
        mixed_precision.set_global_policy("mixed_float16")
        print("mixed precision 활성화: mixed_float16")
    print("=" * 80)


# =====================================================
# 4. 데이터 로딩 (3개 폴더 비율 적용)
# =====================================================
def parse_angle_from_filename(filename: str) -> int:
    """파일명 마지막 _ 뒤 숫자를 라벨로 사용. 예: train_go_00001_090.png -> 90"""
    name_without_ext = os.path.splitext(filename)[0]
    angle_str = name_without_ext.split("_")[-1]
    return int(angle_str)


def load_data_with_ratio(sample_config: dict, seed: int = 42) -> pd.DataFrame:
    """
    train_go / train_left / train_right 3개 폴더에서 sample_config 비율로 PNG 로딩.
    서브샘플링 스크립트와 동일한 로직.
    """
    image_paths, steering_angles, class_names = [], [], []

    for folder_name, max_count in sample_config.items():
        folder_path = os.path.join(DATA_DIR, folder_name)
        if not os.path.exists(folder_path):
            print(f"[WARNING] 폴더 없음: {folder_path}")
            continue

        files = [f for f in sorted(os.listdir(folder_path))
                 if fnmatch.fnmatch(f.lower(), "*.png")]
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
        raise RuntimeError("PNG 이미지 없음. 데이터 폴더 확인.")

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
# 5. 전처리 (서브샘플링 코드와 동일)
# =====================================================
def my_imread(image_path: str) -> np.ndarray:
    """OpenCV로 BGR 읽고 RGB로 변환."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없음: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def img_preprocess(image: np.ndarray) -> np.ndarray:
    """
    NVIDIA 도로 인식용 전처리:
    1) 하단 50%만 사용 (위쪽 하늘/배경 제거)
    2) RGB → YUV (조명 변화 강건)
    3) Gaussian Blur (노이즈 완화)
    4) 200 x 66 리사이즈 (모델 입력 사이즈)
    5) 0~1 정규화
    """
    height = image.shape[0]
    image = image[int(height / 2):, :, :]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = cv2.resize(image, (200, 66))
    image = image.astype(np.float32) / 255.0
    return image


# =====================================================
# 6. 데이터 증강 (Teacher에서 새로 추가)
# =====================================================
def random_augment(image: np.ndarray, angle: int) -> tuple:
    """
    학습 시에만 적용:
    - 좌우반전 (50% 확률) + 라벨 부호 반전 (45 ↔ 135, 90은 그대로 90)
    - 밝기 ±20% 조절 (50% 확률)

    검증/테스트 시에는 절대 호출하지 말 것.
    """
    if random.random() < 0.5:
        image = cv2.flip(image, 1)
        angle = 180 - angle  # 좌회전(45) ↔ 우회전(135), 직진(90)은 그대로

    if random.random() < 0.5:
        factor = 0.8 + random.random() * 0.4   # 0.8 ~ 1.2
        image = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    return image, angle


# =====================================================
# 7. 데이터 생성기 (증강 옵션 추가)
# =====================================================
def image_data_generator(image_paths, steering_angles, batch_size, augment=False):
    """
    무한 제너레이터.
    - augment=True: 학습용 (random_augment 적용)
    - augment=False: 검증용
    매 배치마다 batch_size 만큼 랜덤 인덱스 추출 후 디스크에서 읽어 GPU로 전달.
    """
    image_paths = np.asarray(image_paths)
    steering_angles = np.asarray(steering_angles)
    data_size = len(image_paths)

    while True:
        batch_images, batch_angles = [], []
        indices = np.random.randint(0, data_size, batch_size)

        for idx in indices:
            image = my_imread(image_paths[idx])
            angle = int(steering_angles[idx])

            if augment:
                image, angle = random_augment(image, angle)

            image = img_preprocess(image)
            batch_images.append(image)
            batch_angles.append(angle)

        yield (
            np.asarray(batch_images, dtype=np.float32),
            np.asarray(batch_angles, dtype=np.float32),
        )


# =====================================================
# 8. 모델 정의 (서브샘플링 코드와 동일한 NVIDIA CNN)
# =====================================================
def nvidia_model() -> tf.keras.Model:
    """
    NVIDIA End-to-End Self-Driving Car CNN.
    입력 (66, 200, 3) — img_preprocess 결과와 매칭.
    출력 1차원 — 조향각 회귀 (45/90/135 정수가 아닌 실수 예측).
    """
    model = Sequential(name="Nvidia_CNN_Teacher")
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
    # mixed precision 환경에서 회귀 출력은 float32 권장 (수치 안정성)
    model.add(Dense(1, dtype="float32"))
    model.compile(loss="mse", optimizer=Adam(learning_rate=LEARNING_RATE), metrics=["mae"])
    return model


# =====================================================
# 9. 결과 저장 유틸
# =====================================================
def save_angle_histogram(df: pd.DataFrame, exp_dir: str):
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.hist(df["Angle"], bins=25, width=1)
    ax.set_title("Steering Angle Distribution - Teacher")
    ax.set_xlabel("Angle"); ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(os.path.join(exp_dir, "angle_distribution.png"))
    plt.close(fig)


def save_loss_plot(history, exp_dir: str):
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(history.history["loss"], label="train_loss")
    ax.plot(history.history["val_loss"], label="val_loss")
    ax.set_title("Teacher Training / Validation Loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss"); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(exp_dir, "loss.png"))
    plt.close(fig)


def append_result_csv(row: dict):
    file_exists = os.path.exists(RESULT_CSV)
    with open(RESULT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# =====================================================
# 10. 메인 학습
# =====================================================
def main():
    set_seed(SEED)
    setup_gpu()

    df = load_data_with_ratio(TEACHER_SAMPLES, seed=SEED)
    save_angle_histogram(df, TEACHER_DIR)

    X = df["ImagePath"].values
    y = df["Angle"].values

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=SEED,
        shuffle=True,
        stratify=df["Class"].values,
    )

    steps_per_epoch = max(1, math.ceil(len(X_train) / BATCH_SIZE))
    validation_steps = max(1, math.ceil(len(X_valid) / BATCH_SIZE))
    print("Training data:", len(X_train), " Validation data:", len(X_valid))
    print("steps_per_epoch:", steps_per_epoch, " validation_steps:", validation_steps)

    train_generator = image_data_generator(X_train, y_train, BATCH_SIZE, augment=USE_AUGMENTATION)
    valid_generator = image_data_generator(X_valid, y_valid, BATCH_SIZE, augment=False)

    model = nvidia_model()
    model.summary()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = os.path.join(TEACHER_DIR, "best_model.keras")
    final_keras_path = os.path.join(MODEL_DIR, f"{TEACHER_MODEL_NAME}.keras")
    final_h5_path = os.path.join(MODEL_DIR, f"{TEACHER_MODEL_NAME}.h5")
    history_path = os.path.join(TEACHER_DIR, "history.pkl")

    callbacks = [
        # val_loss가 가장 낮을 때마다 모델 저장
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path, monitor="val_loss",
            save_best_only=True, verbose=1,
        ),
        # val_loss가 10 epoch 동안 개선 안 되면 학습 중단 + best 가중치 복원
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10,
            restore_best_weights=True, verbose=1,
        ),
        # val_loss가 3 epoch 정체되면 LR을 절반으로
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3,
            min_lr=1e-6, verbose=1,
        ),
    ]

    print("학습 시작 — Teacher 모델")
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

    model.save(final_keras_path)
    model.save(final_h5_path)
    with open(history_path, "wb") as f:
        pickle.dump(history.history, f)
    save_loss_plot(history, TEACHER_DIR)

    # 검증셋 배치 단위 예측 (OOM 방지)
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

    result = {
        "model_name": TEACHER_MODEL_NAME,
        "timestamp": timestamp,
        "samples": str(TEACHER_SAMPLES),
        "augmentation": USE_AUGMENTATION,
        "total_count": len(df),
        "batch_size": BATCH_SIZE,
        "epochs_requested": EPOCHS,
        "epochs_ran": len(history.history["loss"]),
        "best_val_loss": float(min(history.history["val_loss"])),
        "valid_mse": float(mse),
        "valid_mae": float(mae),
        "valid_r2": float(r2),
        "elapsed_sec": float(elapsed_sec),
        "model_path": final_h5_path,
    }
    append_result_csv(result)

    print("=" * 80)
    print("Teacher 모델 학습 완료")
    print("저장 경로:", final_h5_path)
    print(f"valid_mse: {mse:.3f}  valid_mae: {mae:.3f}  valid_r2: {r2:.3f}")
    print(f"학습 시간: {elapsed_sec:.1f}초 ({elapsed_sec/60:.1f}분)")
    print("=" * 80)


if __name__ == "__main__":
    main()
