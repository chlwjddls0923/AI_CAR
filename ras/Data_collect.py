import mycamera
import cv2
import os
from gpiozero import DigitalOutputDevice
from gpiozero import PWMOutputDevice

# ── 모터 핀 설정 ──────────────────────────────────────────
PWMA = PWMOutputDevice(18)
AIN1 = DigitalOutputDevice(22)
AIN2 = DigitalOutputDevice(27)

PWMB = PWMOutputDevice(23)
BIN1 = DigitalOutputDevice(25)
BIN2 = DigitalOutputDevice(24)

# ── 모터 제어 함수 ─────────────────────────────────────────
def motor_go(speed):
    AIN1.value = 0; AIN2.value = 1; PWMA.value = speed
    BIN1.value = 0; BIN2.value = 1; PWMB.value = speed

def motor_back(speed):
    AIN1.value = 1; AIN2.value = 0; PWMA.value = speed
    BIN1.value = 1; BIN2.value = 0; PWMB.value = speed

def motor_left(speed):
    AIN1.value = 1; AIN2.value = 0; PWMA.value = 0.0
    BIN1.value = 0; BIN2.value = 1; PWMB.value = speed

def motor_right(speed):
    AIN1.value = 0; AIN2.value = 1; PWMA.value = speed
    BIN1.value = 1; BIN2.value = 0; PWMB.value = 0.0

def motor_stop():
    AIN1.value = 0; AIN2.value = 1; PWMA.value = 0.0
    BIN1.value = 1; BIN2.value = 0; PWMB.value = 0.0

# ── 저장 경로 설정 ─────────────────────────────────────────
# 원본 보존 원칙: 크롭/필터/반전 등 전처리 없이 원본 그대로 저장
# 전처리는 학습 코드에서 처리
BASE_PATH   = "/home/pi/AI_CAR/video"
FILEPATH_GO    = os.path.join(BASE_PATH, "train_go",    "train_go")
FILEPATH_LEFT  = os.path.join(BASE_PATH, "train_left",  "train_left")
FILEPATH_RIGHT = os.path.join(BASE_PATH, "train_right", "train_right")

# 저장 폴더 없으면 자동 생성
for path in [FILEPATH_GO, FILEPATH_LEFT, FILEPATH_RIGHT]:
    os.makedirs(os.path.dirname(path), exist_ok=True)

# ── 촬영 설정 ──────────────────────────────────────────────
SPEED_SET    = 0.5
SAVE_INTERVAL = 5   # 연속 프레임 추출 금지: 5프레임마다 1장 저장


def main():
    camera = mycamera.MyPiCamera(640, 480)

    # 상태별 독립 카운터
    i_go    = 0
    i_left  = 0
    i_right = 0

    frame_count = 0
    carState = "stop"
    DISPLAY_UNIT = 50   # 50장 단위로 표시 업데이트

    while camera.isOpened():

        keyValue = cv2.waitKey(10)

        if keyValue == ord('q'):
            break
        elif keyValue == 82:   # ↑ 직진
            carState = "go"
            motor_go(SPEED_SET)
        elif keyValue == 84:   # ↓ 정지
            carState = "stop"
            motor_stop()
        elif keyValue == 81:   # ← 좌회전
            carState = "left"
            motor_left(SPEED_SET)
        elif keyValue == 83:   # → 우회전
            carState = "right"
            motor_right(SPEED_SET)

        _, image = camera.read()
        image = cv2.flip(image, -1)   # 카메라 장착 방향 보정 (상하좌우 반전)

        # ── 50장 단위로 표시 업데이트 ────────────────────────
        disp_go    = (i_go    // DISPLAY_UNIT) * DISPLAY_UNIT
        disp_left  = (i_left  // DISPLAY_UNIT) * DISPLAY_UNIT
        disp_right = (i_right // DISPLAY_UNIT) * DISPLAY_UNIT

        cv2.putText(image, f"State: {carState}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(image, f"GO:{disp_go}  LEFT:{disp_left}  RIGHT:{disp_right}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow('Original', image)

        # ── 원본 그대로 저장 (전처리 없음) ──────────────────
        # 크롭/YUV변환/블러/리사이즈는 학습 코드에서 처리
        frame_count += 1
        if frame_count % SAVE_INTERVAL == 0:   # 5프레임마다 1장 저장
            if carState == "left":
                cv2.imwrite("%s_%05d_%03d.png" % (FILEPATH_LEFT, i_left, 45), image)
                i_left += 1
            elif carState == "right":
                cv2.imwrite("%s_%05d_%03d.png" % (FILEPATH_RIGHT, i_right, 135), image)
                i_right += 1
            elif carState == "go":
                cv2.imwrite("%s_%05d_%03d.png" % (FILEPATH_GO, i_go, 90), image)
                i_go += 1

    # ── 종료 처리 ──────────────────────────────────────────

    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    finally:
        PWMA.value = 0.0
        PWMB.value = 0.0