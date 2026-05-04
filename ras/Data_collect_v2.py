import mycamera
import cv2
import os
import time
from gpiozero import DigitalOutputDevice
from gpiozero import PWMOutputDevice

PWMA = PWMOutputDevice(18)
AIN1 = DigitalOutputDevice(22)
AIN2 = DigitalOutputDevice(27)

PWMB = PWMOutputDevice(23)
BIN1 = DigitalOutputDevice(25)
BIN2 = DigitalOutputDevice(24)

def motor_go(speed):
    AIN1.value = 0
    AIN2.value = 1
    PWMA.value = speed
    BIN1.value = 0
    BIN2.value = 1
    PWMB.value = speed

def motor_back(speed):
    AIN1.value = 1
    AIN2.value = 0
    PWMA.value = speed
    BIN1.value = 1
    BIN2.value = 0
    PWMB.value = speed
    
def motor_left(speed):
    AIN1.value = 0
    AIN2.value = 1
    PWMA.value = speed * 0.5
    BIN1.value = 0
    BIN2.value = 1
    PWMB.value = speed * 1.5
    
def motor_right(speed):
    AIN1.value = 0
    AIN2.value = 1
    PWMA.value = speed * 1.5
    BIN1.value = 0
    BIN2.value = 1
    PWMB.value = speed * 0.5

def motor_stop():
    AIN1.value = 0
    AIN2.value = 1
    PWMA.value = 0.0
    BIN1.value = 1
    BIN2.value = 0
    PWMB.value = 0.0

speedSet = 0.5

def main():
    camera = mycamera.MyPiCamera(640,480)

    # 상태별 저장 경로 분리
    filepath_go    = "/home/pi/AI_CAR/video/train_go/train_go"
    filepath_left  = "/home/pi/AI_CAR/video/train_left/train_left"
    filepath_right = "/home/pi/AI_CAR/video/train_right/train_right"

    # 저장 폴더 자동 생성
    os.makedirs("/home/pi/AI_CAR/video/train_go",    exist_ok=True)
    os.makedirs("/home/pi/AI_CAR/video/train_left",  exist_ok=True)
    os.makedirs("/home/pi/AI_CAR/video/train_right", exist_ok=True)

    i_go    = 0
    i_left  = 0
    i_right = 0
    frame_count = 0  # 5프레임 간격 저장용 카운터

    carState = "stop"
    while( camera.isOpened() ):
        
        keyValue = cv2.waitKey(10)
        
        if keyValue == ord('q'):
            break
        elif keyValue == 82:
            print("go")
            carState = "go"
            motor_go(speedSet)
        elif keyValue == 84:
            print("stop")
            carState = "stop"
            motor_stop()
        elif keyValue == 81:
            print("left")
            carState = "left"
            motor_left(speedSet)
        elif keyValue == 83:
            print("right")
            carState = "right"
            motor_right(speedSet)
        
        _, image = camera.read()
        image = cv2.flip(image,-1)
        cv2.imshow('Original', image)

        # 5프레임마다 원본 그대로 저장 (전처리 없음)
        frame_count += 1
        if frame_count % 5 == 0:
            if carState == "left":
                cv2.imwrite("%s_%05d_%03d.png" % (filepath_left, i_left, 45), image)
                i_left += 1
            elif carState == "right":
                cv2.imwrite("%s_%05d_%03d.png" % (filepath_right, i_right, 135), image)
                i_right += 1
            elif carState == "go":
                cv2.imwrite("%s_%05d_%03d.png" % (filepath_go, i_go, 90), image)
                i_go += 1
        
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()
    PWMA.value = 0.0
    PWMB.value = 0.0
