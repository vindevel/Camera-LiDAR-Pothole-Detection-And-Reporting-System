import cv2
import numpy as np
import tensorrt as trt
import ctypes
import time
import board
import busio
from adafruit_pca9685 import PCA9685

# 해상도 설정
IMG_WIDTH = 1280
IMG_HEIGHT = 720
CENTER_X = IMG_WIDTH / 2

# TensorRT 엔진 로드
TRT_LOGGER = trt.Logger()
with open("lane_model_resized.trt", "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()
input_idx = engine.get_binding_index("input")
output_idx = engine.get_binding_index("output")
input_shape = engine.get_binding_shape(input_idx)
output_shape = engine.get_binding_shape(output_idx)
input_host = np.empty(input_shape, dtype=np.float32)
output_host = np.empty(output_shape, dtype=np.float32)
input_device = ctypes.c_void_p()
output_device = ctypes.c_void_p()
cuda = ctypes.CDLL("libcudart.so.10.2")
cuda.cudaMalloc(ctypes.byref(input_device), input_host.nbytes)
cuda.cudaMalloc(ctypes.byref(output_device), output_host.nbytes)

# CSI 카메라 오픈
def open_csi_camera():
    return cv2.VideoCapture(
        "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1, format=NV12 ! "
        "nvvidconv flip-method=0 ! video/x-raw, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! appsink",
        cv2.CAP_GSTREAMER
    )

cap = open_csi_camera()
assert cap.isOpened(), "[DEBUG] CSI 카메라 열기 실패"

# PCA9685 초기화
i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c, address=0x60)
pca.frequency = 1000
pca.channels[15].duty_cycle = 0xFFFF  # STBY

# 모터 채널
PWMA, AIN1, AIN2 = 8, 9, 10
PWMB, BIN1, BIN2 = 13, 11, 12

# 왼쪽 바퀴
def set_left_motor(forward=True, speed=0x9999):
    if forward:
        pca.channels[AIN1].duty_cycle = 0xFFFF
        pca.channels[AIN2].duty_cycle = 0x0000
    else:
        pca.channels[AIN1].duty_cycle = 0x0000
        pca.channels[AIN2].duty_cycle = 0xFFFF
    pca.channels[PWMA].duty_cycle = speed

# 오른쪽 바퀴 (반전 필요할 경우 여기서 swap)
def set_right_motor(forward=True, speed=0x9999):
    if forward:
        pca.channels[BIN1].duty_cycle = 0x0000  # 반전!
        pca.channels[BIN2].duty_cycle = 0xFFFF
    else:
        pca.channels[BIN1].duty_cycle = 0xFFFF
        pca.channels[BIN2].duty_cycle = 0x0000
    pca.channels[PWMB].duty_cycle = speed


def set_motor(left_speed, right_speed):
    def set_single_motor(pwm, in1, in2, speed):
        duty = int(min(abs(speed), 1.0) * 0xFFFF)
        pca.channels[pwm].duty_cycle = duty
        if speed > 0:
            pca.channels[in1].duty_cycle = 0xFFFF
            pca.channels[in2].duty_cycle = 0x0000
        elif speed < 0:
            pca.channels[in1].duty_cycle = 0x0000
            pca.channels[in2].duty_cycle = 0xFFFF
        else:
            pca.channels[in1].duty_cycle = 0x0000
            pca.channels[in2].duty_cycle = 0x0000

    set_single_motor(PWMA, AIN2, AIN1, left_speed)
    set_single_motor(PWMB, BIN1, BIN2, right_speed)

# 추론 루프
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 전처리
        img = cv2.resize(frame, (224, 224)).astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1).reshape(1, 3, 224, 224)
        np.copyto(input_host, img)

        # 추론
        cuda.cudaMemcpy(input_device, input_host.ctypes.data, input_host.nbytes, 1)
        bindings = [int(input_device.value), int(output_device.value)]
        context.execute_v2(bindings)
        cuda.cudaMemcpy(output_host.ctypes.data, output_device, output_host.nbytes, 2)

        # 추론 결과 → 조향 판단
        x_norm, y_norm = output_host[0]
        x_pixel = x_norm * (IMG_WIDTH / 2) + (IMG_WIDTH / 2)
        diff = (x_pixel - CENTER_X) / CENTER_X  # -1 ~ 1
        base_speed = 0.4
        turn = diff * 0.3

        # 바퀴 속도 설정
        left_speed = base_speed - turn
        right_speed = base_speed + turn
        set_motor(left_speed, right_speed)

except KeyboardInterrupt:
    print("[INFO]] 사용자 종료 요청")

finally:
    cap.release()
    for ch in [PWMA, AIN1, AIN2, PWMB, BIN1, BIN2]:
        pca.channels[ch].duty_cycle = 0x0000
    pca.deinit()
    cuda.cudaFree(input_device)
    cuda.cudaFree(output_device)
    print("[INFO] 주행 종료 및 자원 해제 완료")
