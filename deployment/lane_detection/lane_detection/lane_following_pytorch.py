import cv2
import torch
import numpy as np
import board
import busio
import time
from adafruit_pca9685 import PCA9685
from torchvision.transforms import Compose, Resize, ToTensor, ToPILImage 

# === [1] 모델 정의 ===
class LaneRegressor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, 2, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, 3, 2, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, 2, 1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 28 * 28, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.net(x)

# === [2] 모델 로드 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LaneRegressor().to(device)
model.load_state_dict(torch.load("lane_model_2.pt", map_location=device))
model.eval()

# === [3] 전처리 정의 ===
transform = Compose([
    ToPILImage(),
    Resize((224, 224)),
    ToTensor()
])

# === [4] PCA9685 설정 ===
i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c, address=0x60)
pca.frequency = 1000
pca.channels[15].duty_cycle = 0xFFFF  # STBY

# 모터 채널
PWMA, AIN1, AIN2 = 8, 9, 10
PWMB, BIN1, BIN2 = 13, 11, 12

def set_motor(left_speed, right_speed):
    def drive(pwm, in1, in2, speed):
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

    drive(PWMA, AIN2, AIN1, left_speed)
    drive(PWMB, BIN1, BIN2, right_speed)

# === [5] 카메라 오픈 ===
gst_str = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1, format=NV12 ! "
    "nvvidconv ! "
    "video/x-raw, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! appsink max-buffers=1 drop=true sync=false"
)
cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
assert cap.isOpened(), "[DEBUG] CSI 카메라 열기 실패"

IMG_WIDTH = 1280
CENTER_X = IMG_WIDTH / 2

# === [6] 주행 루프 ===
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_tensor = transform(frame).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
        x_norm, y_norm = output[0].cpu().numpy()
        x_pixel = x_norm * (IMG_WIDTH / 2) + (IMG_WIDTH / 2)
        diff = (x_pixel - CENTER_X) / CENTER_X  # -1 ~ 1

        # 주행 로직
        base_speed = 0.4
        turn = diff * 0.3
        left_speed = base_speed - turn
        right_speed = base_speed + turn
        set_motor(left_speed, right_speed)

except KeyboardInterrupt:
    print("[INFO] 사용자 종료")

finally:
    cap.release()
    for ch in [PWMA, AIN1, AIN2, PWMB, BIN1, BIN2]:
        pca.channels[ch].duty_cycle = 0x0000
    pca.deinit()
    print("[INFO] 자원 정리 완료 및 정지")
