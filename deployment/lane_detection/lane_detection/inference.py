import cv2
import torch
from torchvision.transforms import Compose, Resize, ToTensor, ToPILImage

import torch
torch.backends.cudnn.benchmark = True

# 모델 클래스 정의 (Colab과 동일하게)
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

# 장치 설정 및 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LaneRegressor().to(device)
model.load_state_dict(torch.load("lane_model_2.pt", map_location=device))
model.eval()

# 전처리 함수 정의
transform = Compose([
    ToPILImage(),
    Resize((224, 224)),
    ToTensor()
])

# CSI 카메라 GStreamer 파이프라인
gst_str = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1, format=NV12 ! "
    "nvvidconv ! "
    "video/x-raw, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! appsink max-buffers=1 drop=true sync=false"
)
cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("카메라 열기 실패. GStreamer 파이프라인 확인 요망.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임 읽기 실패.")
        break

    # 전처리 (Colab과 동일한 방식)
    input_tensor = transform(frame).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        x_norm, y_norm = output[0].cpu().numpy()

    h, w = frame.shape[:2]
    x = int(x_norm * w)
    y = int(y_norm * h)

    # 시각화
    cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)
    cv2.putText(frame, f"({x},{y})", (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    cv2.imshow("Lane Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
