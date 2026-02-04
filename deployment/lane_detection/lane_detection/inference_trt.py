import cv2
import numpy as np
import tensorrt as trt
import ctypes
import time
import os

# 해상도 정의
IMG_WIDTH = 1280
IMG_HEIGHT = 720

# === [1] TensorRT 엔진 로드 ===
TRT_LOGGER = trt.Logger()
with open("lane_model_resized.trt", "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()

# 입력 & 출력 바인딩 인덱스
input_idx = engine.get_binding_index("input")
output_idx = engine.get_binding_index("output")

# 입력/출력 shape
input_shape = engine.get_binding_shape(input_idx)
output_shape = engine.get_binding_shape(output_idx)

# 메모리 할당 (host & device)
input_host = np.empty(input_shape, dtype=np.float32)
output_host = np.empty(output_shape, dtype=np.float32)

input_device = ctypes.c_void_p()
output_device = ctypes.c_void_p()

cuda = ctypes.CDLL('libcudart.so')
cuda.cudaMalloc(ctypes.byref(input_device), input_host.nbytes)
cuda.cudaMalloc(ctypes.byref(output_device), output_host.nbytes)

# === [2] CSI 카메라 열기 ===
def open_csi_camera():
    return cv2.VideoCapture(
        "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1, format=NV12 ! "
        "nvvidconv flip-method=0 ! video/x-raw, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! appsink",
        cv2.CAP_GSTREAMER
    )

cap = open_csi_camera()
assert cap.isOpened(), "CSI 카메라 열기 실패"

# === [3] 역정규화 함수 ===
def denormalize(x_norm, y_norm, width=IMG_WIDTH, height=IMG_HEIGHT):
    cx, cy = width / 2, height / 2
    x = x_norm * cx + cx
    y = y_norm * cy + cy
    return int(x), int(y)

# === [4] 프레임 저장 디렉토리 설정 ===
save_dir = "./samples"
os.makedirs(save_dir, exist_ok=True)

start_time = time.time()
frame_id = 0

# === [5] 추론 + 저장 루프 ===
while time.time() - start_time < 20:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess
    img = cv2.resize(frame, (224, 224)).astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1).reshape(1, 3, 224, 224).astype(np.float32)
    np.copyto(input_host, img)

    # Host → Device
    cuda.cudaMemcpy(input_device, input_host.ctypes.data, input_host.nbytes, 1)

    # Inference
    bindings = [int(input_device.value), int(output_device.value)]
    context.execute_v2(bindings)

    # Device → Host
    cuda.cudaMemcpy(output_host.ctypes.data, output_device, output_host.nbytes, 2)

    # Postprocess
    x_norm, y_norm = output_host[0]
    x, y = denormalize(x_norm, y_norm)

    # Draw and Save
    cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)
    fname = os.path.join(save_dir, f"frame_{frame_id}.jpg")
    cv2.imwrite(fname, frame)
    print(f"[{frame_id}] Saved: {fname}")
    frame_id += 1

    time.sleep(1)  # 1초 간격 저장

# === [6] 자원 해제 ===
cap.release()
cuda.cudaFree(input_device)
cuda.cudaFree(output_device)
