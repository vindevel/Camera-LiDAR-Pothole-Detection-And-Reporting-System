import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import jetson_utils
import numpy as np

class JetsonCameraNode(Node):
    def __init__(self):
        super().__init__('jetson_camera_node')  # ROS2 노드 이름 설정
        self.publisher_ = self.create_publisher(Image, 'camera/image_raw', 10)  # 이미지 퍼블리셔 생성
        self.bridge = CvBridge()  # OpenCV <-> ROS 이미지 변환용 브릿지

        # Jetson의 CSI 카메라 입력 설정
        self.camera = jetson_utils.videoSource("csi://0?flip-method=0")

        self.timer = self.create_timer(0.05, self.timer_callback)  # 약 20Hz 주기로 콜백 실행

    def timer_callback(self):
        img = self.camera.Capture()  # Jetson 카메라 프레임 캡처

        if img is None:
            self.get_logger().warn("[WARNING] 프레임 캡처 실패")  # 프레임 캡처 실패 시 로그
            return

        # CUDA 이미지 → NumPy 배열로 변환
        np_img = jetson_utils.cudaToNumpy(img)

        # RGBA → BGR로 변환 (OpenCV 호환 형식)
        bgr_img = cv2.cvtColor(np_img, cv2.COLOR_RGBA2BGR)

        # 180도 회전 (상하 + 좌우 뒤집기)
        bgr_img = cv2.flip(bgr_img, -1)

        # ROS2 퍼블리시를 위한 이미지 메시지 생성 및 발행
        msg = self.bridge.cv2_to_imgmsg(bgr_img, encoding='bgr8')
        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)  # ROS2 초기화
    node = JetsonCameraNode()  # 노드 인스턴스 생성
    rclpy.spin(node)  # 콜백 함수 반복 실행
    node.destroy_node()  # 종료 시 노드 제거
    rclpy.shutdown()  # ROS2 종료

if __name__ == '__main__':
    main()
