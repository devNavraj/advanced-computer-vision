import sys
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import threading
import argparse

class CamCapture(Node):
    def __init__(self, name, save_path):
        super().__init__(name)
        self.cam_subscription = self.create_subscription(
            Image,
            '/depth_cam/rgb/image_raw',
            self.image_callback,
            1
        )
        self.cv_bridge = CvBridge()
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
        self.save_id = 0
        self.latest_image = None
        self.lock = threading.Lock()

        # Start thread to wait for user input
        self.input_thread = threading.Thread(target=self.wait_for_input)
        self.input_thread.daemon = True
        self.input_thread.start()

    def image_callback(self, msg):
        image_bgr = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        with self.lock:
            self.latest_image = image_bgr
    #This is the part I added, so I can take photos every time I press something
    def wait_for_input(self):
        while rclpy.ok():
            input("Press Enter to capture image...")
            with self.lock:
                if self.latest_image is not None:
                    filename = os.path.join(self.save_path, f'image{self.save_id}.jpg')
                    cv2.imwrite(filename, self.latest_image)
                    print(f"[INFO] Saved: {filename}")
                    self.save_id += 1 
                else:
                    print("[WARN] No image received yet.")

def main(args=None):
    rclpy.init(args=args)
    parser = argparse.ArgumentParser(description='Save images from ROS2 topic')
    parser.add_argument(
        '--save_path',
        type=str,
        default='/home/ubuntu/gary/temp_test/comp8430_week8/captured_images/object'
    )
    parsed_args, unknown = parser.parse_known_args()

    node = CamCapture("capture_sub", parsed_args.save_path)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Shutting down...")
    rclpy.shutdown()
