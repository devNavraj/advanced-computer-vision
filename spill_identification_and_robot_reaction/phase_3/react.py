import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision import models
import PIL.Image as PIL_Image

class VisionMover(Node):
    def __init__(self):
        super().__init__('vision_mover')
        self.cam_subscription = self.create_subscription(Image, '/depth_cam/rgb/image_raw', self.image_callback, 1)
        self.cv_bridge = CvBridge()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.latest_image = None

        # Publisher for movement
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Paths to model and class labels (update these paths as needed)
        self.model_path = '/home/ubuntu/48288209/comp8430_week12/src/react_topic/react_topic/3FineCategoryModel.pth'
        self.class_labels_path = '/home/ubuntu/48288209/comp8430_week12/src/react_topic/react_topic/class_labels.txt'

        # Load class labels
        try:
            with open(self.class_labels_path, "r") as f:
                self.class_labels = [line.strip() for line in f.readlines()]
            self.get_logger().info(f"Loaded {len(self.class_labels)} class labels.")
        except Exception as e:
            self.get_logger().error(f"Failed to load class labels: {e}")
            self.class_labels = [f"Class {i+1}" for i in range(3)]  # fallback

        # Load model
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, len(self.class_labels))
        # The problem last time is that the pth was saved as state_dict, not the actual model. I addressed that here
        try:
            state_dict = torch.load(self.model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            self.get_logger().info(f"Model loaded from {self.model_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")

        model.to(self.device).eval()
        self.model = model

        # Image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        #I only want the robot to take a photo once per second. Gives us a lot of time to react if something happens 
        self.timer = self.create_timer(1.0, self.process_image)

    def image_callback(self, msg):
        try:
            img_bgr = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            self.latest_image = img_rgb
        except Exception as e:
            self.get_logger().warn(f"Image conversion failed: {e}")

    def process_image(self):
        if self.latest_image is None:
            self.get_logger().warn("No image received yet.")
            return

        # Preprocess and classify
        try:
            input_tensor = self.transform(self.latest_image).unsqueeze(0).to(self.device)
        except Exception as e:
            self.get_logger().warn(f"Image transform failed: {e}")
            return

        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.softmax(output, dim=1).squeeze()

        max_prob = torch.max(probs).item()
        predicted_idx = torch.argmax(probs).item()
        predicted_label = self.class_labels[predicted_idx] if predicted_idx < len(self.class_labels) else f"Class {predicted_idx+1}"

        # This is where the classfication result is logged
        if max_prob < 0.75:
            self.get_logger().info(f"Uncertain classification (max prob: {max_prob:.2f}). Staying still.")
            linear_x, angular_z = 0.0, 0.0
        else:
            self.get_logger().info(f"Classified as '{predicted_label}' with confidence {max_prob:.2f}")
            # This is where the movement is decided based on the class
            if predicted_idx == 0:
                linear_x, angular_z = 0.2, 0.0
            elif predicted_idx == 1:
                linear_x, angular_z = 0.2, 1.0
            elif predicted_idx == 2:
                linear_x, angular_z = -0.2, 0.0
            else:
                linear_x, angular_z = 0.0, 0.0
                self.get_logger().warn("Invalid class detected. Stopping.")

        twist = Twist()
        twist.linear.x = linear_x
        twist.angular.z = angular_z
        self.cmd_vel_pub.publish(twist)

        # I want to see visually how the robot sees the spills
        try:
            img_bgr = cv2.cvtColor(self.latest_image, cv2.COLOR_RGB2BGR)
            img_with_label = self.plot_cls_labels(img_bgr, f"{predicted_label}: {max_prob:.2f}")
            cv2.imshow("Classification Results", img_with_label)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().warn(f"Failed to display image: {e}")

    def plot_cls_labels(self, img, cls_text, box_color=(0, 255, 0), text_color=(0, 0, 0)):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        margin = 5
        (text_w, text_h), _ = cv2.getTextSize(cls_text, font, font_scale, thickness)
        top_left = (margin, margin)
        bottom_right = (margin + text_w + 2 * margin, margin + text_h + 2 * margin)
        cv2.rectangle(img, top_left, bottom_right, box_color, thickness=-1)
        text_origin = (top_left[0] + margin, top_left[1] + text_h + margin // 2)
        cv2.putText(img, cls_text, text_origin, font, font_scale, text_color, thickness, lineType=cv2.LINE_AA)
        return img


def main(args=None):
    rclpy.init(args=args)
    node = VisionMover()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Shutting down...")
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()