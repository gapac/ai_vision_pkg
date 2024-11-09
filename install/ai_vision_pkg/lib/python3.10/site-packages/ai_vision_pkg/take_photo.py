import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import sys

class ImageSaver(Node):
    def __init__(self, save_path):
        super().__init__('image_saver')
        self.bridge = CvBridge()
        self.color_image = None
        self.depth_image = None
        self.save_path = save_path
        self.image_count = 0

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.color_sub = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.color_callback,
            10)
        
        self.depth_sub = self.create_subscription(
            Image,
            '/camera/camera/aligned_depth_to_color/image_raw',
            self.depth_callback,
            10)

    def color_callback(self, msg):
        self.color_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, '16UC1')

    def save_images(self):
        if self.color_image is not None and self.depth_image is not None:
            color_path = os.path.join(self.save_path, f'color_image_{self.image_count}.png')
            depth_path = os.path.join(self.save_path, f'depth_image_{self.image_count}.png')
            cv2.imwrite(color_path, self.color_image)
            cv2.imwrite(depth_path, self.depth_image)
            self.get_logger().info(f'Saved color image to {color_path}')
            self.get_logger().info(f'Saved depth image to {depth_path}')
            self.image_count += 1
        else:
            self.get_logger().warn('No images received yet.')

def main(args=None):
    rclpy.init(args=args)
    
    if len(sys.argv) < 2:
        print("Usage: ros2 run ai_vision_pkg take_photo.py <save_path>")
        return

    save_path = sys.argv[1]
    image_saver = ImageSaver(save_path)

    try:
        while rclpy.ok():
            rclpy.spin_once(image_saver)
            input("Press Enter to save images...")
            image_saver.save_images()
    except KeyboardInterrupt:
        pass

    image_saver.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
