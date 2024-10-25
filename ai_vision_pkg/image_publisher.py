# image_publisher/image_publisher.py

import os
import sys
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class ImagePublisher(Node):
    def __init__(self, image_dir, apply_transform=False):
        super().__init__('image_publisher')
        self.publisher_ = self.create_publisher(Image, 'image_topic', 10)
        self.bridge = CvBridge()
        self.image_dir = image_dir
        self.image_paths = self.load_images(image_dir)
        self.index = 0
        self.apply_transform = apply_transform

        # Set the timer at 30 Hz (1/30 seconds per frame)
        self.timer = self.create_timer(1.0 / 30, self.timer_callback)

        if self.apply_transform:
            # Define the transformation matrix based on the provided rotation and translation
            self.transformation_matrix = self.get_transformation_matrix()

    def load_images(self, image_dir):
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        # Load and sort images by name to maintain order
        image_paths = [os.path.join(image_dir, f) for f in sorted(os.listdir(image_dir)) 
                       if os.path.splitext(f)[1].lower() in image_extensions]
        return image_paths

    def get_transformation_matrix(self):
        # Define the rotation matrix (3x3)
        rotation = np.array([
            [0.9999583959579468,  0.008895332925021648, -0.0020127370953559875],
            [-0.008895229548215866, 0.9999604225158691,   6.045500049367547e-05],
            [0.0020131953060626984, -4.254872692399658e-05, 0.9999979734420776]
        ])

        # Define the translation vector (3x1)
        translation = np.array([0.01485931035131216, 0.0010161789832636714, 0.0005317096947692335])

        # Construct the 4x4 transformation matrix
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation
        transformation_matrix[:3, 3] = translation

        return transformation_matrix

    def apply_transformation(self, image):
        # Assume the image is a 2D array of size (H, W) with 3 color channels (BGR format).
        height, width, _ = image.shape

        # Create a mesh grid of coordinates in the image space
        y, x = np.indices((height, width))
        homogeneous_coords = np.stack((x.ravel(), y.ravel(), np.ones_like(x).ravel()))

        # Apply the transformation matrix (assuming 3D transformation with no scaling)
        transformed_coords = self.transformation_matrix @ np.vstack((homogeneous_coords, np.zeros_like(x).ravel()))

        # Extract transformed X and Y coordinates
        x_transformed = transformed_coords[0, :].reshape(height, width)
        y_transformed = transformed_coords[1, :].reshape(height, width)

        # Use OpenCV to remap the image based on transformed coordinates
        transformed_image = cv2.remap(image, x_transformed.astype(np.float32), y_transformed.astype(np.float32), cv2.INTER_LINEAR)

        return transformed_image

    def timer_callback(self):
        if self.image_paths:
            image_path = self.image_paths[self.index]
            self.index = (self.index + 1) % len(self.image_paths)
            self.get_logger().info(f'Publishing: {image_path}')

            # Load the image
            cv_image = cv2.imread(image_path)
            if cv_image is not None:
                # Apply transformation if enabled
                if self.apply_transform:
                    cv_image = self.apply_transformation(cv_image)

                ros_image = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
                self.publisher_.publish(ros_image)
            else:
                self.get_logger().error(f'Failed to load image: {image_path}')
        else:
            self.get_logger().error('No images found in the specified directory.')

def main(args=None):
    rclpy.init(args=args)

    if len(sys.argv) < 2:
        print("Usage: ros2 run image_publisher image_publisher <image_directory> [--transform]")
        return

    image_directory = sys.argv[1]
    if not os.path.isdir(image_directory):
        print(f"The directory {image_directory} does not exist.")
        return

    # Check for optional --transform flag
    apply_transform = '--transform' in sys.argv

    image_publisher = ImagePublisher(image_directory, apply_transform)

    rclpy.spin(image_publisher)

    image_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
