import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import open3d as o3d
import time  # Add this import

class PointCloudSaverNode(Node):
    def __init__(self):
        super().__init__('pointcloud_saver_node')
        
        # Declare and get the depth image topic and save path parameters
        self.declare_parameter('depth_image_topic', '/florence/depth_image')
        self.declare_parameter('save_path', './')
        depth_image_topic = self.get_parameter('depth_image_topic').get_parameter_value().string_value
        self.save_path = self.get_parameter('save_path').get_parameter_value().string_value

        # Directly subscribe to image and depth topics
        self.image_sub = self.create_subscription(Image, '/florence/source_image', self.image_callback, 10)
        self.depth_sub = self.create_subscription(Image, depth_image_topic, self.depth_callback, 10)

        self.bridge = CvBridge()
        self.rgb_image = None
        self.depth_image = None
        self.file_index = 0
        self.get_logger().info("Initialized subscribers.")

    def image_callback(self, image_msg):
        self.get_logger().info('Received image message')
        try:
            # Convert ROS Image message to OpenCV image
            self.rgb_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
            self.get_logger().info('Converted ROS Image message to OpenCV image')
            self.save_pointcloud()
        except Exception as e:
            self.get_logger().error(f"Error converting ROS Image message to OpenCV image: {str(e)}")

    def depth_callback(self, depth_msg):
        self.get_logger().info('Received depth message')
        try:
            # Convert ROS Image message to OpenCV image
            self.depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='16UC1')
            self.get_logger().info('Converted ROS Image message to OpenCV image')
        except Exception as e:
            self.get_logger().error(f"Error converting ROS Image message to OpenCV image: {str(e)}")

    def save_pointcloud(self):
        if self.rgb_image is None or self.depth_image is None:
            self.get_logger().error("No synchronized image and depth data available.")
            return

        try:
            # Convert depth image to meters
            depth_image_meters = self.depth_image.astype(np.float32) / 1000.0

            # Create an RGBD image
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(self.rgb_image),
                o3d.geometry.Image(depth_image_meters),
                convert_rgb_to_intensity=False
            )

            # Create a point cloud from the RGBD image
            pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
                o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
            )
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image,
                pinhole_camera_intrinsic
            )

            # Save the point cloud to a file with an indexed filename
            filename = f"{self.save_path}/segmented_pointcloud_{self.file_index}.ply"
            o3d.io.write_point_cloud(filename, pcd)
            self.get_logger().info(f"Saved point cloud to {filename}")
            self.file_index += 1

            # Wait for 5 seconds before the next capture
            time.sleep(10)
        except Exception as e:
            self.get_logger().error(f"Error saving point cloud: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = PointCloudSaverNode()

    try:
        rclpy.spin(node)
    except Exception as e:
        node.get_logger().error(f"Error during spin: {str(e)}")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        node.get_logger().info("Node destroyed and shutdown complete.")

if __name__ == '__main__':
    main()
