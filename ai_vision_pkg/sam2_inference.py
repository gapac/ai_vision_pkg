import rclpy
from rclpy.node import Node
from vision_msgs.msg import BoundingBox2DArray  # Import the BoundingBox2DArray message type
from sensor_msgs.msg import Image  # Import the Image message type from ROS 2
import cv2
import torch
import numpy as np
import supervision as sv
from cv_bridge import CvBridge
from cv2 import cvtColor, COLOR_BGR2RGB
import os
from PIL import Image as PILImage
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import message_filters

class SAM2InferenceNode(Node):
    def __init__(self):
        super().__init__('sam2_inference_node')
        self.subscription_bbox = self.create_subscription(
            BoundingBox2DArray,
            '/florence/bounding_boxes',
            self.detection_callback,
            10)
        
        # Use message filters to synchronize image and depth topics
        self.image_sub = message_filters.Subscriber(self, Image, '/florence/source_image')
        self.depth_sub = message_filters.Subscriber(self, Image, '/florence/depth_image')
        self.ts = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.depth_sub], 10, 0.1)
        self.ts.registerCallback(self.sync_callback)

        self.publisher_ = self.create_publisher(Image, '/SAM2/segmented_image', 10)
        self.src_publisher_ = self.create_publisher(Image, '/SAM2/source_image', 10)
        self.depth_publisher_ = self.create_publisher(Image, '/SAM2/segmented_depth', 10)
        self.bridge = CvBridge()
        self.bounding_boxes = []  # Initialize bounding_boxes attribute
        self.get_logger().info("Initialized subscribers and publishers.")
        
        # Initialize SAM2 model
        try:
            torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            #self.DEVICE = torch.device('cpu')
            self.CHECKPOINT = "/home/g22/GitHub/sam2/checkpoints/sam2_hiera_large.pt"
            self.CONFIG = "sam2_hiera_l.yaml"
            self.sam2_model = build_sam2(self.CONFIG, self.CHECKPOINT, device=self.DEVICE, apply_postprocessing=False)
            # Load the state dictionary with strict=False to ignore unexpected keys
            state_dict = torch.load(self.CHECKPOINT, map_location=self.DEVICE)
            filtered_state_dict = {k: v for k, v in state_dict.items() if k in self.sam2_model.state_dict()}
            self.sam2_model.load_state_dict(filtered_state_dict, strict=False)
            self.mask_generator = SAM2AutomaticMaskGenerator(self.sam2_model)
            self.predictor = SAM2ImagePredictor(self.sam2_model)
            self.get_logger().info('SAM2 model initialized successfully')
        except Exception as e:
            self.get_logger().error(f"Error initializing SAM2 model: {str(e)}")

    def detection_callback(self, msg):
        self.get_logger().info('Received detection message')
        try:
            self.bounding_boxes = self.parse_bounding_boxes(msg)
            self.get_logger().info(f'Parsed bounding boxes: {self.bounding_boxes}')
        except Exception as e:
            self.get_logger().error(f"Error in detection callback: {str(e)}")

    def parse_bounding_boxes(self, msg):
        bounding_boxes = []
        try:
            for bbox in msg.boxes:
                box = {
                    'x': int(bbox.center.position.x - bbox.size_x / 2),
                    'y': int(bbox.center.position.y - bbox.size_y / 2),
                    'width': int(bbox.size_x),
                    'height': int(bbox.size_y),
                    'label': ''  # Assuming labels are not included in BoundingBox2DArray
                }
                bounding_boxes.append(box)
            self.get_logger().info(f'Parsed bounding boxes: {bounding_boxes}')
        except Exception as e:
            self.get_logger().error(f"Error parsing bounding boxes: {str(e)}")
        return bounding_boxes

    def sync_callback(self, image_msg, depth_msg):
        self.get_logger().info('Received synchronized image and depth messages')
        try:
            # Convert ROS Image message to OpenCV image
            image_bgr = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
            self.rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            self.get_logger().info('Converted ROS Image message to OpenCV image')
        except Exception as e:
            self.get_logger().error(f"Error converting ROS Image message to OpenCV image: {str(e)}")
            return

        try:
            # Initialize variables
            source_image = None
            segmented_image = None
            # Segment the image using SAM2
            self.predictor.set_image(self.rgb_image)
            # Convert bounding boxes to the format expected by SAM2
            boxes = np.array([[box['x'], box['y'], box['x'] + box['width'], box['y'] + box['height']] for box in self.bounding_boxes])
            masks, scores, logits = self.predictor.predict(
                box=boxes,
                multimask_output=False
            )
            # Store masks for use in depth_callback
            self.masks = masks.astype(bool)
            # With one box as input, predictor returns masks of shape (1, H, W);
            # with N boxes, it returns (N, 1, H, W).
            if boxes.shape[0] != 1:
                masks = np.squeeze(masks)
            # Convert masks to xyxy format
            box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
            mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
            detections = sv.Detections(
                xyxy=sv.mask_to_xyxy(masks=masks),
                mask=masks.astype(bool)
            )
            source_image = box_annotator.annotate(scene=self.rgb_image.copy(), detections=detections)
            segmented_image = mask_annotator.annotate(scene=self.rgb_image.copy(), detections=detections)
            self.get_logger().info("Successfully segmented image.")

            # Convert ROS DEPTH Image message to OpenCV image
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='16UC1')
            self.get_logger().info('Converted ROS Depth Image message to OpenCV image')

            # Compress all masks into a single mask
            masks = np.any(masks, axis=0)

            # Apply masks to depth image
            segmented_depth = depth_image.copy()
            segmented_depth[~masks] = 0
            
            
            # Log segmented_depth.shape, segmented_depth.dtype
            self.get_logger().info(f"segmented_depth.shape: {segmented_depth.shape}, segmented_depth.dtype: {segmented_depth.dtype}")
            self.get_logger().info("Successfully segmented depth image.")

            # Convert annotated images back to ROS Image messages
            try:
                annotated_seg_msg = self.bridge.cv2_to_imgmsg(segmented_image, encoding="rgb8")
                annotated_src_msg = self.bridge.cv2_to_imgmsg(source_image, encoding="rgb8")
                segmented_depth_msg = self.bridge.cv2_to_imgmsg(segmented_depth, encoding="16UC1")
                self.publisher_.publish(annotated_seg_msg)
                self.depth_publisher_.publish(segmented_depth_msg)
                self.src_publisher_.publish(image_msg)
                self.get_logger().info("Published annotated and segmented depth images.")
            except Exception as e:
                self.get_logger().error(f"Error converting annotated image to ROS message: {str(e)}")
        except Exception as e:
            self.get_logger().error(f"Error segmenting image: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = SAM2InferenceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()