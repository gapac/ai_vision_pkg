import rclpy
from rclpy.node import Node
from vision_msgs.msg import BoundingBox2DArray  # Import the BoundingBox2DArray message type
import cv2
import torch
import numpy as np
import supervision as sv
from cv_bridge import CvBridge
from cv2 import cvtColor, COLOR_BGR2RGB
import os
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

class SAM2InferenceNode(Node):


    def __init__(self):
        super().__init__('sam2_inference_node')
        self.subscription = self.create_subscription(BoundingBox2DArray,'/florence/bounding_boxes',self.detection_callback,10)
        self.subscription = self.create_subscription(Image, '/image_topic', self.image_callback, 10)
        self.publisher_ = self.create_publisher(Image, '/SAM2/segmented_image', 10)
        self.bridge = CvBridge()
        self.get_logger().info("Initialized subscribers and publishers.")
        # Initialize SAM2 model
        try:
            torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.CHECKPOINT = "/home/g22/GitHub/sam2/checkpoints/sam2.1_hiera_small.pt"
            self.CONFIG = "sam2_hiera_s.yaml"
            self.sam2_model = build_sam2(self.CONFIG, self.CHECKPOINT, device=self.DEVICE, apply_postprocessing=False)
            self.mask_generator = SAM2AutomaticMaskGenerator(self.sam2_model)
            self.predictor = SAM2ImagePredictor(self.sam2_model)
            self.get_logger().info('SAM2 model initialized successfully')
        except Exception as e:
            self.get_logger().error(f"Error initializing SAM2 model: {str(e)}")
    

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


    def detection_callback(self, msg):
        self.get_logger().info('Received detection message')
        try:
            self.bounding_boxes = self.parse_bounding_boxes(msg)
            self.get_logger().info(f'Parsed bounding boxes: {self.bounding_boxes}')
        except Exception as e:
            self.get_logger().error(f"Error in detection callback: {str(e)}")
    

    def image_callback(self, msg):
        self.get_logger().info("Received an image")
        try:
            # Convert ROS Image message to OpenCV image (BGR format)
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.rgb_image = cvtColor(cv_image, COLOR_BGR2RGB)
            # Convert the RGB image to a PIL image
            # pil_image = PILImage.fromarray(rgb_image)
            self.get_logger().info("Converted ROS Image to OpenCV format.")
        except Exception as e:
            self.get_logger().error(f"Error converting ROS Image to OpenCV: {str(e)}")

        try:
            sam2_result = self.mask_generator.generate(self.rgb_image)
            mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
            detections = sv.Detections.from_sam(sam_result=sam2_result)
            segmented_image = mask_annotator.annotate(scene=self.rgb_image.copy(), detections=detections)

            # # Load the image (assuming you have a way to get the image corresponding to the bounding boxes)
            # image_path = '/path/to/your/image.jpg'  # Replace with the actual image path
            # image_bgr = cv2.imread(image_path)
            # if image_bgr is None:
            #     self.get_logger().error(f"Failed to load image from path: {image_path}")
            #     return
            # image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            # self.predictor.set_image(image_rgb)
            # # Convert bounding boxes to the format expected by SAM2
            # boxes = np.array([[box['x'], box['y'], box['x'] + box['width'], box['y'] + box['height']] for box in bounding_boxes])
            # masks, scores, logits = self.predictor.predict(
            #     box=boxes,
            #     multimask_output=False
            # )
            # # With one box as input, predictor returns masks of shape (1, H, W);
            # # with N boxes, it returns (N, 1, H, W).
            # if boxes.shape[0] != 1:
            #     masks = np.squeeze(masks)
            # box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
            # mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
            # detections = sv.Detections(
            #     xyxy=sv.mask_to_xyxy(masks=masks),
            #     mask=masks.astype(bool)
            # )
            # source_image = box_annotator.annotate(scene=image_bgr.copy(), detections=detections)
            # segmented_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)
            # sv.plot_images_grid(
            #     images=[source_image, segmented_image],
            #     grid_size=(1, 2),
            #     titles=['source image', 'segmented image']
            # )
            self.get_logger().info("Secsessfully segmented image.")
        except Exception as e:
            self.get_logger().error(f"Error segmenting image: {str(e)}")

        # Convert annotated PIL image back to ROS Image message
        try:
            #annotated_cv_image = np.array(pil_image)
            annotated_msg = self.bridge.cv2_to_imgmsg(segmented_image, encoding="rgb8")
            self.publisher_.publish(annotated_msg)
            self.get_logger().info("Published annotated image.")
        except Exception as e:
            self.get_logger().error(f"Error converting annotated image to ROS message: {str(e)}")

            return


def main(args=None):
    rclpy.init(args=args)
    node = SAM2InferenceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# import rclpy
# from rclpy.node import Node
# from vision_msgs.msg import BoundingBox2DArray
# import cv2
# import torch
# import numpy as np
# import supervision as sv

# from sam2.build_sam import build_sam2
# from sam2.sam2_image_predictor import SAM2ImagePredictor
# from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


# class SAM2InferenceNode(Node):
#     def __init__(self):
#         super().__init__('sam2_inference_node')
#         self.subscription = self.create_subscription(
#             BoundingBox2DArray,
#             '/florence/bounding_boxes',
#             self.detection_callback,
#             10)
#         self.subscription  # prevent unused variable warning

#     def detection_callback(self, msg):
#         self.get_logger().info('Received detection message')
#         bounding_boxes = self.parse_bounding_boxes(msg)
#         self.get_logger().info(f'Parsed bounding boxes: {bounding_boxes}')

#     def parse_bounding_boxes(self, msg):
#         bounding_boxes = []
#         for bbox in msg.boxes:
#             box = {
#                 'x': int(bbox.center.position.x - bbox.size_x / 2),
#                 'y': int(bbox.center.position.y - bbox.size_y / 2),
#                 'width': int(bbox.size_x),
#                 'height': int(bbox.size_y),
#                 'label': ''  # Assuming labels are not included in BoundingBox2DArray
#             }
#             bounding_boxes.append(box)
#         self.get_logger().info(f'Parsed bounding boxes: {bounding_boxes}')
#         return bounding_boxes

# def main(args=None):
#     rclpy.init(args=args)
#     node = SAM2InferenceNode()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()