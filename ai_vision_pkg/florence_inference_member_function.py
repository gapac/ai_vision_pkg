#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image as PILImage
import numpy as np
import supervision as sv
from io import BytesIO


class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection_node')

        # Initialize ROS 2 subscriber to the /image_raw topic
        self.subscription = self.create_subscription(
            Image,
            '/image_topic',
            self.listener_callback,
            10
        )
        self.subscription  # Prevent unused variable warning

        # Initialize ROS 2 publisher for the processed image topic
        self.publisher_ = self.create_publisher(Image, '/image_with_detections', 10)

        # Initialize CvBridge for converting between ROS Image and OpenCV/PIL images
        self.bridge = CvBridge()

        # Load model and processor for object detection
        CHECKPOINT = "microsoft/Florence-2-large-ft"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(CHECKPOINT, trust_remote_code=True).to(self.device)
        self.processor = AutoProcessor.from_pretrained(CHECKPOINT, trust_remote_code=True)

        self.get_logger().info('ObjectDetectionNode has been initialized')

    def listener_callback(self, msg):
        # Convert the ROS image message to a PIL image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        pil_image = PILImage.fromarray(cv_image)

        # Run inference and get the response
        text = "<OD>"
        task = "<OD>"
        inputs = self.processor(text=text, images=pil_image, return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        response = self.processor.post_process_generation(generated_text, task=task, image_size=pil_image.size)
        detections = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, response, resolution_wh=pil_image.size)

        # Annotate image with bounding boxes and labels
        bounding_box_annotator = sv.BoundingBoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
        label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)

        annotated_image = bounding_box_annotator.annotate(pil_image, detections)
        annotated_image = label_annotator.annotate(annotated_image, detections)
        annotated_image.thumbnail((600, 600))

        # Convert the annotated PIL image back to a ROS message and publish it
        annotated_cv_image = np.array(annotated_image)  # Convert back to OpenCV format
        annotated_ros_image = self.bridge.cv2_to_imgmsg(annotated_cv_image, encoding='rgb8')
        self.publisher_.publish(annotated_ros_image)

        # Log a message
        self.get_logger().info('Published image with object detection annotations')

def main(args=None):
    rclpy.init(args=args)

    # Create the node
    node = ObjectDetectionNode()

    # Spin the node to keep it alive
    rclpy.spin(node)

    # Clean up when shutting down
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
