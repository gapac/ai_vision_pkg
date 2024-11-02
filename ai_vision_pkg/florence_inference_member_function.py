import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from PIL import Image as PILImage
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
import supervision as sv
from cv2 import cvtColor, COLOR_BGR2RGB
from vision_msgs.msg import BoundingBox2D, BoundingBox2DArray, Pose2D
from std_msgs.msg import String


class FlorenceInferenceNode(Node):
    def __init__(self):
        super().__init__('florence_inference_node')

        # Declare and get parameters for selecting tasks
        self.declare_parameter('task', '<OD>')  # Default to Object Detection
        self.task = self.get_parameter('task').get_parameter_value().string_value
        self.get_logger().info(f"Task parameter set to: {self.task}")

        #add parameter text
        self.declare_parameter('text', '')  # Default to Object Detection
        self.text = self.get_parameter('text').get_parameter_value().string_value
        self.get_logger().info(f"Text parameter set to: {self.text}")

        # Initialize Florence model and processor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = ("cpu")
        self.get_logger().info(f"Using device: {self.device}")

        checkpoint = "microsoft/Florence-2-large-ft"
        try:
            self.model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True).to(self.device)
            self.processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
            self.get_logger().info("Successfully loaded model and processor.")
        except Exception as e:
            self.get_logger().error(f"Error loading model or processor: {str(e)}")

        # Set up subscribers and publishers
        self.subscription = self.create_subscription(Image, '/image_topic', self.image_callback, 10)
        self.publisher_ant = self.create_publisher(Image, '/florence/annotated_image', 10)
        self.publisher_src = self.create_publisher(Image, '/florence/source_image', 10)
        self.bbox_publisher = self.create_publisher(BoundingBox2DArray, '/florence/bounding_boxes', 10)
        self.bridge = CvBridge()
        self.get_logger().info("Initialized subscribers and publishers.")
    

    def run_inference(self, image: PILImage, task: str, text: str = ""):
        try:
            inputs = self.processor(text=task+text, images=image, return_tensors="pt").to(self.device)
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3
            )
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            response = self.processor.post_process_generation(generated_text, task=task, image_size=image.size)
            return response 
        except Exception as e:
            self.get_logger().error(f"Error running inference: {str(e)}")
            return response

        # Add this helper function to create BoundingBox2D messages

    def create_bounding_box_msg(self, detections, image_size):
        bbox_array_msg = BoundingBox2DArray()
        # Iterate through each bounding box in the xyxy array
        for bbox_coordinates in detections.xyxy:
            self.get_logger().info(f"Bounding box coordinates: {bbox_coordinates}")
            x_min, y_min, x_max, y_max = bbox_coordinates  # Unpack the coordinates
            self.get_logger().info(f"Bounding box coordinates: {x_min}, {y_min}, {x_max}, {y_max}")
            # Create a BoundingBox2D message
            bbox = BoundingBox2D()
            pose = Pose2D()
            pose.position.x = (x_min + x_max) / 2
            pose.position.y = (y_min + y_max) / 2
            bbox.center = pose
            #convert to float
            bbox.size_x = float(x_max - x_min)
            bbox.size_y = float(y_max - y_min)
            # Append the bounding box to the array message
            self.get_logger().info(f"Bounding box message: {bbox}")
            bbox_array_msg.boxes.append(bbox)
            self.get_logger().info(f"Bounding box array message: {bbox_array_msg}")
        return bbox_array_msg
    
    def image_callback(self, msg):
        self.get_logger().info("Received an image")
        try:
            # Convert ROS Image message to OpenCV image (BGR format)
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.get_logger().info("Converted ROS Image to OpenCV format.")
        except Exception as e:
            self.get_logger().error(f"Error converting ROS Image to OpenCV: {str(e)}")
            return
        try:
            # Convert BGR (OpenCV format) to RGB (PIL format)
            rgb_image = cvtColor(cv_image, COLOR_BGR2RGB)
            # Convert the RGB image to a PIL image
            pil_image = PILImage.fromarray(rgb_image)
            self.get_logger().info("Converted OpenCV image to PIL image in RGB format.")
        except Exception as e:
            self.get_logger().error(f"Error converting OpenCV image to PIL: {str(e)}")
            return

        # Run object detection task
        if self.task == "<OD>":
            try:
                #run inference function
                response = self.run_inference(pil_image,self.task)
                detections = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, response, resolution_wh=pil_image.size)
                bounding_box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
                label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)
                pil_image = bounding_box_annotator.annotate(pil_image, detections)
                pil_image = label_annotator.annotate(pil_image, detections)
                self.get_logger().info("Annotated the image with detections.")
            except Exception as e:
                self.get_logger().error(f"Error in <OD>: {str(e)}")
        
        # caption to phrase grouding task       
        elif self.task == "<CAPTION_TO_PHRASE_GROUNDING>":
            try:
                task = "<MORE_DETAILED_CAPTION>"
                #task = "<DETAILED_CAPTION>"
                #task = "<CAPTION>"
                response = self.run_inference(image=pil_image, task=task)
                text = response[task]

                task = "<CAPTION_TO_PHRASE_GROUNDING>"
                response = self.run_inference(image=pil_image, task=task, text=text)
                detections = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, response, resolution_wh=pil_image.size)
                bounding_box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
                label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)
                pil_image = bounding_box_annotator.annotate(pil_image, detections)
                pil_image = label_annotator.annotate(pil_image, detections)
                self.get_logger().info("Annotated the image for CAPTION_TO_PHRASE_GROUNDING.")
            except Exception as e:
                self.get_logger().error(f"Error annotating image for CAPTION_TO_PHRASE_GROUNDING: {str(e)}")
        
        # caption to region proposal task     
        elif self.task == "<REGION_PROPOSAL>":
            try:
                response = self.run_inference(image=pil_image, task=self.task, text=self.text)
                detections = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, response, resolution_wh=pil_image.size)
                bounding_box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
                label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)
                pil_image = bounding_box_annotator.annotate(pil_image, detections)
                pil_image = label_annotator.annotate(pil_image, detections)
                self.get_logger().info("Annotated the image for REGION_PROPOSAL.")
            except Exception as e:
                self.get_logger().error(f"Error annotating image for REGION_PROPOSAL: {str(e)}")

        # segmentation task     
        elif self.task == "<REFERRING_EXPRESSION_SEGMENTATION>":
            try:
                response = self.run_inference(image=pil_image, task=self.task, text=self.text)
                detections = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, response, resolution_wh=pil_image.size)
                mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
                label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)
                pil_image = mask_annotator.annotate(pil_image, detections)
                pil_image = label_annotator.annotate(pil_image, detections)
                self.get_logger().info("Annotated the image for REFERRING_EXPRESSION_SEGMENTATION.")
            except Exception as e:
                self.get_logger().error(f"Error annotating image for REFERRING_EXPRESSION_SEGMENTATION: {str(e)}")
        
        # Publish bounding boxes
        try:
            self.get_logger().info(f"Detections: {detections}")
            bbox_msg = self.create_bounding_box_msg(detections, pil_image.size)
            self.bbox_publisher.publish(bbox_msg)
            self.get_logger().info("Published bounding boxes.")
        except Exception as e:
            self.get_logger().error(f"Error publishing bounding boxes: {str(e)}")

        # Convert annotated PIL image back to ROS Image message
        try:
            annotated_cv_image = np.array(pil_image)
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated_cv_image, encoding="rgb8")
            self.publisher_ant.publish(annotated_msg)
            self.publisher_src.publish(msg)
            self.get_logger().info("Published florence annotated, source image.")
        except Exception as e:
            self.get_logger().error(f"Error converting annotated image to ROS message: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = FlorenceInferenceNode()
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

