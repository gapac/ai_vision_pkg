import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from PIL import Image as PILImage
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
import supervision as sv

class FlorenceInferenceNode(Node):
    def __init__(self):
        super().__init__('florence_inference_node')

        # Declare and get parameters for selecting tasks
        self.declare_parameter('task', '<OD>')  # Default to Object Detection
        self.task = self.get_parameter('task').get_parameter_value().string_value
        self.get_logger().info(f"Task parameter set to: {self.task}")

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
        self.publisher_ = self.create_publisher(Image, '/florence/annotated_image', 10)
        self.bridge = CvBridge()
        self.get_logger().info("Initialized subscribers and publishers.")

    def image_callback(self, msg):
        self.get_logger().info("Received an image")

        try:
            # Convert ROS Image message to PIL image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.get_logger().info("Converted ROS Image to OpenCV format.")
        except Exception as e:
            self.get_logger().error(f"Error converting ROS Image to OpenCV: {str(e)}")
            return

        try:
            pil_image = PILImage.fromarray(cv_image)
            self.get_logger().info("Converted OpenCV image to PIL image.")
        except Exception as e:
            self.get_logger().error(f"Error converting OpenCV image to PIL: {str(e)}")
            return

        # Prepare inputs for Florence model
        try:
            inputs = self.processor(text=self.task, images=pil_image, return_tensors="pt").to(self.device)
            self.get_logger().info("Prepared inputs for model inference.")
        except Exception as e:
            self.get_logger().error(f"Error preparing inputs for model: {str(e)}")
            return

        # Generate predictions
        try:
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3
            )
            self.get_logger().info("Generated predictions from the model.")
        except Exception as e:
            self.get_logger().error(f"Model inference error: {str(e)}")
            return

        try:
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            response = self.processor.post_process_generation(generated_text, task=self.task, image_size=pil_image.size)
            self.get_logger().info("Processed generated text.")
        except Exception as e:
            self.get_logger().error(f"Error processing generated text: {str(e)}")
            return

        # Annotate the image based on detections (only for Object Detection tasks)
        if self.task == "<OD>":
            try:
                detections = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, response, resolution_wh=pil_image.size)
                bounding_box_annotator = sv.BoundingBoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
                label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)
                pil_image = bounding_box_annotator.annotate(pil_image, detections)
                pil_image = label_annotator.annotate(pil_image, detections)
                self.get_logger().info("Annotated the image with detections.")
            except Exception as e:
                self.get_logger().error(f"Error annotating image: {str(e)}")
        
        # #TODO add a caption to phrase grouding task       
        # else if self.task == "<CTG>":
        #     try:
        #         detections = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, response, resolution_wh=pil_image.size)
        #         bounding_box_annotator = sv.BoundingBoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
        #         label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)
        #         pil_image = bounding_box_annotator.annotate(pil_image, detections)
        #         pil_image = label_annotator.annotate(pil_image, detections)
        #         self.get_logger().info("Annotated the image with detections.")
        #     except Exception as e:
        #         self.get_logger().error(f"Error annotating image: {str(e)}")

        # Convert annotated PIL image back to ROS Image message
        try:
            annotated_cv_image = np.array(pil_image)
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated_cv_image, encoding="bgr8")
            self.publisher_.publish(annotated_msg)
            self.get_logger().info("Published annotated image.")
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

