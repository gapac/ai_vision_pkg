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
import spacy
from ai_vision_pkg.srv import LlamaService


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

        checkpoint = "microsoft/Florence-2-base-ft"
        try:
            self.model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True).to(self.device)
            self.processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
            self.get_logger().info("Successfully loaded model and processor.")
        except Exception as e:
            self.get_logger().error(f"Error loading model or processor: {str(e)}")

        # Initialize LLama service client
        self.llama_client = self.create_client(LlamaService, 'llama_service')
        while not self.llama_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('LLama service not available, waiting again...')

        # Set up subscribers and publishers
        self.subscription = self.create_subscription(Image, '/image_topic', self.image_callback, 10)
        self.publisher_ant = self.create_publisher(Image, '/florence/annotated_image', 10)
        self.publisher_src = self.create_publisher(Image, '/florence/source_image', 10)
        self.bbox_publisher = self.create_publisher(BoundingBox2DArray, '/florence/bounding_boxes', 10)
        self.label_publisher = self.create_publisher(String, '/florence/labels', 10)
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
        # Iterate through each boundin#g box in the xyxy array
        for bbox_coordinates in detections.xyxy:
            #self.get_logger().info(f"Bounding box coordinates: {bbox_coordinates}")
            x_min, y_min, x_max, y_max = bbox_coordinates  # Unpack the coordinates
            #self.get_logger().info(f"Bounding box coordinates: {x_min}, {y_min}, {x_max}, {y_max}")
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
            #self.get_logger().info(f"Bounding box message: {bbox}")
            bbox_array_msg.boxes.append(bbox)
            #self.get_logger().info(f"Bounding box array message: {bbox_array_msg}")
        return bbox_array_msg

    def handle_object_detection(self, pil_image):
        try:
            response = self.run_inference(pil_image, self.task)
            detections = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, response, resolution_wh=pil_image.size)
            bounding_box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
            label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)
            pil_image = bounding_box_annotator.annotate(pil_image, detections)
            pil_image = label_annotator.annotate(pil_image, detections)
            self.get_logger().info("Annotated the image with detections.")
            return detections, pil_image
        except Exception as e:
            self.get_logger().error(f"Error in <OD>: {str(e)}")
            return None, pil_image

    def handle_caption_to_phrase_grounding(self, pil_image, text="<MORE_DETAILEDCAPTION>"):
        try:
            if text == "<MORE_DETAILEDCAPTION>":
                response = self.run_inference(image=pil_image, task=text)
                paragraph = response[task]
                self.get_logger().info(f"Extracted paragraph: {paragraph}")
                text = paragraph
                # import spacy
                # nlp = spacy.load("en_core_web_sm")
                # def extract_and_combine_objects(paragraph):
                #     doc = nlp(paragraph)
                #     all_objects = []
                #     for sent in doc.sents:
                #         for token in sent:
                #             if token.dep_ in ("dobj", "iobj", "pobj"):
                #                 all_objects.append(token.lemma_)
                #     combined_objects = ', '.join(all_objects)
                #     return combined_objects
                # text = extract_and_combine_objects(paragraph)
                # self.get_logger().info(f"Extracted and combined objects: {text}")
            task = "<CAPTION_TO_PHRASE_GROUNDING>"
            response = self.run_inference(image=pil_image, task=task, text=text)
            detections = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, response, resolution_wh=pil_image.size)
            bounding_box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
            label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)
            pil_image = bounding_box_annotator.annotate(pil_image, detections)
            pil_image = label_annotator.annotate(pil_image, detections)
            self.get_logger().info("Annotated the image for CAPTION_TO_PHRASE_GROUNDING.")
            return detections, pil_image
        except Exception as e:
            self.get_logger().error(f"Error annotating image for CAPTION_TO_PHRASE_GROUNDING: {str(e)}")
            return None, pil_image

    def handle_region_proposal(self, pil_image):
        try:
            response = self.run_inference(image=pil_image, task=self.task, text=self.text)
            detections = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, response, resolution_wh=pil_image.size)
            bounding_box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
            label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)
            pil_image = bounding_box_annotator.annotate(pil_image, detections)
            pil_image = label_annotator.annotate(pil_image, detections)
            self.get_logger().info("Annotated the image for REGION_PROPOSAL.")
            return detections, pil_image
        except Exception as e:
            self.get_logger().error(f"Error annotating image for REGION_PROPOSAL: {str(e)}")
            return None, pil_image

    def handle_referring_expression_segmentation(self, pil_image):
        try:
            response = self.run_inference(image=pil_image, task=self.task, text=self.text)
            detections = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, response, resolution_wh=pil_image.size)
            mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
            label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)
            pil_image = mask_annotator.annotate(pil_image, detections)
            pil_image = label_annotator.annotate(pil_image, detections)
            self.get_logger().info("Annotated the image for REFERRING_EXPRESSION_SEGMENTATION.")
            return detections, pil_image
        except Exception as e:
            self.get_logger().error(f"Error annotating image for REFERRING_EXPRESSION_SEGMENTATION: {str(e)}")
            return None, pil_image

    def handle_caption(self, pil_image, task):
        try:
            response = self.run_inference(image=pil_image, task=task)
            caption = response[task]
            self.get_logger().info(f"Generated caption: {caption}")
            return caption, pil_image
        except Exception as e:
            self.get_logger().error(f"Error generating caption: {str(e)}")
            return None, pil_image

    def handle_dense_region_caption(self, pil_image):
        try:
            task = "<DENSE_REGION_CAPTION>"
            response = self.run_inference(image=pil_image, task=task)
            detections = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, response, resolution_wh=pil_image.size)
            bounding_box_annotator = sv.BoundingBoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
            label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)
            pil_image = bounding_box_annotator.annotate(pil_image, detections)
            pil_image = label_annotator.annotate(pil_image, detections)
            self.get_logger().info("Annotated the image for DENSE_REGION_CAPTION.")
            return detections, pil_image
        except Exception as e:
            self.get_logger().error(f"Error annotating image for DENSE_REGION_CAPTION: {str(e)}")
            return None, pil_image

    def call_llama_service(self, paragraph):
        try:
            request = LlamaService.Request()
            request.paragraph = paragraph
            future = self.llama_client.call_async(request)
            rclpy.spin_until_future_complete(self, future)
            if future.result() is not None:
                return future.result().objects
            else:
                self.get_logger().error('LLama service call failed')
                return []
        except Exception as e:
            self.get_logger().error(f"Error calling LLama service: {str(e)}")
            return []

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


        detections = None
        if self.task == "<OD>":
            detections, pil_image = self.handle_object_detection(pil_image)
        elif self.task == "<CAPTION_TO_PHRASE_GROUNDING>":
            detections, pil_image = self.handle_caption_to_phrase_grounding(pil_image)
        elif self.task == "<REGION_PROPOSAL>":
            detections, pil_image = self.handle_region_proposal(pil_image)
        elif self.task == "<REFERRING_EXPRESSION_SEGMENTATION>":
            detections, pil_image = self.handle_referring_expression_segmentation(pil_image)
        elif self.task == "<MORE_DETAILED_CAPTION>" or self.task == "<DETAILED_CAPTION>" or self.task == "<CAPTION>":
            caption, pil_image = self.handle_caption(pil_image, task=self.task)
        elif self.task == "<DENSE_REGION_CAPTION>":
            detections, pil_image = self.handle_dense_region_caption(pil_image)
        elif self.task == "<CUSTOM_CAPTION_SPACY_PHRASE_GROUNDING>":
            paragraph, pil_image = self.handle_caption(pil_image, task="<MORE_DETAILED_CAPTION>")
            # Code that extracts objects from paragraph and changes to singular form
            nlp = spacy.load("en_core_web_sm")
            def extract_and_combine_objects(paragraph):
                doc = nlp(paragraph)
                all_objects = []
                for sent in doc.sents:
                    for token in sent:
                        if token.dep_ in ("dobj", "iobj", "pobj"):
                            all_objects.append(token.lemma_)
                combined_objects = ', '.join(all_objects)
                return combined_objects
            text = extract_and_combine_objects(paragraph)
            self.get_logger().info(f"Extracted and combined objects: {text}")
            detections, pil_image = self.handle_caption_to_phrase_grounding(pil_image, text=text)
        
        elif self.task == "<CUSTOM_CAPTION_ONTOLOGY_PHRASE_GROUNDING>":
            paragraph, pil_image = self.handle_caption(pil_image, task="<MORE_DETAILED_CAPTION>")
            # Service call that sends paragraph to LLama that returns a list of objects
            objects = self.call_llama_service(paragraph)
            text = ', '.join(objects)
            self.get_logger().info(f"Objects from LLama service: {text}")
            detections, pil_image = self.handle_caption_to_phrase_grounding(pil_image, text=text)

        if detections:
            try:
                bbox_msg = self.create_bounding_box_msg(detections, pil_image.size)
                self.bbox_publisher.publish(bbox_msg)
                self.get_logger().info("Published bounding boxes.")
            except Exception as e:
                self.get_logger().error(f"Error publishing bounding boxes: {str(e)}")

            try:
                labels = detections.data['class_name']
                labels_msg = String()
                labels_msg.data = ', '.join(labels)
                self.label_publisher.publish(labels_msg)
                self.get_logger().info("Published labels.")
            except Exception as e:
                self.get_logger().error(f"Error publishing labels: {str(e)}")

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

