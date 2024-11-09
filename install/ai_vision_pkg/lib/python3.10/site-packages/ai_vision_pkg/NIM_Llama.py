import rclpy
from rclpy.node import Node
from ai_vision_pkg.srv import LlamaService
import spacy

class LlamaServiceNode(Node):
    def __init__(self):
        super().__init__('llama_service_node')
        self.srv = self.create_service(LlamaService, 'llama_service', self.handle_llama_service)
        self.nlp = spacy.load("en_core_web_sm")
        self.get_logger().info("LLama service node has been started.")

    def handle_llama_service(self, request, response):
        paragraph = request.paragraph
        objects = self.extract_and_combine_objects(paragraph)
        response.objects = objects.split(', ')
        self.get_logger().info(f"LLAMA response: {response.objects}")
        return response

    def extract_and_combine_objects(self, paragraph):
        doc = self.nlp(paragraph)
        all_objects = []
        for sent in doc.sents:
            for token in sent:
                if token.dep_ in ("dobj", "iobj", "pobj"):
                    all_objects.append(token.lemma_)
        combined_objects = ', '.join(all_objects)
        return combined_objects

def main(args=None):
    rclpy.init(args=args)
    node = LlamaServiceNode()
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
