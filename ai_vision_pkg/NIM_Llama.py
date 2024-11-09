import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import spacy
from openai import OpenAI

class LlamaNIM(Node):
    def __init__(self):
        super().__init__('llama_nim_node')
        self.subscription = self.create_subscription(String, '/florence/paragraph', self.handle_paragraph, 10)
        self.publisher = self.create_publisher(String, '/llama/ontology', 10)
        self.nlp = spacy.load("en_core_web_sm")
        self.paragraph = None
        self.timer = self.create_timer(10.0, self.timer_callback)
        self.get_logger().info("LLama NIM node has been started.")
        
        # Initialize OpenAI client
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key="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"
        )

    def handle_paragraph(self, msg):
        self.paragraph = msg.data
        self.get_logger().info(f"Received paragraph: {self.paragraph}")

    def timer_callback(self):
        if self.paragraph:
            objects = self.prompt_llama_for_ontology(self.paragraph)
            response_msg = String()
            response_msg.data = objects
            self.publisher.publish(response_msg)
            self.get_logger().info(f"Published ontology data: {response_msg.data}")

    def prompt_llama_for_ontology(self, paragraph):
        try:
            completion = self.client.chat.completions.create(
                model="meta/llama-3.1-405b-instruct",
                messages=[{"role": "user", "content": f"Extract objects from the following paragraph: {paragraph}"}],
                temperature=0.2,
                top_p=0.7,
                max_tokens=1024,
                stream=True
            )
            objects = ""
            for chunk in completion:
                if chunk.choices[0].delta.content is not None:
                    objects += chunk.choices[0].delta.content
            return objects
        except Exception as e:
            self.get_logger().error(f"Error prompting LLama for ontology: {str(e)}")
            return ""

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
    node = LlamaNIM()
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
