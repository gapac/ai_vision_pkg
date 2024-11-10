import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import spacy
from openai import OpenAI
import json


class LlamaNIM(Node):
    def __init__(self):
        super().__init__('llama_nim_node')
        self.subscription = self.create_subscription(String, '/florence/paragraph', self.handle_paragraph, 10)
        self.publisher = self.create_publisher(String, '/llama/ontology', 10)
        self.nlp = spacy.load("en_core_web_sm")
        self.paragraph = None
        self.ontology = """{{
                            "objects": [
                                {{
                                    "name": "object1",
                                    "properties": {{
                                        "property1": "value1",
                                        "property2": "value2"
                                    }},
                                    "relationships": [
                                        {{
                                            "type": "relationship1",
                                            "target": "object2"
                                        }}
                                    ]
                                }},
                                {{
                                    "name": "object2",
                                    "properties": {{
                                        "property1": "value1",
                                        "property2": "value2"
                                    }},
                                    "relationships": []
                                }}
                            ]
                            }}"""
        self.previous_paragraph = None
        self.timer = self.create_timer(5.0, self.timer_callback)
        self.get_logger().info("LLama NIM node has been started.")
        
        # Initialize OpenAI client
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key="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"
        )

    def handle_paragraph(self, msg):
        self.paragraph = msg.data
        self.get_logger().info(f"Received paragraph: {self.paragraph}")
    
    def save_ontology(self, ontology, filename):
        with open(filename, 'w') as f:
            json.dump(ontology, f, indent=4)

    def timer_callback(self):
        if self.paragraph and self.paragraph != self.previous_paragraph:
            self.ontology = self.prompt_llama_for_ontology(self.paragraph)
            self.save_ontology(self.ontology, 'ontology.json')
            self.previous_paragraph = self.paragraph
            response_msg = String()
            response_msg.data = self.ontology
            self.publisher.publish(response_msg)
            self.get_logger().info(f"Published ontology data: {response_msg.data}")
        else:
            self.get_logger().info("No new paragraph to process.")

    def prompt_llama_for_ontology(self, paragraph, ):
        try:
            completion = self.client.chat.completions.create(
                model="meta/llama-3.1-405b-instruct",
                messages=[{"role": "user", "content": f""" From description add objects to existing ontology. Return new entries to ontology only if object is not already in ontology, do not add any other text and explanations. Add relevant properties that are typical for thesetypes of objects, even if not explicitly mentioned inthe text.  Description: {paragraph} Existing Ontology: {self.ontology}"""}],
                    #""" Analyze the following paragraph and use the information to identify distinct objects, their properties, and relationships. Add this information to the existing ontology, without deleting objectsadding relevant properties that are typical for thesetypes of objects, even if not explicitly mentioned inthe text. Return only the ontology without any other text and explanationsParagraph: {paragraph}Existing Ontology: {self.ontology}"""}],
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
