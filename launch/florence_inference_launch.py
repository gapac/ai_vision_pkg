from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ai_vision_pkg',
            executable='florence',
            name='florence',
            #parameters=[{'task': '<REFERRING_EXPRESSION_SEGMENTATION>', 'text': 'box'}]
            parameters=[{'task': '<CAPTION_TO_PHRASE_GROUNDING>'}]
            #parameters=[{'task': '<REGION_PROPOSAL>'}]
            #parameters=[{'task': '<OD>'}]
        ),
    ])
