from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'image_topic',
            default_value='/camera/camera/color/image_raw',
            description='Topic to subscribe to for images'
        ),
        Node(
            package='ai_vision_pkg',
            executable='florence',
            name='florence',
            parameters=[{
                # 'task': '<CAPTION_TO_PHRASE_GROUNDING>',

                # 'task': '<CUSTOM_PROMPT_PHRASE_GROUNDING>',
                # 'text': 'computer mouse',

                'task': '<OD>',



                'image_topic': LaunchConfiguration('image_topic')

            #TODO add the image topic parameter to all
            #parameters=[{'task': '<REFERRING_EXPRESSION_SEGMENTATION>', 'text': 'box'}]
            #parameters=[{'task': '<CAPTION_TO_PHRASE_GROUNDING>'}]
            #parameters=[{'task': '<DENSE_REGION_CAPTION>'}]
            #parameters=[{'task': '<CUSTOM_CAPTION_ONTOLOGY_PHRASE_GROUNDING>'}]
            #parameters=[{'task': '<REGION_PROPOSAL>'}]
            #parameters=[{'task': '<OD>'}]
            #parameters=[{'task': '<CUSTOM_PROMPT_PHRASE_GROUNDING>', 'text': 'human', 'image_topic': '/image_topic'}]
            
            }]
        ),
    ])
