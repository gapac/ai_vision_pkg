from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ai_vision_pkg',
            executable='florence',
            name='florence',
            parameters=[{'task': '<OD>'}]
        ),
    ])
