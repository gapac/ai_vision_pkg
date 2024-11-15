from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution, TextSubstitution

def generate_launch_description():
    # Paths and other configurations
    image_folder_path = '/home/g22/Pictures/data_rv_ovire_na_tleh/png'
    usb_cam_config_file = '/home/g22/GitHub/ros2_package_testing_ws/src/usb_cam/config/params_custom1.yaml'
    rviz_config_file = './rviz_config/demonstracija.rviz'

    return LaunchDescription([
        # # Include the inference launch file from ai_vision_pkg
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare('ai_vision_pkg'),
                    'launch',
                    'florence_inference_launch.py'
                ])
            ])
        ),

        # SAM node
        Node(
            package='ai_vision_pkg',
            executable='sam2',
            name='sam2',
            output='screen'
        ),

        # # Launch the image_publisher node with image folder as a parameter
        # Node(
        #     package='ai_vision_pkg',
        #     executable='image_publisher',
        #     name='image_publisher_node',
        #     output='screen',
        #     parameters=[{
        #         'image_folder': image_folder_path,
        #         'apply_transform': False  # Set to True if you want to apply the transformation
        #     }]
        # ),

        # Launch the RViz2 node with a config file
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config_file]
        ),

        # # Launch the usb_cam node with a parameter file
        # Node(
        #     package='usb_cam',
        #     executable='usb_cam_node_exe',
        #     name='usb_cam_node',
        #     output='screen',
        #     parameters=[usb_cam_config_file],
        # ),

        # Include the realsense2 camera launch file with custom parameters
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare('realsense2_camera'),
                    'launch',
                    'rs_launch.py'
                ])
            ]),
            launch_arguments={
                'depth_module.depth_profile': '1280x720x30',
                'pointcloud.enable': 'true',
                'align_depth.enable' : 'true',
                # 'color_fps': '10',
                # 'depth_fps': '10',


            
            }.items()
        ),
    ])
