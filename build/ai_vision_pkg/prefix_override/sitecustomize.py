import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/g22/GitHub/ros2_package_testing_ws/src/ai_vision_pkg/install/ai_vision_pkg'
