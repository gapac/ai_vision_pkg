cmake_minimum_required(VERSION 3.5)
project(ai_vision_pkg)

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/LlamaService.srv"
)

ament_export_dependencies(rosidl_default_runtime)

install(PROGRAMS
  scripts/NIM_Llama.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()