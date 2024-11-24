cmake_minimum_required(VERSION 3.8)
project(yolo_detection)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

# Copy the requirements file
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/requirements.txt
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
)

ament_package()