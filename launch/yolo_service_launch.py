from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='yolo_service',
            executable='yolo_service_node',
            name='yolo_service_node',
            output='screen'
        )
    ])