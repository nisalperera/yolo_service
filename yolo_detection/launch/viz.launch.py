from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument

def generate_launch_description():

    namespace = LaunchConfiguration("namespace")
    namespace_cmd = DeclareLaunchArgument(
        "namespace",
        default_value="yolo",
        description="Namespace for the nodes")

    #
    # NODES
    #
    viz_node_cmd = Node(
        package="yolo_detection",
        executable="viz_node",
        name="viz_node",
        namespace=namespace,
    )

    ld = LaunchDescription()
    ld.add_action(namespace_cmd)
    ld.add_action(viz_node_cmd)

    return ld