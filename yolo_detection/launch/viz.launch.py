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
    visualize = DeclareLaunchArgument(
        "visualize",
        default_value=True,
        description="Whether to use Visualizations"
    )

    #
    # NODES
    #
    detector_node_cmd = Node(
        package="yolo_detection",
        executable="viz_node",
        name="viz_node",
        namespace=namespace,
    )

    ld = LaunchDescription()
    ld.add_action(namespace_cmd)
    ld.add_action(detector_node_cmd)
    ld.add_action(visualize)

    return ld