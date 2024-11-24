import rclpy
from nav2_costmap_2d import Layer, LayeredCostmap
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2
import numpy as np
import tf2_ros

class YOLODetectionLayer(Layer):
    def __init__(self):
        super().__init__('rgb_detection_layer')
        self.enabled_ = True
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Parameters
        self.declare_parameter('enabled', True)
        self.declare_parameter('topic', '/object_detections')
        self.declare_parameter('cost_scaling_factor', 10.0)
        self.declare_parameter('inflation_radius', 0.5)
        
        # Subscribers
        self.detection_sub = self.create_subscription(
            PoseStamped,
            self.get_parameter('topic').value,
            self.detection_callback,
            10
        )
        self.detections = []

    def onInitialize(self):
        self.current_ = True
        self.enabled_ = self.get_parameter('enabled').value
        self.rolling_window = True

    def updateBounds(self, robot_x, robot_y, robot_yaw, min_x, min_y, max_x, max_y):
        if not self.enabled_:
            return
        
        # Update bounds based on detections
        for detection in self.detections:
            px = detection.pose.position.x
            py = detection.pose.position.y
            radius = self.get_parameter('inflation_radius').value
            
            min_x = min(min_x, px - radius)
            min_y = min(min_y, py - radius)
            max_x = max(max_x, px + radius)
            max_y = max(max_y, py + radius)
        
        return min_x, min_y, max_x, max_y

    def updateCosts(self, master_grid):
        if not self.enabled_:
            return
            
        scaling_factor = self.get_parameter('cost_scaling_factor').value
        radius = self.get_parameter('inflation_radius').value
        
        for detection in self.detections:
            # Transform detection to costmap frame
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.global_frame_,
                    detection.header.frame_id,
                    rclpy.time.Time()
                )
                
                # Apply transform to detection position
                wx = detection.pose.position.x + transform.transform.translation.x
                wy = detection.pose.position.y + transform.transform.translation.y
                
                # Convert to map coordinates
                mx, my = self.worldToMap(wx, wy)
                
                # Apply cost using gaussian distribution
                for i in range(int(mx - radius), int(mx + radius)):
                    for j in range(int(my - radius), int(my + radius)):
                        if self.inBounds(i, j):
                            distance = np.hypot(i - mx, j - my)
                            if distance <= radius:
                                cost = int((1.0 - distance/radius) * scaling_factor)
                                master_grid[i][j] = max(master_grid[i][j], cost)
                                
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, 
                   tf2_ros.ExtrapolationException) as e:
                self.get_logger().warn(f'Transform failed: {e}')

    def detection_callback(self, msg):
        self.detections.append(msg)
        # Keep only recent detections
        if len(self.detections) > 10:
            self.detections.pop(0)