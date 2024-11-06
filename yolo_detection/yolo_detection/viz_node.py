# Copyright (C) 2023  Nisal Chinthana Perera

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


from typing import List, Dict

import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy

from cv_bridge import CvBridge

from std_srvs.srv import SetBool
from sensor_msgs.msg import Image

from yolo_msgs.msg import DetectionArray


class VizNode(Node):
    def __init__(self) -> None:
        super().__init__("yolo_viz_node")

        # Declare parameters
        self.declare_parameter("enable", True)
        self.declare_parameter("image_reliability", QoSReliabilityPolicy.RELIABLE)


        # Get parameter values
        self.enable = self.get_parameter("enable").value
        self.reliability = self.get_parameter("image_reliability").value

        # Set up QoS profile
        self.image_qos_profile = QoSProfile(
            reliability=self.reliability,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=5
        )

        # Create publisher, subscriber, and service
        self._pub = self.create_publisher(Image, "viz", 10)
        self._srv = self.create_service(SetBool, "enable", self.enable_callback)
        self._sub = self.create_subscription(
            DetectionArray, 
            "detections",
            self.image_callback,
            self.image_qos_profile
        )

        self.cv_bridge = CvBridge()

        self.logger = self.get_logger()
        self.logger.info(f"{self.__class__.__name__} Node created and initialized")

    def enable_callback(self, request, response):
        self.enable = request.data
        response.success = True
        return response

    def image_callback(self, msg: DetectionArray) -> None:

        if self.enable:
            # convert image visualize
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg.image, desired_encoding='bgr8')
            
            for detection in msg.detections:
                # for bbox in detection.bbox:
                start = (detection.bbox.top.x, detection.bbox.top.y)
                end = (detection.bbox.bottom.x, detection.bbox.bottom.y)
                cv_image = cv2.rectangle(cv_image, start, end, (0, 0, 255), 2)
                cv_image = cv2.putText(cv_image, detection.bbox.class_name, start, cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

            # publish detections
            viz_image = self.cv_bridge.cv2_to_imgmsg(cv_image, encoding='rgb8', header=msg.image.header)
            self._pub.publish(viz_image)

            del cv_image
            del viz_image


def main():
    rclpy.init()
    node = VizNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__=="__main__":
    main()