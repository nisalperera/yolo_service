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
import message_filters

from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy

from cv_bridge import CvBridge

from std_srvs.srv import SetBool
from sensor_msgs.msg import Image

from yolo_msgs.msg import DetectionArray


class YOLOVizNode(Node):
    def __init__(self) -> None:
        super().__init__("yolo_viz_node")

        # Declare parameters
        self.declare_parameter("enable", False)
        self.declare_parameter("image_reliability", QoSReliabilityPolicy.RELIABLE)
        self.declare_parameter("log_image", False)


        # Get parameter values
        self.enable = self.get_parameter("enable").value
        self.reliability = self.get_parameter("image_reliability").value
        self.log_image = self.get_parameter("log_image").value

        # Set up QoS profile
        self.image_qos_profile = QoSProfile(
            reliability=self.reliability,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=5
        )

        # Create publisher, subscriber, and service
        self._pub = self.create_publisher(Image, "viz", self.image_qos_profile)
        self._srv = self.create_service(SetBool, "enable", self.enable_callback)
        self._sub = self.create_subscription(
            DetectionArray, 
            "detections",
            self.image_callback,
            self.image_qos_profile
        )

        # Create stereo publisher, subscriber
        self.right_publish = self.create_publisher(Image, "right_viz", self.image_qos_profile)
        self.left_publish = self.create_publisher(Image, "left_viz", self.image_qos_profile)

        self.right_detections = message_filters.Subscriber(
            self,
            DetectionArray,
            "right_detections",
            qos_profile=self.image_qos_profile
        )
        self.left_detections = message_filters.Subscriber(
            self,
            DetectionArray,
            "left_detections",
            qos_profile=self.image_qos_profile
        )

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.right_detections, self.left_detections], 
            queue_size=10, 
            slop=0.1
        )

        self.ts.registerCallback(self.stereo_image_callback)

        self.cv_bridge = CvBridge()
        
        self.logged = False
        self.logger = self.get_logger()
        self.logger.info(f"{self.__class__.__name__} Node created and initialized.")

    def enable_callback(self, request, response):
        self.enable = request.data
        response.success = True
        return response

    def image_callback(self, msg: DetectionArray) -> None:

        if self.enable:

            if not self.logged:
                self.logger.info("1st Detections received and start processing.")
                self.logged = True

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

    def stereo_image_callback(self, right_msg: DetectionArray, left_msg: DetectionArray) -> None:

        if self.enable:

            if not self.logged:
                self.logger.info("1st Detections received and start processing.")
                self.logged = True

            # convert image visualize
            right_image = self.cv_bridge.imgmsg_to_cv2(right_msg.image, desired_encoding='bgr8')
            
            for detection in right_msg.detections:
                # for bbox in detection.bbox:
                start = (detection.bbox.top.x, detection.bbox.top.y)
                end = (detection.bbox.bottom.x, detection.bbox.bottom.y)
                right_image = cv2.rectangle(right_image, start, end, (0, 0, 255), 2)
                right_image = cv2.putText(right_image, detection.bbox.class_name, start, cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

            # publish detections
            right_viz_image = self.cv_bridge.cv2_to_imgmsg(right_image, encoding='rgb8', header=right_msg.image.header)

            left_image = self.cv_bridge.imgmsg_to_cv2(left_msg.image, desired_encoding='bgr8')
            
            for detection in left_msg.detections:
                # for bbox in detection.bbox:
                start = (detection.bbox.top.x, detection.bbox.top.y)
                end = (detection.bbox.bottom.x, detection.bbox.bottom.y)
                left_image = cv2.rectangle(left_image, start, end, (0, 0, 255), 2)
                left_image = cv2.putText(left_image, detection.bbox.class_name, start, cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

            # publish detections
            left_viz_image = self.cv_bridge.cv2_to_imgmsg(left_image, encoding='rgb8', header=left_msg.image.header)
            
            self.right_publish.publish(right_viz_image)
            self.left_publish.publish(left_viz_image)

            del right_image
            del right_viz_image

            del left_image
            del left_viz_image


def main():
    rclpy.init()
    node = YOLOVizNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__=="__main__":
    main()
