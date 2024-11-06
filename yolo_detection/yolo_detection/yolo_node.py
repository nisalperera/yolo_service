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


from typing import List, Dict, Tuple

import torch

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy

from cv_bridge import CvBridge

from ultralytics import YOLO, NAS
from ultralytics.engine.results import Results
from ultralytics.engine.results import Boxes
from ultralytics.engine.results import Masks
from ultralytics.engine.results import Keypoints

from std_srvs.srv import SetBool
from sensor_msgs.msg import Image
from yolo_msgs.msg import Point2D
from yolo_msgs.msg import BoundingBox2D
from yolo_msgs.msg import OrientedBoundingBox
from yolo_msgs.msg import Mask
from yolo_msgs.msg import KeyPoint2D
from yolo_msgs.msg import KeyPoint2DArray
from yolo_msgs.msg import Detection
from yolo_msgs.msg import DetectionArray


class YOLONode(Node):
    def __init__(self) -> None:
        super().__init__("yolo_node")

        # Declare parameters
        self.declare_parameter("model_type", "YOLO")
        self.declare_parameter("model", "yolov8m.pt")
        self.declare_parameter("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.declare_parameter("threshold", 0.5)
        self.declare_parameter("enable", True)
        self.declare_parameter("image_reliability", QoSReliabilityPolicy.RELIABLE)

        self.type_to_model = {
            "YOLO": YOLO,
            "NAS": NAS
        }

        # Get parameter values
        self.model_type = self.get_parameter("model_type").value
        self.model = self.get_parameter("model").value
        self.device = self.get_parameter("device").value
        self.threshold = self.get_parameter("threshold").value
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
        self._pub = self.create_publisher(DetectionArray, "detections", self.image_qos_profile)
        self._srv = self.create_service(SetBool, "enable", self.enable_callback)
        self._sub = self.create_subscription(
            Image,
            "image_raw",
            self.image_callback,
            self.image_qos_profile
        )

        self.cv_bridge = CvBridge()

        # Initialize YOLO model
        self.yolo = self.type_to_model[self.model_type](self.model)
        if "v10" not in self.model:
            self.yolo.fuse()

        self.logger = self.get_logger()
        self.logger.info(f"{self.__class__.__name__} Node created and initialized")

    def enable_callback(self, request, response):
        self.enable = request.data
        response.success = True
        return response

    def convert_to_detection_msg(self, result, frame_id):
        detection_array = DetectionArray()
        detection_array.header.frame_id = frame_id
        detection_array.header.stamp = self.get_clock().now().to_msg()

        for i, det in enumerate(result):
            detection = Detection()
            detection.frame_id = frame_id
            detection.id = str(i)
            
            # Convert bounding box
            box = det.boxes[0]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            detection.bbox = BoundingBox2D()
            detection.bbox.top = Point2D()
            detection.bbox.top.x = int(x1)
            detection.bbox.top.y = int(y1)
            detection.bbox.bottom = Point2D()
            detection.bbox.bottom.x = int(x2)
            detection.bbox.bottom.y = int(y2)
            detection.bbox.class_id = int(box.cls)
            detection.bbox.class_name = result.names[int(box.cls)]
            detection.bbox.score = float(box.conf)

            # Convert segmentation mask if available
            if hasattr(det, 'masks') and det.masks is not None:
                mask = det.masks[0]
                detection.mask = Mask()
                detection.mask.height = mask.shape[0]
                detection.mask.width = mask.shape[1]
                # Get mask contours
                contours = mask.xy[0]
                for point in contours:
                    p = Point2D()
                    p.x = int(point[0])
                    p.y = int(point[1])
                    detection.mask.data.append(p)

            # Convert keypoints if available
            if hasattr(det, 'keypoints') and det.keypoints is not None:
                keypoints = det.keypoints[0]
                detection.keypoints = KeyPoint2DArray()
                
                for idx, kp in enumerate(keypoints):
                    keypoint = KeyPoint2D()
                    keypoint.id = idx
                    keypoint.point = Point2D()
                    keypoint.point.x = int(kp[0])
                    keypoint.point.y = int(kp[1])
                    keypoint.score = float(kp[2]) if len(kp) > 2 else 1.0
                    detection.keypoints.data.append(keypoint)

            detection_array.detections.append(detection)

        return detection_array

    def image_callback(self, msg: Image) -> None:

        if self.enable:
            # convert image + predict
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            results = self.yolo.predict(
                source=cv_image,
                verbose=False,
                stream=False,
                conf=self.threshold,
                device=self.device
            )
            results: Results = results[0].cpu()

            detections_msg = self.convert_to_detection_msg(
                                results, 
                                msg.header.frame_id
                            )

            # publish detections
            detections_msg.header = msg.header
            detections_msg.image = self.cv_bridge.cv2_to_imgmsg(cv_image, header=msg.header, encoding="rgb8")
            self._pub.publish(detections_msg)

            del results
            del cv_image


def main():
    rclpy.init()
    node = YOLONode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__=="__main__":
    main()
