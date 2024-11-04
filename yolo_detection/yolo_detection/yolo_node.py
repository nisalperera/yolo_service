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


class Yolov8Node(Node):
    def __init__(self) -> None:
        super().__init__("yolo_node")

        # Declare parameters
        self.declare_parameter("model_type", "YOLO")
        self.declare_parameter("model", "yolov8m.pt")
        self.declare_parameter("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.declare_parameter("threshold", 0.5)
        self.declare_parameter("enable", True)
        self.declare_parameter("image_reliability", QoSReliabilityPolicy.BEST_EFFORT)

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
        self._pub = self.create_publisher(DetectionArray, "detections", 10)
        self._srv = self.create_service(SetBool, "enable", self.enable_cb)
        self._sub = self.create_subscription(
            Image,
            "image_raw",
            self.image_cb,
            self.image_qos_profile
        )

        self.cv_bridge = CvBridge()

        # Initialize YOLO model
        self.yolo = self.type_to_model[self.model_type](self.model)
        if "v10" not in self.model:
            self.yolo.fuse()

        self.logger = self.get_logger()
        self.logger.info("Yolov8 Node created and initialized")

    def enable_cb(self, request, response):
        self.enable = request.data
        response.success = True
        return response

    def parse_hypothesis(self, results: Results) -> Dict[List[Dict]]:

        metadata = {}

        if results.boxes:
            hypothesis_list = []
            for box_data in results.boxes:
                meta = {
                    "class_id": int(box_data.cls),
                    "class_name": self.yolo.names[int(box_data.cls)],
                    "score": float(box_data.conf)
                }
                hypothesis_list.append(meta)

            metadata["boxes"] = hypothesis_list

        if results.obb:
            hypothesis_list = []
            for i in range(results.obb.cls.shape[0]):
                meta = {
                    "class_id": int(results.obb.cls[i]),
                    "class_name": self.yolo.names[int(results.obb.cls[i])],
                    "score": float(results.obb.conf[i])
                }
                hypothesis_list.append(meta)

        return metadata

    def parse_boxes(self, results: Results) -> Tuple[List[BoundingBox2D], List[OrientedBoundingBox]]:

        boxes_list = []
        oriented_boxes_list = []

        if results.boxes:
            for box_data in results.boxes:

                msg = BoundingBox2D()

                # get boxes values
                box = box_data.xyxy[0]
                msg.top.x = float(box[0])
                msg.top.y = float(box[1])
                msg.bottom.x = float(box[2])
                msg.bottom.y = float(box[3])

                # append msg
                boxes_list.append(msg)

        elif results.obb:
            for i in range(results.obb.cls.shape[0]):
                msg = OrientedBoundingBox()

                # get boxes values
                box = results.obb.xyxyr[i]
                msg.top.x = float(box[0])
                msg.top.y = float(box[1])
                msg.theta = float(box[4])
                msg.bottom.x = float(box[2])
                msg.bottom.y = float(box[3])

                # append msg
                oriented_boxes_list.append(msg)

        return boxes_list, oriented_boxes_list


    def parse_keypoints(self, results: Results) -> List[KeyPoint2DArray]:

        keypoints_list = []

        points: Keypoints
        for points in results.keypoints:

            msg_array = KeyPoint2DArray()

            if points.conf is None:
                continue

            for kp_id, (p, conf) in enumerate(zip(points.xy[0], points.conf[0])):

                if conf >= self.threshold:
                    msg = KeyPoint2D()

                    msg.id = kp_id + 1
                    msg.point.x = float(p[0])
                    msg.point.y = float(p[1])
                    msg.score = float(conf)

                    msg_array.data.append(msg)

            keypoints_list.append(msg_array)

        return keypoints_list

    def image_cb(self, msg: Image) -> None:

        self.logger.info(f"YOLO object detection service enabled: {self.enable}")
        if self.enable:
            # convert image + predict
            self.logger.info(f"Image recieved: {msg != None}")
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg)
            results = self.yolo.predict(
                source=cv_image,
                verbose=False,
                stream=False,
                conf=self.threshold,
                device=self.device
            )
            results: Results = results[0].cpu()

            if results.boxes or results.obb:
                hypothesis = self.parse_hypothesis(results)
                boxes, oriented_boxes = self.parse_boxes(results)

            if results.keypoints:
                keypoints = self.parse_keypoints(results)

            # create detection msgs
            detections_msg = DetectionArray()

            for i in range(len(results)):

                aux_msg = Detection()

                if results.boxes or results.obb:
                    aux_msg.class_id = hypothesis[i]["class_id"]
                    aux_msg.class_name = hypothesis[i]["class_name"]
                    aux_msg.score = hypothesis[i]["score"]

                    aux_msg.bbox = boxes[i]

                if results.masks:
                    aux_msg.mask = masks[i]

                if results.keypoints:
                    aux_msg.keypoints = keypoints[i]

                detections_msg.detections.append(aux_msg)

            # publish detections
            detections_msg.header = msg.header
            detections_msg.image = self.cv_bridge.cv2_to_imgmsg(cv_image)
            self._pub.publish(detections_msg)

            del results
            del cv_image


def main():
    rclpy.init()
    node = Yolov8Node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__=="__main__":
    main()
