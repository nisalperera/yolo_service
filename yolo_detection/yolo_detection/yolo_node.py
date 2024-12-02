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
import message_filters

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy

from cv_bridge import CvBridge

from ultralytics import YOLO, NAS
from ultralytics.engine.results import Results

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
        self.declare_parameter("to_posestamped", False)
        self.declare_parameter("to_pointcloud", False)
        self.declare_parameter("log_image", False)
        self.declare_parameter("visualize", False)
        self.declare_parameter("stereo_vision", False)
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
        self._to_posestamped = self.get_parameter("to_posestamped").value
        self._to_pointcloud = self.get_parameter("to_pointcloud").value
        self.log_image = self.get_parameter("log_image").value
        self.visualize = self.get_parameter("visualize").value
        stereo_vision = self.get_parameter("stereo_vision").value


        # Set up QoS profile
        self.image_qos_profile = QoSProfile(
            reliability=self.reliability,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=5
        )

        # Create publisher, subscriber, and service
        self._srv = self.create_service(SetBool, "enable", self.enable_callback)
        if stereo_vision:
            self.right_publish = self.create_publisher(DetectionArray, "right_detections", self.image_qos_profile)
            self.left_publish = self.create_publisher(DetectionArray, "left_detections", self.image_qos_profile)

            self.right_sub = message_filters.Subscriber(
                self,
                Image,
                "/right_camera/image_raw",
                qos_profile=self.image_qos_profile
            )
            self.left_sub = message_filters.Subscriber(
                self,
                Image,
                "/left_camera/image_raw",
                qos_profile=self.image_qos_profile
            )

            self.ts = message_filters.ApproximateTimeSynchronizer(
                [self.right_sub, self.left_sub], 
                queue_size=10, 
                slop=0.1
            )

            self.ts.registerCallback(self.stereo_image_callback)
        
        else:
            self._publish = self.create_publisher(DetectionArray, "detections", self.image_qos_profile)
            self._subscribe = self.create_subscription(
                Image,
                "image_raw",
                self.single_image_callback,
                qos_profile=self.image_qos_profile
            )

        self.cv_bridge = CvBridge()

        # Initialize YOLO model
        self.yolo = self.type_to_model[self.model_type](self.model)
        if "v10" not in self.model:
            self.yolo.fuse()

        self.logger = self.get_logger()
        self.logged = {"stereo_image_callback": False, "single_image_callback": False}
        self.logger.info(f"{self.__class__.__name__} Node created and initialized. Using {'Stereo Vision' if stereo_vision else 'Mono Vision'}")

    def enable_callback(self, request, response):
        self.enable = request.data
        response.success = True
        return response

    def convert_to_detection_msg(self, result, frame_id=None):
        detection_array = DetectionArray()
        if frame_id:
            detection_array.header.frame_id = frame_id
        detection_array.header.stamp = self.get_clock().now().to_msg()

        for i in range(len(result)):
            self.logger.debug(f"Boxes: {result.boxes.xyxy[i]}, Tracking ID: {result.boxes.id[i]}")
            self.logger.debug(f"Masks: {result.masks.xy[i]}, Tracking ID: {result.boxes.id[i]}")
            self.logger.debug(f"Keypoints: {result.keypoints}, Tracking ID: {result.boxes.id[i]}")
            detection = Detection()
            if frame_id:
                detection.header.frame_id = frame_id

            detection.id = int(result.boxes.id[i].item())
            
            x1, y1, x2, y2 = result.boxes.xyxy[i].cpu().numpy()
            
            detection.bbox = BoundingBox2D()
            detection.bbox.top = Point2D()
            detection.bbox.top.x = int(x1)
            detection.bbox.top.y = int(y1)
            detection.bbox.bottom = Point2D()
            detection.bbox.bottom.x = int(x2)
            detection.bbox.bottom.y = int(y2)
            detection.bbox.class_id = int(result.boxes.cls[i])
            detection.bbox.class_name = result.names[int(result.boxes.cls[i])]
            detection.bbox.score = float(result.boxes.conf[i])

            # Convert segmentation mask if available
            if hasattr(result, 'masks') and result.masks is not None:
                mask = result.masks[i]
                detection.mask = Mask()
                detection.mask.height = mask.shape[0]
                detection.mask.width = mask.shape[1]
                detection.mask.class_id = result.boxes.cls[i]
                # Get mask contours
                contours = mask.xy[i]
                for point in contours:
                    p = Point2D()
                    p.x = int(point[0])
                    p.y = int(point[1])
                    detection.mask.data.append(p)

            # Convert keypoints if available
            if hasattr(result, 'keypoints') and result.keypoints is not None:
                keypoints = result.keypoints[i]
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

    def to_posestamped(self) -> str:
        return
    
    def stereo_predict(self, right_image, left_image, right_image_header, left_image_header) -> None:
        results: Results = self.yolo.track(
                source=[left_image, right_image],
                verbose=False,
                stream=False,
                conf=self.threshold,
                device=self.device
            )

        left_detections_msg = self.convert_to_detection_msg(
                                results[0].cpu()
                            )
        
        right_detections_msg = self.convert_to_detection_msg(
                                results[1].cpu()
                            )

        # publish detections
        if self.visualize:
            right_detections_msg.image = self.cv_bridge.cv2_to_imgmsg(right_image, header=right_image_header, encoding="rgb8")
            left_detections_msg.image = self.cv_bridge.cv2_to_imgmsg(left_image, header=left_image_header, encoding="rgb8")
        else:
            right_detections_msg.image = None
            left_detections_msg.image = None

        self.right_publish.publish(right_detections_msg)
        self.left_publish.publish(left_detections_msg)

        del results
        del right_detections_msg
        del left_detections_msg

    def stereo_image_callback(self, right_img_msg: Image, left_img_msg: Image) -> None:

        if self.enable:
            if not self.logged:
                self.logger.info("1st Image received for Stereo Vision mode and performing detections.")
                self.logged = True

            right_image = self.cv_bridge.imgmsg_to_cv2(right_img_msg, desired_encoding='bgr8')
            right_image_header = right_img_msg.header
            left_image = self.cv_bridge.imgmsg_to_cv2(left_img_msg, desired_encoding='bgr8')
            left_image_header = left_img_msg.header
            self.stereo_predict(right_image, left_image, right_image_header, left_image_header)

    def single_predict(self, cv_image) -> None:
        results: Results = self.yolo.predict(
                source=cv_image,
                verbose=False,
                stream=False,
                conf=self.threshold,
                device=self.device
            )

        detections_msg = self.convert_to_detection_msg(
                                results[0].cpu()
                            )

        # publish detections
        if self.visualize:
            detections_msg.image = self.cv_bridge.cv2_to_imgmsg(cv_image, header=self.image_header, encoding="rgb8")
        else:
            detections_msg.image = None

        self._publish(detections_msg)

        del results
        del detections_msg

    def single_image_callback(self, img_msg: Image) -> None:

        if self.enable:
            if not self.logged:
                self.logger.info("1st Image received for Mono Vision mode and performing detections.")
                self.logged = True

            self.image_header = img_msg.header
            self.single_predict(self.cv_bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8'))


def main():
    rclpy.init()
    node = YOLONode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__=="__main__":
    main()
