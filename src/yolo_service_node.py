import rclpy
from rclpy.node import Node
from yolo_service.srv import ModelParams, DetectionResult
from sensor_msgs.msg import Image

from ultralytics import YOLO

class YoloServiceNode(Node):

    def __init__(self):
        super().__init__('yolo_service_node')
        self.srv = self.create_service(ModelParams, 'run_yolo', self.run_yolo_callback)

    def run_yolo_callback(self, request, response):
        # Parse model parameters
        model_name = request.model_name
        task = request.task
        model_version = request.model_version

        if task == "det":
            model_name = 'yolo'+model_name+model_version+'.pt'
        else: model_name = 'yolo'+model_name+model_version+task+'.pt'

        # Load and run the YOLO model (pseudo-code)
        # model = load_model(model_name, model_version)
        # results = model.run(task, request.image)
        cv_image = self.bridge.imgmsg_to_cv2(request.image, "bgr8")
        model = YOLO(model_name)
        results = model(cv_image)

        # Populate response with dummy data for demonstration
        response.success = True
        response.message = "Model executed successfully"
        return response

def main(args=None):
    rclpy.init(args=args)
    node = YoloServiceNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()