# YoloDetectionResult.msg

# Class probability
int32 class_id
string class_name
float64 score

# ID for tracking
string id

# 2D bounding box surrounding the object in pixels
yolo_service/BoundingBox2D bbox

# 3D bounding box surrounding the object in meters
yolo_service/BoundingBox3D bbox3d

# Segmentation mask of the detected object
yolo_service/Mask mask

# Keypoints for human pose estimation
yolo_service/KeyPoint2D[] keypoints

# 3D Keypoints for human pose estimation
yolo_service/KeyPoint3D[] keypoints3d