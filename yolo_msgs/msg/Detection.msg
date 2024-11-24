# defines a YOLO detection result

# ID for tracking
int32 id

# 2D bounding box surrounding the object in pixels
yolo_msgs/BoundingBox2D bbox

# 3D bounding box surrounding the object in meters
yolo_msgs/BoundingBox3D bbox3d

# segmentation mask of the detected object
# it is only the boundary of the segmented object
yolo_msgs/Mask mask

# keypoints for human pose estimation
yolo_msgs/KeyPoint2DArray keypoints

# keypoints for human pose estimation
yolo_msgs/KeyPoint3DArray keypoints3d

yolo_msgs/OrientedBoundingBox obb