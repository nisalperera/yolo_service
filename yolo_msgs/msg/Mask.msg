# segmentation mask for one instance

# size of the original image
int32 height
int32 width

int32 class_id

# mask data represeted by the points of the border of the mask
yolo_msgs/Point2D[] data

