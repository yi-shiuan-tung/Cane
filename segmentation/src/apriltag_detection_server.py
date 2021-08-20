#!/usr/bin/env python3
import rospy
import dt_apriltags
import cv2
import transformations
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from segmentation.srv import ApriltagPose, ApriltagPoseResponse


class Detector:

    def __init__(self):
        self.detector = dt_apriltags.Detector(families='tagStandard41h12', nthreads=4, quad_decimate=1.0,
                                              quad_sigma=0.0, refine_edges=1, decode_sharpening=0.25)
        self.camera_info = rospy.wait_for_message("/camera/color/camera_info", CameraInfo)
        self.camera_params = [self.camera_info.K[0], self.camera_info.K[4], self.camera_info.K[2],
                              self.camera_info.K[5]]
        self.cv_bridge = CvBridge()

    def get_current_rgb_image(self):
        rgb = rospy.wait_for_message("/camera/color/image_raw", Image)
        return self.cv_bridge.imgmsg_to_cv2(rgb, desired_encoding="passthrough")

    def get_apriltag_pose(self, tag_id, tag_size=0.073):
        gray = cv2.cvtColor(self.get_current_rgb_image(), cv2.COLOR_RGB2GRAY)

        detections = self.detector.detect(gray, estimate_tag_pose=True, camera_params=self.camera_params,
                                          tag_size=tag_size)
        detection_id = None
        detection = None
        for idx, det in enumerate(detections):
            if det.tag_id == tag_id:
                detection_id = idx
                detection = det
                break
        if detection_id is None:
            return None

        matrix = detection.pose_R
        # turn matrix into 4x4 homogeneous matrix
        matrix = np.hstack([matrix, np.zeros((3, 1))])
        matrix = np.vstack([matrix, np.array([0, 0, 0, 1])])
        quaternion = transformations.quaternion_from_matrix(matrix)
        return {"position": detection.pose_t, "orientation": quaternion}


def handle_requests(req):
    detector = Detector()
    pose = detector.get_apriltag_pose(req.tag_id)
    if pose is None:
        return ApriltagPoseResponse(0, 0, 0, 0, 0, 0, 0)
    x, y, z = pose["position"]
    a, b, c, d = pose["orientation"]
    return ApriltagPoseResponse(x, y, z, a, b, c, d)


def server():
    rospy.init_node("apriltag_detection")
    rospy.Service("apriltag_detection", ApriltagPose, handle_requests)
    rospy.spin()


if __name__ == "__main__":
    server()
