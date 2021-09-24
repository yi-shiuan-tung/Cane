#!/usr/bin/env python3
import cv2
import message_filters
import numpy as np
import rospy
from cv_bridge import CvBridge
from obj_inference.msg import Objects
from segmentation.msg import Prediction
from sensor_msgs.msg import Image, CameraInfo
import matplotlib.pyplot as plt
import pyrealsense2

bridge = CvBridge()


def annotate_image(rgb, centers, position_x, position_y, position_z):
    for i, center in enumerate(centers):
        rgb = cv2.circle(rgb, (int(center[0]), int(center[1])), 5, (0, 55, 255), -1)
        rgb = cv2.putText(
            rgb,
            "{:.2f} {:.2f}".format(position_x[i], position_y[i]),
            fontScale=0.5,
            org=(int(center[0]), int(center[1] + 20)),
            thickness=1,
            color=(194, 66, 245),
            fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
        )
    return rgb


class Visualizer:
    r""" """

    def __init__(self):
        self.camera_info = rospy.wait_for_message("/camera/color/camera_info", CameraInfo)
        rospy.Subscriber("/seg/prediction", Prediction, self.callback)

    def callback(self, pred):
        rgb = pred.color_img
        rgb_img = bridge.imgmsg_to_cv2(rgb, desired_encoding="passthrough")

        centers = np.array(pred.centers).reshape(-1, 2)
        rgb = annotate_image(rgb_img, centers, pred.position_x, pred.position_y, pred.position_z)
        cv2.imshow("input", rgb)
        cv2.setMouseCallback("input", self.click_callback)
        key = cv2.waitKey(200)
        if key == 27 & 0xff:
            cv2.destroyAllWindows()

    def convert_depth_to_phys_coord_using_realsense(self, x, y, depth):
        _intrinsics = pyrealsense2.intrinsics()
        _intrinsics.width = self.camera_info.width
        _intrinsics.height = self.camera_info.height
        _intrinsics.ppx = self.camera_info.K[2]
        _intrinsics.ppy = self.camera_info.K[5]
        _intrinsics.fx = self.camera_info.K[0]
        _intrinsics.fy = self.camera_info.K[4]
        _intrinsics.model = pyrealsense2.distortion.none
        _intrinsics.coeffs = [i for i in self.camera_info.D]
        result = pyrealsense2.rs2_deproject_pixel_to_point(_intrinsics, [x, y], depth)
        return result

    # Define a function to show the image in an OpenCV Window
    def show_image(self, img):
        cv2.imshow("Image Window", img)
        cv2.waitKey(3)

    # Define a callback for the Image message
    def image_callback(self, pred, img_msg):
        # log some info about the image topic
        rospy.loginfo(img_msg.header)

        # Try to convert the ROS Image message to a CV2 Image
        cv_image = bridge.imgmsg_to_cv2(img_msg, "passthrough")

        # Show the converted image
        self.show_image(cv_image)

    def click_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            vector = self.convert_depth_to_phys_coord_using_realsense(x, y, 1.4732)
            print("[%0.2f, %0.2f]" % (-vector[1] + 0.66, -vector[0] + 0.19))


if __name__ == "__main__":
    rospy.init_node("visualize")
    viz = Visualizer()
    rospy.spin()
