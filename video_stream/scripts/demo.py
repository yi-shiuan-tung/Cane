#!/usr/bin/env python
import rospy
import time
from param_setup import setup
import pyrealsense2 as rs
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import sys
import os
import multiprocessing

sys.path.insert(0, os.path.abspath("src/"))
from streamer import ROSBag, RawImages, RealSense, main


cv2.namedWindow("Depth")
cv2.namedWindow("RGB")
bridge = CvBridge()


def rgb_callback(img):
    img_msg = bridge.imgmsg_to_cv2(img, "passthrough")
    cv2.imshow("RGB", img_msg)

    key = cv2.waitKey(20)
    if key == 27:
        sys.exit()


def depth_callback(img):
    img_msg = bridge.imgmsg_to_cv2(img, "passthrough")
    cv2.imshow(
        "Depth",
        cv2.applyColorMap(
            cv2.normalize(img_msg, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U),
            cv2.COLORMAP_JET,
        ),
    )

    key = cv2.waitKey(20)

    if key == 27:
        sys.exit()


if __name__ == "__main__":

    # Set up parameter server
    setup(
        bag_file="/home/epiphyte/Documents/Research/cane_ws/src/cane/segmentation/bags/rgbd.bag"
    )

    # Main function of streamer
    rgb_dir = rospy.Subscriber("/video_stream/rgb_img", Image, rgb_callback)
    depth_dir = rospy.Subscriber("/video_stream/depth_img", Image, depth_callback)
    main()
