#!/usr/bin/env python3
from typing import Tuple, Union

import numpy as np
import rospy
import torch
from cv_bridge import CvBridge
from obj_inference.msg import Objects
from scipy import ndimage, stats
from segmentation.msg import Prediction
from sensor_msgs.msg import Image

REALSENSE_FOCAL_LENGTH = 1.93  # millimeters
REALSENSE_SENSOR_HEIGHT_MM = 2.454  # millimeters
REALSENSE_SENSOR_WIDTH_MM = 3.896  # millimeters
REALSENSE_REZ_HEIGHT = 800  # pixels
REALSENSE_REZ_WIDTH = 1280  # pixels


class Distance_Inference:
    def __init__(self):
        self.input_sub = rospy.Subscriber("/seg/prediction", Prediction, self.callback)
        self.inference_pub = rospy.Publisher(
            "/inference/obj_inference", Objects, queue_size=3
        )
        self.bridge = CvBridge()

    def get_distance(
        self,
        depth_map: np.ndarray,
        mask: np.ndarray,
        centers: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""Get the distance of each object from a numpy array
        Args:
            mask (np.ndarray):  Binary masks of found objects
            depth_map (np.ndarray):  Depth map input from camera
            centers (np.ndarray):  Centers of objects, output from Detectron

        Returns:
            distances (np.ndarray): distances of each object
            sizes (np.ndarray): sizes of each object
            relative_positions (np.ndarray): positions of each object

        Inputs:
            - **mask** of shape `(H, W, 1)`, where each object in mask has different integer value.

        Outputs:
            - **relative_positions** of shape `(objects_found, 3)`, which are direction vectors
                that tell where the object is in relation to the camera.
        """
        num_objects = mask.shape[0]

        object_sizes = np.zeros((num_objects))
        rel_positions = np.zeros((num_objects, 2))

        fill_center = False
        # only detectron model inputs the center of each object,
        # thus if we get an empty array we must fill it
        if centers.sum() == 0:
            fill_center = True
            centers = np.zeros((num_objects, 2))
        else:
            # Otherwise we have to unflatten array
            centers = centers.reshape(num_objects, 2)

        ### rel_positions[i][1] is distances to objects
        for i, obj_mask in enumerate(mask):
            # Trimmed mean for distance to reject outliers
            obj = obj_mask * depth_map
            rel_positions[i][1] = stats.trim_mean(
                obj[obj > 0], proportiontocut=0.2, axis=None
            )

            # Sum only width
            object_px_size = np.max(obj_mask.sum(1))
            obj_width_sensor = (REALSENSE_SENSOR_WIDTH_MM * object_px_size) / (
                REALSENSE_REZ_WIDTH
            )
            object_sizes[i] = (
                rel_positions[i][1] * obj_width_sensor
            ) / REALSENSE_FOCAL_LENGTH

            # Calc width of FOV at known distance
            field_width = (
                REALSENSE_SENSOR_WIDTH_MM * rel_positions[i][1]
            ) / REALSENSE_FOCAL_LENGTH

            if fill_center:
                centers[i] = ndimage.measurements.center_of_mass(obj_mask)

            rel_positions[i][0] = (centers[i][0] / REALSENSE_REZ_WIDTH) * field_width
            #  rel_positions[i][2] = (centers[i][1] / REALSENSE_REZ_HEIGHT) * field_width

        return (object_sizes, rel_positions)

    def callback(self, pred: Prediction) -> None:
        print("Inference heard message!", pred.header.stamp)
        dep = self.bridge.imgmsg_to_cv2(pred.depth_map, desired_encoding="passthrough")
        #  mask = self.bridge.imgmsg_to_cv2(pred.mask, desired_encoding="passthrough")

        # Has shape (n_objects, H, W)
        mask = np.frombuffer(pred.mask, dtype=np.uint8).reshape(
            pred.mask_channels,
            pred.mask_width,
            pred.mask_height,
        )

        sizes, positions = self.get_distance(
            depth_map=dep,
            mask=mask,
            centers=np.array(pred.centers),
        )
        #  print("\t\tObject inf: ")
        #  print("Positions: ", positions.shape)
        #  print("sizes: ", sizes.shape)
        #  print("scores: ", pred.scores)
        obj = Objects()
        obj.positions = positions.ravel()
        obj.sizes = sizes
        obj.header = pred.header
        obj.labels = pred.labels
        obj.scores = pred.scores
        self.inference_pub.publish(obj)


if __name__ == "__main__":
    dist_inf = Distance_Inference()
    rospy.init_node("object_inference")

    while not rospy.is_shutdown():
        rospy.spin()
