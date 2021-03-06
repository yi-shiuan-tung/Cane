import torch
import numpy as np
from scipy import stats
from scipy import ndimage
from pandas import unique

import rospy
from sensor_msgs.msg import Image
from typing import Union, Tuple

from segmentation.msg import Prediction
from obj_inference.msg import Object

REALSENSE_FOCAL_LENGTH = 1.93  # millimeters
REALSENSE_SENSOR_HEIGHT_MM = 2.454  # millimeters
REALSENSE_SENSOR_WIDTH_MM = 3.896  # millimeters
REALSENSE_REZ_HEIGHT = 800  # pixels
REALSENSE_REZ_WIDTH = 1280  # pixels


class Distance_Inference:
    def __init__(self):
        self.input_sub = rospy.Subscriber("/seg/prediction", Prediction, self.callback)
        self.inference_pub = rospy.Publisher(
            "/inference/obj_inference", Object, queue_size=3
        )

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

        # Get all unique object vals in mask
        unique_vals = unique(mask.ravel())
        num_objects = len(unique_vals)

        object_sizes = np.zeros((num_objects))
        rel_positions = np.zeros((num_objects, 2))

        fill_center = False
        # only detectron model inputs the center of each object,
        # thus if we get an empty array we must fill it
        if centers.sum() == 0:
            fill_center = True
            centers = np.zeros((num_objects, 2))

        ### rel_positions[i][1] is distances to objects
        for i in range(num_objects):
            # Single object mask
            #  obj_mask = np.ma.masked_where(mask == unique_vals[i], mask)
            obj_mask = np.ma.masked_equal(unique_vals[i], mask).mask

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
        sizes, positions = self.get_distance(
            depth_map=pred.depth_map,
            mask=pred.mask,
            centers=pred.centers,
        )
        obj = Object()
        obj.positions = positions
        obj.labels = pred.labels
        obj.sizes = sizes
        self.inference_pub.publish(obj)
