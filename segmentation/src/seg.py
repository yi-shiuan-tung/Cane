#!/usr/bin/env python3
from abc import ABC, abstractmethod
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from typing import Tuple, Union
import numpy as np
import rospy
import torch
import torch.nn as nn
from cv_bridge import CvBridge
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from segmentation.msg import Prediction
from sensor_msgs.msg import Image, CameraInfo
from torch import Tensor
from detectron2 import model_zoo
import pyrealsense2
import cv2


AVAILABLE_MODELS = [
    "timm-efficientnet-b0",
    "timm-efficientnet-b3",
    "timm-efficientnet-b4",
    "timm-efficientnet-b7",
    "detectron",
]


class AbstractModel(ABC):
    r""" """

    @abstractmethod
    def get_class_names(
        self,
        index: Union[int, np.ndarray],
    ) -> Union[Tuple[str, ...], str]:
        ...

    @abstractmethod
    def full_output(
        self, x: Tensor
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[str, ...]]:
        ...


class EfficientNet:
    r""" """

    def __init__(self):
        pass


class DetectronModel(DefaultPredictor, AbstractModel):
    """Detectron Model interface"""

    def __init__(
        self,
        model_weights: str,
    ) -> None:
        self.config = self.setup_predictor_config(
            model_weights=model_weights,
        )
        super().__init__(self.config)
        self.class_names = MetadataCatalog.get("hrc_train").class_names
        self.num_classes = len(self.class_names)

    def setup_predictor_config(
        self,
        model_weights: str,
    ):
        """
        Setup config and return predictor. See config/defaults.py for more options
        """
        cfg = get_cfg()

        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_weights
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3

        return cfg

    def forward(self, x: Tensor) -> dict:
        r"""Returns full detectron output

        Args:
            x (torch.Tensor): Input image

        Returns:
            dict: Detectron output, prediction masks + other information
        """
        return self.__call__(x)

    def get_class_names(
        self,
        index: np.ndarray,
    ) -> Union[Tuple[str, ...], str]:

        names = []
        for idx in index:
            names.append(self.class_names[idx])
        return tuple(names)

    def format_mask(
        self,
        mask: np.ndarray,
    ) -> np.ndarray:
        r"""Formats input mask to be sent over ROS publisher
        Args:
            mask (np.ndarray): Output from DefaultPredictor
        Returns:
            np.ndarray: mask compressed to shape (H, W, 1)

        Inputs:
            - **mask**: Has shape `(num_objects, H, W)`, where each image is 0's and 1's.

        Outputs:
            - np.uint8: New mask compressed to be (H, W, 1), where each different object found
            has it's unique integer
        """
        compressed_mask = np.zeros((mask.shape[1], mask.shape[2]), dtype=np.uint8)
        for i in range(len(mask)):
            compressed_mask += mask[i] * (i + 1)
        return compressed_mask

    def full_output(
        self, x: Tensor
    ):
        pred = self.forward(x)["instances"]
        centers = pred.pred_boxes.get_centers().cpu().numpy()
        mask = pred.pred_masks.cpu().numpy().astype(np.uint8)
        labels = self.get_class_names(pred.pred_classes)
        scores = pred.scores.cpu().numpy()

        return mask, centers, labels, scores


class SegmentationModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        weights: str,
    ) -> None:
        super(SegmentationModel, self).__init__()

        if model_name not in AVAILABLE_MODELS:
            raise ValueError("Input model not available, select one from list")

        MetadataCatalog.get("hrc_train").set(thing_classes=['foot', 'leg', 'top'])

        # Create model
        # TODO: UPDATE timm model
        if model_name.find("timm") != -1:
            pass
        else:
            self.model = DetectronModel(model_weights=weights)

        self.predictor = DefaultPredictor(self.model.config)
        self.pred_pub = rospy.Publisher("/seg/prediction", Prediction, queue_size=3)
        self.camera_info = rospy.wait_for_message("/camera/color/camera_info", CameraInfo)
        self.cv_bridge = CvBridge()
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_1000)
        self.aruco_params = cv2.aruco.DetectorParameters_create()
        rospy.Subscriber("/camera/color/image_raw", Image, self.callback)
        self.aruco_map = {
            1: "small screw",
            2: "nut",
            3: "big screw",
            4: "tray"
        }

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

    def callback(self, rgb: Image) -> None:
        r"""ROS Subscriber to input RGB image + depth maps
        Args:
            rgb (sensor_msgs.Image): Image message
        """
        rgb_img = self.cv_bridge.imgmsg_to_cv2(rgb, desired_encoding="passthrough")

        (mask, centers, labels, scores) = self.model.full_output(rgb_img)
        centers = np.array(centers)[np.array(scores) > 0.5]
        labels = np.array(labels)[np.array(scores) > 0.5].tolist()

        pub_img = Prediction()
        pub_img.header = rgb.header
        pub_img.scores = scores
        pub_img.color_img = rgb

        corners, ids, rejected = cv2.aruco.detectMarkers(rgb_img, self.aruco_dict, parameters=self.aruco_params)

        aruco_centers = []
        aruco_labels = []

        if len(corners) > 0:
            for (marker_corner, marker_id) in zip(corners, ids):
                top_left, top_right, bottom_right, bottom_left = marker_corner.reshape((4, 2))
                top_left = (int(top_left[0]), int(top_left[1]))
                bottom_right = (int(bottom_right[0]), int(bottom_right[1]))

                center_x = int((top_left[0] + bottom_right[0])/2.0)
                center_y = int((top_left[1] + bottom_right[1])/2.0)
                try:
                    aruco_labels.append(self.aruco_map[marker_id[0]])
                    aruco_centers.append((center_x, center_y))
                except KeyError:
                    continue
            centers = np.vstack([centers, aruco_centers])

        position_x = []
        position_y = []
        position_z = []

        for center in centers:
            vector = self.convert_depth_to_phys_coord_using_realsense(center[0], center[1], 1.4732)
            position_x.append(-vector[1] + 0.66)
            position_y.append(-vector[0] + 0.19)
            position_z.append(vector[2])
        pub_img.centers = centers.ravel()
        pub_img.labels = labels + aruco_labels
        pub_img.position_x = position_x
        pub_img.position_y = position_y
        pub_img.position_z = position_z
        self.pred_pub.publish(pub_img)


if __name__ == "__main__":
    rospy.init_node("segmentation")
    weights = rospy.get_param("model_weights")
    model_type = rospy.get_param("segmentation_model")
    seg_model = SegmentationModel(
        model_name=model_type,
        weights=weights,
    )

    while not rospy.is_shutdown():
        rospy.spin()
