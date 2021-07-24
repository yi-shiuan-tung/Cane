#!/usr/bin/env python3
import time
from abc import ABC, abstractmethod
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import message_filters
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
from scipy import ndimage, stats
from image_geometry import PinholeCameraModel
import tf2_ros
import pyrealsense2
from geometry_msgs.msg import Point, PointStamped
import tf2_geometry_msgs


REALSENSE_FOCAL_LENGTH = 1.93  # millimeters
REALSENSE_SENSOR_HEIGHT_MM = 2.454  # millimeters
REALSENSE_SENSOR_WIDTH_MM = 3.896  # millimeters
REALSENSE_REZ_HEIGHT = 800  # pixels
REALSENSE_REZ_WIDTH = 1280  # pixels

if TYPE_CHECKING:
    from segmentation.detectron2.detectron2.config.config import CfgNode

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
        config_file: str,
    ) -> None:
        self.config = self.setup_predictor_config(
            model_weights=model_weights,
            config_file=config_file,
        )
        super().__init__(self.config)
        self.class_names = MetadataCatalog.get("hrc_train").class_names
        self.num_classes = len(self.class_names)

    def setup_predictor_config(
        self,
        model_weights: str,
        config_file: str,
    ) -> "CfgNode":
        """
        Setup config and return predictor. See config/defaults.py for more options
        """
        cfg = get_cfg()

        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        # Mask R-CNN ResNet101 FPN weights
        cfg.MODEL.WEIGHTS = model_weights
        # This determines the resizing of the image. At 0, resizing is disabled.
        cfg.INPUT.MIN_SIZE_TEST = 0
        cfg.INPUT.FORMAT = "BGR"

        cfg.SOLVER.IMS_PER_BATCH = 2
        cfg.SOLVER.BASE_LR = 0.00025
        cfg.SOLVER.MAX_ITER = 1000
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6

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


class DistanceInference:
    def __init__(self):
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
        rel_positions = np.zeros((num_objects, 3))

        fill_center = False
        # only detectron model inputs the center of each object,
        # thus if we get an empty array we must fill it
        if centers.sum() == 0:
            fill_center = True
            centers = np.zeros((num_objects, 2))
        else:
            # Otherwise we have to unflatten array
            centers = centers.reshape(num_objects, 2)

        # rel_positions[i][1] is distances to objects
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
            rel_positions[i][2] = (centers[i][1] / REALSENSE_REZ_HEIGHT) * field_width

        return object_sizes, rel_positions


class SegmentationModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        weights: str,
        config_file: str,
    ) -> None:
        super(SegmentationModel, self).__init__()

        if model_name not in AVAILABLE_MODELS:
            raise ValueError("Input model not available, select one from list")

        MetadataCatalog.get("hrc_train").set(thing_classes=['chair-back', 'chair-back-cover', 'chair-bracket', 'leg',
                                                            'seat', 'top'])

        # Create model
        # TODO: UPDATE timm model
        if model_name.find("timm") != -1:
            pass
        else:
            self.model = DetectronModel(model_weights=weights, config_file=config_file)

        self.predictor = DefaultPredictor(self.model.config)
        self.pred_pub = rospy.Publisher("/seg/prediction", Prediction, queue_size=3)
        self.camera_info = rospy.wait_for_message("/camera/color/camera_info", CameraInfo)
        self.camera_model = PinholeCameraModel()
        self.camera_model.fromCameraInfo(self.camera_info)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        # self.trans = self.tf_buffer.lookup_transform("base", "camera_color_frame", rospy.Time(0), rospy.Duration(20.0))
        rgb_sub = message_filters.Subscriber("/camera/color/image_raw", Image)
        dep_sub = message_filters.Subscriber("/camera/depth/image_rect_raw", Image)
        synch = message_filters.ApproximateTimeSynchronizer([rgb_sub, dep_sub], 5, 0.5)
        synch.registerCallback(self.callback)
        self.cv_bridge = CvBridge()

        self.distance_infer = DistanceInference()

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

    def callback(self, rgb: Image, depth_map: Image) -> None:
        r"""ROS Subscriber to input RGB image + depth maps
        Args:
            rgb (sensor_msgs.Image): Image message
            depth_map (sensor_msgs.Image)
        """
        rgb_img = self.cv_bridge.imgmsg_to_cv2(rgb, desired_encoding="passthrough")
        depth_img = self.cv_bridge.imgmsg_to_cv2(depth_map, desired_encoding="passthrough")

        (mask, centers, labels, scores) = self.model.full_output(rgb_img)
        outputs = self.predictor(rgb_img)
        v = Visualizer(rgb_img, metadata=MetadataCatalog.get("hrc_train"))
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        pub_img = Prediction()
        pub_img.header = rgb.header
        pub_img.scores = scores
        pub_img.centers = centers.ravel()
        pub_img.labels = labels
        pub_img.labeled_img = self.cv_bridge.cv2_to_imgmsg(out.get_image(), encoding="passthrough")

        position_x = []
        position_y = []
        position_z = []
        for center in centers:
            vector = self.convert_depth_to_phys_coord_using_realsense(center[0], center[1], 1.45)
            position_x.append(vector[0]+0.65)
            position_y.append(-vector[1]-0.14)
            position_z.append(vector[2])
        pub_img.position_x = position_x
        pub_img.position_y = position_y
        pub_img.position_z = position_z
        self.pred_pub.publish(pub_img)


if __name__ == "__main__":
    rospy.init_node("segmentation")
    weights = rospy.get_param("model_weights")
    model_type = rospy.get_param("segmentation_model")
    config = rospy.get_param("model_config")
    seg_model = SegmentationModel(
        model_name=model_type,
        weights=weights,
        config_file=config,
    )

    while not rospy.is_shutdown():
        rospy.spin()
