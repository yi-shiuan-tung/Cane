import torch
import torch.nn as nn
from torch import Tensor
import segmentation_models_pytorch as smp
import sys
from typing import Union
import os


from segmentation.detectron2.detectron2.engine import DefaultPredictor
from segmentation.detectron2.detectron2.config import get_cfg
from segmentation.detectron2.detectron2.utils.visualizer import (
    GenericMask,
    Visualizer,
    ColorMode,
)
from segmentation.detectron2.detectron2.structures import Boxes, RotatedBoxes


AVAILABLE_MODELS = [
    "timm-efficientnet-b0",
    "timm-efficientnet-b3",
    "timm-efficientnet-b4",
    "timm-efficientnet-b7",
    "detectron",
]


def make_timm_model(
    model_name: str,
    weights: str,
    in_channels: int,
    out_channels: int,
) -> nn.Module:
    return smp.DeepLabV3Plus(
        encoder_name=model_name,
        encoder_weights=weights,
        activation="sigmoid",
        in_channels=in_channels,
        classes=out_channels,
    )


class DetectronModel(DefaultPredictor):
    """Detectron Model interface"""

    # TODO: UPdate to input yaml file and weight files
    def __init__(
        self,
        model_weights: str = "/home/epiphyte/Documents/Research/cane/segmentation/detectron2/weights/model_final_a54504.pkl",
        config_file: str = "/home/epiphyte/Documents/Research/cane/segmentation/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml",
    ) -> None:
        self.config = self.setup_predictor_config(
            model_weights=model_weights,
            config_file=config_file,
        )
        super().__init__(self.config)

        self.class_names = self.metadata.class_names
        self.num_classes = len(self.class_names)

    def setup_predictor_config(self, model_weights: str, config_file: str):
        """
        Setup config and return predictor. See config/defaults.py for more options
        """
        cfg = get_cfg()

        cfg.merge_from_file(config_file)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        # Mask R-CNN ResNet101 FPN weights
        cfg.MODEL.WEIGHTS = model_weights
        # This determines the resizing of the image. At 0, resizing is disabled.
        cfg.INPUT.MIN_SIZE_TEST = 0

        return cfg

    def forward(self, x: Tensor) -> Tensor:
        r"""Returns only the prediction masks

        Args:
            x (torch.Tensor): Input image

        Returns:
            torch.Tensor: Prediction masks per class object
        """
        return self.__call__(x)["instances"].pred_masks

    def full_output(self, x: Tensor) -> Tensor:
        r"""Returns full detectron output

        Args:
            x (torch.Tensor): Input image

        Returns:
            dict: prediction masks + other information
        """
        return self.__call__(x)


class SegmentationModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        weights: str,
        config: dict,
    ) -> None:
        super(SegmentationModel, self).__init__()

        if model_name not in AVAILABLE_MODELS:
            raise ValueError("Input model not available, select one from list")

        # Create model
        if model_name.find("timm") != -1:
            if ("in_channels" not in config.keys()) or (
                "out_channels" not in config.keys()
            ):
                raise ValueError("Need in_channels and out_channels in config dict.")
            self.model = make_timm_model(
                model_name,
                weights,
                config["in_channels"],
                config["out_channels"],
            )
        else:
            self.model = DetectronModel()

    def full_output(self, x: Tensor) -> Tensor:
        return self.model.full_output(x)

    def format_out(self, x: Tensor):
        r"""
        Args:
        Returns:

            masks
            bounding_boxes
            pred_classes
            scores
        """
        pred = self.model.full_output(x)["instances"]
        return (
            pred.pred_masks,
            pred.pred_boxes,
            pred.pred_classes,
            pred.scores,
        )

    def get_class_names(self, index: int) -> str:
        return self.model.class_names[index]

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
