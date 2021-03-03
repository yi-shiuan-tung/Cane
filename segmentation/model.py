import torch
import torch.nn as nn
from torch import Tensor
import segmentation_models_pytorch as smp
import sys
import os

sys.path.insert(0, os.path.abspath("detectron2"))
import detectron


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


class DetectronModel(nn.Module):
    """Detectron Model interface"""

    def __init__(
        self,
        yaml_file: str,
        weights: str,
    ) -> None:
        super(DetectronModel, self).__init__()
        # TODO: UPdate to input yaml file and weight files
        self.model = detectron.Detectron()

    def forward(self, x: Tensor) -> Tensor:
        r"""Returns only the prediction masks

        Args:
            x (torch.Tensor): Input image

        Returns:
            torch.Tensor: Prediction masks per class object
        """
        return self.model(x)["instances"].pred_masks

    def full_output(self, x: Tensor) -> Tensor:
        r"""Returns full detectron output

        Args:
            x (torch.Tensor): Input image

        Returns:
            dict: prediction masks + other information
        """
        return self.model(x)


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
            self.model = DetectronModel(config["yaml"], weights)

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

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
