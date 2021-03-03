import torch
import numpy as np
from scipy import stats

#  import rospy
from typing import Union, Tuple

REALSENSE_FOCAL_LENGTH = 1.93  # millimeters
REALSENSE_SENSOR_HEIGHT_MM = 2.454  # millimeters
REALSENSE_SENSOR_WIDTH_MM = 3.896  # millimeters
REALSENSE_REZ_HEIGHT = 800  # pixels
REALSENSE_REZ_WIDTH = 1280  # pixels


class Distance_Inference:
    def __init__(self):
        # self.input_preds = rospy.Subscriber("/seg/prediction", prediction, None)
        pass

    def _get_dist_numpy(
        self,
        masks: np.ndarray,
        depth_map: np.ndarray,
        centers: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""Get the distance of each object from a numpy array
        Args:
            masks (np.ndarray):  Binary masks of found objects
            depth_map (np.ndarray):  Depth map input from camera
            centers (np.ndarray):  Centers of objects, output from Detectron

        Returns:
            distances (np.ndarray): distances of each object
            sizes (np.ndarray): sizes of each object
            relative_positions (np.ndarray): positions of each object

        Inputs:
            - **masks** of shape `(objects_found, H, W)`: where `H, W` are the dimensions of
            the input image. Each mask only has values (0, 1)

        Outputs:
            - **relative_positions** of shape `(objects_found, 3)`, which are direction vectors
                that tell where the object is in relation to the camera.
        """
        distances = np.zeros((masks.shape[0]))
        object_sizes = np.zeros((masks.shape[0]))
        rel_positions = np.zeros((masks.shape[0], 2))

        for i, mask in enumerate(masks):
            # Trimmed mean for distance to reject outliers
            obj = mask * depth_map
            distances[i] = stats.trim_mean(obj[obj > 0], proportiontocut=0.2, axis=None)

            # Sum only width
            object_px_size = np.max(mask.sum(1))
            obj_width_sensor = (REALSENSE_SENSOR_WIDTH_MM * object_px_size) / (
                REALSENSE_REZ_WIDTH
            )
            object_sizes[i] = (distances[i] * obj_width_sensor) / REALSENSE_FOCAL_LENGTH

            # Calc width of FOV at known distance
            field_width = (
                REALSENSE_SENSOR_WIDTH_MM * distances[i]
            ) / REALSENSE_FOCAL_LENGTH

            rel_positions[i][0] = (centers[i][0] / REALSENSE_REZ_WIDTH) * field_width
            rel_positions[i][1] = distances[i]
            #  rel_positions[i][2] = (centers[i][1] / REALSENSE_REZ_HEIGHT) * field_width

        return (distances, object_sizes, rel_positions)

    def _get_dist_detectron_obj(
        self,
        preds: dict,
        depth_map: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""Get the distance of each object from the detectron output
        Args:
            preds (dict):  Output of Detectron model

        Returns:
            positions (np.ndarray): positions of each object
            distances (np.ndarray): distances of each object
        """
        return self._get_dist_numpy(
            masks=preds["instances"].pred_masks.cpu().numpy(),
            depth_map=depth_map,
            centers=preds["instances"].pred_boxes.cpu().numpy(),
        )

    def get_distance(
        self,
        depth_map: np.ndarray,
        mask: Union[dict, np.ndarray, torch.Tensor],
        centers: Union[np.ndarray, torch.Tensor],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if isinstance(mask, dict):
            return self._get_dist_detectron_obj(preds=mask, depth_map=depth_map)

        elif isinstance(mask, torch.Tensor):
            return self._get_dist_numpy(
                masks=mask.cpu().detach().numpy(),
                depth_map=depth_map,
                centers=centers.cpu().numpy(),
            )
        elif isinstance(mask, np.ndarray):
            return self._get_dist_numpy(
                masks=mask, depth_map=depth_map, centers=centers
            )
        else:
            raise TypeError(
                f"Input to object inference method 'get_distance()' is of incompatible type: {type(mask)}"
            )
