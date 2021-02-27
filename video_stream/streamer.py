import pyrealsense2 as rs
import numpy as np
import cv2
from abs import ABC, abstractmethod
from typing import Optional, Tuple, Any
import warnings

try:
    import rospy
    import rosbag
    import itertools
except ImportError:
    raise ImportError("Failed to import rospy, try sourcing workspace")


class VideoStream(ABC):
    @abstractmethod
    def __init__(self):
        ...

    @abstractmethod
    def get_frame(self):
        ...

    @abstractmethod
    def wait_for_frame(self):
        ...

    @abstractmethod
    def start(self):
        ...

    @abstractmethod
    def stop(self):
        ...


class RealSense(VideoStream):
    r"""RealSense Camera input, either streaming directly from the cam or using the intel bagfile"""

    def __init__(
        self, input_file: Optional[str] = None, visualize: bool = False
    ) -> None:
        # RealSense Pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # If we are streaming from bag file
        if input_file is not None:
            # Tell config that we will use a recorded device from file to be used by the pipeline through playback.
            self.config.enable_device_from_file(input_file)

        # TODO: Change parameters of this
        self.config.enable_stream(rs.stream.depth, rs.format.z16, 30)

        # Create cv2 window if we are visualizing input stream
        if visualize:
            self.input_viz_win = cv2.namedWindow("Input Stream", cv2.WINDOW_AUTOSIZE)

    def start(self) -> None:
        """Start stream"""
        self.pipeline.start(self.config)

    def stop(self) -> None:
        """Stop stream"""
        self.pipeline.stop(self.config)

    def wait_for_frame(self):
        return self.pipeline.wait_for_frames()

    def get_frame(self):
        return self.wait_for_frame()


class ROSBag(VideoStream):
    r"""Stream video from ROS bagfile

    Ref:
        IntelRealSense/librealsense python examples:
        https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/read_bag_example.py

    """

    def __init__(
        self,
        input_file: str,
        topics: list = [
            "/camera/color/image_raw/compressed",
            "/camera/depth/image_rect_raw/compressed",
        ],
        visualize: bool = False,
    ) -> None:

        self.bag = rosbag.Bag(input_file)
        self.index = -1

        # List of topics to subscribe to
        self.generators = []
        self.length = -1

        for topic in topics:

            # Raise exception if topic is not in bag file
            if topic not in self.bag.get_type_and_topic_info()[1].keys():
                raise ValueError(f"Input topic {topic} is not found in bagfile")

            # Raise warning if topic lengths do not match
            if (self.length != -1) and (
                self.length != self.bag.get_message_count(topic)
            ):
                warnings.warn("Some topics have different lengths...", Warning)

            else:
                self.length = self.bag.get_message_count(topic)

            self.generators.append(self.bag.read_messages(topics=topic))

        # Create cv2 window if we are visualizing input stream
        if visualize:
            self.input_viz_win = cv2.namedWindow("Input Stream", cv2.WINDOW_AUTOSIZE)

    def __len__(self) -> int:
        return self.length

    def __getitem__(
        self,
        index: int,
    ) -> Tuple[Any, ...]:

        return_list = []
        for gen in self.generators:
            return_list.append(next(itertools.islice(gen, index, None)))
        return tuple(return_list)

    def start(self):
        pass

    def stop(self):
        pass

    def get_frame(self) -> Tuple[Any, ...]:
        self.index += 1
        return self.__getitem__(self.index)

    def wait_for_frame(self) -> Tuple[Any, ...]:
        return self.get_frame()
