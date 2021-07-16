#!/usr/bin/env  python3
import glob
import itertools
import os
import time
import warnings
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, Union

import cv2
import numpy as np
import pyrealsense2 as rs
import rosbag
import rospy
from cv_bridge import CvBridge
from video_stream.msg import Stream


class VideoStream(ABC):
    @abstractmethod
    def __init__(self):
        ...

    @abstractmethod
    def get_frame(self):
        ...

    @abstractmethod
    def wait_for_frame(self) -> Union[None, Tuple[Any, ...]]:
        ...

    @abstractmethod
    def start(self):
        ...

    @abstractmethod
    def stop(self):
        ...


class RawImages(VideoStream):
    r"""Stream input from two seperate folders, of depth and rgb images"""

    def __init__(
        self,
        rgb_dir: str,
        depth_dir: str,
        visualize: bool = False,
        publish_rate: Union[float, int] = 2,
    ):

        # Check if directories exist
        if os.path.isdir(rgb_dir) and os.path.isdir(depth_dir):
            self.rgb_dir = rgb_dir
            self.depth_dir = depth_dir
        else:
            raise ValueError(
                f"RGB or Depth directory not found: \n\t{rgb_dir}\n\t{depth_dir}"
            )

        # Set config variables
        self.index = 1
        self.publish_rate = publish_rate
        self.length = min(
            len(glob.glob1(depth_dir, "*.png")), len(glob.glob1(rgb_dir, "*.png"))
        )

        # Image publishers
        self.input_pub = rospy.Publisher(
            "/video_stream/input_imgs", Stream, queue_size=2
        )

        if visualize:
            self.input_viz_win = cv2.namedWindow("Input Stream")

    def __del__(self):
        cv2.destroyWindow("Input Stream")

    def viz(self, rgb, depth):
        cv2.imshow(
            "Input Stream",
            np.concatenate(
                (
                    rgb,
                    np.zeros((rgb.shape[0], 10, 3), dtype=np.uint8),
                    cv2.applyColorMap(
                        cv2.normalize(
                            depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
                        ),
                        cv2.COLORMAP_JET,
                    ),
                ),
                1,
            ),
        )

    def __len__(self) -> int:
        return self.length

    def __getitem__(
        self,
        index: int,
    ) -> Union[None, Tuple[Any, ...]]:

        if index >= self.length:
            return None

        rgb = cv2.imread(
            self.rgb_dir + "/frame%06i.png" % self.index,
            cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH,
        )

        depth = cv2.imread(
            self.depth_dir + "/frame%06i.png" % self.index,
            cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH,
        )
        self.index += 1

        # If one of the files did not exist
        if rgb is None or depth is None:
            return self.__getitem__(self.index)

        else:
            return (rgb, depth)

    def start(self):
        pass

    def stop(self):
        pass

    def get_frame(self) -> Union[None, Tuple[Any, ...]]:
        self.index += 1
        return self.__getitem__(self.index)

    def wait_for_frame(self) -> Union[None, Tuple[Any, ...]]:
        return self.get_frame()

    def loop(self) -> None:
        while not rospy.is_shutdown():
            start_t = time.time()
            rgb, dep = self.wait_for_frame()
            msg = Stream()
            msg.rgb = rgb.message
            msg.depth_map = dep.message
            msg.id = self.index
            self.input_pub.publish(msg)
            time.sleep(self.publish_rate - start_t)


class RealSense(VideoStream):
    r"""RealSense Camera input, either streaming directly from the cam or using the intel bagfile"""

    def __init__(
        self,
        input_file: Optional[str] = None,
        visualize: bool = False,
        publish_rate: Union[int, float] = 2,
    ) -> None:
        # RealSense Pipeline
        self.publish_rate = publish_rate
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.bridge = CvBridge()
        self.index = -1

        # If we are streaming from bag file
        if input_file is not None:
            # Tell config that we will use a recorded device from file to be used by the pipeline through playback.
            self.config.enable_device_from_file(input_file)

        # TODO: Change parameters of this
        self.config.enable_stream(rs.stream.depth, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, rs.format.bgr8, 30)

        self.input_pub = rospy.Publisher(
            "/video_stream/input_imgs", Stream, queue_size=2
        )
        # Create cv2 window if we are visualizing input stream
        if visualize:
            self.input_viz_win = cv2.namedWindow("Input Stream")

    def __del__(self):
        cv2.destroyWindow("Input Stream")

    def viz(self, rgb, depth):
        cv2.imshow(
            "Input Stream",
            np.concatenate(
                (
                    rgb,
                    np.zeros((rgb.shape[0], 10, 3), dtype=np.uint8),
                    cv2.applyColorMap(
                        cv2.normalize(
                            depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
                        ),
                        cv2.COLORMAP_JET,
                    ),
                ),
                1,
            ),
        )

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

    def loop(self) -> None:
        while not rospy.is_shutdown():
            start_t = time.time()
            while True:
                frame = self.wait_for_frame()
                color_frame = frame.get_color_frame()
                depth_frame = frame.get_depth_frame()
                if color_frame and depth_frame:
                    break
            rgb_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            rgb_msg = self.bridge.cv2_to_imgmsg(rgb_image, encoding="passthrough")
            depth_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding="passthrough")
            self.index += 1
            msg = Stream()
            msg.rgb = rgb_msg
            msg.depth_map = depth_msg
            msg.id = self.index
            self.input_pub.publish(msg)
            time.sleep(self.publish_rate + time.time() - start_t)


class ROSBag(VideoStream):
    r"""Stream video from ROS bagfile

    Ref:
        IntelRealSense/librealsense python examples:
        https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/read_bag_example.py

    """

    def __init__(
        self,
        input_file: str,
        topics: list,
        visualize: bool = False,
        publish_rate: Union[int, float] = 2,
    ) -> None:

        self.bag = rosbag.Bag(input_file)
        self.index = -1
        self.publish_rate = publish_rate

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

        self.input_pub = rospy.Publisher(
            "/video_stream/input_imgs", Stream, queue_size=2
        )
        # Create cv2 window if we are visualizing input stream
        if visualize:
            self.input_viz_win = cv2.namedWindow("Input Stream")

    def __del__(self):
        cv2.destroyWindow("Input Stream")
        self.bag.close()

    def viz(self, rgb, depth):
        cv2.imshow(
            "Input Stream",
            np.concatenate(
                (
                    rgb,
                    np.zeros((rgb.shape[0], 10, 3), dtype=np.uint8),
                    cv2.applyColorMap(
                        cv2.normalize(
                            depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
                        ),
                        cv2.COLORMAP_JET,
                    ),
                ),
                1,
            ),
        )

    def __len__(self) -> int:
        return self.length

    def __getitem__(
        self,
        index: int,
    ) -> Union[Tuple[Any, ...], None]:

        if index >= self.length:
            return None

        return_list = []
        for gen in self.generators:
            # This snippet of code just indexes a generator
            try:
                return_list.append(next(itertools.islice(gen, index, None)))
            except StopIteration:
                return None
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

    def loop(self) -> None:
        while not rospy.is_shutdown():
            start_t = time.time()
            rgb, dep = self.wait_for_frame()
            print("video_stream publishing image", self.index)
            msg = Stream()
            msg.rgb = rgb.message
            msg.depth_map = dep.message
            msg.id = self.index
            self.input_pub.publish(msg)
            time.sleep(self.publish_rate - (time.time() - start_t))


def main():
    rospy.init_node("video_stream")
    stream_type = rospy.get_param("video_stream")
    visualize = rospy.get_param("visualize")
    video_stream = None

    if stream_type.lower() == "rosbag":
        bag_file = rospy.get_param("bag_file")
        publish_rate = rospy.get_param("publish_rate")
        topics = rospy.get_param("topics")
        video_stream = ROSBag(
            input_file=bag_file,
            topics=topics,
            visualize=visualize,
            publish_rate=publish_rate,
        )

    elif stream_type.lower() == "realsense":
        video_stream = RealSense(visualize=visualize)
        video_stream.start()

    elif stream_type.lower() == "rawimages":
        publish_rate = rospy.get_param("publish_rate")
        rgb_dir = rospy.get_param("rgb_dir")
        depth_dir = rospy.get_param("depth_dir")

        video_stream = RawImages(
            rgb_dir=rgb_dir, depth_dir=depth_dir, visualize=visualize
        )

    # Loop, publishing images
    video_stream.loop()


if __name__ == "__main__":
    main()
