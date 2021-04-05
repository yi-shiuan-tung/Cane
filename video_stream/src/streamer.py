#!/usr/bin/env  python
import glob
import itertools
import os
import signal
import subprocess
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
        raise NotImplementedError

    @abstractmethod
    def get_frame(self):
        raise NotImplementedError

    @abstractmethod
    def wait_for_frame(self) -> Union[None, Tuple[Any, ...]]:
        raise NotImplementedError

    @abstractmethod
    def start(self):
        raise NotImplementedError

    @abstractmethod
    def stop(self):
        raise NotImplementedError


class RawImages(VideoStream):
    r"""Stream input from two seperate folders, of depth and rgb images"""

    def __init__(
        self,
        rgb_dir: str,
        depth_dir: str,
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
            msg.header = rgb.message.header
            self.input_pub.publish(msg)
            time.sleep(self.publish_rate - start_t)


class RealSense(VideoStream):
    r"""RealSense Camera input, either streaming directly from the cam or using the intel bagfile"""

    def __init__(
        self,
        input_file: Optional[str] = None,
        publish_rate: Union[int, float] = 2,
    ) -> None:
        # RealSense Pipeline
        self.publish_rate = publish_rate
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.index = -1

        # If we are streaming from bag file
        if input_file is not None:
            # Tell config that we will use a recorded device from file to be used by the pipeline through playback.
            self.config.enable_device_from_file(input_file)

        # TODO: Change parameters of this
        self.config.enable_stream(rs.stream.depth, rs.format.z16, 30)

        self.input_pub = rospy.Publisher(
            "/video_stream/input_imgs", Stream, queue_size=2
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
            rgb, dep = self.wait_for_frame()
            self.index += 1
            msg = Stream()
            msg.rgb = rgb.message
            msg.depth_map = dep.message
            msg.header = rgb.message.header
            self.input_pub.publish(msg)
            time.sleep(self.publish_rate - start_t)


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
        publish_rate: Union[int, float] = 2,
    ) -> None:

        self.bag = rosbag.Bag(input_file)
        self.index = -1
        self.publish_rate = publish_rate

        # List of topics to subscribe to
        self.generators = []
        self.length = -1

        if len(topics) > 0:
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
        else:
            pass

        self.input_pub = rospy.Publisher(
            "/video_stream/input_imgs", Stream, queue_size=2
        )

    def __del__(self):
        self.bag.close()

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
            msg = Stream()
            msg.rgb = rgb.message
            msg.depth_map = dep.message
            msg.header = rgb.message.header
            print("video_stream publishing image", msg.header.stamp)
            self.input_pub.publish(msg)
            time.sleep(self.publish_rate - (time.time() - start_t))


class ROSBagSP(VideoStream):
    """Start subprocess to play ROSbag file as user specified publish_rate."""

    def __init__(
        self,
        input_file: str,
        publish_rate_hz: Union[int, float] = 2,
    ) -> None:

        path = os.path.abspath(input_file)
        hz = "--hz={}".format(publish_rate_hz)
        DVNULL = open(os.devnull, "w")
        print("Starting to publish from ROSBag")
        self.proc = subprocess.Popen(["rosbag", "play", hz, input_file], stdout=DVNULL)

    def __del__(self):
        self.proc.kill()

    def start(self):
        pass

    def stop(self):
        pass

    def get_frame(self) -> Tuple[Any, ...]:
        pass

    def wait_for_frame(self) -> Tuple[Any, ...]:
        pass

    def loop(self):
        rospy.spin()


def main():
    rospy.init_node("video_stream")
    stream_type = rospy.get_param("video_stream")
    video_stream = None

    if stream_type.lower() == "rosbag":
        bag_file = rospy.get_param("bag_file")
        publish_rate = rospy.get_param("publish_rate")
        video_stream = ROSBagSP(
            input_file=bag_file,
            publish_rate_hz=publish_rate,
        )

    elif stream_type.lower() == "realsense":
        video_stream = RealSense()

    elif stream_type.lower() == "rawimages":
        publish_rate = rospy.get_param("publish_rate")
        rgb_dir = rospy.get_param("rgb_dir")
        depth_dir = rospy.get_param("depth_dir")

        video_stream = RawImages(rgb_dir=rgb_dir, depth_dir=depth_dir)

    # Loop, publishing images
    video_stream.loop()


if __name__ == "__main__":
    main()
