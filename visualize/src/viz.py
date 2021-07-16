#!/usr/bin/env python
import cv2
import message_filters
import numpy as np
import rospy
from cv_bridge import CvBridge
from obj_inference.msg import Objects
from segmentation.msg import Prediction
from sensor_msgs.msg import Image

bridge = CvBridge()


def annotate_image(rgb, centers, labels, scores, dists):
    for i, center in enumerate(centers):
        rgb = cv2.circle(rgb, (int(center[0]), int(center[1])), 5, (0, 55, 255), -1)
        rgb = cv2.putText(
            rgb,
            labels[i] + "{:.2f}".format(scores[i]),
            fontScale=0.65,
            org=(int(center[0]), int(center[1])),
            thickness=1,
            color=(194, 66, 245),
            fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
        )
        rgb = cv2.putText(
            rgb,
            "{:.2f}".format(dists[i]),
            fontScale=0.65,
            org=(int(center[0]), int(center[1] + 20)),
            thickness=1,
            color=(194, 66, 245),
            fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
        )
    return rgb


def callback(pred, inf, rgb, dep):
    print("Visualize recieved messages")
    sizes = np.array(inf.sizes)
    scores = np.array(inf.scores)
    labels = inf.labels
    positions = np.array(inf.positions).reshape(sizes.shape[0], 2)

    rgb_img = bridge.imgmsg_to_cv2(rgb, desired_encoding="passthrough")
    dep_img = bridge.imgmsg_to_cv2(dep, desired_encoding="passthrough")

    height = pred.mask_height
    width = pred.mask_width
    chan = pred.mask_channels
    centers = np.array(pred.centers).reshape(chan, 2)
    mask = np.frombuffer(pred.mask, dtype=np.uint8).reshape(chan, width, height).sum(0)
    rgb = annotate_image(rgb_img, centers, labels, scores, positions[:, 1])
    masskk = cv2.applyColorMap(
        cv2.normalize(
            mask[:, :, np.newaxis], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        ),
        cv2.COLORMAP_JET,
    )
    depppthh = cv2.applyColorMap(
        cv2.normalize(dep_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U),
        cv2.COLORMAP_JET,
    )
    divider = np.zeros((480, 10, 3), dtype=np.uint8)
    concatenn = np.concatenate((masskk, divider, depppthh, divider, rgb), 1)
    cv2.imshow("input", concatenn)
    key = cv2.waitKey(20)
    if key == 27:
        cv2.destroyAllWindows()


class Visualizer:
    r""" """

    def __init__(self):
        pred_sub = message_filters.Subscriber("/seg/prediction", Prediction)
        inf_sub = message_filters.Subscriber("/inference/obj_inference", Objects)
        rgb_sub = message_filters.Subscriber("/camera/color/image_raw", Image)
        dep_sub = message_filters.Subscriber("/camera/depth/image_rect_raw", Image)

        ts = message_filters.ApproximateTimeSynchronizer(
            [pred_sub, inf_sub, rgb_sub, dep_sub],
            10,
            0.1,
        )
        ts.registerCallback(callback)


if __name__ == "__main__":
    viz = rospy.get_param("visualize")

    if viz:
        rospy.init_node("visualize")
        cv2.namedWindow("input")
        viz = Visualizer()
        rospy.spin()
