import os, sys
import cv2
import time
import torch
import numpy as np
import imageio as io
import pygame

sys.path.insert(0, os.path.abspath(".."))

from video_stream.streamer import RawImages
from obj_inference.inference import Distance_Inference
from segmentation.model import SegmentationModel
from slam import FastSLAM

#  from segmentation.detectron2.detectron2.data import MetadataCatalog

def graph_pred(mask):
    actual_mask = np.zeros((mask[0].shape[0], mask[0].shape[1]))
    for a in mask:
        tmp = a + 1
        actual_mask += tmp

    return actual_mask


def draw(window, positions):
    for pos in positions:
        norm_pos = pos / np.linalg.norm(pos)
        norm_pos *= 480
        pygame.draw.circle(window, (0, 255, 0), (norm_pos[0], norm_pos[1]), 4)


if __name__ == "__main__":
    model = SegmentationModel("detectron", "", {"yaml": 0})
    obj_inf = Distance_Inference()

    streamer = RawImages(
        rgb_dir="../segmentation/rgbd_output/RGB",
        depth_dir="../segmentation/rgbd_output/depth",
    )

    cv2.namedWindow("test")
    #  FPS = 30
    #  WINDOWWIDTH = 500
    #  WINDOWHEIGHT = 500
    #  COLOR = {
    #  "white": (255, 255, 255),
    #  "black": (0, 0, 0),
    #  "green": (0, 255, 0),
    #  "blue": (0, 0, 255),
    #  "red": (255, 0, 0),
    #  "purple": (128, 0, 128),
    #  }

    SLAM = FastSLAM(x=250, y=250, orien=0)
    #  window = pygame.display.set_mode((WINDOWWIDTH, WINDOWHEIGHT), 0, 32)
    with torch.no_grad():
        frame = 1

        rgb, depth = streamer.get_frame()

        while frame is not None:
            rgb, depth = streamer.get_frame()

            pred, bound_box, labels, scores = model.format_out(rgb)
            print("=======================================")
            print("LABELS< SCORESc")
            print((labels))
            print((scores))
            print("=======================================")

            mask = graph_pred(pred.cpu().numpy())

            vals, counts = np.unique(mask, return_counts=True)

            vals, counts = np.unique(depth, return_counts=True)
            dists, sizes, rel_positions = obj_inf.get_distance(
                mask=pred, depth_map=depth, centers=bound_box.get_centers()
            )
            print("CENTERS OF BOXES", bound_box.get_centers())
            print("dists: ", dists)
            print("Sizes: ", sizes)
            print("REL POS: ", rel_positions)
            print("")
            #  window.fill((255, 255, 255))
            #  draw(window, rel_positions)
            #  pygame.display.update()
            print("rel_positions", rel_positions.shape)
            rel_positions *= 380 / rel_positions.max()
            for pos in rel_positions:
                print("====")
                dis = np.linalg.norm(pos)
                direction = np.arctan2(pos[1], pos[0])
                print("pos", pos)
                print("dis", dis)
                print("Direct", direction)
                SLAM.new_obs([(dis, direction)])

            # yapf: disable
            for b in bound_box:
                rgb = cv2.rectangle(rgb, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (52, 235, 125), 2)
            # yapf: enable

            for i, cent in enumerate(bound_box.get_centers()):
                rgb = cv2.circle(rgb, (int(cent[0]), int(cent[1])), 5, (0, 55, 255), -1)
                rgb = cv2.putText(
                    rgb,
                    model.get_class_names(labels[i])+ "-{:.2f}".format(scores[i]),
                    fontScale=0.65,
                    org=(int(cent[0]), int(cent[1])),
                    thickness=1,
                    color=(194, 66, 245),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                )

                rgb = cv2.putText(
                    rgb,
                    "{:.2f}m".format(dists[i] / 1000),
                    fontScale=0.65,
                    org=(int(cent[0]), int(cent[1]) + 20),
                    thickness=1,
                    color=(194, 66, 245),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                )

            print("CONCAT")
            print("CONCAT")
            # yapf: disable
            print("MASK TYPE", rgb.dtype)
            divider = np.zeros((480, 10, 3), dtype=np.uint8)
            masskk = cv2.applyColorMap(cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U), cv2.COLORMAP_JET)
            depppthh = cv2.applyColorMap(cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U), cv2.COLORMAP_JET)
            concatenn = np.concatenate((masskk, divider, depppthh, divider, rgb), 1)
            print("CONCAT")
            print(concatenn.shape)
            cv2.imshow("test", concatenn)


            key = cv2.waitKey(20)
            if key == 27:  # exit on ESC
                break
            time.sleep(0.2)
        cv2.destroyWindow("test")
