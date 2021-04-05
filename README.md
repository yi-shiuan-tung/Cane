

### Table of Contents
---

- [Important ROS Topics](#important-ros-topics)
- [Building](#building)
- [Running](#running)
- [Config File](#config-file)
- [Building the Documentation](#building-the-documentation)
- [SLAM Library](#slam-library)
- [Semantic Segmentation Library](#semantic-segmentation-library)
- [Todo](#todo)

---

## Project Layout
```
.
├── video_stream                 # Streams video from bagfile or Realsense Cam.
├── segmentation                 # Takes streamed video and outputs prediction masks + classes
├── obj_inference                # Extracts distance and center point from pred. masks
└── slam                         # Uses center points of objects to map + localize
```


## Important ROS Topics

- [`/seg/prediction`](./segmentation/msg/Prediction.msg) -- Segmentation masks, object centers, labels
- [`/video_stream/input_imgs`](./video_stream/msg/Stream.msg) -- Depth map + RGB input image
- [`/inference/obj_inference`](./obj_inference/msg/Objects.msg) -- Distances, relative positions of objects, labels




## Dependencies

- ROS Melodic, built with Python3
- OpenCV >= 4.4.0


## Building

### Install Detectron2
 
```bash
$ git clone https://github.com/facebookresearch/detectron2.git && cd detectron2
$ python setup.py install
```
Then [Download Detectron weights](https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x/137260431/model_final_a54504.pkl), 
for the R50-FPN Mask R-CNN for COCO InstanceSegmentation, all model weights can be found on the [Detectron GitHub](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md)

Then download the [model metric file](https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x/137260431/metrics.json).




### Building project
```bash
### SOURCE ROS FIRST ###

# make cane workspace 
$ mkdir -p cane_ws/src && cd cane_ws/src
$ git clone https://github.com/clbeggs/Cane.git

# cd to cane_ws
$ cd .. && catkin_make

# Source cane_ws
$ source devel/setup.bash  
```

## Running

First, [edit the config file](#config-file)

```bash
# Start segmentation model node
$ roslaunch segmentation segmentation.launch

# Start object inference node
$ roslaunch obj_inference inference.launch

# Start video stream
$ roslaunch video_stream streamer.launch
```

## Config File

Entries to edit:
- RGB + Depth map input source -- [video_stream](./video_stream/README.md)  
    - `ROSBag` - Input video from bag file
    - `RawFiles` - input video from rgb directory + depth directory
        - requires `rgb_dir` and `depth_dir` specifications in config file.
    - `RealSense` - input video from connected RealSense Came
- Semantic Segmentation Model -- [segmentation](./segmentation/README.md)
    - `detectron` - Facebook's [Detectron2 model](https://github.com/facebookresearch/detectron2)


#### Example:
```yaml
##################################
####### Video Input Config #######
##################################
# How the input images are fed to the model
video_stream: "ROSBag"                            # Valid inputs: ["RawImages", "RealSense"]
publish_rate: 2                                   # Publish an image every <publish_rate> seconds
visualize: True                                   # Visualize input images via cv2.namedWindow

### Necessary for all input methods ###
rgb_input_topic: "/camera/depth/image_rect_raw/compressed"   # RGB Input topic to subscribe to 
depth_input_topic: "/camera/color/image_raw/compressed"      #  Depth input topic to subscribe to

### ROSBag Config ###
bag_file: "./segmentation/bags/rgbd.bag"          # Input ROSbag file, only necessary for "ROSBag" video_stream


### RawFiles Config ###
rgb_dir: ".../path/to/rgb_dir"
depth_dir: ".../path/to/depth_dir"


#####################################
##### Segmentation model config #####
#####################################
segmentation_model: "detectron"                  

# Weights and config for Detectron2
model_weights: "...path/to/weights/model_final_a54504.pkl"
model_config: "...path/to/weights/mask_rcnn_R_50_FPN_1x.yaml"
```


## Building the Documentation

## TODO: 

- Make video streamer asynch, use multiprocessing like torch DataLoader
- example script
- Sphinx documentation
