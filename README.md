

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
├── lib                          # External libraries
├── video_stream                 # Streams video from bagfile or Realsense Cam.
├── segmentation                 # Takes streamed video and outputs prediction masks + classes
├── obj_inference                # Extracts distance and center point from pred. masks
└── slam                         # Uses center points of objects to map + localize

```


## Important ROS Topics

```bash
- /seg/prediction                   # Model output
- /video_stream/rgb_img             # RGB input image
- /video_stream/depth_img           # Depth image input
- /inference/obj_inference          # Distances, relative positions of objects
```


## Output

 - label
 - distance
 - Relative location from cane
 - class probability
 - obj. width




## Dependencies

- pygame
- ROS Melodic, built with Python3


## Building

## Running

First, [edit the config file](#config-file)

```bash


```


## Config File

Entries to edit:
- RGB + Depth map input source - [video_stream](./video_stream/README.md)  
    - `ROSBag` - Input video from bag file
    - `RawFiles` - input video from rgb directory + depth directory
    - `RealSense` - input video from connected RealSense Came
- Semantic Segmentation Model - [segmentation](./segmentation/README.md)

#### Example:
```yaml
### Video Input Config
video_stream: "ROSBag"
bag_file: "./segmentation/bags/rgbd.bag"
topics: 
    - "/camera/depth/image_rect_raw/compressed"
    - "/camera/color/image_raw/compressed"
visualize: True

### Segmentation model config
segmentation_model: "detectron"
```


## Building the Documentation


## SLAM Library:

[nwang57/FastSLAM](https://github.com/nwang57/FastSLAM)


## Semantic Segmentation Library


## TODO: 

- Make video streamer asynch, use multiprocessing like torch DataLoader
- example script
- Sphinx documentation
