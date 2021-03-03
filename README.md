


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

## SLAM Library:

[nwang57/FastSLAM](https://github.com/nwang57/FastSLAM)


## Semantic Segmentation


## Output

 - label
 - distance
 - Relative location from cane
 - class probability
 - obj. width




## Dependencies

- pygame
- ROS Melodic, built with Python3


## Building + Running




## TODO: 

- Make video streamer asynch, use multiprocessing like torch DataLoader
- example script
- Sphinx documentation
