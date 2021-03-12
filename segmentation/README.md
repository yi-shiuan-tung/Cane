


# Segmentation Model


## Prediction ROS message

```bash
- Prediction.msg

std_msgs/Header header
string[] labels                 # labels of each detected object
float32[] scores                # Confidence scores of each obj
float32[] centers               # centers of objects (if using detectron model)
sensor_msgs/Image depth_map     # input depth map

### Masks of each object, has to be stored as flat array
### but once recieved, is reshaped to be (mask_channels, width, height)
### where mask_channels == n_objects
uint8[] mask
uint16 mask_height
uint16 mask_width
uint16 mask_channels
```




