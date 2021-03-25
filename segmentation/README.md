


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


---
**NOTE**

The `Prediction.mask` field MUST be resized when recieving. ROS messages only accept
1 dimensional arrays. 

Ex:

```python
def subscriber_callback(seg_output):
    height = pred.mask_height
    width = pred.mask_width
    channels = pred.mask_channels

    # using dtype=np.uint8 to maintain same type
    mask = np.frombuffer(pred.mask, dtype=np.uint8).reshape(channels, width, height)
```

---



