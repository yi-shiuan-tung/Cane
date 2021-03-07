


# Segmentation Model


## Prediction ROS message

```bash
- Prediction.msg

string[] labels                 # labels of each detected object
float32[] centers               # centers of objects (if using detectron model)
sensor_msgs/Image mask          # output masks of model (int's)
sensor_msgs/Image depth_map     # input depth map
```

The `sensor_msgs/Image mask` entry is an image with integers denoting the different detected objects.
The entries in the labels array correspond to the number in the mask.

e.g.:
```bash
mask = [
    [0  1  2],
    [1  1  1],
    [2  1  1]]
labels = ["car", "bird"]

"""
Where the mask is 1 is the 'car' object,
and where the mask is 2 is the 'bird' object.
0's in the mask correspond to nothing.
"""
```





