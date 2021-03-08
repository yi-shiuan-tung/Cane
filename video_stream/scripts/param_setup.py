import rospy


def setup(bag_file: str):
    rospy.set_param("video_stream", "ROSBag")
    rospy.set_param("bag_file", bag_file)
    rospy.set_param(
        "topics", ["/camera/color/image_raw", "/camera/depth/image_rect_raw"]
    )
    rospy.set_param("use_sim_time", True)
    rospy.set_param("visualize", False)
    rospy.set_param("publish_rate", 1)
