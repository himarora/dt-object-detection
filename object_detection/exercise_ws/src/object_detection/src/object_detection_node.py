#!/usr/bin/env python3
import numpy as np
import rospy
import rospkg

from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from sensor_msgs.msg import CompressedImage, Image
from duckietown_msgs.msg import Twist2DStamped, LanePose, WheelsCmdStamped, BoolStamped, FSMState, StopLineReading, \
    AntiInstagramThresholds
from image_processing.anti_instagram import AntiInstagram
import cv2
from object_detection.model import Wrapper
from cv_bridge import CvBridge


class ObjectDetectionNode(DTROS):

    def __init__(self, node_name):

        # Initialize the DTROS parent class
        super(ObjectDetectionNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.PERCEPTION
        )

        # Construct publishers
        self.pub_obj_dets = rospy.Publisher(
            "~duckie_detected",
            BoolStamped,
            queue_size=1,
            dt_topic_type=TopicType.PERCEPTION
        )

        self.pub_d_viz_obj = rospy.Publisher(
            "~/debug/viz_obj/compressed", CompressedImage, queue_size=1,
            dt_topic_type=TopicType.DEBUG
        )

        # Construct subscribers
        self.sub_image = rospy.Subscriber(
            "~image/compressed",
            CompressedImage,
            self.image_cb,
            buff_size=10000000,
            queue_size=1
        )

        self.sub_thresholds = rospy.Subscriber(
            "~thresholds",
            AntiInstagramThresholds,
            self.thresholds_cb,
            queue_size=1
        )

        self.ai_thresholds_received = False
        self.anti_instagram_thresholds = dict()
        self.ai = AntiInstagram()
        self.bridge = CvBridge()

        model_file = rospy.get_param('~model_file', '.')
        rospack = rospkg.RosPack()
        model_file_absolute = rospack.get_path('object_detection') + model_file
        self.model_wrapper = Wrapper(model_file_absolute, n_classes=4)
        self.initialized = True
        self.classes = ["duckie", "cone", "truck", "bus"]
        self.log("Initialized!")

    def thresholds_cb(self, thresh_msg):
        self.anti_instagram_thresholds["lower"] = thresh_msg.low
        self.anti_instagram_thresholds["higher"] = thresh_msg.high
        self.ai_thresholds_received = True

    def image_cb(self, image_msg):
        if not self.initialized:
            return

        # TODO to get better hz, you might want to only call your wrapper's predict function only once ever 4-5 images?
        # This way, you're not calling the model again for two practically identical images. Experiment to find a good number of skipped
        # images.

        # Decode from compressed image with OpenCV
        try:
            image = self.bridge.compressed_imgmsg_to_cv2(image_msg)
        except ValueError as e:
            self.logerr('Could not decode image: %s' % e)
            return

        # Perform color correction
        if self.ai_thresholds_received:
            image = self.ai.apply_color_balance(
                self.anti_instagram_thresholds["lower"],
                self.anti_instagram_thresholds["higher"],
                image
            )

        image = cv2.resize(image, (224, 224))
        # bboxes, classes, scores = self.model_wrapper.predict(image)

        msg = BoolStamped()
        msg.header = image_msg.header
        msg.data = self.det2bool(bboxes, classes)
        print(f"Decision: {msg.data}\n")
        self.pub_obj_dets.publish(msg)

    def det2bool(self, bboxes, classes):
        # TODO remove these debugging prints
        # print(bboxes)
        # print(classes)
        clas = 0    # look for duckies
        min_area = 5000
        y_range = (50, 190)
        for i, (box, c) in enumerate(zip(bboxes, classes)):
            area = (box[2] - box[0]) * (box[3] - box[1])
            center = ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)
            if c == clas:
                print(f"{self.classes[c]} {center} {area:.2f}")
            if area > min_area and c == clas and y_range[0] < center[0] < y_range[1]:
                return True
        return False



if __name__ == "__main__":
    # Initialize the node
    object_detection_node = ObjectDetectionNode(node_name='object_detection_node')
    # Keep it spinning
    rospy.spin()
