import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import rospy 
import cv2
from detecto.core import Model
from detecto.utils import filter_top_predictions


class Detector(object):
    def __init__(self):
        self._model = Model()  
    def predict(self, img):
        return self._model.predict(img)
    
class Webcam(object):
    def __init__(self):
        self._device = cv2.VideoCapture(2)   
        
    def __del__(self):
        self._device.release()
        
    def get_frame(self):
        _, frame = self._device.read()
        return frame

class DetectorNode(object):
    def __init__(self, node_name, camera, detector, threshold):
        rospy.init_node(node_name, anonymous=False)
        self._camera = camera
        self._detector = detector
        self._rate = rospy.Rate(5)
        self._score_threshold = threshold       
    def run(self):
        while not rospy.is_shutdown():
            frame = self._camera.get_frame()
            predictions = self._detector.predict(frame)
            self.draw_bbox(frame, predictions)
            cv2.imshow("window", frame)
            cv2.waitKey(10)
            self._rate.sleep()
            
    def draw_bbox(self, img, p):
        for i in range(len(p[0])):
            label, bbox, probs = p[0][i], p[1][i], p[2][i]
            if probs < self._score_threshold:
                return
            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 3)
            cv2.putText(img, label, (int(x_min), int(y_min)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))

camera = Webcam()
detector = Detector()

node = DetectorNode("detector_node", camera, detector, 0.85)

node.run()
