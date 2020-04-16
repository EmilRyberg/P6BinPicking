from vision.yolo.detector import Detector
from PIL import Image
import numpy as np

if __name__ == '__main__':
    detector = Detector('vision/yolo/config/yolov3-tiny.cfg', 'vision/yolo/weights/yolov3-tiny_final.weights')
    image = Image.open('vision/yolo/images/color1574246897.8219955.png')
    np_image = np.array(image)
    detector.predict(np_image)