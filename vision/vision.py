from vision.yolo.detector import Detector
from PIL import Image as pimg
from PIL import ImageDraw
from cv2 import cv2
import glob
import imutils
import numpy as np
import scipy.misc
import os
from vision.orientation.orientation_detector import OrientationDetectorNet
from utils.image_shifter import RuntimeShifter
from aruco import Calibration
import torch
from vision.segmentation.detector import InstanceDetector
import torch.utils.model_zoo
import os

YOLOCFGPATH = 'vision/yolo/'
IMAGE_NAME = "webcam_capture.png"
ORIENTATION_MODEL_PATH = "orientation_cnn.pth"


class Vision:
    def __init__(self, segmentation_weight_path):
        self.current_directory = os.getcwd()
        yolo_cfg_path_absolute = self.current_directory + YOLOCFGPATH
        self.image_path = self.current_directory + "/" + IMAGE_NAME
        self.mask_path = self.current_directory + "/masks/"
        """self.detector = Detector(os.path.join(yolo_cfg_path_absolute, 'cfg/obj.data'),
                                 os.path.join(yolo_cfg_path_absolute, 'cfg/yolov3-tiny.cfg'),
                                 os.path.join(yolo_cfg_path_absolute, 'yolov3-tiny_final.weights'))"""
        self.counter = 0
        self.first_run = True
        self.results = None
        self.orientationCNN = OrientationDetectorNet()
        #self.orientationCNN.load_state_dict(torch.load(ORIENTATION_MODEL_PATH))
        #self.shifter = image_shifter.RuntimeShifter
        self.calibrate = Calibration()
        self.segmentation_detector = InstanceDetector(segmentation_weight_path)

    def __del__(self):
        pass

    def find_parts(self, class_id, fuse_index=-1):
        class_id1, class_id2 = class_id
        part = (-1, -1, -1, -1, -1)
        # result is an array of dictionaries
        found_class_index = 0
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in self.results:
            if (class_id1 == cls_pred or class_id2 == cls_pred) and cls_conf > 0.6:
                if fuse_index > -1 and fuse_index != found_class_index:
                    found_class_index += 1
                    continue
                width = x2 - x1
                height = y2 - y1
                x_coord = width / 2 + x1
                y_coord = height / 2 + y1
                if height > width:
                    orientation = OrientationEnum.VERTICAL.value
                    grip_width = width * 0.58
                elif width > height:
                    orientation = OrientationEnum.HORIZONTAL.value
                    grip_width = height * 0.58
                else:
                    orientation = OrientationEnum.HORIZONTAL.value
                    grip_width = height * 0.58
                    print("[W] Could not determine orientation, using 1 as default")
                #new_part_id = convert_to_part_id(part_class)
                part = (cls_pred, x_coord, y_coord, orientation, grip_width)
                break
        print(part)
        return part

    def segment(self, np_img):
        results = self.segmentation_detector.predict(np_img)
        classes = ["PCB", "BottomCover", "BlueCover", "WhiteCover", "BlackCover"]
        masks = []
        for i in range(len(results["instances"].pred_classes)):
            mask_image = results["instances"].pred_masks[i].cpu().numpy()
            mask_image = np.asarray(mask_image * 255, dtype=np.uint8)
            moments = cv2.moments(mask_image)
            cX = int(moments["m10"] / moments["m00"])
            cY = int(moments["m01"] / moments["m00"])
            center = (cX, cY)
            area = moments["m00"]
            part = classes[results['instances'].pred_classes[i]]
            score = results['instances'].scores[i]
            mask = {"part": part, "score": score, "area": area, "center": center, "ignored": False, "ignore_reason": "", "mask": mask_image}
            masks.append(mask)
        return masks

    def detect_object(self):
        np_img = pimg.open(self.image_path)
        self.results = self.detector.predict(np_img)
        self.draw_boxes(self.results)

    def draw_boxes(self, results):
        source_img = pimg.open(self.image_path).convert("RGBA")
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in self.results:
            if cls_conf > 0.6:
                width = x2 - x1
                height = y2 - y1
                x_coord = width / 2 + x1
                y_coord = height / 2 + y1
                draw = ImageDraw.Draw(source_img)
                draw.rectangle(((x1, y1), (x2, y2)), fill=None, outline=(200, 0, 150), width=6)
                draw.text((x_coord, y_coord), convert_from_part_id(int(cls_pred)))
        source_img.save('boundingboxes.png')

    def is_facing_right(self, np_image):
        pil_image = pimg.fromarray(np_image)
        resized_image = pil_image.resize((224, 224))
        resized_image_np = np.array(resized_image) / 255
        image_tensor = torch.from_numpy(resized_image_np).permute(2, 0, 1).float()
        image_tensor = image_tensor.unsqueeze(0)
        self.orientationCNN.eval()
        with torch.no_grad():
            prediction = self.orientationCNN(image_tensor)
        result = prediction[0][0] >= 0.5
        print("[INFO] Part is facing right. {}".format(result))
        return result

    def get_image_path(self):
        return self.image_path

    def find_part_for_grasp(self):
        masks = glob.glob(self.mask_path + "*")
        number_of_masks = len(masks)
        print(f"There are {number_of_masks} masks")
        contour_sizes = []
        for index, file_path in enumerate(masks):
            mask = pimg.open(file_path)
            mask = np.array(mask)
            print(f"Finding contours on image {index + 1}/{number_of_masks}")
            contours = self.find_contour(mask)
            for c in contours:
                area = cv2.contourArea(c)
                if area < 5000:
                    continue
                else:
                    contour_sizes.append(area)

        print(contour_sizes)
        part_to_grasp = contour_sizes.index(max(contour_sizes))
        print(part_to_grasp)
        return part_to_grasp


if __name__ == "__main__":
    hey = Vision()
    masks = glob.glob(hey.mask_path + "*")
    part_to_grasp = hey.find_part_for_grasp()
    mask = pimg.open(masks[part_to_grasp])
    mask = np.array(mask)
    color_image = cv2.imread("color1582023984.5763314-0.png")

    depth = cv2.imread("depth.png")
    #depth = image_shifter.shift_image(depth)
    dim = (720, 1280)
    mask = cv2.resize(mask, dim)
    mask_contours = hey.find_contour(mask)
    x, y = hey.find_center(mask_contours)
    z = hey.get_z(x, y, depth)
    print(x, y, z)
    x, y, z = hey.calibrate.calibrate(color_image, x, y, z)
    print(x, y, z)
    #hey.vector_normal(x, y, img, depth)