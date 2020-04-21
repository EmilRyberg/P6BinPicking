from darknetpy.detector import Detector
import pyrealsense2 as rs
from PIL import Image as pimg
from PIL import ImageDraw
import cv2
import glob
import imutils 
import numpy as np
import scipy.misc
from enums import PartEnum, OrientationEnum
import os
from orientation_detector import OrientationDetector
from class_converter import convert_to_part_id
from utils import image_shifter
from aruco import Calibration
from surface_normal import SurfaceNormals


YOLOCFGPATH = '/DarkNet/'
IMAGE_NAME = "webcam_capture.png"
ORIENTATION_MODEL_PATH = "orientation_cnn.hdf5"

class Vision:
    def __init__(self):
        self.rs_pipeline = rs.pipeline()
        self.current_directory = os.getcwd()
        yolo_cfg_path_absolute = self.current_directory + YOLOCFGPATH
        self.image_path = self.current_directory + "/" + IMAGE_NAME
        self.mask_path = self.current_directory + "/masks/"
        self.detector = Detector(yolo_cfg_path_absolute + 'cfg/obj.data', yolo_cfg_path_absolute + 'cfg/yolov3-tiny.cfg', yolo_cfg_path_absolute + 'yolov3-tiny_final.weights')
        self.counter = 0
        self.first_run = True
        self.results = None
        self.orientationCNN = OrientationDetector(ORIENTATION_MODEL_PATH)
        self.shifter = image_shifter.RuntimeShifter
        self.calibrate = Calibration()

    def __del__(self):
        # Stop streaming
        self.rs_pipeline.stop()

    def capture_image(self):
        if self.first_run:
            cfg = rs.config()
            # cfg.enable_stream(realsense.stream.depth, 1280, 720, realsense.format.z16, 30)
            cfg.enable_stream(rs.stream.color, 1920, 1080, rs.format.rgb8, 30)

            profile = self.rs_pipeline.start(cfg)
            sensors = profile.get_device().query_sensors()
            rgb_camera = sensors[1]
            rgb_camera.set_option(rs.option.white_balance, 4600)
            rgb_camera.set_option(rs.option.exposure, 80)
            #rgb_camera.set_option(rs.option.saturation, 65)
            #rgb_camera.set_option(rs.option.contrast, 50)


            frames = None
            # wait for autoexposure to catch up
            for i in range(90):
                frames = self.rs_pipeline.wait_for_frames()
            self.first_run = False

        frames = self.rs_pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        color_image_ready_to_save = pimg.fromarray(color_image, 'RGB')
        color_image_ready_to_save.save(self.image_path)
        return color_image

    def find_parts(self, class_id, fuse_index=-1):
        class_id1, class_id2 = class_id
        part = (-1, -1, -1, -1, -1)
        # result is an array of dictionaries
        found_class_index = 0
        for i in range(len(self.results)):
            d = self.results[i]
            if (d['class'] == class_id1 or d['class'] == class_id2) and d['prob'] > 0.6:
                if fuse_index > -1 and fuse_index != found_class_index:
                    found_class_index += 1
                    continue
                part_class = d['class']
                prob = d['prob']
                width = d['right'] - d['left']
                height = d['bottom'] - d['top']
                x_coord = width / 2 + d['left']
                y_coord = height / 2 + d['top']
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
                new_part_id = convert_to_part_id(part_class)
                part = (new_part_id, x_coord, y_coord, orientation, grip_width)
                break
        print(part)
        return part

    def detect_object(self):
        results = self.detector.detect(self.image_path)
        self.draw_boxes(self.results)

    def draw_boxes(self, results):
        source_img = pimg.open(self.image_path).convert("RGBA")
        for i in range(len(results)):
            d = results[i]
            if d['prob'] > 0.6:
                classify = d['class']
                prob = d['prob']
                width = d['right'] - d['left']
                height = d['bottom'] - d['top']
                x_coord = width / 2 + d['left']
                y_coord = height / 2 + d['top']
                draw = ImageDraw.Draw(source_img)
                draw.rectangle(((d['left'], d['top']), (d['right'], d['bottom'])), fill=None, outline=(200, 0, 150), width=6)
                draw.text((x_coord, y_coord), d['class'])
        source_img.save('boundingboxes.png')

    def is_facing_right(self, np_image):
        result = self.orientationCNN.is_facing_right(np_image)
        print("[INFO] Part is facing right. {}".format(result))
        return result

    def get_image_path(self):
        return self.image_path

    def find_contour(self, mask):
        kernel = np.ones((10,10),np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        #cv2.imshow("MORPH", mask)
        #cv2.waitKey(0)

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        return cnts

    def find_center(self, cnts):
        for c in cnts:
            # compute the center of the contour
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # draw the contour and center of the shape on the image
            #mask_coloured = cv2.cvtColor(mask,cv2.COLOR_GRAY2RGB)
            #cv2.drawContours(mask_coloured, [c], -1, (0, 255, 0), 2)
            #cv2.circle(mask_coloured, (cX, cY), 7, (255, 0, 255), -1)
            #cv2.putText(mask_coloured, "center", (cX - 20, cY - 20),
                        #cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            # show the image
            #cv2.imshow("CENTER", mask_coloured)
            #cv2.waitKey(0)
            return cX, cY

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
    #x, y, z = hey.calibrate.calibrate(color_image, x, y, z)
    #print(x, y, z)
    SurfaceNormals.vector_normal(x, y, mask_contours, depth)


