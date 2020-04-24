from vision.yolo.detector import Detector
import pyrealsense2 as rs
from PIL import Image as pimg
from PIL import ImageDraw
import cv2
import glob
import imutils
import numpy as np
import scipy.misc
from controller.enums import PartEnum, OrientationEnum
import os
from vision.orientation.orientation_detector import OrientationDetectorNet
from controller.class_converter import convert_to_part_id, convert_from_part_id
from utils import image_shifter
from aruco import Calibration
import torch

import os

YOLOCFGPATH = 'vision/yolo/'
IMAGE_NAME = "webcam_capture.png"
ORIENTATION_MODEL_PATH = "orientation_cnn.pth"

class Vision:
    def __init__(self):
        self.rs_pipeline = rs.pipeline()
        self.current_directory = os.getcwd()
        yolo_cfg_path_absolute = self.current_directory + YOLOCFGPATH
        self.image_path = self.current_directory + "/" + IMAGE_NAME
        self.mask_path = self.current_directory + "/masks/"
        self.detector = Detector(os.path.join(yolo_cfg_path_absolute, 'cfg/obj.data'),
                                 os.path.join(yolo_cfg_path_absolute, 'cfg/yolov3-tiny.cfg'),
                                 os.path.join(yolo_cfg_path_absolute, 'yolov3-tiny_final.weights'))
        self.counter = 0
        self.first_run = True
        self.results = None
        self.orientationCNN = OrientationDetectorNet()
        self.orientationCNN.load_state_dict(torch.load(ORIENTATION_MODEL_PATH))
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
            rgb_camera.set_option(rs.option.saturation, 65)
            rgb_camera.set_option(rs.option.contrast, 50)


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

    def get_z(self, x, y, depth_image):
        z = depth_image[x,y,0]
        print(z)
        return z

    def vector_normal(self, x, y, mask, depthimage):
        A = np.array([x, y, self.get_z(x, y, depthimage)])
        B = np.array([x-10, y-10, self.get_z(x-10, y-10, depthimage)])
        C = np.array([x+10, y-10, self.get_z(x+10, y-10, depthimage)])
        D = np.array([x-10, y+10, self.get_z(x-10, y+10, depthimage)])


        vector1 = D-B
        vector2 = C-B
        normal_vector = np.cross(vector1, vector2)
        print(normal_vector)
        """
        #DEBUG CODE FOR VISUALISATION
        a, b, c = normal_vector

        # This evaluates a * x3 + b * y3 + c * z3 which equals d
        d = np.dot(normal_vector, D)

        print('The equation is {0}x + {1}y + {2}z = {3}'.format(a, b, c, d))

        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x = np.linspace(B[0], C[0], D[0])
        y = np.linspace(B[1], C[1], D[1])
        X, Y = np.meshgrid(x, y)

        Z = (d - a * X - b * Y) / c

        # plot the mesh. Each array is 2D, so we flatten them to 1D arrays
        ax.plot(X.flatten(),
        Y.flatten(),
        Z.flatten(), 'bo ')

        # plot the original points. We use zip to get 1D lists of x, y and z
        # coordinates.
        ax.plot(*zip(B, C, D), color='r', linestyle=' ', marker='o')
        vector_origin = B
        ax.quiver(vector_origin[0], vector_origin[1], vector_origin[2], normal_vector[0], normal_vector[1], normal_vector[2])
        # adjust the view so we can see the point/plane alignment
        ax.view_init(10, 20)
        plt.tight_layout()
        #plt.savefig('images/plane.png')
        plt.show()"""

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