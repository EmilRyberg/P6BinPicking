###from darknetpy.detector import Detector
###import pyrealsense2 as rs
from PIL import Image as pimg
from PIL import ImageDraw
import cv2
import glob
import imutils 
import numpy as np
import scipy.misc
from enums import PartEnum, OrientationEnum
import os
###from orientation_detector import OrientationDetector
from class_converter import convert_to_part_id
from utils import image_shifter
from aruco import Calibration


###YOLOCFGPATH = '/DarkNet/'
IMAGE_NAME = "webcam_capture.png"
ORIENTATION_MODEL_PATH = "orientation_cnn.hdf5"

class Vision:
    def __init__(self):
        ###self.rs_pipeline = rs.pipeline()
        self.current_directory = os.getcwd()
        ###yolo_cfg_path_absolute = self.current_directory + YOLOCFGPATH
        self.image_path = self.current_directory + "/" + IMAGE_NAME
        self.mask_path = self.current_directory + "/masks/"
        ###self.detector = Detector(yolo_cfg_path_absolute + 'cfg/obj.data', yolo_cfg_path_absolute + 'cfg/yolov3-tiny.cfg', yolo_cfg_path_absolute + 'yolov3-tiny_final.weights')
        self.counter = 0
        self.first_run = True
        self.results = None
        ###self.orientationCNN = OrientationDetector(ORIENTATION_MODEL_PATH)
        self.shifter = image_shifter.RuntimeShifter
        self.calibrate = Calibration()

    #def __del__(self):
        # Stop streaming
        ###self.rs_pipeline.stop()

    def capture_image(self):
        if self.first_run:
            ###cfg = rs.config()
            # cfg.enable_stream(realsense.stream.depth, 1280, 720, realsense.format.z16, 30)
            ###cfg.enable_stream(rs.stream.color, 1920, 1080, rs.format.rgb8, 30)

            ###profile = self.rs_pipeline.start(cfg)
            ###sensors = profile.get_device().query_sensors()
            ###rgb_camera = sensors[1]
            ###rgb_camera.set_option(rs.option.white_balance, 4600)
            ###rgb_camera.set_option(rs.option.exposure, 80)
            #rgb_camera.set_option(rs.option.saturation, 65)
            #rgb_camera.set_option(rs.option.contrast, 50)


            frames = None
            # wait for autoexposure to catch up
            ###for i in range(90):
                ###frames = self.rs_pipeline.wait_for_frames()
            ###self.first_run = False

        ###frames = self.rs_pipeline.wait_for_frames()
        ###color_frame = frames.get_color_frame()

        # Convert images to numpy arrays
        ###color_image = np.asanyarray(color_frame.get_data())
        ###color_image_ready_to_save = pimg.fromarray(color_image, 'RGB')
        ###color_image_ready_to_save.save(self.image_path)
        ###return color_image

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

    ###def detect_object(self):
        ###self.results = self.detector.detect(self.image_path)
        ###self.draw_boxes(self.results)

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

    ###def is_facing_right(self, np_image):
        ###result = self.orientationCNN.is_facing_right(np_image)
        ###print("[INFO] Part is facing right. {}".format(result))
        ###return result

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

    def find_point_in_mask(self, centre_x, centre_y, mask_contours, point_number):
        mask_contour = 0
        for c in mask_contours:
            area = cv2.contourArea(c)
            if area < 5000:
                continue
            else:
                mask_contour = c
        print(centre_x, centre_y)
        print(mask_contour)
        max_x = 0
        max_y = 0
        for i in range(len(mask_contour)):
            if mask_contour[i][0][0]>max_x:
                max_x = mask_contour[i][0][0]
            if mask_contour[i][0][1]>max_y:
                max_y = mask_contour[i][0][1]

        x_offset = int((max_x - centre_x) / 2)
        y_offset = int((max_y - centre_y) / 2)

        if point_number == 1:
            return centre_x + x_offset, centre_y
        elif point_number == 2:
            return centre_x, centre_y + y_offset
        else:
            print("Invalid point number specified")
            exit(-1)

    def vector_normal(self, x, y, mask_contours, depthimage):
        A = np.array([x, y, self.get_z(x, y, depthimage)])
        Bx, By = self.find_point_in_mask(x, y, mask_contours, 1)
        B = np.array([Bx, By, self.get_z(Bx, By, depthimage)])
        Cx, Cy = self.find_point_in_mask(x, y, mask_contours, 2)
        C = np.array([Cx, Cy, self.get_z(Cx, Cy, depthimage)])

        vector2 = C-A
        vector1 = B-A
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
    #x, y, z = hey.calibrate.calibrate(color_image, x, y, z)
    #print(x, y, z)
    hey.vector_normal(x, y, mask_contours, depth)


