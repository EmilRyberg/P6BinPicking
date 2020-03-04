###from darknetpy.detector import Detector
###import pyrealsense2 as rs
from PIL import Image as pimg
from PIL import ImageDraw
import cv2
import imutils 
import numpy as np
from enums import PartEnum, OrientationEnum
import os
###from orientation_detector import OrientationDetector
from class_converter import convert_to_part_id
from utils import image_shifter

###YOLOCFGPATH = '/DarkNet/'
IMAGE_NAME = "webcam_capture.png"
ORIENTATION_MODEL_PATH = "orientation_cnn.hdf5"

class Vision:
    def __init__(self):
        ###self.rs_pipeline = rs.pipeline()
        self.current_directory = os.getcwd()
        ###yolo_cfg_path_absolute = self.current_directory + YOLOCFGPATH
        self.image_path = self.current_directory + "/" + IMAGE_NAME
        ###self.detector = Detector(yolo_cfg_path_absolute + 'cfg/obj.data', yolo_cfg_path_absolute + 'cfg/yolov3-tiny.cfg', yolo_cfg_path_absolute + 'yolov3-tiny_final.weights')
        self.counter = 0
        self.first_run = True
        self.results = None
        ###self.orientationCNN = OrientationDetector(ORIENTATION_MODEL_PATH)
        self.shifter = image_shifter.RuntimeShifter

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

    def find_center(self, mask):
        kernel = np.ones((5,5),np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        cv2.imshow("Image", mask)
        cv2.waitKey(0)


        contour = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = imutils.grab_contours(contour)
        M = cv2.moments(contour)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        cv2.drawContours(mask, [0], -1, (0, 255, 0), 2)
        cv2.circle(mask, (center[0], center[1]), 7, (255, 255, 255), -1)
        cv2.putText(mask, "center", (center[0] - 20, center[1] - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.imshow("Image", mask)
        cv2.waitKey(0)
        return center

    def get_z(self, x, y, depth_image):
        z = depth_image.at<char>(point[0],point[1])
        return z

    def vector_normal(self, centerpoint, mask, depthimage):
        A = np.array([centerpoint[0], centerpoint[1], 0])
        B = np.array([centerpoint[0]-100, centerpoint[1]-100, 0])
        C = np.array([centerpoint[0]+100, centerpoint[1]-100, 0])
        D = np.array([centerpoint[0]-100, centerpoint[1]+100, 0])

        A[2] = self.get_z(A[0], A[1], depthimage)
        B[2] = self.get_z(B[0], B[1], depthimage)
        C[2] = self.get_z(C[0], C[1], depthimage)
        D[2] = self.get_z(D[0], D[1], depthimage)

        vector1 = D-B
        vector2 = C-B
        normal_vector = np.cross(vector1, vector2)

        """ DEBUG CODE FOR VISUALISATION
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

        # adjust the view so we can see the point/plane alignment
        ax.view_init(0, 22)
        plt.tight_layout()
        plt.savefig('images/plane.png')
        plt.show()
        """




if __name__ == "__main__":
    hey = Vision()
    while True:
        img = cv2.imread("/Users/AlbertOlesen/Documents/Skole/Uni/P6/Project/mask829.png")
        hey.find_center(img)
        ###hey.capture_image()
        ###hey.detect_object()
        input()