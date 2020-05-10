import os
import numpy as np
import cv2
from PIL import Image as pimg
import imutils
from aruco import Calibration
from scipy.spatial.transform import Rotation


class SurfaceNormals:
    def __init__(self):
        self.aruco = Calibration()

    def find_point_in_mask(self, centre_x, centre_y, mask_contours, point_number):
        mask_contour = None
        for c in mask_contours:
            area = cv2.contourArea(c)
            if area < 1000:
                continue
            else:
                mask_contour = c
        assert mask_contour.any() != None, "Couldn't find large enough mask contour"
        #print(centre_x, centre_y)
        #print(mask_contour)
        max_x = 0
        max_y = 0
        for i in range(len(mask_contour)):
            if mask_contour[i][0][0]>max_x:
                max_x = mask_contour[i][0][0]
            if mask_contour[i][0][1]>max_y:
                max_y = mask_contour[i][0][1]

        x_offset = int((max_x - centre_x) *0.8)
        y_offset = int((max_y - centre_y) *0.8)


        if point_number == 1:
            return centre_x + x_offset, centre_y
        elif point_number == 2:
            return centre_x, centre_y + y_offset
        else:
            print("Invalid point number specified")
            exit(-1)

    def get_z(self, x, y, depth_image):
        #TODO make finidng camera offset automatic
        z = 1000 - depth_image[y, x] * 10  # get value in mm
        return z

    def get_tool_orientation_matrix(self, np_mask, np_depth_image, np_reference_image):
        center, normal_vector = self.vector_normal(np_mask, np_depth_image, np_reference_image)
        tool_direction = normal_vector * -1
        s = np.sin(np.pi / 2)
        c = np.cos(np.pi / 2)
        s2 = np.sin(-np.pi / 2)
        c2 = np.cos(-np.pi / 2)
        Ry = np.array([[c, 0, s],
                       [0, 1, 0],
                       [-s, 0, c]])
        Rz = np.array([[c2, -s2, 0],
                       [s2, c2, 0],
                       [0, 0, 1]])
        x_vector = np.dot(Ry, tool_direction)
        x_vector = x_vector / np.linalg.norm(x_vector)
        y_vector = np.dot(Rz, x_vector)
        y_vector = y_vector / np.linalg.norm(y_vector)
        matrix = np.append(x_vector.reshape((3, 1)), y_vector.reshape((3, 1)), axis=1)
        matrix = np.append(matrix, tool_direction.reshape((3, 1)), axis=1)
        rot = Rotation.from_matrix(matrix)
        rotvec = rot.as_rotvec()

        reference_z = np.array([0, 0, 1])
        relative_angle_to_z = np.arccos(np.clip(np.dot(reference_z, normal_vector), -1.0, 1.0))
        return center, rotvec, normal_vector, relative_angle_to_z

    def vector_normal(self, np_mask, np_depthimage, np_reference_image):
        # depth = image_shifter.shift_image(depth)
        #pimg.fromarray(np_reference_image).show()
        pil_depth = pimg.fromarray(np_depthimage)
        pil_depth = pil_depth.resize((1920, 1080))
        #pil_depth.show()
        #pimg.fromarray(np_mask).show()
        np_depthimage = np.asarray(pil_depth)
        mask_contours = self.find_contour(np_mask)
        Ax, Ay = self.find_center(mask_contours)
        Az = self.get_z(Ax, Ay, np_depthimage)
        A = self.aruco.calibrate(np_reference_image, Ax, Ay, Az)
        Bx, By = self.find_point_in_mask(Ax, Ay, mask_contours, 1)
        Bz = self.get_z(Bx, By, np_depthimage)
        B = self.aruco.calibrate(np_reference_image, Bx, By, Bz)
        Cx, Cy = self.find_point_in_mask(Ax, Ay, mask_contours, 2)
        Cz = self.get_z(Cx, Cy, np_depthimage)
        C = self.aruco.calibrate(np_reference_image, Cx, Cy, Cz)

        vector2 = C-A
        vector1 = B-A
        normal_vector = np.cross(vector2, vector1)
        normal_vector = normal_vector / np.linalg.norm(normal_vector)

        #print(normal_vector)
        return A, normal_vector


    def find_contour(self, np_mask):
        mask = np_mask.copy()
        kernel = np.ones((10, 10), np.uint8)
        #cv2.imshow("a", mask)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        #cv2.imshow("b", mask)


        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        #c = cv2.drawContours(mask, cnts[0], -1, [120, 0, 0], thickness=2)
        #cv2.imshow("c", c)
        # cv2.waitKey()
        cnts = imutils.grab_contours(cnts)
        return cnts

    def find_center(self, cnts):
        if len(cnts) == 0:
            return -1, -1
        c = cnts[0]
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return cX, cY


if __name__ == "__main__":
    sn = SurfaceNormals()
    np_mask = np.asarray(pimg.open("m.BMP"))
    #np_mask = cv2.cvtColor(np_mask, cv2.COLOR_RGB2GRAY)
    np_depth = np.asarray(pimg.open("d.BMP"))
    #np_depth = cv2.cvtColor(np_depth, cv2.COLOR_RGB2GRAY)
    np_reference = np.asarray(pimg.open("r.BMP"))
    a = sn.get_tool_orientation_matrix(np_mask, np_depth, np_reference)
    pass
