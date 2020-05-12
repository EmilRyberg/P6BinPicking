import os
import numpy as np
import cv2
from PIL import Image as pimg
import imutils
from aruco import Calibration
from scipy.spatial.transform import Rotation
from vision.vision import Vision


class SurfaceNormals:
    def __init__(self):
        self.aruco = Calibration()
        self.vision = Vision()

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

    def get_gripper_orientation(self, np_mask, np_depth_image, np_reference_image, rotation_around_self_z=0, debug=False):
        # Getting the img ready for PCA
        mat = np.argwhere(np_mask != 0)
        mat[:, [0, 1]] = mat[:, [1, 0]]
        mat = np.array(mat).astype(np.float32)  # have to convert type for PCA

        pil_depth = pimg.fromarray(np_depth_image)
        pil_depth = pil_depth.resize((1920, 1080))
        np_depth_image = np.asarray(pil_depth)

        mean, eigenvectors = cv2.PCACompute(mat, mean=np.array([]))  # computing PCA
        #print("eigenvectors", eigenvectors)

        center_img_space = np.array(np.round(mean[0]), dtype=np.int32)
        long_vector_point = np.array(np.round(center_img_space + eigenvectors[0] * 20), dtype=np.int32)
        short_vector_point = np.array(np.round(center_img_space + eigenvectors[1] * 20), dtype=np.int32)
        print(f'center: {center_img_space}, long: {long_vector_point}, short: {short_vector_point}')

        center_z = self.get_z(center_img_space[0], center_img_space[1], np_depth_image)
        center = self.aruco.calibrate(np_reference_image, center_img_space[0], center_img_space[1], center_z)
        long_vector_z = self.get_z(long_vector_point[0], long_vector_point[1], np_depth_image)
        long_vector = self.aruco.calibrate(np_reference_image, long_vector_point[0], long_vector_point[1], long_vector_z)
        short_vector_z = self.get_z(short_vector_point[0], short_vector_point[1], np_depth_image)
        short_vector = self.aruco.calibrate(np_reference_image, short_vector_point[0], short_vector_point[1], short_vector_z)

        vector1 = long_vector - center  # from tests this should be x (if normal is pointing in)
        vector2 = short_vector - center  # and this y
        vector1 = vector1 / np.linalg.norm(vector1)
        vector2 = vector2 / np.linalg.norm(vector2)
        normal_vector_in = np.cross(vector1, vector2)
        normal_vector_in = normal_vector_in / np.linalg.norm(normal_vector_in)
        normal_vector_out = normal_vector_in * -1

        matrix = np.append(vector1.reshape((3, 1)), vector2.reshape((3, 1)), axis=1)
        matrix = np.append(matrix, normal_vector_in.reshape((3, 1)), axis=1)

        theta = rotation_around_self_z
        ux, uy, uz = normal_vector_in[0], normal_vector_in[1], normal_vector_in[2]
        W = np.array([[0, -uz, uy],
                      [uz, 0, -ux],
                      [-uy, ux, 0]])
        I = np.identity(3)
        R = I + np.sin(theta) * W + (1 - np.cos(theta)) * (W @ W)
        # print(R)
        orientation = R @ matrix
        rotvec = Rotation.from_matrix(orientation).as_rotvec()
        # print(rotvec)

        reference_z = np.array([0, 0, 1])
        relative_angle_to_z = np.arccos(np.clip(np.dot(reference_z, normal_vector_out), -1.0, 1.0))

        # print(normal_vector)
        # return A, normal_vector_out
        if debug:
            # debug/test stuff
            np_mask_255 = np_mask * 255
            rgb_img = cv2.cvtColor(np_mask_255, cv2.COLOR_GRAY2BGR)
            cv2.circle(rgb_img, tuple(center_img_space), 5, 0)
            cv2.line(rgb_img, tuple(center_img_space), tuple(mean[0] + eigenvectors[0] * 20), (0, 0, 255))
            cv2.line(rgb_img, tuple(center_img_space), tuple(mean[0] + eigenvectors[1] * 20), (0, 255, 0))
            cv2.imshow("out", rgb_img)
            cv2.waitKey(0)

        return center, rotvec, normal_vector_out, relative_angle_to_z

    def vector_normal(self, np_mask, np_depthimage, np_reference_image, rotation_around_self_z=0):
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

        vector2 = C-A #pointing down = effective -x
        vector1 = B-A #pointing right = effective y
        vector1 = vector1 / np.linalg.norm(vector1)
        vector2 = vector2 / np.linalg.norm(vector2)
        normal_vector_out = np.cross(vector2, vector1)
        normal_vector_out = normal_vector_out / np.linalg.norm(normal_vector_out)
        normal_vector_in = normal_vector_out * -1

        vector2 = vector2 * -1 # convert to effective x
        matrix = np.append(vector2.reshape((3, 1)), vector1.reshape((3, 1)), axis=1)
        matrix = np.append(matrix, normal_vector_in.reshape((3, 1)), axis=1)
        #print(matrix)

        # https://math.stackexchange.com/questions/142821/matrix-for-rotation-around-a-vector
        theta = rotation_around_self_z
        ux, uy, uz = normal_vector_in[0], normal_vector_in[1], normal_vector_in[2]
        W = np.array([[0, -uz, uy],
                      [uz, 0, -ux],
                      [-uy, ux, 0]])
        I = np.identity(3)
        R = I + np.sin(theta) * W + (1-np.cos(theta))*(W @ W)
        #print(R)
        orientation = R @ matrix
        rotvec = Rotation.from_matrix(orientation).as_rotvec()
        #print(rotvec)

        reference_z = np.array([0, 0, 1])
        relative_angle_to_z = np.arccos(np.clip(np.dot(reference_z, normal_vector_out), -1.0, 1.0))

        #print(normal_vector)
        #return A, normal_vector_out
        return A, rotvec, normal_vector_out, relative_angle_to_z


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
    #a = sn.get_tool_orientation_matrix(np_mask, np_depth, np_reference)
    sn.vector_normal(np_mask, np_depth, np_reference)
    pass
