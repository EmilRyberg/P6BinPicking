import os
import numpy as np
import cv2
from PIL import Image as pimg
import imutils
from aruco import Calibration
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

        x_offset = int((max_x - centre_x) / 2)
        y_offset = int((max_y - centre_y) / 2)

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

        theta_z = self.vision.long_axis_rotation(np_mask)
        s_z = np.sin(theta_z)
        c_z = np.cos(theta_z)

        Ry = np.array([[c, 0, s],
                       [0, 1, 0],
                       [-s, 0, c]])
        Rz = np.array([[c_z, -s_z, 0],
                       [s_z, c_z, 0],
                       [0, 0, 1]])
        x_vector = np.dot(Ry, tool_direction)
        x_vector = x_vector / np.linalg.norm(x_vector)
        y_vector = np.dot(Rz, x_vector)
        y_vector = y_vector / np.linalg.norm(y_vector)
        matrix = np.append(x_vector.reshape((3, 1)), y_vector.reshape((3, 1)), axis=1)
        matrix = np.append(matrix, tool_direction.reshape((3, 1)), axis=1)
        return center, matrix

    def vector_normal(self, np_mask, np_depthimage, np_reference_image):
        # depth = image_shifter.shift_image(depth)
        pil_depth = pimg.fromarray(np_depthimage)
        pil_depth = pil_depth.resize((1920, 1080))
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

        print(normal_vector)
        return A, normal_vector
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

    def find_contour(self, np_mask):
        mask = np_mask.copy()
        kernel = np.ones((10, 10), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
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
    np_mask = np.array(pimg.open("bottom.png"))
    np_depth = np.asarray(pimg.open("depth.png"))
    np_reference = np.asarray(pimg.open("ref.png"))
    sn.vector_normal(np_mask, np_depth, np_reference)
