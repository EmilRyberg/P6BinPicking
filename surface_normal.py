import os
import numpy as np
import cv2

class SurfaceNormals:
    def __init__(self):
        self.current_directory = os.getcwd()
        self.mask_path = self.current_directory + "/masks/"

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

    def get_z(self, x, y, depth_image):
        z = depth_image[x,y,0]
        print(z)
        return z

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
