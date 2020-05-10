import cv2
import numpy as np
import math
import time
from controllers.ur_controller.P6BinPicking.aruco import Calibration



class BoxDetector:
    def __init__(self):
        self.mean = np.array([24, 87, 126], np.uint8) #simulation thresholds
        self.std = np.array([5, 5, 5], np.uint8) #simulation thresholds
        self.lower_thresh = self.mean - self.std
        self.upper_thresh = self.mean + self.std
        self.calibration = Calibration()


    def find_box(self, img, debug=False, get_mask=False):
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_thresh = (int(self.lower_thresh[0]), int(self.lower_thresh[1]), int(self.lower_thresh[2]))
        upper_thresh = (int(self.upper_thresh[0]), int(self.upper_thresh[1]), int(self.upper_thresh[2]))
        img_thresholded = cv2.inRange(img_hsv, lower_thresh, upper_thresh)
        kernel = np.ones((3, 3), np.uint8)
        kernel2 = np.ones((5, 5), np.uint8)
        kernel3 = np.ones((7, 7), np.uint8)
        img_morph = cv2.morphologyEx(img_thresholded, cv2.MORPH_CLOSE, kernel)
        img_morph = cv2.morphologyEx(img_morph, cv2.MORPH_OPEN, kernel)
        img_morph = cv2.dilate(img_morph, kernel, iterations=3)
        img_morph = cv2.morphologyEx(img_morph, cv2.MORPH_CLOSE, kernel)
        img_morph = cv2.morphologyEx(img_morph, cv2.MORPH_CLOSE, kernel2)
        img_morph = cv2.morphologyEx(img_morph, cv2.MORPH_CLOSE, kernel3)

        img_floodfill = img_morph.copy()
        h, w = img_floodfill.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(img_floodfill, mask, (0, 0), 255)
        img_floodfill_inv = cv2.bitwise_not(img_floodfill)
        img_binary_final = img_morph | img_floodfill_inv

        contours, hierarchy = cv2.findContours(img_binary_final, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        max_contour_area = 0
        largest_contour = None
        for contour in contours:
            contour_area = cv2.contourArea(contour)
            if contour_area > max_contour_area:
                max_contour_area = contour_area
                largest_contour = contour

        epsilon = 0.1 * cv2.arcLength(largest_contour, True)
        approx_rect_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
        final_rect = np.array(approx_rect_contour).reshape((4, -1))

        if debug:
            print("conts", approx_rect_contour)
            print("final rect", final_rect)
            cv2.imshow("thresholded", img_thresholded)
            cv2.imwrite("thresholded.png", img_thresholded)
            cv2.imshow("morph", img_morph)
            cv2.imwrite("morph.png", img_thresholded)
            cv2.imshow("flood", img_floodfill)
            cv2.imshow("bin final", img_binary_final)
            cv2.imwrite("bin_final.png", img_binary_final)
            img_cont = img.copy()
            cv2.drawContours(img_cont, contours, -1, (0, 255, 0), 2)
            cv2.imshow("contours", img_cont)
            cv2.imwrite("contours.png", img_cont)
            img_cont2 = img.copy()
            cv2.drawContours(img_cont2, [approx_rect_contour], -1, (0, 0, 255), 2)
            cv2.imshow("contours2", img_cont2)
            cv2.imwrite("lower_approximation_contour.png", img_cont2)
            cv2.waitKey(0)

        if get_mask==True:
            return final_rect, img_binary_final
        else:
            return final_rect

    def get_average_pixel_value(self, img):
        #img = cv2.imread(image_name, cv2.IMREAD_COLOR)
        roi_x, roi_y, roi_width, roi_height = cv2.selectROI("select", img)
        cv2.destroyWindow("select")

        img_crop = img[roi_y:(roi_y + roi_height), roi_x:(roi_x + roi_width)]
        cv2.imshow("cropped", img_crop)
        img_crop_hsv = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)

        mean, std = cv2.meanStdDev(img_crop_hsv)
        print(f'mean: {mean}, std: {std}')

        cv2.waitKey(0)

    def box_grasp_location(self, cv2_image):
        box_location = self.find_box(cv2_image)
        grasp_x = box_location[2][0] + int(((box_location[3][0] - box_location[2][0]) / 2))
        grasp_y = box_location[2][1] + int(((box_location[3][1] - box_location[2][1]) / 2))
        vector = [box_location[3][0] - box_location[2][0], box_location[3][1] - box_location[2][1]]
        box_start_vector = [0, -483]  # hardcoded
        dot_product = np.array(box_start_vector) @ np.array(vector)
        norm_dot_product = np.linalg.norm(np.array(box_start_vector) * np.linalg.norm(np.array(vector)))
        angle = np.arccos(dot_product / norm_dot_product)
        # print(angle)
        # print(grasp_x,grasp_y)
        grasp_location = self.calibration.calibrate(cv2_image, grasp_x, grasp_y, 130)
        return grasp_location, angle


if __name__ == "__main__":
    boxy = BoxDetector()
    img = cv2.imread("sim_box.jpg", cv2.IMREAD_COLOR)
    start = time.time()
    result = boxy.find_box(img)
    stop = time.time()
    diff = stop - start
    print(f"Result {result} tooks {diff*1000}ms")