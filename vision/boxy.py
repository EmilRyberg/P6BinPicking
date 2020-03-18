import cv2
import numpy as np
import math


class Boxy:
    def __init__(self):
        self.mean = np.array([10, 130, 115], np.uint8)
        self.std = np.array([4, 30, 40], np.uint8)
        self.lower_thresh = self.mean - self.std
        self.upper_thresh = self.mean + self.std

    def find_box(self, image_name, debug=False):
        img = cv2.imread(image_name, cv2.IMREAD_COLOR)
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
        #img_morph = cv2.dilate(img_morph, kernel2)
        img_morph = cv2.morphologyEx(img_morph, cv2.MORPH_CLOSE, kernel)
        #img_morph = cv2.morphologyEx(img_morph, cv2.MORPH_OPEN, kernel)
        img_morph = cv2.morphologyEx(img_morph, cv2.MORPH_CLOSE, kernel2)
        #img_morph = cv2.morphologyEx(img_morph, cv2.MORPH_OPEN, kernel2)
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

        rect = cv2.minAreaRect(approx_rect_contour)
        rect_points = cv2.boxPoints(rect)
        final_rect = np.array(approx_rect_contour).reshape((4, -1))

        if debug:
            print("rect", rect_points)
            print("conts", approx_rect_contour)
            print("final rect", final_rect)
            cv2.imshow("thresholded", img_thresholded)
            cv2.imshow("morph", img_morph)
            cv2.imshow("flood", img_floodfill)
            cv2.imshow("bin final", img_binary_final)
            #cv2.imshow("cunt", img_contours)
            img_cont = img.copy()
            cv2.drawContours(img_cont, contours, -1, (0, 255, 0), 2)
            cv2.imshow("contours", img_cont)
            img_cont2 = img.copy()
            cv2.drawContours(img_cont2, [approx_rect_contour], -1, (0, 0, 255), 2)
            cv2.imshow("contours2", img_cont2)

            img_best_rect = img.copy()
            cv2.line(img_best_rect, (rect_points[0][0], rect_points[0][1]), (rect_points[1][0], rect_points[1][1]), (0, 0, 255), 2)
            cv2.line(img_best_rect, (rect_points[1][0], rect_points[1][1]), (rect_points[2][0], rect_points[2][1]), (0, 0, 255), 2)
            cv2.line(img_best_rect, (rect_points[2][0], rect_points[2][1]), (rect_points[3][0], rect_points[3][1]), (0, 0, 255), 2)
            cv2.line(img_best_rect, (rect_points[3][0], rect_points[3][1]), (rect_points[0][0], rect_points[0][1]), (0, 0, 255), 2)

            cv2.imshow("best rect", img_best_rect)
            cv2.waitKey(0)

        return final_rect

    def get_average_pixel_value(self, image_name):
        img = cv2.imread(image_name, cv2.IMREAD_COLOR)
        roi_x, roi_y, roi_width, roi_height = cv2.selectROI("select", img)
        cv2.destroyWindow("select")
        #cv2.rectangle(img, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (255, 0, 0), thickness=2)
        #cv2.imshow("ROI", img)
        img_crop = img[roi_y:(roi_y + roi_height), roi_x:(roi_x + roi_width)]
        cv2.imshow("cropped", img_crop)
        img_crop_hsv = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)

        mean, std = cv2.meanStdDev(img_crop_hsv)
        print(f'mean: {mean}, std: {std}')

        cv2.waitKey(0)

    def group_points(self, points, max_distance):
        grouped_points = []
        points_left_to_check = points.copy()
        while len(points_left_to_check) > 0:
            if len(points_left_to_check) < 3:
                break
            pivot = points_left_to_check.pop()
            new_pool = []
            current_group = [[int(round(pivot[0])), int(round(pivot[1]))]]
            while len(points_left_to_check) > 0:
                point_to_check = points_left_to_check.pop()
                dist = math.sqrt((pivot[0] - point_to_check[0])**2 + (pivot[1] - point_to_check[1])**2)
                if dist <= max_distance:
                    current_group.append([int(round(point_to_check[0])), int(round(point_to_check[1]))])
                else:
                    new_pool.append(point_to_check)
            points_left_to_check = new_pool
            if len(current_group) >= 5:
                grouped_points.append(current_group)
        return grouped_points

    def score_rect(self, img_hsv, rect):
        score = 0
        for index, point in enumerate(rect):
            line_to_next_normalised = (0, 0)
            line_to_next = (0, 0)
            if index < len(rect)-1:
                line_to_next = (rect[index+1][0] - point[0], rect[index+1][1] - point[1])
            else:
                line_to_next = (rect[0][0] - point[0], rect[0][1] - point[1])
            norm = math.sqrt(line_to_next[0]**2 + line_to_next[1]**2)
            line_to_next_normalised = (int(line_to_next[0] / norm), int(line_to_next[1] / norm))

            samples_pr_line = 20
            for t in range(0, samples_pr_line):
                check_point = (point[0] + (t/float(samples_pr_line)) * line_to_next[0], point[1] + (t/float(samples_pr_line)) * line_to_next[1])
                check_point = (int(check_point[0]), int(check_point[1]))
                #print("check point: ", check_point)
                pixel = img_hsv[check_point[1], check_point[0]]
                pixel_score = 0
                #print("pixel: ", pixel)
                if int(self.lower_thresh[0]) < pixel[0] < int(self.upper_thresh[0])\
                        and int(self.lower_thresh[1]) < pixel[1] < int(self.upper_thresh[1])\
                        and int(self.lower_thresh[2]) < pixel[2] < int(self.upper_thresh[2]):
                    score += 3
                else:
                    score -= 1
        diagonal1 = (rect[2][0] - rect[0][0], rect[2][1] - rect[0][1])
        diagonal2 = (rect[3][0] - rect[1][0], rect[3][1] - rect[1][1])
        samples_pr_diagonal = 20
        for t in range(0, samples_pr_diagonal):
            check_point = (rect[0][0] + (t / float(samples_pr_diagonal)) * diagonal1[0],
                           rect[0][1] + (t / float(samples_pr_diagonal)) * diagonal1[1])
            check_point = (int(check_point[0]), int(check_point[1]))
            check_point2 = (rect[1][0] + (t / float(samples_pr_diagonal)) * diagonal2[0],
                           rect[1][0] + (t / float(samples_pr_diagonal)) * diagonal2[1])
            check_point2 = (int(check_point[0]), int(check_point[1]))
            # print("check point: ", check_point)
            pixel = img_hsv[check_point[1], check_point[0]]
            pixel2 = img_hsv[check_point2[1], check_point2[0]]
            if int(self.lower_thresh[0]) < pixel[0] < int(self.upper_thresh[0])\
                    and int(self.lower_thresh[1]) < pixel[1] < int(self.upper_thresh[1])\
                    and int(self.lower_thresh[2]) < pixel[2] < int(self.upper_thresh[2]):
                score += 1
            if int(self.lower_thresh[0]) < pixel2[0] < int(self.upper_thresh[0])\
                    and int(self.lower_thresh[1]) < pixel2[1] < int(self.upper_thresh[1])\
                    and int(self.lower_thresh[2]) < pixel2[2] < int(self.upper_thresh[2]):
                score += 1
        return score


if __name__ == "__main__":
    boxy = Boxy()
    boxy.find_box("color1582019554.5333703-0.png", debug=True)
    #boxy.get_average_pixel_value("color1582019554.5333703-0.png")