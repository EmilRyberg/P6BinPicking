import cv2
import numpy as np
import math


class Boxy:
    def __init__(self):
        self.mean = np.array([10, 130, 115], np.uint8)
        self.std = np.array([2, 20, 15], np.uint8)
        self.lower_thresh = self.mean - self.std
        self.upper_thresh = self.mean + self.std

    def find_box(self, image_name):
        img = cv2.imread(image_name, cv2.IMREAD_COLOR)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_thresh = (int(self.lower_thresh[0]), int(self.lower_thresh[1]), int(self.lower_thresh[2]))
        upper_thresh = (int(self.upper_thresh[0]), int(self.upper_thresh[1]), int(self.upper_thresh[2]))
        #print(lower_thresh)
        #print(upper_thresh)
        #13.46, 127.5, 127.5
        #
        img_thresholded = cv2.inRange(img_hsv, lower_thresh, upper_thresh)
       # img_thresholded = cv2.cvtColor(img_thresholded_hsv, cv2.COLOR_HSV2BGR)
        cv2.imshow("thresholded", img_thresholded)
        dilate_kernel = np.ones((3, 3), np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        kernel2 = np.ones((5, 5), np.uint8)
        img_morph = cv2.dilate(img_thresholded, dilate_kernel)
        img_morph = cv2.morphologyEx(img_morph, cv2.MORPH_OPEN, kernel)
        img_morph = cv2.morphologyEx(img_morph, cv2.MORPH_CLOSE, kernel)
        img_morph = cv2.morphologyEx(img_morph, cv2.MORPH_CLOSE, kernel2)
        img_morph = cv2.morphologyEx(img_morph, cv2.MORPH_OPEN, kernel2)
        cv2.imshow("morph", img_morph)

        lines = cv2.HoughLines(img_morph, 1, np.pi/180, 320)
        #print(lines)
        lines_xy = []
        img_lines = img.copy()
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 2000 * (-b))
            y1 = int(y0 + 2000 * a)
            x2 = int(x0 - 2000 * (-b))
            y2 = int(y0 - 2000 * a)
            lines_xy.append([x1, y1, x2, y2])

            cv2.line(img_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imshow("lines", img_lines)

        #find intersections
        intersections = []
        for index, line in enumerate(lines):
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            #rho = x cos(theta) + y sin(theta)
            other_lines = [x for i, x in enumerate(lines) if i != index]
            for other_line in other_lines:
                rho_other, theta_other = other_line[0]
                A = np.array([[np.cos(theta), np.sin(theta)], [np.cos(theta_other), np.sin(theta_other)]])
                b = np.array([rho, rho_other])
                try:
                    x = np.linalg.solve(A, b)
                    if np.allclose(np.dot(A, x), b):
                        intersections.append(x)
                except np.linalg.LinAlgError:
                    continue
                    #print("WARNING: LinAlgError")


        #print(f"intersections: {len(intersections)}")
        intersections = [inter for inter in intersections if inter[0] >= 0 and inter[1] >= 0 and inter[0] < img.shape[1] and inter[1] < img.shape[0]]
        #print(f"intersections after: {len(intersections)}")

        img_intersections = img.copy()
        for intersection in intersections:
            x = int(round(intersection[0]))
            y = int(round(intersection[1]))
            cv2.circle(img_intersections, (x, y), 2, (255, 0, 0), -1)

        grouped_intersections = self.group_points(intersections, 150)

        print(f"groups: {len(grouped_intersections)}")

        for group in grouped_intersections:
            center = np.mean(np.array(group), axis=0)
            center = [int(center[0]), int(center[1])]

            max_dist = 0
            for index, point in enumerate(group):
                dist = math.sqrt((point[0] - center[0])**2 + (point[1] - center[1])**2)
                if dist > max_dist:
                    max_dist = dist
            cv2.circle(img_intersections, (center[0], center[1]), int(max_dist) + 10, (0, 255, 0), 1)

        cv2.imshow("points", img_intersections)

        anchor_points = []
        for group in grouped_intersections:
            local_groups = self.group_points(group, 15)
            for local_group in local_groups:
                group_center = np.mean(np.array(local_group), axis=0)
                if group_center[0] < 0 or group_center[0] > img.shape[1] or group_center[1] < 0 or group_center[1] > img.shape[0]:
                    continue
                anchor_points.append((int(group_center[0]), int(group_center[1])))

        anchor_img = img.copy()
        for point in anchor_points:
            cv2.circle(anchor_img, point, 2, (255, 0, 0), -1)
        cv2.imshow("anchor points", anchor_img)
        area_threshold = 200000
        rectangles = []
        print("starting rects")
        start_index = 0
        #intersections_int = [[100, 100], [500, 100], [500, 500], [123, 123], [123, 123]]
        max_area = 0
        for index, point in enumerate(anchor_points):#[0:int(round(len(intersections)/2))]
            print("outer loop index:", index)
            other_points = [p for i, p in enumerate(anchor_points) if i != index]
            for index2, point2 in enumerate(other_points):
                #print(f"inner loop 1 index {group_index2} out of {len(other_groups)}")
                line_vector = [point2[0] - point[0], point2[1] - point[1]]
                last_points = [p for i, p in enumerate(other_points) if i != index2]
                for index3, point3 in enumerate(last_points):
                    other_line_vector = [point3[0] - point2[0], point3[1] - point2[1]]
                    #print(f"line vectors: {line_vector} - {other_line_vector}")
                    dot = line_vector[0] * other_line_vector[0] + line_vector[1] * other_line_vector[1]
                    #print("dot", dot)
                    if -500 < dot < 500:
                        r = (point[0] - point2[0], point[1] - point2[1])
                        x4 = point3[0] + r[0]
                        y4 = point3[1] + r[1]
                        if x4 < 0 or x4 >= img.shape[1] or y4 < 0 or y4 >= img.shape[0]:
                            continue
                        length1 = math.sqrt(line_vector[0]**2 + line_vector[1]**2)
                        length2 = math.sqrt(other_line_vector[0]**2 + other_line_vector[1]**2)
                        area = length1 * length2
                        if area > max_area:
                            max_area = area
                        if area >= area_threshold:
                            rectangles.append([(point[0], point[1]), (point2[0], point2[1]), (point3[0], point3[1]), (x4, y4)])
        print("max area", max_area)
        best_rectangles = []
        best_rectangle = None
        biggest_area = 0
        max_score = -10000
        for rect in rectangles:
            r1 = (rect[1][0] - rect[0][0], rect[1][1] - rect[0][1])
            r2 = (rect[2][0] - rect[1][0], rect[2][1] - rect[1][1])
            length1 = math.sqrt(r1[0] ** 2 + r1[1] ** 2)
            length2 = math.sqrt(r2[0] ** 2 + r2[1] ** 2)
            aspect_ratio = 0
            if length1 > length2:
                aspect_ratio = length1 / length2
            else:
                aspect_ratio = length2 / length1
            if 1.15 < aspect_ratio < 1.21:
                best_rectangles.append(rect)
                score = self.score_rect(img_hsv, rect)
                if score > max_score:
                    max_score = score
                    best_rectangle = rect
        print(f"best rectangles count: {len(best_rectangles)}")
        #img_rects = img.copy()
        image_rects = []
        cv2.namedWindow("rect_image")
        for index, rect in enumerate(best_rectangles):
            print(f"index: {index} - rect: {rect}")
            img_rect = img.copy()
            cv2.line(img_rect, rect[0], rect[1], (255, 0, 0))
            cv2.line(img_rect, rect[1], rect[2], (255, 0, 0))
            cv2.line(img_rect, rect[2], rect[3], (255, 0, 0))
            cv2.line(img_rect, rect[3], rect[0], (255, 0, 0))
            print("SCORE: " + str(self.score_rect(img_hsv, rect)))
            image_rects.append(img_rect)
            #break

        def trackBarChange(val):
            cv2.imshow("rect_image", image_rects[val])
        cv2.createTrackbar("rectimg", "rect_image", 0, len(image_rects)-1, trackBarChange)
        cv2.imshow("rect_image", image_rects[0])
        img_best_rect = img.copy()
        cv2.line(img_best_rect, best_rectangle[0], best_rectangle[1], (0, 0, 255), 2)
        cv2.line(img_best_rect, best_rectangle[1], best_rectangle[2], (0, 0, 255), 2)
        cv2.line(img_best_rect, best_rectangle[2], best_rectangle[3], (0, 0, 255), 2)
        cv2.line(img_best_rect, best_rectangle[3], best_rectangle[0], (0, 0, 255), 2)
        cv2.imshow("best rect", img_best_rect)

        cv2.waitKey(0)
        return best_rectangle


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
                if int(self.lower_thresh[0]) < pixel[0] < int(self.upper_thresh[0]):
                    pixel_score += 1
                if int(self.lower_thresh[1]) < pixel[1] < int(self.upper_thresh[1]):
                    pixel_score += 1
                if int(self.lower_thresh[2]) < pixel[2] < int(self.upper_thresh[2]):
                    pixel_score += 1
                if pixel_score == 3:
                    #pixel_score *= 10
                    score += 2
                elif pixel_score <= 1:
                    score -= 1
                elif pixel_score == 0:
                    score -= 2
        return score

if __name__ == "__main__":
    boxy = Boxy()
    boxy.find_box("color1582020110.6481073-0.png")
    #boxy.get_average_pixel_value("color1582019554.5333703-0.png")