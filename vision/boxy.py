import cv2
import numpy as np
import math


class Boxy:
    def __init__(self):
        self.mean = np.array([10, 130, 115], np.uint8)
        self.std = np.array([2, 20, 15], np.uint8)
        self.lower_thresh = self.mean - self.std
        self.upper_thresh = self.mean + self.std

    def find_box(self, image_name, debug=False, save_video=False):
        img = cv2.imread(image_name, cv2.IMREAD_COLOR)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_thresh = (int(self.lower_thresh[0]), int(self.lower_thresh[1]), int(self.lower_thresh[2]))
        upper_thresh = (int(self.upper_thresh[0]), int(self.upper_thresh[1]), int(self.upper_thresh[2]))
        img_thresholded = cv2.inRange(img_hsv, lower_thresh, upper_thresh)
        if debug and not save_video:
            cv2.imshow("thresholded", img_thresholded)
        dilate_kernel = np.ones((3, 3), np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        kernel2 = np.ones((5, 5), np.uint8)
        img_morph = cv2.dilate(img_thresholded, dilate_kernel)
        img_morph = cv2.morphologyEx(img_morph, cv2.MORPH_OPEN, kernel)
        img_morph = cv2.morphologyEx(img_morph, cv2.MORPH_CLOSE, kernel)
        img_morph = cv2.morphologyEx(img_morph, cv2.MORPH_CLOSE, kernel2)
        img_morph = cv2.morphologyEx(img_morph, cv2.MORPH_OPEN, kernel2)

        if debug and not save_video:
            cv2.imshow("morph", img_morph)

        lines = cv2.HoughLines(img_morph, 1, np.pi/180, 320)
        img_lines = None
        if debug:
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
                cv2.line(img_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)

            if not save_video:
                cv2.imshow("lines", img_lines)

        #find intersections
        intersections = []
        for index, line in enumerate(lines):
            rho, theta = line[0]
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

        intersections = [inter for inter in intersections if inter[0] >= 0 and inter[1] >= 0 and inter[0] < img.shape[1] and inter[1] < img.shape[0]]

        img_intersections = None
        if debug:
            img_intersections = img.copy()
            for intersection in intersections:
                x = int(round(intersection[0]))
                y = int(round(intersection[1]))
                cv2.circle(img_intersections, (x, y), 2, (255, 0, 0), -1)

        grouped_intersections = self.group_points(intersections, 150)

        print(f"groups amount: {len(grouped_intersections)}")

        for group in grouped_intersections:
            center = np.mean(np.array(group), axis=0)
            center = [int(center[0]), int(center[1])]

            max_dist = 0
            for index, point in enumerate(group):
                dist = math.sqrt((point[0] - center[0])**2 + (point[1] - center[1])**2)
                if dist > max_dist:
                    max_dist = dist
            if debug:
                cv2.circle(img_intersections, (center[0], center[1]), int(max_dist) + 10, (0, 255, 0), 1)

        if debug and not save_video:
            cv2.imshow("points", img_intersections)

        anchor_points = []
        for group in grouped_intersections:
            local_groups = self.group_points(group, 15)
            for local_group in local_groups:
                group_center = np.mean(np.array(local_group), axis=0)
                if group_center[0] < 0 or group_center[0] > img.shape[1] or group_center[1] < 0 or group_center[1] > img.shape[0]:
                    continue
                anchor_points.append((int(group_center[0]), int(group_center[1])))
        img_anchor = None
        if debug:
            img_anchor = img.copy()
            for point in anchor_points:
                cv2.circle(img_anchor, point, 2, (255, 0, 0), -1)

            if not save_video:
                cv2.imshow("anchor points", img_anchor)

        area_threshold = 200000
        rectangles = []
        max_area = 0
        box_images = []
        img_with_boxes = img_anchor.copy()
        total_index = 0
        saved_boxes = 0
        print(f"anchor points: {len(anchor_points)}")
        for index, point in enumerate(anchor_points):
            other_points = [p for i, p in enumerate(anchor_points) if i != index]
            checked_lines = []
            for index2, point2 in enumerate(other_points):
                line_vector = [point2[0] - point[0], point2[1] - point[1]]
                last_points = [p for i, p in enumerate(other_points) if i != index2]
                for index3, point3 in enumerate(last_points):
                    total_index += 1
                    other_line_vector = [point3[0] - point2[0], point3[1] - point2[1]]
                    #is_saving_img = False
                    if debug and save_video and index2 % 10 == 0 and index3 % 10 == 0:
                        #is_saving_img = True
                        #print(f"Saving image for point {total_index}/{len(anchor_points)*len(other_points)*len(last_points)}")
                        #cur_img = img_with_boxes.copy()
                        checked_lines.append([(point[0], point[1]), (point2[0], point2[1]), (point2[0], point2[1]), (point3[0], point3[1])])
                        #cv2.line(cur_img, (point[0], point[1]), (point2[0], point2[1]), (0, 0, 0), 1)
                        #cv2.line(cur_img, (point2[0], point2[1]), (point3[0], point3[1]), (0, 0, 0), 1)
                        #box_images.append(cur_img)
                    dot = line_vector[0] * other_line_vector[0] + line_vector[1] * other_line_vector[1]
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
                            if debug and save_video and saved_boxes % 10 == 0:
                                img_with_box = img_with_boxes.copy()
                                for i in range(0, len(checked_lines), 4):
                                    checked_line = checked_lines[i]
                                    cur_img = img_with_boxes.copy()
                                    cv2.line(cur_img, checked_line[0], checked_line[1], (0, 0, 0), 1)
                                    cv2.line(cur_img, checked_line[2], checked_line[3], (0, 0, 0), 1)
                                    box_images.append(cur_img)
                                cv2.line(img_with_box, (point[0], point[1]), (point2[0], point2[1]), (0, 0, 255), 2)
                                cv2.line(img_with_box, (point2[0], point2[1]), (point3[0], point3[1]), (0, 0, 255), 2)
                                cv2.line(img_with_box, (point3[0], point3[1]), (x4, y4), (0, 0, 255), 2)
                                cv2.line(img_with_box, (x4, y4), (point[0], point[1]), (0, 0, 255), 2)
                                cv2.line(img_with_boxes, (point[0], point[1]), (point2[0], point2[1]), (0, 0, 255), 2)
                                cv2.line(img_with_boxes, (point2[0], point2[1]), (point3[0], point3[1]), (0, 0, 255), 2)
                                cv2.line(img_with_boxes, (point3[0], point3[1]), (x4, y4), (0, 0, 255), 2)
                                cv2.line(img_with_boxes, (x4, y4), (point[0], point[1]), (0, 0, 255), 2)
                                box_images.append(img_with_box)
                            saved_boxes += 1
        print("max area", max_area)
        best_rectangles = []
        best_rectangle = None
        max_score = -10000
        for rect in rectangles:
            r1 = (rect[1][0] - rect[0][0], rect[1][1] - rect[0][1])
            r2 = (rect[2][0] - rect[1][0], rect[2][1] - rect[1][1])
            length1 = math.sqrt(r1[0] ** 2 + r1[1] ** 2)
            length2 = math.sqrt(r2[0] ** 2 + r2[1] ** 2)
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

        image_rects = []
        if debug and not save_video:
            cv2.namedWindow("rect_image")
        def track_bar_change(val):
            cv2.imshow("rect_image", image_rects[val])

        img_best_boxes = img.copy()
        if debug:
            for index, rect in enumerate(best_rectangles):
                img_rect = img.copy() if not save_video else img_best_boxes
                cv2.line(img_rect, rect[0], rect[1], (255, 0, 0))
                cv2.line(img_rect, rect[1], rect[2], (255, 0, 0))
                cv2.line(img_rect, rect[2], rect[3], (255, 0, 0))
                cv2.line(img_rect, rect[3], rect[0], (255, 0, 0))

                if not save_video:
                    print("SCORE: " + str(self.score_rect(img_hsv, rect)))
                    image_rects.append(img_rect)
                    cv2.createTrackbar("rectimg", "rect_image", 0, len(image_rects) - 1, track_bar_change)
                    cv2.imshow("rect_image", image_rects[0])

        if debug:
            img_best_rect = img.copy()
            cv2.line(img_best_rect, best_rectangle[0], best_rectangle[1], (0, 0, 255), 2)
            cv2.line(img_best_rect, best_rectangle[1], best_rectangle[2], (0, 0, 255), 2)
            cv2.line(img_best_rect, best_rectangle[2], best_rectangle[3], (0, 0, 255), 2)
            cv2.line(img_best_rect, best_rectangle[3], best_rectangle[0], (0, 0, 255), 2)

            if not save_video:
                cv2.imshow("best rect", img_best_rect)
                cv2.waitKey(0)
            else:
                writer = cv2.VideoWriter('test.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, (img.shape[1], img.shape[0]))
                for i in range(30):
                    writer.write(img)
                img_thresholded_rgb = cv2.cvtColor(img_thresholded, cv2.COLOR_GRAY2BGR)
                for i in range(20):
                    writer.write(img_thresholded_rgb)
                img_morph_rgb = cv2.cvtColor(img_morph, cv2.COLOR_GRAY2BGR)
                for i in range(20):
                    writer.write(img_morph_rgb)
                for i in range(20):
                    writer.write(img_lines)
                for i in range(20):
                    writer.write(img_intersections)
                for i in range(20):
                    writer.write(img_anchor)
                for box_img in box_images:
                    writer.write(box_img)
                for i in range(45):
                    writer.write(img_best_boxes)
                for i in range(75):
                    writer.write(img_best_rect)
                writer.release()

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
    boxy.find_box("color1582019554.5333703-0.png", debug=True, save_video=True)
    #boxy.get_average_pixel_value("color1582019554.5333703-0.png")