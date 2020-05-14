from vision.vision import Vision
from enums import PartEnum, PartCategoryEnum
from vision.surface_normal import SurfaceNormals
from PIL import Image as pimg
import cv2
import numpy as np
import random
from camera_interface import CameraInterface
from vision.box_detector import BoxDetector
from scipy.spatial.transform import Rotation


class Controller:
    def __init__(self, move_robot, camera_interface: CameraInterface, segmentation_weight_path):
        self.move_robot = move_robot
        self.camera = camera_interface
        self.vision = Vision(segmentation_weight_path)
        self.box_detector = BoxDetector()
        self.surface_normals = SurfaceNormals()
        self.detected_objects = None
        self.masks = None
        self.unsuccessful_grip_counter = 0
        self.unsuccessful_grip_shake_counter = 0
        self.area_threshold = 2000000
        self.score_threshold = 0.6
        self.reference_image = None
        self.depth_image = None
        self.part_position_details = None
        self.first_box_move = 0
        self.picked_parts = {PartCategoryEnum.BOTTOM_COVER.value: 0, PartCategoryEnum.PCB.value: 0, PartCategoryEnum.BLACK_COVER.value: 0, PartCategoryEnum.BLUE_COVER.value: 0, PartCategoryEnum.WHITE_COVER.value: 0}

        print("[I] Controller running")

    def main_flow(self, debug=False):
        at_least_one_of_each_part_picked = False
        while not at_least_one_of_each_part_picked:
            self.capture_images()
            self.masks = self.vision.segment(self.reference_image)
            self.move_robot.move_to_home_suction(speed=3)
            picked_part = self.pick_and_place_part(part="any", debug=debug)
            self.picked_parts[picked_part] += 1
            at_least_one_of_each_part_picked = True
            for key, value in self.picked_parts.items():
                if value == 0:
                    at_least_one_of_each_part_picked = False

        print("Successfully picked 5 different parts")

        #self.pick_and_place_part("pcb")

    def pick_and_place_part(self, part, debug=False):
        success = False
        self.unsuccessful_grip_shake_counter = 0
        mask = None
        while success == False and self.unsuccessful_grip_shake_counter < 3:
            print("finding part: ", part)
            mask = self.find_best_mask_by_part(part, self.masks, debug)
            if mask == False:
                print("no appropriate part found")
                print(f"shaking, try {self.unsuccessful_grip_shake_counter+1}/3")
                self.move_box()
                self.move_robot.move_out_of_view()
                self.capture_images()
                self.masks = self.vision.segment(self.reference_image)
                self.unsuccessful_grip_shake_counter += 1
            else:
                np_mask = np.asarray(mask["mask"])
                if debug:
                    applied_mask = cv2.bitwise_and(self.reference_image, self.reference_image, mask=np_mask)
                    cv2.imshow("picking", cv2.resize(applied_mask, (1280, 720)))
                    cv2.waitKey()

                if part == PartCategoryEnum.PCB.value:
                    center, part_orientation, normal_vector, relative_angle_to_z = self.part_position_details
                    print(f'gripping {mask["part"]} at {center} with suction')
                    self.move_robot.set_tcp(self.move_robot.suction_tcp)
                    default_orientation = Rotation.from_euler("xyz", [0, 3.14, 0]).as_rotvec() # rotz=0 is down left, counterclockwise positive
                    self.move_robot.movej([-60, -60, -110, -190, -70, 100], vel=2) #down right above box
                    approach_center = center + 200*normal_vector
                    self.move_robot.movel2(approach_center, part_orientation) #pre pick above box
                    self.move_robot.movel2(center, part_orientation, vel=0.2) #pick
                    self.move_robot.enable_suction()
                    lift_orientation = Rotation.from_euler("xyz", [0, 3.14, 1.57]).as_rotvec()
                    self.move_robot.movel2([center[0], center[1], 300], lift_orientation, vel=0.2) #lift straight up
                    self.move_robot.movej([-60, -60, -110, -190, 20, 100], vel=2) #down left above box
                    self.move_robot.movel2([200, -200, 300], default_orientation, vel=1) #north of box, above
                    self.move_robot.movel2([200, -200, 50], default_orientation, vel=0.4) #move straight down
                    self.move_robot.disable_suction()
                    self.move_robot.movel2([200, -200, 300], default_orientation, vel=1) #move straight up
                    print("success")
                    success = True
                else:
                    center, rotvec, normal_vector, relative_angle_to_z, short_vector = self.surface_normals.get_gripper_orientation(np_mask, self.depth_image, self.reference_image, 0)
                    print(f'gripping {mask["part"]} at {center} with gripper')
                    self.move_robot.set_tcp(self.move_robot.gripper_tcp)
                    self.move_robot.move_to_home_gripper(speed=2)
                    self.move_robot.movel([0, -300, 300, 0, np.pi, 0], vel=0.8)
                    approach_center = center + 200*normal_vector
                    pose_approach = np.concatenate((approach_center, rotvec))
                    self.move_robot.movel(pose_approach)
                    pose_pick = np.concatenate((center - 5*normal_vector, rotvec))
                    self.move_robot.close_gripper(50)
                    self.move_robot.movel(pose_pick)
                    self.move_robot.close_gripper(15, speed=1)
                    self.move_robot.movel([center[0], center[1], 300, 0, np.pi, 0])
                    self.move_robot.movel([200, -200, 300, 0, np.pi, 0])
                    self.move_robot.movel([200, -200, 50, 0, np.pi, 0])
                    self.move_robot.open_gripper()
                    self.move_robot.movel([200, -200, 300, 0, np.pi, 0])
                    print("success")
                    success = True
        picked_part = None
        if self.unsuccessful_grip_shake_counter >= 3:
            if part == "any":
                part = input("manually pick any part and enter the picked part")
            else:
                input(f"manually pick {part} and press enter")
            picked_part = part
        else:
            picked_part = mask["part"]
        return picked_part

    def find_best_mask_by_part(self, part, masks, debug=False):
        #first cull images by some basic criteria
        for index, mask in enumerate(masks):
            if mask["area"] < self.area_threshold:
                mask["ignored"] = True
                mask["ignore_reason"] += "too small, "
            if mask["score"] < self.score_threshold:
                mask["ignored"] = True
                mask["ignore_reason"] += "low confidence, "
            x, y = mask["center"]
            box_location, box_mask = self.box_detector.find_box(self.reference_image, get_mask=True)
            if box_mask[y, x] == 0:
                mask["ignored"] = True
                mask["ignore_reason"] += "outside box, "

            if mask["part"] != PartCategoryEnum.PCB.value and (mask["part"] == part or part == "any") and not mask["ignored"]:
                center, rotvec, normal_vector, relative_angle_to_z, short_vector_2d = self.surface_normals.get_gripper_orientation(
                    mask["mask"], self.depth_image, self.reference_image, 0)
                mask_center = np.asarray(mask["center"], dtype=np.int32)
                pil_depth = pimg.fromarray(self.depth_image)
                pil_depth = pil_depth.resize((1920, 1080))
                np_rescaled_depth_image = np.asarray(pil_depth)
                points_checking = []
                points_on_mask = []
                is_valid, mask_point, points_checked = self.check_for_valid_gripper_point(mask_center, mask, short_vector_2d, np_rescaled_depth_image)
                points_checking.extend(points_checked)
                points_on_mask.append(mask_point)

                if not mask["ignored"]:  # only check if first side is clear
                    is_valid, mask_point, points_checked = self.check_for_valid_gripper_point(mask_center, mask,
                                                                                              -short_vector_2d,
                                                                                              np_rescaled_depth_image)
                    points_checking.extend(points_checked)
                    points_on_mask.append(mask_point)

                if len(points_checking) > 0 and debug:
                    rgb_img = cv2.cvtColor(mask["mask"], cv2.COLOR_GRAY2BGR)
                    for point in points_checking:
                        cv2.circle(rgb_img, tuple(point), 2, (0, 0, 255), -1)
                    for point in points_on_mask:
                        cv2.circle(rgb_img, tuple(point), 2, (255, 0, 0), -1)
                    cv2.imshow('points being checked', rgb_img)
                    cv2.waitKey(0)

        #next, find largest mask of matching part type
        highest_index = -1
        highest_area = -1
        for index, mask in enumerate(masks):
            if (mask["part"] == part or part == "any") and mask["area"] > highest_area and mask["ignored"] == False:
                self.part_position_details = self.surface_normals.vector_normal(mask["mask"], self.depth_image, self.reference_image, rotation_around_self_z=0.78) #0.78=down right, clockwise positive
                if self.part_position_details[3] < 0.8: # check how flat it is (relative angle to reference z)
                    highest_index = index
                    highest_area = mask["area"]
                else:
                    mask["ignored"] = True
                    mask["ignore_reason"] += "not flat enough, "
        if highest_index == -1: #if none are acceptable
            return False
        else:
            return masks[highest_index]

    def capture_images(self):
        self.move_robot.move_out_of_view(speed=3)
        self.reference_image = self.camera.get_image()
        self.depth_image = self.camera.get_depth()

    def move_box(self):
        grasp_location, angle = self.box_detector.box_grasp_location(self.reference_image)
        self.move_robot.set_tcp(self.move_robot.gripper_tcp)
        self.move_robot.move_to_home_gripper()
        rot = Rotation.from_euler("XYZ", [0, 3.14, 2.35-angle])
        rot = rot.as_rotvec()
        self.move_robot.movel([grasp_location[0], grasp_location[1], grasp_location[2]+20, rot[0], rot[1], rot[2]], vel=0.5)
        self.move_robot.movel([grasp_location[0], grasp_location[1], grasp_location[2]-80, rot[0], rot[1], rot[2]])
        self.move_robot.grasp_box()
        self.move_robot.movel([grasp_location[0], grasp_location[1], grasp_location[2] - 60, rot[0], rot[1], rot[2]])
        self.move_robot.movel([grasp_location[0]+80, grasp_location[1]+80, grasp_location[2]-60, rot[0], rot[1], rot[2]], vel=0.4)
        self.move_robot.movel([grasp_location[0], grasp_location[1], grasp_location[2]-60, rot[0], rot[1], rot[2]], vel=0.4)
        self.move_robot.movel([grasp_location[0], grasp_location[1], grasp_location[2] - 70, rot[0], rot[1], rot[2]])
        self.move_robot.open_gripper()
        self.move_robot.movel([grasp_location[0], grasp_location[1], grasp_location[2] +40, rot[0], rot[1], rot[2]])
        #print(grasp_location)

    def choose_action(self):
        print("Please write a command (write 'help' for a list of commands):")
        command = input()
        if command == "help":
            print(
                "Possible commands are: \nblack: assemble phone with black cover \nwhite: assemble phone with white cover \n"
                + "blue: assemble phone with blue cover \nzero: put the robot in zero position \nquit: close the program")
        elif command == "black":
            self.main_flow(PartEnum.BLACKCOVER.value)
        elif command == "white":
            self.main_flow(PartEnum.WHITECOVER.value)
        elif command == "blue":
            self.main_flow(PartEnum.BLUECOVER.value)
        elif command == "zero":
            self.move_robot.move_out_of_view()
        elif command == "quit":
            return True
        else:
            print("Invalid command, please try again")
        return False

    def has_object_between_fingers(self):
        return self.move_robot.get_gripper_distance() > 0.001

    def check_for_valid_gripper_point(self, mask_center, mask, direction_to_check, np_scaled_depth_image, depth_margin=8):
        points_checking = []
        point_on_mask = None
        is_valid_grasp = True
        for i in range(30, 200):
            point_to_check = np.array(np.round(mask_center + direction_to_check * i), dtype=np.int32)
            if mask["mask"][point_to_check[1], point_to_check[0]] == 0:
                point_on_mask = np.array(np.round(point_to_check - direction_to_check * 10), dtype=np.int32)
                point_on_mask_z = self.surface_normals.get_z(point_on_mask[0], point_on_mask[1],
                                                             np_scaled_depth_image)
                offset_point = np.array(np.round(point_to_check + direction_to_check * 8), dtype=np.int32)
                offset_point_z = self.surface_normals.get_z(offset_point[0], offset_point[1],
                                                            np_scaled_depth_image)
                rotated_direction_to_check = np.array([-direction_to_check[1], direction_to_check[0]])
                offset_point_2 = np.array(np.round(offset_point + rotated_direction_to_check * 5), dtype=np.int32)
                offset_point_2_z = self.surface_normals.get_z(offset_point_2[0], offset_point_2[1],
                                                              np_scaled_depth_image)
                offset_point_3 = np.array(np.round(offset_point - rotated_direction_to_check * 5), dtype=np.int32)
                offset_point_3_z = self.surface_normals.get_z(offset_point_3[0], offset_point_3[1],
                                                              np_scaled_depth_image)
                offset_point_4 = np.array(np.round(offset_point + direction_to_check * 4), dtype=np.int32)
                offset_point_4_z = self.surface_normals.get_z(offset_point_4[0], offset_point_4[1],
                                                            np_scaled_depth_image)
                offset_point_5 = np.array(np.round(offset_point_4 + rotated_direction_to_check * 5), dtype=np.int32)
                offset_point_5_z = self.surface_normals.get_z(offset_point_5[0], offset_point_5[1],
                                                              np_scaled_depth_image)
                offset_point_6 = np.array(np.round(offset_point_4 - rotated_direction_to_check * 5), dtype=np.int32)
                offset_point_6_z = self.surface_normals.get_z(offset_point_6[0], offset_point_6[1],
                                                              np_scaled_depth_image)

                z_points = [
                    offset_point_z,
                    offset_point_2_z,
                    offset_point_3_z,
                    offset_point_4_z,
                    offset_point_5_z,
                    offset_point_6_z,
                ]

                points_checking = [
                    offset_point,
                    offset_point_2,
                    offset_point_3,
                    offset_point_4,
                    offset_point_5,
                    offset_point_6
                ]

                for z_point in z_points:
                    diff = point_on_mask_z - z_point
                    if diff <= depth_margin:
                        print(
                            f'z difference is less than {depth_margin} mm for one or more points. diff: {diff}, on mask: {point_on_mask_z}, offset: {z_point}')
                        mask["ignored"] = True
                        mask["ignore_reason"] += "not enough space for fingers, "
                        is_valid_grasp = False
                        break

                break
        return is_valid_grasp, point_on_mask, points_checking

if __name__ == "__main__":
    from simulation_camera_interface import SimulationCamera
    from simulation_connector import SimulationConnector
    connector = SimulationConnector(2000)
    camera = SimulationCamera(connector)
    controller = Controller(connector, camera, "vision/model_final_sim.pth")
    controller.main_flow(debug=False)