from vision.vision import Vision
from controller.enums import PartEnum, PartCategoryEnum
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

        print("[I] Controller running")

    def main_flow(self, colour_part_id, debug=False):
        self.capture_images()
        self.masks = self.vision.segment(self.reference_image)
        self.move_robot.move_to_home(speed=3)
        self.pick_and_place_part(PartCategoryEnum.WHITE_COVER.value, debug)

        self.capture_images()
        self.masks = self.vision.segment(self.reference_image)
        self.move_robot.move_to_home(speed=3)
        self.pick_and_place_part(PartCategoryEnum.PCB.value, debug)

        self.capture_images()
        self.masks = self.vision.segment(self.reference_image)
        self.move_robot.move_to_home(speed=3)
        self.pick_and_place_part(PartCategoryEnum.BLACK_COVER.value, debug)

        #self.pick_and_place_part("pcb")

    def pick_and_place_part(self, part, debug=False):
        success = False
        self.unsuccessful_grip_shake_counter = 0
        while success == False and self.unsuccessful_grip_shake_counter < 3:
            print("finding part: ", part)
            mask = self.find_best_mask_by_part(part, self.masks)
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
                    center, rotvec, normal_vector, relative_angle_to_z = self.part_position_details
                    print(f'gripping {mask["part"]} at {center} with suction')
                    self.move_robot.set_tcp(self.move_robot.suction_tcp)
                    self.move_robot.movel([0, -300, 300, 0, 3.14, 0], vel=0.8)
                    approach_center = center + 200*normal_vector
                    pose_approach = np.concatenate((approach_center, rotvec))
                    self.move_robot.movel(pose_approach)
                    pose_pick = np.concatenate((center, rotvec))
                    self.move_robot.movel(pose_pick)
                    self.move_robot.enable_suction()
                    self.move_robot.movel([center[0], center[1], 300, 0, 3.14, 0])
                    self.move_robot.movel([200, -200, 300, 0, 3.14, 0])
                    self.move_robot.movel([200, -200, 50, 0, 3.14, 0])
                    self.move_robot.disable_suction()
                    self.move_robot.movel([200, -200, 300, 0, 3.14, 0])
                    print("success")
                    success = True
                else:
                    center, rotvec, normal_vector, relative_angle_to_z, short_vector = self.surface_normals.get_gripper_orientation(np_mask, self.depth_image, self.reference_image, 0)
                    print(f'gripping {mask["part"]} at {center} with gripper')
                    self.move_robot.set_tcp(self.move_robot.gripper_tcp)
                    self.move_robot.movel([0, -300, 300, 0, np.pi, 0], vel=0.8)
                    approach_center = center + 200*normal_vector
                    pose_approach = np.concatenate((approach_center, rotvec))
                    self.move_robot.movel(pose_approach)
                    pose_pick = np.concatenate((center - 25*normal_vector, rotvec))
                    self.move_robot.close_gripper(50)
                    self.move_robot.movel(pose_pick)
                    self.move_robot.close_gripper()
                    self.move_robot.movel([center[0], center[1], 300, 0, np.pi, 0])
                    self.move_robot.movel([200, -200, 300, 0, np.pi, 0])
                    self.move_robot.movel([200, -200, 50, 0, np.pi, 0])
                    self.move_robot.open_gripper()
                    self.move_robot.movel([200, -200, 300, 0, np.pi, 0])
                    print("success")
                    success = True
                """
                if random.randint(0, 10) > 9:
                    success = True
                    print("success")
                else:
                    print("failed")
                    mask["ignored"] = True"""
        if self.unsuccessful_grip_shake_counter >= 3:
            input(f"manually pick {part} and press enter")

    def find_best_mask_by_part(self, part, masks):
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
            if box_mask[y,x] == 0:
                mask["ignored"] = True
                mask["ignore_reason"] += "outside box, "

            if mask["part"] != PartCategoryEnum.PCB.value:
                center, rotvec, normal_vector, relative_angle_to_z, short_vector_2d = self.surface_normals.get_gripper_orientation(
                    mask["mask"], self.depth_image, self.reference_image, 0)
                print('mask center: ', mask["center"])
                mask_center = np.asarray(mask["center"], dtype=np.int32)
                print('mask shape: ', mask["mask"].shape)
                pil_depth = pimg.fromarray(self.depth_image)
                pil_depth = pil_depth.resize((1920, 1080))
                np_rescaled_depth_image = np.asarray(pil_depth)
                points_checking = []
                print('short vector 2d:', short_vector_2d)
                for i in range(30, 200):
                    point_to_check = np.array(np.round(mask_center + short_vector_2d * i), dtype=np.int32)
                    if mask["mask"][point_to_check[1], point_to_check[0]] == 0:
                        print('found zero point', point_to_check)
                        point_on_mask = np.array(np.round(point_to_check - short_vector_2d * 10), dtype=np.int32)
                        point_on_mask_z = self.surface_normals.get_z(point_on_mask[0], point_on_mask[1],
                                                                     np_rescaled_depth_image)
                        offset_point = np.array(np.round(point_to_check + short_vector_2d * 5), dtype=np.int32)
                        offset_point_z = self.surface_normals.get_z(offset_point[0], offset_point[1],
                                                                    np_rescaled_depth_image)
                        z_difference = point_on_mask_z - offset_point_z
                        points_checking.append((point_on_mask, offset_point))
                        if z_difference <= 10:
                            print(
                                f'height difference is less than 10 mm. diff: {z_difference}, on mask: {point_on_mask_z}, offset: {offset_point_z}')
                            mask["ignored"] = True
                            mask["ignore_reason"] += "not enough space for fingers, "

                        break
                if not mask["ignored"]:  # only check if first side is clear
                    for i in range(30, 200):
                        point_to_check = np.array(np.round(mask_center - short_vector_2d * i), dtype=np.int32)
                        if mask["mask"][point_to_check[1], point_to_check[0]] == 0:
                            print('found zero point', point_to_check)
                            point_on_mask = np.array(np.round(point_to_check + short_vector_2d * 10), dtype=np.int32)
                            point_on_mask_z = self.surface_normals.get_z(point_on_mask[0], point_on_mask[1],
                                                                         np_rescaled_depth_image)
                            offset_point = np.array(np.round(point_to_check - short_vector_2d * 5), dtype= np.int32)
                            offset_point_z = self.surface_normals.get_z(offset_point[0], offset_point[1],
                                                                        np_rescaled_depth_image)
                            z_difference = point_on_mask_z - offset_point_z
                            points_checking.append((point_on_mask, offset_point))
                            if z_difference <= 10:
                                print(
                                    f'height difference is less than 10 mm. diff: {z_difference}, on mask: {point_on_mask_z}, offset: {offset_point_z}')
                                mask["ignored"] = True
                                mask["ignore_reason"] += "not enough space for fingers, "

                            break
                if len(points_checking) > 0:
                    rgb_img = cv2.cvtColor(mask["mask"], cv2.COLOR_GRAY2BGR)
                    for point_on_m, point_outside_m in points_checking:
                        cv2.circle(rgb_img, tuple(point_on_m), 2, (255, 0, 0), -1)
                        cv2.circle(rgb_img, tuple(point_outside_m), 2, (0, 0, 255), -1)
                    cv2.imshow('points being checked', rgb_img)
                    cv2.waitKey(0)

        #next, find largest mask of matching part type
        highest_index = -1
        highest_area = -1
        for index, mask in enumerate(masks):
            if mask["part"] == part and mask["area"] > highest_area and mask["ignored"] == False:
                self.part_position_details = self.surface_normals.vector_normal(mask["mask"], self.depth_image, self.reference_image, rotation_around_self_z=-1.57)
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
        self.move_robot.move_to_home()
        rot = Rotation.from_euler("XYZ", [0, 3.14, 2.35-angle])
        rot = rot.as_rotvec()
        self.move_robot.movel([grasp_location[0], grasp_location[1], grasp_location[2]+20, rot[0], rot[1], rot[2]], vel=0.5)
        self.move_robot.movel([grasp_location[0], grasp_location[1], grasp_location[2]-80, rot[0], rot[1], rot[2]])
        self.move_robot.grasp_box()
        if self.first_box_move == 0:
            self.move_robot.movel([grasp_location[0], grasp_location[1], grasp_location[2] - 60, rot[0], rot[1], rot[2]])
            self.move_robot.movel([grasp_location[0]+80, grasp_location[1]+80, grasp_location[2]-60, rot[0], rot[1], rot[2]], vel=0.4)
            self.move_robot.movel([grasp_location[0], grasp_location[1], grasp_location[2]-60, rot[0], rot[1], rot[2]], vel=0.4)
            self.move_robot.movel([grasp_location[0], grasp_location[1], grasp_location[2] - 70, rot[0], rot[1], rot[2]])
            self.first_box_move = 1
        else:
            self.move_robot.movel([grasp_location[0]-15, grasp_location[1]-15, grasp_location[2]-80, rot[0], rot[1], rot[2]])
            self.first_box_move = 0
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


if __name__ == "__main__":
    from simulation_camera_interface import SimulationCamera
    from simulation_connector import SimulationConnector
    connector = SimulationConnector(2000)
    camera = SimulationCamera(connector)
    controller = Controller(connector, camera)
    controller.main_flow(1)