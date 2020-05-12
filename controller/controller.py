from vision.vision import Vision
#from enums import PartEnum
from vision.surface_normal import SurfaceNormals
from PIL import Image as pimg
import cv2
import numpy as np
import random
from camera_interface import CameraInterface
from vision.box_detector import BoxDetector
from scipy.spatial.transform import Rotation


class Controller:
    def __init__(self, move_robot, camera_interface: CameraInterface):
        self.move_robot = move_robot
        self.camera = camera_interface
        self.vision = Vision()
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

    def main_flow(self, colour_part_id):
        self.capture_images()
        self.masks = self.vision.segment(self.reference_image)
        self.move_robot.move_to_home(speed=3)
        self.pick_and_place_part("WhiteCover")

        self.capture_images()
        self.masks = self.vision.segment(self.reference_image)
        self.move_robot.move_to_home(speed=3)
        self.pick_and_place_part("PCB")

        self.capture_images()
        self.masks = self.vision.segment(self.reference_image)
        self.move_robot.move_to_home(speed=3)
        self.pick_and_place_part("BlueCover")

        #self.pick_and_place_part("pcb")

    def pick_and_place_part(self, part):
        success = False
        self.unsuccessful_grip_shake_counter = 0
        while success == False and self.unsuccessful_grip_shake_counter <= 3:
            mask = self.find_best_mask_by_part(part, self.masks)
            if mask == False:
                print("no appropriate part found")
                print(f"shaking, try {self.unsuccessful_grip_shake_counter}/3")
                self.move_box()
                self.move_robot.move_out_of_view()
                self.capture_images()
                self.masks = self.vision.segment(self.reference_image)
                self.unsuccessful_grip_shake_counter +=1
            else:
                np_mask = np.asarray(mask["mask"])
                applied_mask = cv2.bitwise_and(self.reference_image, self.reference_image, mask=np_mask)
                cv2.imshow("picking", cv2.resize(applied_mask, (1280, 720)))
                cv2.waitKey()
                center, rotvec, normal_vector, relative_angle_to_z = self.part_position_details
                print(f'gripping {mask["part"]} at {center}')
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