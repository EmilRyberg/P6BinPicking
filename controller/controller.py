from move_robot.move_robot import MoveRobot
from vision.vision import Vision
from enums import PartEnum
from aruco import Calibration
from vision.surface_normal import SurfaceNormals
from PIL import Image as pimg
import cv2
import numpy as np
import random
from simulation_connector import SimulationConnector
from camera_interface import CameraInterface

NUMBER_OF_PARTS = 4
FIXTURE_X = 255
FIXTURE_Y = -320
UR_IP = "192.168.1.148"


class Controller:
    def __init__(self):
        #self.move_robot = MoveRobot(UR_IP)
        self.move_robot = SimulationConnector(2000)
        self.camera = CameraInterface(mode="simulation", simulation_connector_instance=self.move_robot)
        self.vision = Vision()
        self.surface_normals = SurfaceNormals()
        self.detected_objects = None
        self.masks = None
        self.unsuccessful_grip_counter = 0
        self.unsuccessful_grip_shake_counter = 0
        self.area_threshold = 2000000
        self.score_threshold = 0.6
        self.reference_image = None
        self.depth_image = None

        print("[I] Controller running")

    def main_flow(self, colour_part_id):
        self.capture_images()
        self.masks = self.vision.segment(self.reference_image)
        self.move_robot.move_to_home(speed=3)
        self.pick_and_place_part("BlueCover")

        self.capture_images()
        self.masks = self.vision.segment(self.reference_image)
        self.move_robot.move_to_home(speed=3)
        self.pick_and_place_part("PCB")

        self.capture_images()
        self.masks = self.vision.segment(self.reference_image)
        self.move_robot.move_to_home(speed=3)
        self.pick_and_place_part("WhiteCover")

        #self.pick_and_place_part("pcb")

    def pick_and_place_part(self, part):
        success = False
        self.unsuccessful_grip_shake_counter = 0
        while success == False and self.unsuccessful_grip_shake_counter <= 3:
            mask = self.find_best_mask_by_part(part, self.masks)
            if mask == False:
                print("no appropriate part found")
                #shake
                #detect
                print(f"shaking, try {self.unsuccessful_grip_shake_counter}/3")
                for mask in self.masks:
                    mask["ignored"] = False
                self.unsuccessful_grip_shake_counter +=1
            else:
                np_mask = np.asarray(mask["mask"])
                applied_mask = cv2.bitwise_and(self.reference_image, self.reference_image, mask=np_mask)
                cv2.imshow("picking", cv2.cvtColor(cv2.resize(applied_mask, (1280, 720)), cv2.COLOR_RGB2BGR))
                cv2.waitKey()
                pose, normal_vector = self.surface_normals.vector_normal(np_mask, self.depth_image, self.reference_image)
                print(f'gripping {mask["part"]} at {pose}')
                self.move_robot.set_tcp(self.move_robot.suction_tcp)
                self.move_robot.movel([pose[0], pose[1], 200, 0, 0, 0])
                self.move_robot.movel([pose[0], pose[1], pose[2]+5, 0, 0, 0])
                self.move_robot.enable_suction()
                self.move_robot.movel([pose[0], pose[1], 200, 0, 0, 0])
                self.move_robot.movel([-350, -300, 200, 0, 0, 0])
                self.move_robot.disable_suction()
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
            #TODO replace with box detector
            if x<700 or x>1210 or y<310 or y>830:
                mask["ignored"] = True
                mask["ignore_reason"] += "outside box, "

        #next, find largest mask of matching part type
        highest_index = -1
        highest_area = -1
        for index, mask in enumerate(masks):
            if mask["part"] == part and mask["area"] > highest_area and mask["ignored"] == False:
                highest_index = index
                highest_area = mask["area"]
        if highest_index == -1: #if none are acceptable
            return False
        else:
            return masks[highest_index]

    def capture_images(self):
        self.move_robot.move_out_of_view(speed=3)
        self.reference_image = self.camera.get_image()
        self.depth_image = self.camera.get_depth()

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

if __name__ == "__main__":
    controller = Controller()
    controller.main_flow(1)