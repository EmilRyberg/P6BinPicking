from controller.part_id_to_name_converter import part_id_to_name
from move_robot.move_robot import MoveRobot
from vision.vision import Vision
from enums import PartEnum
from controller.class_converter import convert_from_part_id
from aruco import Calibration
from PIL import Image as pimg
import numpy as np

NUMBER_OF_PARTS = 4
FIXTURE_X = 255
FIXTURE_Y = -320
UR_IP = "192.168.1.148"


class Controller:
    def __init__(self):
        self.np_image = None
        self.move_robot = MoveRobot(UR_IP)
        self.vision = Vision()
        self.aruco = Calibration()
        self.detected_objects = None

        print("[I] Controller running")

    def main_flow(self, colour_part_id):
        self.get_image()
        self.detected_objects = self.vision.detect_object()

        z_offset = 0
        self.pick_and_place_part(PartEnum.BACKCOVER.value, z_offset)

        z_offset = 20
        self.pick_and_place_part(PartEnum.PCB.value, z_offset)

        z_offset = 30
        self.pick_and_place_part(colour_part_id, z_offset)

        self.move_robot.move_out_of_view()

    def pick_and_place_part(self, part_id, z_offset):
        #NOTE: mostly pseudocode
        #pick best candidate
            #sort by area
            #pick top area
            #if area below threshold
                #shake and retry
        #calculate principal axis/center and vector
        #calibrate with aruco
        #attempt grip
            #if failed n times pick next best candidate and retry
        #place on table

        self.move_robot.align()

    def get_image(self):
        self.move_robot.move_out_of_view()
        self.np_image = self.vision.capture_image()

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
