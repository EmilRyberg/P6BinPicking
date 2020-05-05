from controller.controller import Controller
from move_robot.move_robot import MoveRobot
from safety.safety import Safety
from real_camera_interface import RealCamera
import threading

UR_IP = "192.168.1.148"

if __name__ == "__main__":
    move_robot = MoveRobot(UR_IP)
    camera_interface = RealCamera()
    controller = Controller(move_robot, camera_interface)
    safety_control = Safety()
    quit_program = False

    safety_thread = threading.Thread(target=safety_control.check_distances, args=(controller, 0), daemon=True)
    safety_thread.start()

    while not quit_program:
        quit_program = controller.choose_action()

