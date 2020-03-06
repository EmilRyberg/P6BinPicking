from controller.controller import Controller
from safety.safety import Safety
import threading

if __name__ == "__main__":
    controller = Controller()
    safety_control = Safety()
    quit_program = False

    safety_thread = threading.Thread(target=safety_control.check_distances, args=(controller, 0), daemon=True)
    safety_thread.start()

    while not quit_program:
        quit_program = controller.choose_action()

