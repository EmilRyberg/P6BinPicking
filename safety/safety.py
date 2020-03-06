from safety.pyurg import pyurg
import threading
import serial
import time

STOP_THRESHOLD_LOW = 10
STOP_THRESHOLD_HIGH = 500
SLOW_THRESHOLD_LOW = 501
SLOW_THRESHOLD_HIGH = 1000


class Safety:
    def __init__(self):
        self.ignore_indexes = []
        self.person_close_counter = 0
        self.person_close = False
        self.person_approaching_counter = 0
        self.person_approaching = False
        self.stop_counter = 0
        self.slow_counter = 0
        self.led_thread_running = False
        self.scanner = pyurg.UrgDevice()
        if not self.scanner.connect():
            print('Could not connect to laser scanner.')
            exit(0)

        self.led = serial.Serial('/dev/ttyUSB0', 9600)
        time.sleep(3)
        self.led.write("r0.g0.b200\n".encode('ascii'))

        # Getting reference values to ignore
        data, timestamp = self.scanner.capture()
        for i in range(len(data)):
            if i == 0:
                continue
            elif i == len(data) - 2:
                break
            elif STOP_THRESHOLD_LOW < data[i - 1] < SLOW_THRESHOLD_HIGH \
                    and STOP_THRESHOLD_LOW < data[i] < SLOW_THRESHOLD_HIGH \
                    and STOP_THRESHOLD_LOW < data[i + 1] < SLOW_THRESHOLD_HIGH:
                self.add_to_ignore_index_if_not_exist(i)
                self.add_to_ignore_index_if_not_exist(i - 1)
                self.add_to_ignore_index_if_not_exist(i + 1)

    def add_to_ignore_index_if_not_exist(self, index):
        if index not in self.ignore_indexes:
            self.ignore_indexes.append(index)

    def check_distances(self, controller, x):
        while True:
            data, timestamp = self.scanner.capture()
            self.person_close_counter = 0
            self.person_approaching_counter = 0
            if timestamp == -1:
                print('Could not get laser scanner data: slowing down and attempting reconnect')
                controller.move_robot.set_speed(0.5)
                for i in range(5):
                    if not self.scanner.connect():
                        print('Could not reconnect to laser scanner, attempt number: ', i)
                    else:
                        print("Reconnected to laser scanner, resuming robot")
                        controller.move_robot.set_speed(100)
                        break
                    print("Could not reconnect to laser scanner, stopping system")
                    controller.move_robot.stop_all()
            else:
                for i in range(len(data)):
                    if i < 5:
                        continue
                    elif i == len(data) - 7:
                        break

                    skip_iteration = False
                    for x in range(-5, 6):
                        if i + x in self.ignore_indexes:
                            skip_iteration = True
                            break

                    if skip_iteration:
                        continue

                    for x in range(-5, 6):
                        if STOP_THRESHOLD_LOW < data[i + x] < STOP_THRESHOLD_HIGH:
                            self.stop_counter += 1
                            if self.stop_counter > 9:
                                self.person_close_counter += 1
                        else:
                            self.stop_counter = 0
                            continue

                    for x in range(-5, 6):
                        if SLOW_THRESHOLD_LOW < data[i] < SLOW_THRESHOLD_HIGH:
                            self.slow_counter += 1
                            i = i+1
                            if self.slow_counter > 9:
                                self.person_approaching_counter += 1
                        else:
                            self.slow_counter = 0
                            continue

                if self.person_close_counter > 1:  # To avoid environmental noise from stopping the robot
                    self.person_close = True
                    print("Warning: person close, stopping robot")
                    controller.move_robot.set_speed(0.5)
                    self.led.write("g0.b0\n".encode('ascii'))
                    self.led.write("r200\n".encode('ascii'))
                    self.person_close_counter = 0
                elif self.person_approaching_counter > 1:  # To avoid environmental noise from slowing the robot
                    self.person_approaching = True
                    print("Warning: person close, slowing down robot")
                    controller.move_robot.set_speed(25)
                    self.led.write("r250.g50.b0\n".encode('ascii'))
                    self.person_approaching_counter = 0
                elif self.person_close_counter < 2 and self.person_approaching_counter < 2:
                    self.person_close = False
                    self.person_approaching = False
                    self.led.write("r0.g200.b0\n".encode('ascii'))

                    controller.move_robot.set_speed(100)



if __name__ == "__main__":
    safety = Safety()
    while True:
        safety.check_distances()
