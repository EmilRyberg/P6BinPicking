import socket
import time
import math
import urx
#from controller.enums import PartEnum, OrientationEnum

GRIPPER_IP = "192.168.1.118"
GRIPPER_PORT = 1000


class MoveRobot:
    def __init__(self, ip):
        self.suction_enable_pin = 6
        self.home_pose_l = [35, -300, 300, 0, 0, -0.8]
        self.home_pose = [-60, -60, -110, -100, 90, -60]
        # self.move_out_of_view_pose = [-350, -35, 300, 3.14, 0, 0]
        self.move_out_of_view_pose = [-150, -60, -110, -100, 90, -60]
        self.default_orientation = [0, 0, 0]
        self.gripper_tcp = [0, 0, 0.1535, 2.9024, -1.2023, 0]
        self.fuse_tcp = [0.057, -0.00109, 0.13215, -1.7600, -0.7291, 1.7601]
        self.suction_tcp = [-0.12, 0, 0.095, 0, 1.5707, 0]
        self.current_part_id = None
        self.grip_has_been_called_flag = False
        self.moved_to_camera_flag = False

        self.align_fuse_point_1 = [258.3808266269915, 182.66080196127277, 50.755338740619685, 0.5129225399673327, -0.5681073061405235, -0.021312928850932115]
        self.align_fuse_point_2 = [269.56669707049855, 192.5271576116136, 36.52450508933613, 0.5352153513500437, -0.5851532800726972, -0.022296119825804664]
        self.align_fuse_point_3 = [256.79863096450663, 180.323917259804, 38.04563142826764, 0.5128000908988749, -0.5681263381546497, -0.021276095366553276]
        self.align_fuse_point_4 = [267.24826717836964, 189.57576475222854, 23.53934054782497, 0.5129307828342723, -0.5681049276113338, -0.02124040568020565]

        self.align_pcb_1 = [2, -62, -108, -97, 89, 46]  # joint values
        self.align_pcb_2 = [381, -12, 272, 0.61, -1.51, 0.64]  # Cartesian coordinates
        self.align_pcb_3 = [388, 2, 280, 0.6154, -1.5228, 0.62]  # Cartesian coordinates
        self.align_pcb_4 = [-6, -51, -114, -96, 89, 44]

        self.align_pcb_flipped_1 = [22, -86, -84, -98, 89, -110]
        self.align_pcb_flipped_2 = [373, -16.5, 257, 2.405, 1.018, 2.52]  # Cartesian coordinates
        self.align_pcb_flipped_3 = [373, -16.5, 300, 2.405, 1.018, 2.52]  # Cartesian coordinates
        self.align_pcb_flipped_4 = [-16, -55, -107, -104, 87, 29]

        self.align_pcb_pick_1 = [22, -57, -150, -70, 53, 61]  # Joint values
        self.align_pcb_pick_2 = [362, -23, 47, 0.177, -1.1775, 1.194]  # Cartesian coordinates
        self.align_pcb_pick_3 = [336.5, -47, 67.5, 0.177, -1.1775, 1.194]  # Cartesian

        self.align_cover_1 = [122.5, -71.6, 105.5, -84, -13, -60]  # joint
        self.align_cover_2 = [560, -253, 134.7, 1.4433, -0.333, -1.095]  # Cartesian

        self.align_cover_flipped_1 = [3.5, -143, -45, -186, 53, 120]  # joint
        self.align_cover_flipped_2 = [546, -265, 136.5, 3.124, 0.44, 2.146]  # Cartesian
        self.align_cover_flipped_3 = [113, -75, 105, -83, -19, -61]

        self.align_cover_pick_1 = [531, -265, 160, 1.32, -0.334, -1.13]  # Cartesian
        self.align_cover_pick_2 = [539.9, -287.6, 99.3, 1.397, -0.316, -1.096]  # Cartesian
        self.align_cover_pick_3 = [563.4, -268.1, 92.3, 1.397, -0.316, -1.096]  # Cartesian
        self.align_cover_pick_4 = [564.5, -267.4, 93.6, 1.402, -0.343, -1.087]
        self.align_cover_pick_5 = [539, -287, 101, 1.402, -0.343, -1.087]
        self.align_cover_pick_6 = [-130, -103, 155, -180, 1, -60]
        self.align_cover_pick_7 = [-130, -100, -130, -180, 1, -60]
        self.align_cover_pick_8 = [-61, -62, -107, -100, 89, -61]  # joint

        self.test_back_loc = (-255, -280)
        self.test_pcb_loc = (-174, -362)
        self.test_fuse_1_loc = (-138, -278)
        self.test_fuse_2_loc = (-100, -315)
        self.test_top_loc = (-107, -425)

        self.camera_pose_gripper = [-60, -60, -110, -100, -90, -75]
        self.camera_pose_suction = [-5, -40, -100, -140, 0, -170]

        self.pcb_singularity_avoidance = [-70, -70, -107, -180, -147, 90]

        done = False
        counter = 0
        while not done:
            try:
                self.robot = urx.Robot(ip)
            except Exception as e:
                if e.args[0] == "timed out":
                    print("[ERROR] Connection to the robot timed out. Check IP address")
                    exit(1)
                elif e.args[0] == "Did not receive a valid data packet from robot in 0.5" and counter < 10:
                    time.sleep(0.5)
                    print("Robot didn't respond, retrying, try %s/10" % counter)
                    counter += 1
                else:
                    print(e)
                    exit(1)
            else:
                done = True
        done = False
        counter = 0
        while not done:
            try:
                self.gripper = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.gripper.connect((GRIPPER_IP, GRIPPER_PORT))
            except Exception as e:
                counter += 1
                print(e)
                print("[WARNING] Couldn't connect to gripper, try %s" % counter)
                time.sleep(0.2)
                if counter > 10:
                    print("[FATAL] Couldn't connect to gripper, exiting.")
                    self.gripper.close()
                    exit(1)
            else:
                done = True

        self.disable_suction()
        # self.move_gripper(0)
        self.move_gripper(50)

    def __del__(self):
        self.stop_all()

    def stop_all(self):
        self.robot.stop()
        msg = "bye()\n"
        msg = msg.encode()
        self.gripper.send(msg)
        self.gripper.close()
        self.disable_suction()
        self.robot.close()
        print("[INFO] Safely stopped robot and gripper")

    def getl(self):
        temp = self.robot.getl()
        return [temp[0] * 1000, temp[1] * 1000, temp[2] * 1000, temp[3], temp[4], temp[5]]

    def movel(self, pose, acc=1.0, vel=0.2, wait=True, relative=False):
        pose_local = pose.copy()
        print("goal pose in mm: ", pose_local)
        pose_local[0] *= 0.001
        pose_local[1] *= 0.001
        pose_local[2] *= 0.001
        print(pose_local)
        self.robot.movel(pose_local, acc=acc, vel=vel, wait=wait, relative=relative)

    def movej(self, pose, acc=1.0, vel=0.1, wait=True, relative=False):
        pose_local = pose.copy()
        print("pose in deg: ", pose_local)
        for i in range(6):
            pose_local[i] = math.radians(pose_local[i])
        print("pose in radians: ", pose_local)
        self.robot.movej(pose_local, acc, vel, wait, relative)

    def move_to_home(self, speed=1.0):
        self.movej(self.home_pose, acc=1.0, vel=speed)

    def move_to_home_l(self, speed=1.0):
        self.movel(self.home_pose_l, acc=1.0, vel=speed)

    def move_to_camera(self, speed=1.0, is_pcb=False):
        self.moved_to_camera_flag = True
        if is_pcb:
            self.movej(self.camera_pose_suction, acc=1.0, vel=speed)
        else:
            self.movej(self.camera_pose_gripper, acc=1.0, vel=speed)

    def move_out_of_view(self, speed=1.0):
        self.movej(self.move_out_of_view_pose, acc=1.0, vel=speed)

    def open_gripper(self, width=100):
        msg = "release()\n"
        msg = msg.encode()
        self.gripper.send(msg)
        time.sleep(1)
        if width < 5:
            self.move_gripper(width)

    def close_gripper(self, width=0):
        msg = "grip(40," + str(width) + "\n"
        msg = msg.encode()
        self.gripper.send(msg)
        time.sleep(2)
        #TODO get feedback from gripper
        success = True
        return success

    def move_gripper(self, position):
        msg = "move({})\n".format(position)
        msg = msg.encode()
        self.gripper.send(msg)
        time.sleep(1)

    def enable_suction(self):
        self.robot.set_digital_out(self.suction_enable_pin, True)

    def disable_suction(self):
        self.robot.set_digital_out(self.suction_enable_pin, False)

    def grip(self, x, y, z, rx, ry, rz, part_id, width):  # 0 = part horizontal, 1 = part vertical
        self.move_gripper(100)
        self.move_to_home()
        rx = math.radians(rx) #these will only be needed if we call grip with rx,ry and rz as degrees
        ry = math.radians(ry)
        rz = math.radians(rz)
        orientation_vector = [rx, ry, rz]
        self.current_part_id = part_id
        self.grip_has_been_called_flag = True
        if part_id == PartEnum.PCB.value or part_id == PartEnum.PCB_FLIPPED.value:
            print("gripping PCB")
            self.move_to_home()
            self.robot.set_tcp(self.suction_tcp)
            self.move_to_home_l()
            self.movel([x, y, 300] + orientation_vector, vel=1)
            self.movel([x, y, z+40]+orientation_vector,vel=0.2)
            self.enable_suction()
            self.movel([x, y, z] + orientation_vector, vel=0.2)
            self.movel([x, y, z+40]+orientation_vector,vel=0.2)
            self.movel([x, y, 300] + orientation_vector, vel=0.2)
        elif part_id == PartEnum.FUSE.value:
            print("gripping fuse")
            #orientation_vector[2] = rz+90  might need to do some angle offsets for fuses when we get to those
            self.move_to_home()
            self.robot.set_tcp(self.fuse_tcp)
            self.move_to_home_l()
            self.movel([x, y, 300] + orientation_vector, vel=1)
            self.movel([x, y, z+20] + orientation_vector, vel=0.2)
            self.movel([x, y, z] + orientation_vector, vel=0.2)
            self.close_gripper()
            self.movel([x, y, z+20] + orientation_vector, vel=0.2)
            self.movel([x, y, 300] + orientation_vector, vel=0.2)
        else:  # covers
            print("gripping cover")
            self.move_to_home()
            self.robot.set_tcp(self.gripper_tcp)
            self.move_to_home_l()
            self.movel([x, y, 300] + orientation_vector, vel=1) 
            self.movel([x, y, z+20]+orientation_vector, vel=0.2)
            self.open_gripper()
            self.movel([x, y, z] + orientation_vector, vel=0.2)
            self.close_gripper()
            self.movel([x, y, 20] + orientation_vector, vel=0.2) 
            self.movel([x, y, 300]+orientation_vector, vel=1)

    def assemble(self, x=320, y=-350, z=0, rotated=False, fuse_id=0):
        if self.moved_to_camera_flag:
            self.move_to_home()
            self.moved_to_camera_flag = False
        if self.current_part_id in (PartEnum.BACKCOVER.value, PartEnum.BACKCOVER_FLIPPED.value): #back cover
            self.move_to_home_l()
            if rotated:
                angle = 3.14
            else:
                angle = 0
            self.movel([x, y, z + 20, 0, 0, angle], vel=1)
            self.movel([x, y, z + 0.5, 0, 0, angle], vel=0.2)
            self.open_gripper(width=20)
            self.movel([x, y, z + 20, 0, 0, angle], vel=0.2)
        elif self.current_part_id in (PartEnum.PCB.value, PartEnum.PCB_FLIPPED.value): #PCB
            self.move_to_home_l()
            if rotated:
                angle = 3.14
                self.movej(self.pcb_singularity_avoidance,vel=2)
            else:
                angle = 0
            self.movel([325 - 320 + x, -353 - -350 + y, 20 + z, 0, 0, -0.24 + angle], vel=1)
            self.movel([325 - 320 + x, -353 - -350 + y, -6 + z, 0, 0, -0.24 + angle])
            self.disable_suction()
            self.movel([325 - 320 + x, -353 - -350 + y, 20 + z, 0, 0, -0.24 + angle])
        elif self.current_part_id == PartEnum.FUSE.value: #Fuses
            self.move_to_home_l()
            if fuse_id == 0:
                self.movel([298.4 - 320 + x, -322 - -350 + y, z + 40, 0, 0, 3.14])
                self.movel([298.4 - 320 + x, -322 - -350 + y, z + 11, 0, 0, 3.14], vel=0.05)
                self.open_gripper(width=7)
                self.movel([298.4 - 320 + x, -322 - -350 + y, z + 40, 0, 0, 3.14])
            else:
                self.movel([306.8 - 320 + x, -330.8 - -350 + y, z + 40, 0, 0, 3.14])
                self.movel([306.8 - 320 + x, -330.8 - -350 + y, z + 11, 0, 0, 3.14], vel=0.05)
                self.open_gripper(width=7)
                self.movel([306.8 - 320 + x, -330.8 - -350 + y, z + 40, 0, 0, 3.14])
            self.move_gripper(30)
        else:  # top covers
            self.move_to_home_l()
            if rotated:
                angle = 3.14
            else:
                angle = 0
            self.movel([x, y, z + 40, 0, 0, angle])
            self.movel([x, y, z + 13, 0, 0, angle])
            self.open_gripper(width=20)
            self.movel([x, y, z + 40, 0, 0, angle])
            self.movel([x, y, z + 40, 0, 0, 1.57], vel=0.2, wait=False)
            self.close_gripper()
            self.movel([x, y, z + 19, 0, 0, 1.57], vel=0.05)
            self.movel([x, y, z + 40, 0, 0, 1.57])
            self.open_gripper()

    def align(self):
        if not self.grip_has_been_called_flag:
            print("[FATAL] align has been called before grip, exiting")
            self.stop_all()
        if self.current_part_id == PartEnum.FUSE.value: #fuses
            self.move_to_home_l()
            self.movel(self.align_fuse_point_1, vel=1)
            self.movel(self.align_fuse_point_2, vel=0.2)
            self.open_gripper(width=10)
            self.movel(self.align_fuse_point_3, vel=1)
            self.movel(self.align_fuse_point_4, vel=0.2)
            self.close_gripper()
            self.movel(self.align_fuse_point_3, vel=0.2)
        elif self.current_part_id in (PartEnum.BACKCOVER.value, PartEnum.BLACKCOVER.value, PartEnum.BLUECOVER.value, PartEnum.WHITECOVER.value):  # covers
            self.move_to_home_l()
            self.movej([50, -115, 85, -155, 84, 2], vel=1)
            self.movej(self.align_cover_1, vel=1)
            self.movel(self.align_cover_2, vel=0.2)
            self.open_gripper(width=10)
            self.movej(self.align_cover_1, vel=1)
            self.movel(self.align_cover_pick_1, vel=1)
            self.movel(self.align_cover_pick_2, vel=1)
            self.movel(self.align_cover_pick_3, vel=0.2)
            self.close_gripper()
            self.movel(self.align_cover_pick_4, vel=0.2)
            self.movel(self.align_cover_pick_5, vel=0.05)
            self.movej(self.align_cover_pick_6, vel=3, acc=3)
            self.movej(self.align_cover_pick_7, vel=3, acc=3)
            self.movej(self.align_cover_pick_8, vel=1)
        elif self.current_part_id in (PartEnum.BACKCOVER_FLIPPED.value, PartEnum.BLACKCOVER_FLIPPED.value, PartEnum.WHITECOVER_FLIPPED.value, PartEnum.BLUECOVER_FLIPPED.value): #covers flipped
            self.move_to_home_l()
            self.movej(self.align_cover_flipped_1, vel=1)
            self.movel(self.align_cover_flipped_2, vel=0.2)
            self.open_gripper(width=10)
            self.movej(self.align_cover_flipped_1, vel=1)
            # self.movel(self.align_cover_pick_1, vel=1)
            self.movej(self.align_cover_flipped_3, vel=1)  # big turn
            self.movel(self.align_cover_pick_2, vel=1)
            self.movel(self.align_cover_pick_3, vel=0.2)
            self.close_gripper()
            self.movel(self.align_cover_pick_4, vel=0.2)
            self.movel(self.align_cover_pick_5, vel=0.05)
            self.movej(self.align_cover_pick_6, vel=3, acc=2)
            self.movej(self.align_cover_pick_7, vel=3, acc=2)
            self.movej(self.align_cover_pick_8, vel=1)
        elif self.current_part_id == PartEnum.PCB.value: #PCB
            self.move_to_home_l()
            self.movej(self.align_pcb_1, vel=1)
            self.movel(self.align_pcb_2, vel=0.2)
            self.disable_suction()
            # self.movel(self.align_pcb_3, vel=1)
            self.movej(self.align_pcb_4, vel=1)
            self.movej(self.align_pcb_pick_1, vel=1)
            self.enable_suction()
            self.movel(self.align_pcb_pick_2, vel=0.2)
            self.movel(self.align_pcb_pick_3, vel=0.2)
        elif self.current_part_id == PartEnum.PCB_FLIPPED.value: #PCB flipped
            self.movej(self.align_pcb_flipped_1, vel=1)
            self.movel(self.align_pcb_flipped_2, vel=0.2)
            self.disable_suction()
            self.movej(self.align_pcb_flipped_1, vel=1)
            self.movej(self.align_pcb_flipped_4, vel=1)
            self.movej(self.align_pcb_pick_1, vel=1)
            self.enable_suction()
            self.movel(self.align_pcb_pick_2, vel=0.2)
            self.movel(self.align_pcb_pick_3, vel=0.2)

    def get_gripper_distance(self):
        raise NotImplementedError()


if __name__ == "__main__":
    robot = MoveRobot("192.168.1.148")
    time.sleep(1)
    print("init done")

    robot.move_to_home()
    robot.grip(robot.test_back_loc[0], robot.test_back_loc[1], OrientationEnum.VERTICAL.value, PartEnum.BACKCOVER.value)
    robot.align()
    #robot.move_to_camera()
    robot.assemble()

    robot.grip(robot.test_pcb_loc[0], robot.test_pcb_loc[1], OrientationEnum.VERTICAL.value, PartEnum.PCB.value)
    robot.align()
    #robot.move_to_camera(is_pcb=True)
    robot.assemble()

    robot.grip(robot.test_fuse_1_loc[0], robot.test_fuse_1_loc[1], OrientationEnum.VERTICAL.value, PartEnum.FUSE.value)
    robot.align()
    robot.assemble(fuse_id=0)
    robot.grip(robot.test_fuse_2_loc[0], robot.test_fuse_2_loc[1], OrientationEnum.VERTICAL.value, PartEnum.FUSE.value)
    robot.align()
    robot.assemble(fuse_id=1)

    robot.grip(robot.test_top_loc[0], robot.test_top_loc[1], OrientationEnum.VERTICAL.value, PartEnum.BLACKCOVER_FLIPPED.value)
    robot.align()
    robot.assemble()

    robot.move_to_home_l()

    robot.stop_all()