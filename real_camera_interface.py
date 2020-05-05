import numpy as np
from PIL import Image as pimg
from camera_interface import CameraInterface
import pyrealsense2 as rs


class RealCamera(CameraInterface):
    def __init__(self):
        self.rs_pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        cfg.enable_stream(rs.stream.color, 1920, 1080, rs.format.rgb8, 30)
        profile = self.rs_pipeline.start(cfg)
        sensors = profile.get_device().query_sensors()
        rgb_camera = sensors[1]
        rgb_camera.set_option(rs.option.white_balance, 4600)
        rgb_camera.set_option(rs.option.exposure, 80)
        rgb_camera.set_option(rs.option.saturation, 65)
        rgb_camera.set_option(rs.option.contrast, 50)
        for i in range(30):
            self.rs_pipeline.wait_for_frames()

    def __del__(self):
        self.rs_pipeline.stop()

    def get_image(self):
        frames = self.rs_pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        return color_image

    def get_depth(self):
        frames = self.rs_pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        return depth_frame