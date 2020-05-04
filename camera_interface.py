import numpy as np
from PIL import Image as pimg

class CameraInterface:
    def __init__(self, mode, simulation_connector_instance=None):
        self.mode = mode
        if mode == "simulation":
            self.connector = simulation_connector_instance
            assert simulation_connector_instance != None, "simulation_connector_instance cannot be empty in simulation mode"
        elif mode == "real":
            import pyrealsense2 as rs
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
        else:
            raise Exception(f"invalid mode: {mode}")

    def __del__(self):
        if self.mode == "real":
            self.rs_pipeline.stop()

    def get_image(self):
        if self.mode == "simulation":
            image = self.connector.get_image()
            return image
        else:
            frames = self.rs_pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())
            return color_image

    def get_depth(self):
        if self.mode == "simulation":
            depth = self.connector.get_depth()
            return depth
        else:
            frames = self.rs_pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()