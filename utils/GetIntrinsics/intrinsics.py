import pyrealsense2 as rs

pipeline = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
cfg.enable_stream(rs.stream.color, 1920, 1080, rs.format.rgb8, 30)
profile = pipeline.start(cfg)

profile_depth = profile.get_stream(rs.stream.depth)  # Fetch stream profile for depth stream
profile_color = profile.get_stream(rs.stream.color)
intr_depth = profile_depth.as_video_stream_profile().get_intrinsics()  # Downcast to video_stream_profile and fetch intrinsics
intr_color = profile_color.as_video_stream_profile().get_intrinsics()
print(intr_depth)
print(intr_color)

"""output: 
    width: 1280, height: 720, ppx: 632.528, ppy: 349.938, fx: 890.942, fy: 890.942, model: 4, coeffs: [0, 0, 0, 0, 0]
width: 1920, height: 1080, ppx: 956.754, ppy: 545.3, fx: 1393.77, fy: 1393.14, model: 2, coeffs: [0, 0, 0, 0, 0]
"""

camera_matrix = np.array([[1393.77, 0, 956.754], [0, 1393.14, 545.3], [0, 0, 1]])
