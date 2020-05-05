class CameraInterface:
    def get_image(self):
        raise NotImplementedError()

    def get_depth(self):
        raise NotImplementedError()