from detectorlib import icamera

class Detector:
    def __init__(self, a_camera: icamera.ICamera) -> None:
        self.icamera = a_camera
