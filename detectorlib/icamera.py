'''Интерфейс для работы с камерой'''

import cv2

from typing import Any


class ICamera:
    def __init__(self, a_camera_id=0) -> None:
        self.vid = cv2.VideoCapture(a_camera_id)

    @property
    def frame(self) -> Any:
        _, frame = self.vid.read()
        return frame
