import math
import cv2
import numpy as np

from typing import Any, Tuple
from detectorlib import utils


class ClockReader:
    def __init__(self) -> None:
        self._circle_detector = _CircleDetector()

    def get_time(self, a_image):
        pass

    def get_time_with_image(self, a_image) -> Tuple[str, Any]:
        circle = self._circle_detector.get_circle(a_image)
        return "", circle


class _CircleDetector:
    def __init__(self) -> None:
        pass

    def get_circle(self, a_image, a_select: bool = True):
        height, width, _ = a_image.shape
        diagonal = math.sqrt(height ** 2 + width ** 2)

        gray = cv2.cvtColor(a_image, cv2.COLOR_BGR2GRAY)
        blur = self._blur_image(gray)

        circles = cv2.HoughCircles(
            image=blur,
            method=cv2.HOUGH_GRADIENT_ALT,
            dp=1,
            minDist=10,
            param1=250,
            param2=0.9,
        )

        if circles is not None:
            circles = np.round(circles[0, :]).astype('int')
            circles = self._filter_circles((height, width), circles)

            _, x, y, r = circles[0]

            rect_x = x - r
            rect_y = y - r
            crop_image = a_image[rect_y:(rect_y + 2 * r), rect_x:(rect_x + 2 * r)]

            if a_select:
                cv2.circle(a_image, (x, y), r, (0, 255, 0), 4)
                cv2.rectangle(a_image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
        else:
            crop_image = a_image

        return crop_image

    def _blur_image(self, a_image):
        blur = cv2.GaussianBlur(a_image, (7, 7), 1.5)
        return blur

    def _filter_circles(self, a_size_image: Tuple[int, int], a_circles):
        height, width = a_size_image
        center = (height // 2, width // 2)

        circles = [
            (utils.distance_between_points(center, (x, y)), x, y, r) for x, y, r in a_circles
        ]

        return circles
