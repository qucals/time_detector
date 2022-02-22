'''Распознавание написано только для секундомера'''

import math
import cv2
import numpy as np

from typing import Any, Tuple
from skimage.morphology import skeletonize

from detectorlib import utils


class ClockReader:
    def __init__(self) -> None:
        self._circle_detector = _CircleDetector()
        self._hand_detector = _HandDetector()

    def get_time(self, a_image):
        pass

    def get_time_with_image(self, a_image) -> Tuple[str, Any, Any]:
        circle = self._circle_detector.detect(
            a_image, a_max_radius=220, a_select=False)
        small_circle = self._circle_detector.detect(
            circle.copy(), a_max_radius=60, a_select=False)

        circle = self._hand_detector.detect(
            circle.copy(), a_min_length=80, a_max_length=120)
        small_circle = self._hand_detector.detect(
            small_circle.copy(), a_min_length=20, a_max_length=25,
            a_show=True, a_id_contour=0,
        )

        return "", circle, small_circle


class _CircleDetector:
    def __init__(self) -> None:
        pass

    def detect(self, a_image, a_max_radius=None, a_select: bool = True) -> Any:
        height, width, _ = a_image.shape
        diagonal = math.sqrt(height ** 2 + width ** 2)

        if a_max_radius is None:
            a_max_radius = int(diagonal)

        gray = cv2.cvtColor(a_image, cv2.COLOR_BGR2GRAY)
        blur = self._blur_image(gray)

        circles = cv2.HoughCircles(
            image=blur,
            method=cv2.HOUGH_GRADIENT_ALT,
            dp=1,
            minDist=10,
            param1=250,
            param2=0.9,
            # minRadius=55,
            # maxRadius=220,
            maxRadius=a_max_radius
        )

        if circles is not None:
            circles = np.round(circles[0, :]).astype('int')
            circles = self._filter_circles((height, width), circles)

            _, x, y, r = circles[0]
            cut_image = self._cut_area_outside(a_image.copy(), (x, y, r))

            rect_x = x - r
            rect_y = y - r
            crop_image = cut_image[rect_y:(
                rect_y + 2 * r), rect_x:(rect_x + 2 * r)]

            if a_select:
                cv2.circle(a_image, (x, y), r, (0, 255, 0), 4)
                cv2.rectangle(a_image, (x - 5, y - 5),
                              (x + 5, y + 5), (0, 128, 255), -1)

            result_image = crop_image
        else:
            result_image = a_image

        return result_image

    def _blur_image(self, a_image):
        blur = cv2.GaussianBlur(a_image, (11, 11), 1.5)
        return blur

    def _filter_circles(self, a_size_image: Tuple[int, int], a_circles):
        height, width = a_size_image
        center = (height // 2, width // 2)

        circles = [
            (utils.distance_between_points(center, (x, y)), x, y, r) for x, y, r in a_circles
        ]

        return circles

    def _cut_area_outside(self, a_image, a_circle: Tuple[int, int, int]) -> Any:
        '''Удаляет территорию вне круга'''

        height, width, _ = a_image.shape
        x, y, r = a_circle

        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(mask, (x, y), r, 255, -1, 8, 0)
        result = cv2.bitwise_and(a_image, a_image, mask=mask)

        return result


class _HandDetector:
    def __init__(self) -> None:
        pass

    def detect(self, a_image, a_min_length, a_max_length, a_id_contour=2, a_show=False) -> Any:
        gray = cv2.cvtColor(a_image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 1.5)

        thresh = cv2.threshold(
            blur, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        thresh = 255 - thresh

        cntrs_info = []
        contours = cv2.findContours(
            thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)

        contours = contours[0] if len(contours) == 2 else contours[1]
        index = 0
        for cntr in contours:
            area = cv2.contourArea(cntr)
            cntrs_info.append((index, area))
            index = index + 1

        # sort contours by area
        def takeSecond(elem):
            return elem[1]
        cntrs_info.sort(key=takeSecond, reverse=True)

        # get third largest contour
        arms = np.zeros_like(thresh)
        index_third = cntrs_info[a_id_contour][0]
        cv2.drawContours(arms, [contours[index_third]], 0, (1), -1)

        arms_thin = skeletonize(arms)
        arms_thin = (255 * arms_thin).clip(0, 255).astype(np.uint8)

        # if a_show:
        #     return arms_thin

        # get hough lines and draw on copy of input
        result = a_image.copy()
        lineThresh = 40
        minLineLength = a_min_length
        maxLineGap = a_max_length
        lines = cv2.HoughLinesP(arms_thin,
                                1,
                                np.pi/180,
                                lineThresh,
                                None,
                                minLineLength,
                                maxLineGap
                                )

        if lines is not None:
            for [line] in lines:
                x1 = line[0]
                y1 = line[1]
                x2 = line[2]
                y2 = line[3]
                cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)

        return result
