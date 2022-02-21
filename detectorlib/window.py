'''Класс для показа окна с камерой'''

import cv2
import logging

from detectorlib import clock_reader, icamera

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.DEBUG, datefmt='%d-%b-%y %H:%M:%S')


class Window:
    def __init__(self) -> None:
        logging.debug('Инициализация камеры')
        self.icamera = icamera.ICamera()

        logging.debug('Инициализация считывателя времени')
        self.clock_reader = clock_reader.ClockReader()

    def show(self):
        logging.debug('Запуск окна камеры')

        while True:
            frame = self.icamera.frame
            # time = self.clock_reader.get_time(frame)
            time, frame = self.clock_reader.get_time_with_image(frame)

            # print(f'Распознанное время: {time}')

            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        logging.debug('Закрытие окна камеры')

        cv2.destroyAllWindows()

