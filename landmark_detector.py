import numpy as np
import cv2
import time
import os

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from hands import Hand


class LandmarkDetector:
    MODEL_FILE_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")

    def __init__(self, number_of_hands: int = 1, running_mode=vision.RunningMode.LIVE_STREAM, result_callback=None):
        self.model_file_path = self.MODEL_FILE_PATH

        self.options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(self.model_file_path),
            running_mode=running_mode,
            num_hands=number_of_hands,
            result_callback=result_callback,
        )

        self.landmarker = vision.HandLandmarker.create_from_options(self.options)
        self.latest_frame = None
        self.latest_hands = []
        self.latest_timestamp = 0

    @staticmethod
    def draw_landmarks_from_hands(drawn_image, hands) -> list:
        drawn_image = cv2.cvtColor(drawn_image, cv2.COLOR_RGB2BGR)

        for hand in hands:
            for _, points in hand.fingers.items():
                line_points = []
                for pt in points:
                    x, y, _ = pt
                    u, v = int(x * drawn_image.shape[1]), int(y * drawn_image.shape[0])
                    cv2.circle(drawn_image, center=(u, v), radius=5, color=(0, 255, 0))

                    line_points.append([(u, v)])

                line_points = np.array(line_points, dtype=np.int32)
                cv2.polylines(drawn_image, [line_points], isClosed=False, color=(0, 255, 0), thickness=2)

        return drawn_image

    def _result_cb(self, result, output_img: mp.Image, timestamp_ms: int) -> None:
        self.latest_hands = Hand.from_landmarker_result(result)

    def get_latest_data(self):
        return (self.latest_frame, self.latest_hands)


class LiveLandmarkDetector(LandmarkDetector):
    def __init__(self, number_of_hands: int = 1):
        super().__init__(number_of_hands=number_of_hands, running_mode=vision.RunningMode.LIVE_STREAM, result_callback=self._result_cb)

    def process_frame(self, frame):
        self.latest_frame = frame

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame,
        )

        self.latest_timestamp = max(int(time.time() * 1000), self.latest_timestamp)
        self.landmarker.detect_async(mp_image, self.latest_timestamp)


class ImageLandmarkDetector(LandmarkDetector):
    def __init__(self, number_of_hands: int = 1):
        super().__init__(number_of_hands=number_of_hands, running_mode=vision.RunningMode.IMAGE, result_callback=None)

    def process_frame(self, frame):
        self.latest_frame = frame

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame,
        )

        result = self.landmarker.detect(mp_image)
        self.latest_hands = Hand.from_landmarker_result(result)


class VideoLandmarkDetector(LandmarkDetector):
    def __init__(self, number_of_hands: int = 1):
        super().__init__(number_of_hands=number_of_hands, running_mode=vision.RunningMode.VIDEO, result_callback=None)

    def process_frame(self, frame):
        self.latest_frame = frame

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame,
        )

        self.latest_timestamp = max(int(time.time() * 1000), self.latest_timestamp)
        result = self.landmarker.detect_for_video(mp_image, self.latest_timestamp)
        self.latest_hands = Hand.from_landmarker_result(result)