import cv2
import numpy as np
import time

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from hands import Hand

class LiveLandmarkDetector:
    def __init__(self, number_of_hands = 1):
        self.model_file_path = "hand_landmarker.task"

        self.options = vision.HandLandmarkerOptions(
            base_options = python.BaseOptions(self.model_file_path),
            running_mode = vision.RunningMode.LIVE_STREAM,
            num_hands = number_of_hands,
            result_callback = self.result_cb
        )

        self.landmarker = vision.HandLandmarker.create_from_options(self.options)
        self.latest_frame = None
        self.latest_hands = []

    def result_cb(self, result, output_img: mp.Image, timestamp_ms: int) -> None: 
        self.latest_hands = Hand.from_landmarker_result(result)
    
    def process_frame(self):
        mp_image = mp.Image(
            image_format = mp.ImageFormat.SRGB,
            data = np.array(self.latest_frame),
        )

        self.landmarker.detect_async(mp_image, int(time.time() * 1000))

    def draw_landmarks(self, drawn_image) -> list:     
        if self.latest_hands is None or self.latest_frame is None:
            return

        drawn_image = cv2.cvtColor(drawn_image, cv2.COLOR_RGB2BGR)

        for hand in self.latest_hands:
            raised_fingers = hand.get_raised_fingers()
            for name, points in hand.fingers.items():
                for pt in points:
                    x, y, _ = pt
                    u, v = int(x * drawn_image.shape[1]), int(y * drawn_image.shape[0])

                    cv2.circle(drawn_image, center=(u,v), radius = 5, color=(0,255,0))
                
                if name in raised_fingers:
                    finger_angle = hand.calculate_finger_angle(name)
                    cv2.putText(drawn_image, f"{finger_angle}", (u,v), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(255,0,0), thickness=2)

        return drawn_image
    
    def start_capture(self, dimensions, draw_landmarks = True) -> None:
        cam = cv2.VideoCapture(0)
        cam.set(3, dimensions[0])
        cam.set(4, dimensions[1])

        while cam.isOpened():
            success, frame = cam.read()

            if success:
                self.latest_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            self.process_frame()

            if draw_landmarks:
                image_copy = self.latest_frame.copy()
                image_copy = self.draw_landmarks(image_copy)
                
                cv2.imshow('frame', cv2.flip(image_copy, 1))

            if cv2.waitKey(1) == ord('q'):
                break

        cam.release()
        cv2.destroyAllWindows()
    