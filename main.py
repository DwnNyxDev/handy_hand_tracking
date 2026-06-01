from landmark_detector import LiveLandmarkDetector
    
if __name__ == "__main__":
    landmark_detector = LiveLandmarkDetector(number_of_hands=1)
    landmark_detector.start_capture(
        dimensions = (1280, 720)
    )
    