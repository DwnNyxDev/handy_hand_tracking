from hand_tracking.landmark_detector import LiveLandmarkDetector
import cv2

def start_capture(dimensions, landmark_detector) -> None:
    cam = cv2.VideoCapture(0)
    cam.set(3, dimensions[0])
    cam.set(4, dimensions[1])

    while True:
        success, frame = cam.read()

        if success:
            landmark_detector.process_frame(frame)
        
        latest_frame, latest_hands = landmark_detector.get_latest_data()
        
        if latest_frame is not None:
            drawn_image = latest_frame.copy()
            drawn_image = landmark_detector.draw_landmarks_from_hands(drawn_image, latest_hands)

            # Flip image horizontally so left and right match that of real world
            drawn_image = cv2.flip(drawn_image, 1)

            # Convert image from bgr to rgb
            drawn_image = cv2.cvtColor(drawn_image, cv2.COLOR_BGR2RGB) 

            cv2.imshow('frame', drawn_image)

        if cv2.waitKey(1) == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

def main() -> None:
    landmark_detector = LiveLandmarkDetector(number_of_hands=2)

    print("\nPress (q) to stop capture\n")
    start_capture(
        dimensions=(1280, 720),
        landmark_detector=landmark_detector,
    )

if __name__ == "__main__":
    main()
    
    
    