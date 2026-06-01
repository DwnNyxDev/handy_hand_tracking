import numpy as np

class Hand:

    FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]

    def __init__(self):
        self.side = None
        self.wrist = ()
        self.fingers = {
            finger:[] for finger in Hand.FINGER_NAMES
        }

    def calculate_finger_angle(self, finger_name) -> float:
        assert finger_name in Hand.FINGER_NAMES, f"{finger_name} not in {Hand.FINGER_NAMES}"

        finger_pts = self.fingers[finger_name]
        finger_x, finger_y = [pt[0] for pt in finger_pts], [pt[1] for pt in finger_pts]

        dy = finger_y[1] - finger_y[0]
        dx = finger_x[1] - finger_x[0]
        
        angle_radians = np.atan2(dy,dx)

        if angle_radians < 0:
            angle_radians += np.pi/2
            
        return angle_radians
    
    def get_raised_fingers(self) -> list[type[str]]:
        ret_list = []
        for name, points in self.fingers.items():
            first_phalanx_vec = np.array(points[1]) - np.array(points[0])
            last_phalanx_vec = np.array(points[3]) - np.array(points[2])
            
            norm_first = np.linalg.norm(first_phalanx_vec)
            norm_last = np.linalg.norm(last_phalanx_vec)

            if norm_first == 0 or norm_last.item() == 0:
                continue

            angle_btwn = np.arccos(
                np.dot(first_phalanx_vec,last_phalanx_vec) / (norm_first * norm_last)
            )

            if angle_btwn < 0.5:
                ret_list.append(name)
        
        return ret_list

    @staticmethod
    def from_landmarker_result(result) -> list["Hand"]:
        new_hands = []
        for hand_index in range(len(result.handedness)):
            side = result.handedness[hand_index][0].display_name
            landmarks = result.hand_landmarks[hand_index]

            hand_instance = Hand()
            hand_instance.side = side
            hand_instance.wrist = (landmarks[0].x, landmarks[0].y, landmarks[0].z)

            for i, finger in enumerate(Hand.FINGER_NAMES):
                finger_start = 1 + 4*i
                finger_end = finger_start + 4

                hand_instance.fingers[finger] = np.array([(landmark.x, landmark.y, landmark.z) for landmark in landmarks[finger_start:finger_end]])

            new_hands.append(hand_instance)
        
        return new_hands

    def __str__(self):
        return f"Side: {self.side}\nWrist: {self.wrist}\nFingers: {self.fingers}"