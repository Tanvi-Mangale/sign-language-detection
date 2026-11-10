import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils


def make_holistic():
    return mp_holistic.Holistic(min_detection_confidence=0.5,
                                min_tracking_confidence=0.5)


def hand_points(hand, nose, scale):
    """42 numbers per hand (x,y of 21 points), measured from the nose."""
    if hand is None:
        return [0.0] * 42
    pts = []
    for lm in hand.landmark:
        pts.append((lm.x - nose[0]) / scale)
        pts.append((lm.y - nose[1]) / scale)
    return pts


def get_frame_features(results):
    """84 numbers per frame: left hand + right hand, relative to the body."""
    if results.pose_landmarks is None:
        return [0.0] * 84
    pose = results.pose_landmarks.landmark
    nose = (pose[0].x, pose[0].y)
    left_shoulder, right_shoulder = pose[11], pose[12]
    scale = ((left_shoulder.x - right_shoulder.x) ** 2 +
             (left_shoulder.y - right_shoulder.y) ** 2) ** 0.5
    if scale < 0.01:
        scale = 0.01
    return (hand_points(results.left_hand_landmarks, nose, scale) +
            hand_points(results.right_hand_landmarks, nose, scale))