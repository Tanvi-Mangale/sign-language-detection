import cv2
import mediapipe as mp
import csv

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.5)

MAX_SAMPLES = 300   # samples saved per recording


def get_features(hand):
    """Turn 21 hand points into 63 numbers that describe only the hand SHAPE."""
    wrist = hand.landmark[0]
    coords = []
    for lm in hand.landmark:
        # subtract the wrist so the position in the frame doesn't matter
        coords.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
    biggest = max(abs(v) for v in coords)
    if biggest == 0:
        biggest = 1
    # divide so the distance from the camera doesn't matter
    return [v / biggest for v in coords]


current_label = None
recording = False
rec_count = 0

cap = cv2.VideoCapture(0)

with open("data.csv", "a", newline="") as f:
    writer = csv.writer(f)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        hand_found = False
        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            hand_found = True
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            # save one sample per frame while recording
            if recording and current_label is not None:
                writer.writerow([current_label] + get_features(hand))
                rec_count += 1
                if rec_count >= MAX_SAMPLES:
                    recording = False   # auto-stop

        # ---- text on screen ----
        status = "RECORDING" if recording else "Paused"
        cv2.putText(frame, f"Letter: {current_label}   {status}   Saved: {rec_count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, "Letter key = choose | SPACE = start/stop | ESC = quit",
                    (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        cv2.imshow("Collect data", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:                       # ESC = quit
            break
        elif key == 32:                     # SPACE = start / stop
            if current_label is not None:
                recording = not recording
                if recording:
                    rec_count = 0
        elif 97 <= key <= 122:              # a-z = choose the letter
            current_label = chr(key).upper()
            recording = False
            rec_count = 0

cap.release()
cv2.destroyAllWindows()