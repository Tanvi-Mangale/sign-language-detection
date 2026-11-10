import cv2
import mediapipe as mp
import joblib
from collections import deque, Counter

model = joblib.load("model.pkl")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.5)


def get_features(hand):
    """Same function as in collect_data.py - it must stay identical."""
    wrist = hand.landmark[0]
    coords = []
    for lm in hand.landmark:
        coords.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
    biggest = max(abs(v) for v in coords)
    if biggest == 0:
        biggest = 1
    return [v / biggest for v in coords]


recent = deque(maxlen=10)   # last 10 predictions

cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    text = "No hand"
    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        features = [get_features(hand)]
        letter = model.predict(features)[0]
        confidence = model.predict_proba(features).max()

        recent.append(letter)
        stable_letter = Counter(recent).most_common(1)[0][0]  # majority vote
        text = f"{stable_letter}  ({confidence * 100:.0f}%)"
    else:
        recent.clear()

    cv2.putText(frame, text, (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    cv2.imshow("Sign language detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()