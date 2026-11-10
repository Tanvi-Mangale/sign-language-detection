import cv2
import mediapipe as mp
import joblib
import time
import pyttsx3
from collections import deque, Counter

model = joblib.load("model.pkl")
engine = pyttsx3.init()

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.5)

HOLD_SECONDS = 1.5      # how long to hold a sign to add the letter
MIN_CONFIDENCE = 0.6    # ignore weak guesses


def get_features(hand):
    wrist = hand.landmark[0]
    coords = []
    for lm in hand.landmark:
        coords.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
    biggest = max(abs(v) for v in coords)
    if biggest == 0:
        biggest = 1
    return [v / biggest for v in coords]


recent = deque(maxlen=10)
sentence = ""
candidate = None
candidate_start = 0
added = False

cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    current = "-"
    progress = 0

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        features = [get_features(hand)]
        letter = model.predict(features)[0]
        confidence = model.predict_proba(features).max()

        if confidence >= MIN_CONFIDENCE:
            recent.append(letter)
        stable = Counter(recent).most_common(1)[0][0] if recent else None
        current = stable if stable else "-"

        if stable != candidate:
            candidate = stable
            candidate_start = time.time()
            added = False
        elif stable and not added:
            progress = (time.time() - candidate_start) / HOLD_SECONDS
            if progress >= 1:
                sentence += stable
                added = True
    else:
        recent.clear()
        candidate = None
        added = False

    # ---- draw on screen ----
    cv2.putText(frame, f"Sign: {current}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    # progress bar for holding a sign
    bar = int(min(progress, 1) * 200)
    cv2.rectangle(frame, (20, 65), (220, 80), (255, 255, 255), 1)
    cv2.rectangle(frame, (20, 65), (20 + bar, 80), (0, 255, 0), -1)

    cv2.putText(frame, sentence[-30:] + "_", (20, 430),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    cv2.putText(frame, "SPACE=space  BACKSPACE=delete  ENTER=speak  C=clear  Q=quit",
                (20, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Sign to speech", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == 32:                 # SPACE
        sentence += " "
    elif key == 8:                  # BACKSPACE
        sentence = sentence[:-1]
    elif key == 13:                 # ENTER = speak
        if sentence.strip():
            engine.say(sentence.lower())
            engine.runAndWait()
    elif key == ord('c'):           # clear
        sentence = ""

cap.release()
cv2.destroyAllWindows()