import cv2
import time
import joblib
import numpy as np
from collections import deque, Counter
from features import make_holistic, get_frame_features, mp_draw, mp_holistic

model = joblib.load("words_model.pkl")
holistic = make_holistic()

FRAMES = 30             # same as in recording
MIN_CONFIDENCE = 0.6    # below this, count it as "no sign"
VOTES_NEEDED = 7        # 7 of the last 10 guesses must agree
HOLD_DISPLAY = 1.5      # seconds to keep showing the word

frames = deque(maxlen=FRAMES)
recent = deque(maxlen=10)
shown = "..."
shown_time = 0
guess = "-"
guess_conf = 0
counter = 0

cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb)

    mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_draw.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_draw.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    frames.append(get_frame_features(results))
    counter += 1

    # predict every 3 frames once we have 30 frames
    if len(frames) == FRAMES and counter % 3 == 0:
        x = np.array(frames).flatten().reshape(1, -1)
        proba = model.predict_proba(x)[0]
        best = int(np.argmax(proba))
        guess = model.classes_[best]
        guess_conf = proba[best]

        recent.append(guess if guess_conf >= MIN_CONFIDENCE else "NONE")
        top, votes = Counter(recent).most_common(1)[0]
        if top != "NONE" and votes >= VOTES_NEEDED:
            shown = top
            shown_time = time.time()

    # clear the word after a short time
    if shown != "..." and time.time() - shown_time > HOLD_DISPLAY:
        shown = "..."
        recent.clear()

    # big word = the confirmed result, small line = the raw live guess
    cv2.putText(frame, shown, (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 0), 3)
    cv2.putText(frame, f"guess: {guess} ({guess_conf * 100:.0f}%)", (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, "Q = quit", (20, 465),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Word recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()