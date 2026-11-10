import cv2
import csv
import os
import time
import numpy as np
from features import make_holistic, get_frame_features, mp_draw, mp_holistic

WORDS = ["HELLO", "I LOVE YOU", "THANK YOU", "YES", "NO", "PLEASE", "NONE"]
FRAMES_PER_CLIP = 30     # one clip = 30 frames
COUNTDOWN = 2            # seconds to get ready
FILE = "gestures.csv"

holistic = make_holistic()

# count clips already saved
counts = {w: 0 for w in WORDS}
if os.path.exists(FILE):
    with open(FILE) as f:
        for row in csv.reader(f):
            if row and row[0] in counts:
                counts[row[0]] += 1

current = 0          # index of the selected word
state = "idle"       # idle -> countdown -> recording -> idle
t0 = 0
clip = []

cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb)

    # draw body and hands
    mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_draw.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_draw.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    if state == "countdown":
        left = COUNTDOWN - (time.time() - t0)
        if left <= 0:
            state = "recording"
            clip = []
        else:
            cv2.putText(frame, str(int(left) + 1), (280, 260),
                        cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 8)

    elif state == "recording":
        clip.append(get_frame_features(results))
        cv2.putText(frame, "SIGN NOW!", (200, 260),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
        if len(clip) >= FRAMES_PER_CLIP:
            row = [WORDS[current]] + list(np.array(clip).flatten())
            with open(FILE, "a", newline="") as f:
                csv.writer(f).writerow(row)
            counts[WORDS[current]] += 1
            state = "idle"

    # menu on the left
    for i, w in enumerate(WORDS):
        color = (0, 255, 0) if i == current else (255, 255, 255)
        cv2.putText(frame, f"{i + 1}: {w}  ({counts[w]})", (10, 25 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(frame, "1-7 = choose word | SPACE = record clip | ESC = quit",
                (10, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    cv2.imshow("Record words", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == 32 and state == "idle":
        state = "countdown"
        t0 = time.time()
    elif ord('1') <= key <= ord('7') and state == "idle":
        current = key - ord('1')

cap.release()
cv2.destroyAllWindows()