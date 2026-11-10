import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        # Point 0 = wrist, point 8 = tip of the index finger
        wrist = hand.landmark[0]
        index_tip = hand.landmark[8]
        print("Wrist:", round(wrist.x, 2), round(wrist.y, 2),
              "| Index tip:", round(index_tip.x, 2), round(index_tip.y, 2))

    cv2.imshow("Points test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()