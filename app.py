import tkinter as tk
from tkinter import ttk
import threading
import time
from collections import deque, Counter

import cv2
import joblib
import numpy as np
import mediapipe as mp
from PIL import Image, ImageTk
from features import make_holistic, get_frame_features, mp_draw, mp_holistic

# ---------- load models ----------
alpha_model = joblib.load("model.pkl")        # alphabet
word_model = joblib.load("words_model.pkl")   # words

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.5)
holistic = make_holistic()

BG = "#1e1e2e"
FG = "#ffffff"
ACCENT = "#7dd3a8"


def alpha_features(hand):
    """Same function used when recording the alphabet."""
    wrist = hand.landmark[0]
    coords = []
    for lm in hand.landmark:
        coords.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
    biggest = max(abs(v) for v in coords) or 1
    return [v / biggest for v in coords]


def speak_text(text):
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.say(text.lower())
        engine.runAndWait()
    except Exception as e:
        print("Speech not available:", e, "-> run: pip install pyttsx3")


class App:
    def __init__(self, root):
        self.root = root
        root.title("Sign Language Detection")
        root.configure(bg=BG)
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.cap = cv2.VideoCapture(0)
        self.mode = tk.StringVar(value="words")
        self.sentence = ""

        # word-mode state
        self.frames = deque(maxlen=30)
        self.recent_w = deque(maxlen=10)
        self.cooldown_until = 0
        self.counter = 0
        # alphabet-mode state
        self.recent_a = deque(maxlen=10)
        self.candidate = None
        self.cand_start = 0
        self.added = False
        # display
        self.detected = "..."
        self.detected_until = 0

        self.build_ui()
        self.update()

    # ---------- layout ----------
    def build_ui(self):
        self.video = tk.Label(self.root, bg="black")
        self.video.grid(row=0, column=0, padx=12, pady=12)

        panel = tk.Frame(self.root, bg=BG, width=340)
        panel.grid(row=0, column=1, sticky="n", padx=(0, 12), pady=12)

        tk.Label(panel, text="Sign Language Detection", bg=BG, fg=ACCENT,
                 font=("Segoe UI", 16, "bold")).pack(anchor="w", pady=(0, 10))

        tk.Label(panel, text="Mode", bg=BG, fg=FG,
                 font=("Segoe UI", 10)).pack(anchor="w")
        for text, value in (("Words (gestures)", "words"),
                            ("Alphabet (fingerspelling)", "alphabet")):
            tk.Radiobutton(panel, text=text, value=value, variable=self.mode,
                           command=self.reset_state, bg=BG, fg=FG,
                           selectcolor="#333355", activebackground=BG,
                           activeforeground=FG,
                           font=("Segoe UI", 11)).pack(anchor="w")

        tk.Label(panel, text="Detected", bg=BG, fg=FG,
                 font=("Segoe UI", 10)).pack(anchor="w", pady=(14, 0))
        self.detected_label = tk.Label(panel, text="...", bg=BG, fg=ACCENT,
                                       font=("Segoe UI", 30, "bold"))
        self.detected_label.pack(anchor="w")

        self.progress = ttk.Progressbar(panel, length=300, maximum=100)
        self.progress.pack(anchor="w", pady=(2, 0))
        tk.Label(panel, text="(alphabet: hold the sign until the bar fills)",
                 bg=BG, fg="#aaaaaa", font=("Segoe UI", 8)).pack(anchor="w")

        tk.Label(panel, text="Sentence", bg=BG, fg=FG,
                 font=("Segoe UI", 10)).pack(anchor="w", pady=(14, 0))
        self.sentence_label = tk.Label(panel, text="|", bg="#2b2b40", fg=FG,
                                       font=("Segoe UI", 16), width=26, height=5,
                                       wraplength=300, justify="left",
                                       anchor="nw", padx=8, pady=8)
        self.sentence_label.pack(anchor="w")

        row = tk.Frame(panel, bg=BG)
        row.pack(anchor="w", pady=10)
        for text, cmd in (("Space", self.add_space), ("Backspace", self.backspace),
                          ("Clear", self.clear), ("Speak", self.speak)):
            tk.Button(row, text=text, command=cmd, width=8,
                      font=("Segoe UI", 10, "bold"), bg="#3b3b5c", fg=FG,
                      activebackground="#55557a", activeforeground=FG,
                      relief="flat").pack(side="left", padx=3)

    # ---------- sentence helpers ----------
    def refresh_sentence(self):
        self.sentence_label.config(text=self.sentence + "|")

    def add_text(self, t):
        self.sentence += t
        self.refresh_sentence()

    def add_space(self):
        self.add_text(" ")

    def backspace(self):
        self.sentence = self.sentence[:-1]
        self.refresh_sentence()

    def clear(self):
        self.sentence = ""
        self.refresh_sentence()

    def speak(self):
        if self.sentence.strip():
            threading.Thread(target=speak_text, args=(self.sentence,),
                             daemon=True).start()

    def reset_state(self):
        self.frames.clear()
        self.recent_w.clear()
        self.recent_a.clear()
        self.candidate = None
        self.added = False
        self.detected = "..."
        self.progress["value"] = 0

    # ---------- word mode ----------
    def process_words(self, frame, rgb):
        results = holistic.process(rgb)
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_draw.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_draw.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        self.frames.append(get_frame_features(results))
        self.counter += 1

        if len(self.frames) == 30 and self.counter % 3 == 0:
            x = np.array(self.frames).flatten().reshape(1, -1)
            proba = word_model.predict_proba(x)[0]
            best = int(np.argmax(proba))
            label = word_model.classes_[best]
            conf = proba[best]

            self.recent_w.append(label if conf >= 0.6 else "NONE")
            top, votes = Counter(self.recent_w).most_common(1)[0]
            if top != "NONE" and votes >= 7 and time.time() > self.cooldown_until:
                self.add_text(top + " ")
                self.detected = top
                self.detected_until = time.time() + 1.5
                self.cooldown_until = time.time() + 2.0   # avoid adding it twice
                self.recent_w.clear()

        if time.time() > self.detected_until:
            self.detected = "..."
        self.progress["value"] = 0

    # ---------- alphabet mode ----------
    def process_alpha(self, frame, rgb):
        results = hands.process(rgb)
        progress = 0
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            feats = [alpha_features(hand)]
            letter = alpha_model.predict(feats)[0]
            conf = alpha_model.predict_proba(feats).max()
            if conf >= 0.6:
                self.recent_a.append(letter)
            stable = Counter(self.recent_a).most_common(1)[0][0] if self.recent_a else None

            if stable != self.candidate:
                self.candidate = stable
                self.cand_start = time.time()
                self.added = False
            elif stable and not self.added:
                progress = min((time.time() - self.cand_start) / 1.5, 1)
                if progress >= 1:
                    self.add_text(stable)
                    self.added = True
            self.detected = stable or "..."
        else:
            self.recent_a.clear()
            self.candidate = None
            self.added = False
            self.detected = "..."
        self.progress["value"] = progress * 100

    # ---------- main loop ----------
    def update(self):
        ok, frame = self.cap.read()
        if ok:
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.mode.get() == "words":
                self.process_words(frame, rgb)
            else:
                self.process_alpha(frame, rgb)

            self.detected_label.config(text=self.detected)
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            photo = ImageTk.PhotoImage(img)
            self.video.config(image=photo)
            self.video.image = photo
        self.root.after(10, self.update)

    def on_close(self):
        self.cap.release()
        self.root.destroy()


root = tk.Tk()
App(root)
root.mainloop()