# Sign Language Recognition System

A real-time desktop app that recognises sign language and converts it into text and speech, built to help reduce the communication gap between deaf and hearing people.

## Features
- **Alphabet mode:** fingerspelling for 24 letters (A–Y, without J)
- **Word mode:** gesture recognition for HELLO, I LOVE YOU, THANK YOU, YES, NO, PLEASE
- **Sentence builder:** recognised words and letters are joined into a sentence
- **Text-to-speech:** the app can read the sentence aloud
- Desktop UI built with Tkinter, with a live camera view

## How it works
1. **MediaPipe** tracks hand (and body) landmarks from the webcam.
2. The landmarks are converted into numbers that describe the hand shape, measured relative to the wrist (alphabet) or the face and shoulders (words), so position and distance from the camera don't matter.
3. A **Random Forest classifier** (scikit-learn) is trained on a custom dataset recorded by me.
   - Alphabet: single frames (63 features per sample)
   - Words: 30-frame clips using both hands (2,520 features per clip)
4. In the app, predictions are smoothed with majority voting so the output doesn't flicker.
5. A **NONE** class teaches the word model what "not signing" looks like, so it doesn't guess a word when nobody is signing.

## Results
| Model | Dataset | Test accuracy |
|---|---|---|
| Alphabet (24 letters) | 8,148 samples | 99.51% |
| Words (6 words + NONE) | 300 clips | 85.0% |

Accuracy is measured on a held-out 20% test split. The alphabet score is very high partly because samples come from the same recording sessions as the training data, so live accuracy varies with lighting, camera position and the signer. Among the words, I LOVE YOU is the weakest (about 50% recall), and the word test set is small (60 clips).

## What I learned along the way
- Some letters (A/T, M/N/S) look almost identical as hand shapes. Cleaner, more varied data fixed this better than changing the model.
- Words need motion, so they are recorded as 30-frame clips rather than single photos.
- Adding a "no sign" class was necessary to stop false detections.

## Tech stack
Python 3.11, OpenCV, MediaPipe 0.10.14, scikit-learn, NumPy, pandas, Tkinter, pyttsx3

## Run it
```bash
git clone https://github.com/Tanvi-Mangale/sign-language-detection.git
cd sign-language-detection
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```
The app needs a webcam. The first start can take up to 30 seconds.

## Project files
| File | Purpose |
|---|---|
| `collect_data.py` | Record alphabet samples |
| `train_model.py` | Train the alphabet model |
| `record_words.py` | Record word gesture clips |
| `train_words.py` | Train the word model |
| `features.py` | Landmark feature extraction for words |
| `app.py` | The final desktop app |

## Limitations
- J and Z are not supported (they need motion).
- Trained on a single signer, so accuracy may drop for other people.
- Words are recognised one by one, without sign language grammar.

## Future work
- Add Indian Sign Language
- Add more words and more signers
- Use a sequence model (LSTM) for motion-heavy signs