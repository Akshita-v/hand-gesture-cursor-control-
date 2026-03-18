# Gesture Mouse Controller (MediaPipe Tasks API)

A real-time hand-gesture control project built with MediaPipe Tasks + Random Forest.

It supports webcam-based:
- Cursor movement
- Click
- Scroll up/down
- Activation/pause control states

---

## Features

- Real-time hand landmark detection using MediaPipe Tasks (`hand_landmarker.task`)
- Gesture classification with `RandomForestClassifier`
- Gesture stabilization using frame-window majority voting
- Cursor smoothing and deadzone filtering to reduce jitter
- Safe activation model:
  - `0` (open palm) enables control
  - `3` (fist) pauses control

### Gesture Map (`main.py`)

- `0` = Activate system (open palm)
- `1` = Click (pinch)
- `2` = Move cursor (index)
- `3` = Pause system (fist)
- `4` = Scroll up (two fingers)
- `5` = Scroll down (three fingers)

---

## Project Structure

- `data_collector.py` → Collect labeled hand-landmark samples into CSV
- `train_model.py` → Train model, print metrics, save model + encoder
- `main.py` → Run main real-time gesture controller
- `main_control.py` → Diagnostic/validation controller (alternate mapping)
- `mouse.py` → Alternate controller script with swipe-style key actions
- `volume.py` → Simple volume-control demo using hand landmarks
- `hand_data.csv` → Dataset (label + normalized landmarks)
- `gesture_model.pkl` → Trained gesture classifier
- `label_encoder.pkl` → Label encoder
- `hand_landmarker.task` → MediaPipe hand model

---

## Requirements

- Python 3.9+ (3.10+ recommended)
- Webcam
- Windows/macOS/Linux (current setup in this workspace is Windows)

Install dependencies:

```bash
pip install opencv-python mediapipe numpy pandas scikit-learn joblib pyautogui
```

---

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install opencv-python mediapipe numpy pandas scikit-learn joblib pyautogui
```

If model files are missing, generate them using the training flow below.

---

## How to Collect Data

1. Open `data_collector.py`
2. Set `LABEL_ID` to the gesture class you want to record
3. Run:

```bash
python data_collector.py
```

In the camera window:
- Press `S` to save one sample frame
- Press `Q` to quit

Repeat for each label with balanced sample counts.

---

## Data Format

Each row in `hand_data.csv`:
- Column `0`: gesture label
- Remaining columns: normalized landmark coordinates (wrist-relative + scaled)

---

## How to Train

Run:

```bash
python train_model.py
```

Training flow:
- Load CSV dataset
- Encode labels (`LabelEncoder`)
- Train/test split (`test_size=0.2`)
- Train Random Forest
- Print accuracy + classification report
- Retrain on full dataset and save:
  - `gesture_model.pkl`
  - `label_encoder.pkl`

---

## Last Recorded Performance Report

From the latest project output:
- Overall Accuracy: **94.67%**

| Gesture ID | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| 0 | 0.86 | 1.00 | 0.92 | 18 |
| 1 | 1.00 | 0.87 | 0.93 | 15 |
| 2 | 1.00 | 0.86 | 0.92 | 7 |
| 3 | 1.00 | 1.00 | 1.00 | 8 |
| 4 | 1.00 | 0.92 | 0.96 | 12 |
| 5 | 1.00 | 1.00 | 1.00 | 10 |
| 6 | 0.83 | 1.00 | 0.91 | 5 |

Summary:
- Accuracy: `0.95` (75 test samples)
- Macro Avg: Precision `0.96` | Recall `0.95` | F1 `0.95`
- Weighted Avg: Precision `0.95` | Recall `0.95` | F1 `0.95`

---

## Run the Gesture Controller

```bash
python main.py
```

Startup behavior:
- Show Palm (`0`) to start control
- Show Fist (`3`) to pause control

Controls:
- Press `q` in the OpenCV window to exit

---

## Command Reference

```bash
# Collect samples
python data_collector.py

# Train classifier and save model files
python train_model.py

# Run primary controller
python main.py

# Run diagnostic/alternate scripts
python main_control.py
python mouse.py
python volume.py
```

---

## Current Runtime Tuning (`main.py`)

- `SMOOTHING = 0.17` → controls cursor interpolation smoothness
- `CURSOR_DEADZONE = 2` → ignores tiny movement jitter
- `GESTURE_WINDOW = 6` → voting window for stable gesture output
- `SCROLL_COOLDOWN = 0.12` → minimum time gap between scroll events

These values are tuned for smoother control and fewer accidental actions.

---

## Troubleshooting

1. **Camera not opening**
   - Close other camera-using apps and rerun.

2. **Model files missing (`gesture_model.pkl`, `label_encoder.pkl`)**
   - Run `python train_model.py` after collecting samples.

3. **Landmarker file missing (`hand_landmarker.task`)**
   - Keep it in project root, or run `data_collector.py` once (it auto-downloads).

4. **Cursor feels jittery**
   - Increase `GESTURE_WINDOW`
   - Increase `CURSOR_DEADZONE`
   - Fine-tune `SMOOTHING`

5. **Prediction quality is low for some classes**
   - Add more balanced samples for weaker classes
   - Capture under different lighting/angles
   - Retrain after adding data

---

## Safety Notes

- `main.py` sets `pyautogui.FAILSAFE = False`, so keep one hand ready to pause quickly.
- Use gesture `3` (fist) to stop control immediately.
- Always test new models slowly before normal use.

---

## Quick Start (One Flow)

```bash
# 1) collect data (optional)
python data_collector.py

# 2) train model
python train_model.py

# 3) run controller
python main.py
```

---

## Notes

This README reflects current behavior in this workspace (`main.py`, `train_model.py`, and related scripts).
If you later add app-launch gestures (Chrome), this structure can be extended with those mappings and safety gates.

---

## Known Limitations

- Performance depends heavily on lighting, background clutter, and webcam quality.
- Single-hand setup (`num_hands=1`) means multi-hand gestures are not supported.
- Gesture labels are numeric, so data consistency during collection is very important.
- This is tuned for desktop control; accidental actions are still possible if gestures are ambiguous.

---

## Roadmap (Optional Improvements)

- Add confidence-based gating for high-risk actions.
- Add an in-app calibration step before control starts.
- Export training metrics automatically to a file after each training run.
- Add a lightweight GUI for changing runtime tuning values without editing code.
