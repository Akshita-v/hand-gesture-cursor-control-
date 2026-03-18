# Gesture Mouse Controller (MediaPipe Tasks API)

Control your mouse and basic desktop actions using real-time hand gestures detected from webcam input.

This project combines:
- **MediaPipe Hand Landmarker (Tasks API)** for 21 hand keypoints
- **Custom gesture classifier** (`RandomForest`) trained on normalized landmark data
- **PyAutoGUI** for system actions (move, click, scroll, key press)

---

## Features

- Real-time hand tracking with on-screen landmark visualization
- Gesture stabilization (majority vote over recent frames) for smoother behavior
- Gesture-based cursor movement, click, and scrolling
- Activation/pause gestures so control is not always live
- Data collection + model training scripts included

---

## Project Structure

- `main.py`  
   Primary application: stable gesture-based mouse control.

- `main_control.py`  
   Validation/diagnostic controller for checking gesture predictions and logic.

- `data_collector.py`  
   Captures normalized hand landmarks and appends labeled samples to `hand_data.csv`.

- `train_model.py`  
   Trains classifier + label encoder and saves:
   - `gesture_model.pkl`
   - `label_encoder.pkl`

- `mouse.py`  
   Alternate gesture control script with swipe left/right key actions.

- `volume.py`  
   Simple hand-gesture volume control demo.

- `hand_landmarker.task`  
   MediaPipe model file used for hand landmark detection.

---

## Requirements

- Python 3.9+ (recommended: 3.10 or newer)
- Webcam
- Windows/macOS/Linux (current setup in this workspace is Windows)

Install Python packages:

```bash
pip install opencv-python mediapipe numpy pandas scikit-learn joblib pyautogui
```

---

## Setup (Windows PowerShell)

1. Create virtual environment

```powershell
python -m venv .venv
```

2. Activate environment

```powershell
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies

```powershell
pip install opencv-python mediapipe numpy pandas scikit-learn joblib pyautogui
```

4. Ensure files exist in project root:
- `hand_landmarker.task`
- `gesture_model.pkl` and `label_encoder.pkl` (generate with training pipeline below if missing)

---

## Quick Start

Run the main controller:

```bash
python main.py
```

Press `q` in the OpenCV window to quit.

---

## Gesture Mapping

### `main.py` (primary app)

- `0` → Open Palm → **Activate control**
- `1` → Pinch → **Click**
- `2` → Index finger → **Move cursor**
- `3` → Fist → **Pause control**
- `4` → Two fingers → **Scroll up**
- `5` → Three fingers → **Scroll down**

### `main_control.py` (diagnostic variant)

This script uses a different mapping in its action logic:

- `0` Enable control
- `1` Disable control
- `2` Move cursor
- `3` Click
- `4` Scroll
- `5` Left arrow key
- `6` Right arrow key

---

## Train Your Own Gesture Model

If model files are missing, outdated, or you want better accuracy for your hand/camera:

1. **Collect data per gesture label**
    - Open `data_collector.py`
    - Set `LABEL_ID` for the gesture you are recording
    - Run:

    ```bash
    python data_collector.py
    ```

    - Press `s` to save a frame sample
    - Press `q` to stop
    - Repeat for each gesture label

2. **Train classifier**

    ```bash
    python train_model.py
    ```

    This prints a test report and saves:
    - `gesture_model.pkl`
    - `label_encoder.pkl`

3. **Run controller**

    ```bash
    python main.py
    ```

---

## Last Recorded Training Report

From current `train_model.py` output notes:
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
- Accuracy: `0.95` on 75 test samples
- Macro Avg: Precision `0.96` | Recall `0.95` | F1 `0.95`
- Weighted Avg: Precision `0.95` | Recall `0.95` | F1 `0.95`

---

## Troubleshooting

- **Camera not opening**
   - Close apps using webcam and rerun script.

- **`gesture_model.pkl` / `label_encoder.pkl` missing**
   - Run `python train_model.py` after collecting data.

- **`hand_landmarker.task` missing**
   - Keep `hand_landmarker.task` in root, or run `data_collector.py`/`mouse.py` once (they auto-download if absent).

- **Mouse movement too jittery**
   - Tune smoothing values in `main.py` (`SMOOTHING`, `CURSOR_DEADZONE`, `GESTURE_WINDOW`).

- **Accidental OS actions while testing**
   - Use pause gesture quickly (`3` in `main.py`) and keep one hand visible.

---

## Notes

- `main.py` is the recommended entry point for day-to-day usage.
- `main_control.py`, `mouse.py`, and `volume.py` are useful for testing or alternative behavior.
