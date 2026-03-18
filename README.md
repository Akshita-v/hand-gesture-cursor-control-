# Gesture Mouse Controller (MediaPipe Tasks API)

Control your mouse and basic desktop actions using real-time hand gestures detected from webcam input.

This project combines:
- **MediaPipe Hand Landmarker (Tasks API)** for 21 hand keypoints
- **Custom gesture classifier** (`RandomForest`) trained on normalized landmark data
- **PyAutoGUI** for system actions (move, click, scroll, key press)

In short: your webcam captures hand landmarks, the model predicts the gesture label, and the app maps that label to a mouse or keyboard action.

---

## Features

- Real-time hand tracking with on-screen landmark visualization
- Gesture stabilization (majority vote over recent frames) for smoother behavior
- Gesture-based cursor movement, click, and scrolling
- Activation/pause gestures so control is not always live
- Data collection + model training scripts included

---

## First-Time Workflow (Recommended)

If you are running this project for the first time, follow this order:

1. Set up the environment and install dependencies.
2. (Optional) Collect your own gesture data using `data_collector.py`.
3. Train the model with `train_model.py`.
4. Run the app with `main.py`.

If model files already exist and work well, you can directly run `main.py`.

---

## Project Structure

- `main.py`  
   Primary application for stable gesture-based mouse control.

- `main_control.py`  
   Diagnostic/validation controller for testing gesture predictions.

- `data_collector.py`  
   Collects normalized landmark samples and appends them to `hand_data.csv`.

- `train_model.py`  
   Trains the gesture classifier and saves:
   - `gesture_model.pkl`
   - `label_encoder.pkl`

- `mouse.py`  
   Alternate controller with swipe-based left/right key actions.

- `volume.py`  
   Standalone hand-gesture volume control demo.

- `hand_landmarker.task`  
   MediaPipe model file used to detect hand landmarks.

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

Tip: keep all scripts and model files in the same project root directory unless you update file paths in code.

---

## Quick Start

Run the main controller:

```bash
python main.py
```

Press `q` in the OpenCV window to quit.

Safety note: because PyAutoGUI controls your system cursor, test gestures slowly first and use the pause gesture when needed.

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

Use `main.py` for everyday use, and `main_control.py` mainly for checking whether labels/actions are being recognized as expected.

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
   - Capture samples in different lighting and hand angles for better robustness
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

- **Low prediction accuracy**
  - Re-collect balanced data for each label and retrain.
  - Avoid mixing similar-looking gestures without enough samples.

---

## Notes

- `main.py` is the recommended entry point for day-to-day usage.
- `main_control.py`, `mouse.py`, and `volume.py` are useful for testing or alternative behavior.
