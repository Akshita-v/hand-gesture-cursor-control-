# Gesture Mouse Controller (MediaPipe Tasks API)

This project uses hand landmarks + a trained ML classifier to control the mouse and basic actions with gestures.

## Main Files
- `main.py` - Main real-time gesture mouse controller.
- `main_control.py` - **Validation/diagnostic script** used to check whether the model is detecting gestures correctly.
- `data_collector.py` - Collects normalized landmark samples and appends them to `hand_data.csv`.
- `train_model.py` - Trains the gesture model and saves `gesture_model.pkl` + `label_encoder.pkl`.

## Gesture Map (main.py)
- `0` = Palm -> Activate control
- `1` = Pinch -> Click
- `2` = Index -> Move cursor
- `3` = Fist -> Pause control
- `4` = Two fingers -> Scroll up
- `5` = Three fingers -> Scroll down

## Training Accuracy (train_model.py)
Latest reported test performance:
- Overall Accuracy: **94.67%**

Per-gesture breakdown:

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
- Accuracy: 0.95 (75 samples)
- Macro Avg: Precision 0.96 | Recall 0.95 | F1 0.95
- Weighted Avg: Precision 0.95 | Recall 0.95 | F1 0.95

## Setup
1. Create and activate virtual environment.
2. Install dependencies:
   ```bash
   pip install opencv-python mediapipe numpy pandas scikit-learn joblib pyautogui
   ```
3. Run the app:
   ```bash
   python main.py
   ```

## Notes for GitHub
- `main_control.py` is for detection verification/debugging, not the primary user app.
- `.gitignore` excludes local data/model artifacts, virtual environment files, and requested scripts (`volume.py`, `mouse.py`).
