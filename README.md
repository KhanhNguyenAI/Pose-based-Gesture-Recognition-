# 🧠 Real-Time Hand Gesture Recognition using MediaPipe

This project demonstrates how to collect gesture data (e.g., Rock-Paper-Scissors), extract hand keypoints using MediaPipe, and train a machine learning model to classify gestures in real time using webcam input.

---

## 🔧 Step-by-Step Implementation

### ✅ Step 1: Environment Setup

Install required packages:

```bash
pip install mediapipe opencv-python numpy pandas scikit-learn
```

---

### ✅ Step 2: Data Collection with MediaPipe

Use your webcam to record specific hand gestures (e.g., "rock", "paper", "scissors") and extract 21 hand landmarks using MediaPipe. Each frame is labeled and saved to a CSV file.

Repeat for each gesture (change label accordingly):

```python
# Capture keypoints and label each frame as 'rock', 'paper', or 'scissors'
# Save the data as hand_data_<label>.csv
```

---

### ✅ Step 3: Combine and Preprocess Data

Merge all gesture data files into one dataset:

```python
import pandas as pd

rock = pd.read_csv("hand_data_rock.csv")
paper = pd.read_csv("hand_data_paper.csv")
scissors = pd.read_csv("hand_data_scissors.csv")

all_data = pd.concat([rock, paper, scissors])
all_data.to_csv("hand_gesture_data.csv", index=False)
```

---

### ✅ Step 4: Train a Classification Model

Train a gesture classifier using RandomForest:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X = all_data.iloc[:, :-1]
y = all_data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)
```

Save the model for later use (optional):

```python
import joblib
joblib.dump(model, "gesture_model.pkl")
```

---

### ✅ Step 5: Real-Time Gesture Detection

Use your webcam to detect hand gestures in real time:

```python
# Capture keypoints with MediaPipe in real time
# Use the trained model to predict the gesture
# Display the predicted label on the video stream
```

---

## 📦 Tools & Libraries Used

- MediaPipe (Hand landmark detection)
- OpenCV (Webcam capture & visualization)
- Pandas / NumPy (Data manipulation)
- Scikit-learn (Gesture classification model)

---
