import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

data = []
label = "paper"  # Hoặc "paper", "scissors", thay đổi theo từng đợt ghi

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            keypoints = []
            for lm in handLms.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
            keypoints.append(label)
            data.append(keypoints)

            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Capture", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Lưu ra CSV
df = pd.DataFrame(data)
df.to_csv(f"hand_data_{label}.csv", index=False)
