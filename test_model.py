import joblib
import cv2
import mediapipe as mp
import numpy as np

model = joblib.load("rf_model.pkl")  # nếu bạn đã lưu mô hình
# hoặc dùng model từ bước 4

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
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
            if len(keypoints) == 63:
                prediction = model.predict([keypoints])[0]
                cv2.putText(img, prediction, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3)

            mp.solutions.drawing_utils.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Prediction", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# joblib.dump(model, "rf_model.pkl")
