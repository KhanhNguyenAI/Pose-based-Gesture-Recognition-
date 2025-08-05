from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd
import joblib
data = pd.read_csv("hand_gesture_data.csv")
X = data.iloc[:, :-1]  # tất cả cột trừ cột cuối
y = data.iloc[:, -1]   # nhãn

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

joblib.dump(model, "rf_model.pkl")
