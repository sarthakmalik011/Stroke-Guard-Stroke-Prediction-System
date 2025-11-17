# ============================================================
# 🧠 Stroke Prediction – Model Evaluation
# Includes:
#  - Kaggle dataset download
#  - Model training
#  - Accuracy/loss curves, confusion matrix, classification report
# ============================================================

# ------------------------------------------------------------
# 1️⃣ SETUP KAGGLE ACCESS (run once)
# ------------------------------------------------------------

uploaded = files.upload()  # Upload your kaggle.json here

!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# ------------------------------------------------------------
# 2️⃣ DOWNLOAD DATASET FROM KAGGLE
# ------------------------------------------------------------
!kaggle datasets download -d fedesoriano/stroke-prediction-dataset -p /content/
!unzip -o /content/stroke-prediction-dataset.zip -d /content/

# ------------------------------------------------------------
# 3️⃣ IMPORT LIBRARIES
# ------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import History

# ------------------------------------------------------------
# 4️⃣ LOAD AND PREPROCESS DATA
# ------------------------------------------------------------
data = pd.read_csv("/content/healthcare-dataset-stroke-data.csv")

# Handle missing BMI values
data["bmi"].fillna(data["bmi"].median(), inplace=True)

# Encode categorical columns
categorical = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
le = LabelEncoder()
for col in categorical:
    data[col] = le.fit_transform(data[col])

# Split features and label
X = data.drop(['id', 'stroke'], axis=1)
y = data['stroke']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# ------------------------------------------------------------
# 5️⃣ BUILD SIMPLE NEURAL NETWORK
# ------------------------------------------------------------
model = Sequential([
    Dense(32, input_dim=X_train.shape[1], activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# ------------------------------------------------------------
# 6️⃣ TRAIN MODEL + CAPTURE HISTORY
# ------------------------------------------------------------
history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                    epochs=30, batch_size=32, verbose=1)


# ------------------------------------------------------------
# 7️⃣ PLOT TRAINING & VALIDATION ACCURACY / LOSS CURVES
# ------------------------------------------------------------
plt.figure(figsize=(12,5))

# Accuracy curve
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
plt.title('Training & Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)


# Loss curve
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
plt.title('Training & Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.show()

