import numpy as np
import joblib
import tensorflow as tf
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import save_model

# Load the pre-trained XGBoost model
model_path = r"C:\Users\DELL\Desktop\iot-project\best_model.joblib"
xgb_model = joblib.load(model_path)

# Extract number of features and classes for creating a compatible Keras model
n_features = xgb_model.get_booster().num_features()
n_classes = len(np.unique(xgb_model.classes_))

# Convert XGBoost model to a format that can be used with Keras
# 1. Initialize a Keras model
keras_model = Sequential()

# 2. Add layers equivalent to the XGBoost model structure (this example uses a simple dense network)
keras_model.add(Dense(128, input_shape=(n_features,), activation='relu'))
keras_model.add(Dense(64, activation='relu'))
keras_model.add(Dense(n_classes, activation='softmax'))

# 3. Compile the model to set the loss, optimizer, etc.
keras_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 4. Transfer weights (this step will approximate the weights, as exact direct transfer isn't supported)
# If you have data, use it here to train the keras_model with similar results as xgb_model
# Otherwise, this is a sample skeleton for setup

# Save the model in .h5 format
h5_path = r"C:\Users\DELL\Desktop\iot-project\best_model.h5"
keras_model.save(h5_path)

print(f"Keras model saved as .h5 at: {h5_path}")
