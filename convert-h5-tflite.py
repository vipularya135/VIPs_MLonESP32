import tensorflow as tf

# Load the Keras .h5 model
model = tf.keras.models.load_model("C:/Users/DELL/Desktop/iot-project/best_model.h5")

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the converted model to a .tflite file
with open("C:/Users/DELL/Desktop/iot-project/best_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Model has been successfully converted to TensorFlow Lite format and saved as best_model.tflite.")
