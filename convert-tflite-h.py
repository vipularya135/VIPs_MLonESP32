import tensorflow as tf

# Load the TFLite model from file
with open("best_model.tflite", "rb") as f:
    tflite_model = f.read()

# Function to convert TFLite model to C header file format
def hex_to_c_array(model, var_name):
    hex_array = ', '.join(f'0x{b:02X}' for b in model)
    c_array = f"const unsigned char {var_name}[] = {{\n{hex_array}\n}};"
    return c_array

# Write the TFLite model to a header file
header_file_name = "best_model.h"
with open(header_file_name, "w") as file:
    file.write(hex_to_c_array(tflite_model, "Model"))
