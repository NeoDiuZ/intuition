import tensorflow as tf

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="your_model.tflite")

# Allocate memory for the model
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Print input details
print("Input details:")
print(input_details)

# Print output details
print("\nOutput details:")
print(output_details)
