import tensorflow as tf
import numpy as np
import cv2
import pickle
import sys

# Load the trained model
model = tf.keras.models.load_model("animal_classifier.h5")

# Load class labels
with open("train_test_split.pkl", "rb") as f:
    class_labels = {0: 'Cat', 1: 'Dog', 2: 'Dolphin', 3: 'Giraffe', 4: 'Bear',
                    5: 'Zebra', 6: 'Panda', 7: 'Tiger', 8: 'Bird', 9: 'Kangaroo',
                    10: 'Horse', 11: 'Cow', 12: 'Deer', 13: 'Lion', 14: 'Elephant'}

print("Class Labels:", class_labels)


# Function to preprocess the image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Error: Could not read image. Make sure the path is correct!")
        sys.exit(1)
    
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Get image path from command-line argument
if len(sys.argv) < 2:
    print("❌ Error: Please provide an image path!")
    print("Usage: python3 test_model.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]

# Preprocess and predict
image = preprocess_image(image_path)
prediction = model.predict(image)
predicted_class = class_labels[np.argmax(prediction)]

# Display the result
print(f"✅ Predicted Animal: {predicted_class}")
print("Raw Prediction Output:", prediction)


