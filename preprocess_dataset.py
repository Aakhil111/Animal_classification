import os
import cv2
import numpy as np
import pickle

# Path to the dataset folder
dataset_path = "dataset"

# Define image size (128x128 pixels)
IMAGE_SIZE = (128, 128)

# Get the list of classes
classes = [cls for cls in os.listdir(dataset_path) if 
os.path.isdir(os.path.join(dataset_path, cls))]
class_labels = {cls: i for i, cls in enumerate(classes)}  # Assign numeric labels

# Lists to store image data and labels
image_data = []
labels = []

# Process each class
for cls in classes:
    class_path = os.path.join(dataset_path, cls)
    images = [img for img in os.listdir(class_path) if img.lower().endswith(('.jpg', 
'.png', '.jpeg'))]

    for img_name in images:
        img_path = os.path.join(class_path, img_name)

        # Load image
        img = cv2.imread(img_path)
        img = cv2.resize(img, IMAGE_SIZE)  # Resize
        img = img / 255.0  # Normalize pixel values (0-1)

        image_data.append(img)
        labels.append(class_labels[cls])

# Convert to NumPy arrays
image_data = np.array(image_data, dtype=np.float32)
labels = np.array(labels, dtype=np.int32)

# Save the processed data
with open("processed_dataset.pkl", "wb") as f:
    pickle.dump((image_data, labels, class_labels), f)

print("✅ Dataset preprocessing complete! Processed data saved as 'processed_dataset.pkl'.")


