import os
import matplotlib.pyplot as plt
import cv2
import random

# Path to the dataset folder
dataset_path = "dataset"

# Get the list of animal classes
classes = [cls for cls in os.listdir(dataset_path) if 
os.path.isdir(os.path.join(dataset_path, cls))]

# Select 3 random classes to visualize
random_classes = random.sample(classes, 3)

# Create a figure
plt.figure(figsize=(10, 5))

for i, cls in enumerate(random_classes):
    class_path = os.path.join(dataset_path, cls)
    images = [img for img in os.listdir(class_path) if img.lower().endswith(('.jpg', 
'.png', '.jpeg'))]

    # Select a random image from the class
    img_name = random.choice(images)
    img_path = os.path.join(class_path, img_name)

    # Read and display the image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    plt.subplot(1, 3, i+1)
    plt.imshow(img)
    plt.title(cls)
    plt.axis("off")

plt.show()

