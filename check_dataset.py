import os

# Path to the dataset folder
dataset_path = "/Users/aakhilmohamed/intpro/Animal_Identification/dataset"

# Check if the dataset folder exists
if not os.path.exists(dataset_path):
    print("❌ Dataset folder not found! Make sure 'dataset' exists inside 'Animal_Identification'.")

    exit()

# List all folders inside 'dataset'
classes = [cls for cls in os.listdir(dataset_path) if 
os.path.isdir(os.path.join(dataset_path, cls))]

# Print the number of classes
print(f"✅ Number of animal classes: {len(classes)}")
print("Classes:", classes)

# Check number of images in each class
image_counts = {}
for cls in classes:
    class_path = os.path.join(dataset_path, cls)
    images = [img for img in os.listdir(class_path) if img.lower().endswith(('.jpg', 
'.png', '.jpeg'))]
    image_counts[cls] = len(images)

print("📸 Image count per class:", image_counts)

