import pickle
import numpy as np
from sklearn.model_selection import train_test_split

# Load preprocessed dataset
with open("processed_dataset.pkl", "rb") as f:
    image_data, labels, class_labels = pickle.load(f)

# Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(image_data, labels, test_size=0.2, 
random_state=42)

# Save the split datasets
with open("train_test_split.pkl", "wb") as f:
    pickle.dump((X_train, X_test, y_train, y_test, class_labels), f)

print(f"✅ Dataset split complete!")
print(f"🔹 Training set: {len(X_train)} images")
print(f"🔹 Testing set: {len(X_test)} images")

