import os
import pickle
import numpy as np
from PIL import Image

# Path to extracted CIFAR folder
DATA_PATH = "cifar-10-batches-py"
OUTPUT_PATH = "dataset/train"

# Class labels
label_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Create class folders
for label in label_names:
    os.makedirs(os.path.join(OUTPUT_PATH, label), exist_ok=True)

def load_batch(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
    return dict

img_count = 0

# Process training batches
for i in range(1, 6):
    batch = load_batch(f"{DATA_PATH}/data_batch_{i}")

    data = batch[b'data']
    labels = batch[b'labels']

    for img, label in zip(data, labels):
        img = img.reshape(3, 32, 32)
        img = np.transpose(img, (1, 2, 0))  # CHW → HWC

        class_name = label_names[label]
        img_path = os.path.join(OUTPUT_PATH, class_name, f"{img_count}.png")

        Image.fromarray(img).save(img_path)
        img_count += 1

        # 🔥 LIMIT DATA FOR CI
        if img_count >= 1000:
            break
    if img_count >= 1000:
        break

print("Done. Images saved:", img_count)