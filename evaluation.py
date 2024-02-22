from util_funcs import load_image
from find_phone import find_phone
import os
from math import sqrt

labels_path = './find_phone/labels.txt'

with open(labels_path, 'r') as file:
    lines = file.readlines()

labels = []
for line in lines:
    parts = line.strip().split(' ')
    if len(parts) == 3:
        img_path, x, y = parts
        labels.append((img_path, float(x), float(y)))

image_paths = [os.path.join("./find_phone", label[0]) for label in labels]

count = 0

for label in labels:
    image_path = os.path.join("./find_phone", label[0])
    pred = find_phone(image_path)
    ground_truth = (label[1], label[2])

    loss = sqrt(pow(pred[0]-ground_truth[0], 2) + pow(pred[1]-ground_truth[1], 2))
    count += 1 if loss <= 0.05 else 0

accuracy = count / len(labels)
print("="*6, f"{accuracy*100:.0f}% of predictions are within 0.05 normalized range", "="*6)