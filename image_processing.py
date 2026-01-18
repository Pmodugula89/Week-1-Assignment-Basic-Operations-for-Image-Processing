import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Load the image
image_path = 'images/sample.jpg'

if not os.path.exists(image_path):
    print(f"Error: Image file not found at {image_path}")
    exit()

image = cv2.imread(image_path)
if image is None:
    print("Error: Failed to load image. Check if the file is corrupted or unsupported.")
    exit()

# Convert to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display original image
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")
plt.show()

# Rotate the image by 45 degrees
(h, w) = image.shape[:2]
center = (w // 2, h // 2)
angle = 45
scale = 1.0
rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))

plt.imshow(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB))
plt.title("Rotated Image (45°)")
plt.axis("off")
plt.show()

# Scale the image by 1.5x
scaled_image = cv2.resize(image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

plt.imshow(cv2.cvtColor(scaled_image, cv2.COLOR_BGR2RGB))
plt.title("Scaled Image (1.5x)")
plt.axis("off")
plt.show()

# ---------------------------------------------------------
# Simulate different focal lengths (correct zoom-based method)
# ---------------------------------------------------------

focal_lengths = [50, 100, 200]
zoom_factors = [f / 50 for f in focal_lengths]  # relative zoom

plt.figure(figsize=(12, 4))

for i, zoom in enumerate(zoom_factors):
    # Compute new size
    new_w = int(w * zoom)
    new_h = int(h * zoom)

    # Resize (zoom)
    zoomed = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Crop or pad back to original size
    if zoom >= 1:
        # Crop center
        start_x = (new_w - w) // 2
        start_y = (new_h - h) // 2
        zoomed = zoomed[start_y:start_y+h, start_x:start_x+w]
    else:
        # Pad to original size
        pad_x = (w - new_w) // 2
        pad_y = (h - new_h) // 2
        zoomed = cv2.copyMakeBorder(
            zoomed, pad_y, pad_y, pad_x, pad_x,
            cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )

    plt.subplot(1, 3, i+1)
    plt.imshow(cv2.cvtColor(zoomed, cv2.COLOR_BGR2RGB))
    plt.title(f"Focal Length: {focal_lengths[i]}")
    plt.axis("off")

plt.show()
