import cv2
import numpy as np
import matplotlib.pyplot as plt


def DetectImageEdges(image_obj, threshold_val1, threshold_val2):

    # Check if the image was loaded successfully
    if image_obj is None:
        print("Error: Image not found or unable to load.")
        return

    # Convert the image to grayscale
    gray = cv2.cvtColor(image_obj, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    img_edges = cv2.Canny(blurred, threshold1=threshold_val1, threshold2=threshold_val2)

    return img_edges


# Load the image
image_path1 = 'Img1.png'
image_path2 = 'Img4.png'
img1 = cv2.imread(image_path1)
img2 = cv2.imread(image_path2)

edges1 = DetectImageEdges(img1, 50, 150)
edges2 = DetectImageEdges(img2, 50, 640)

# Display the original and edge-detected images
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
plt.title("Original Image 1")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(edges1, cmap='gray')
plt.title("Edge Detection 1")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
plt.title("Original Image 2")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(edges2, cmap='gray')
plt.title("Edge Detection 2")
plt.axis("off")

plt.show()
