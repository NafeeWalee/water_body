import cv2
import os
import numpy as np

DIR = rf"{os.path.dirname(__file__)}"

# Load the images
mask_image_path = rf"{DIR}\datasets\Mask\water_body_1.jpg"
original_image_path = rf"{DIR}\datasets\noMask\water_body_1.jpg"

# Read the images
mask_img = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)
original_img = cv2.imread(original_image_path)

# Ensure the mask is binary (0 and 255)
_, mask_img = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)

# Create an empty image for the result with the same shape as the original
result_img = np.zeros_like(original_img)

# Use the mask to copy pixels from the original image to the result image
result_img[mask_img == 255] = original_img[mask_img == 255]

# Save the result image
result_image_path = 'path_to_result_image.png'
cv2.imwrite(result_image_path, result_img)


def extract_unique_colors(image_path):
    # Load the image
    img = cv2.imread(image_path)
    # Reshape the image to a list of pixels
    pixels = img.reshape(-1, 3)
    # Find the unique colors
    unique_colors = np.unique(pixels, axis=0)
    # Display the number of unique colors
    print(f"Total number of unique colors: {len(unique_colors)}")
    return unique_colors

def color_distance(c1, c2):
    return np.sqrt(np.sum((c1 - c2) ** 2))

def find_similar_colors(target_image_path, reference_colors, threshold=50):
    # Load the target image
    target_img = cv2.imread(target_image_path)
    # Copy the target image to highlight similar colors
    highlighted_img = target_img.copy()
    # Reshape the target image to a list of pixels
    pixels = target_img.reshape(-1, 3)
    # Initialize a mask for similar colors
    mask = np.zeros((pixels.shape[0],), dtype=bool)
    # Calculate distances and find similar colors
    for ref_color in reference_colors:
        distances = np.linalg.norm(pixels - ref_color, axis=1)
        mask = mask | (distances < threshold)
    # Highlight similar colors in the image (e.g., setting to white)
    highlighted_img = highlighted_img.reshape(-1, 3)
    highlighted_img[mask] = [255, 255, 255]  # Highlight similar pixels
    highlighted_img = highlighted_img.reshape(target_img.shape)
    return highlighted_img

# Extract unique colors from the reference image
reference_colors = extract_unique_colors(result_image_path)

# Find similar colors in a target image
target_image_path = rf"{DIR}\datasets\Mask\water_body_2.jpg"
highlighted_image = find_similar_colors(target_image_path, reference_colors, threshold=50)

# Save and display the highlighted image
highlighted_image_path = 'path_to_highlighted_image.png'
cv2.imwrite(highlighted_image_path, highlighted_image)

cv2.imshow('Highlighted Image', highlighted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()