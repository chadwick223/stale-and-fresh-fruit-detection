import cv2
import numpy as np

def preprocess_image(image_path, target_size=(224, 224)):
    # Load the image
    image = cv2.imread(image_path)
    
    # Resize the image using bilinear interpolation
    image_resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    
    # Convert to HSV color space (better for color-based differentiation)
    image_hsv = cv2.cvtColor(image_resized, cv2.COLOR_BGR2HSV)
    
    # Apply Histogram Equalization to the V (value) channel for better contrast
    h, s, v = cv2.split(image_hsv)
    v_eq = cv2.equalizeHist(v)
    image_hsv_eq = cv2.merge((h, s, v_eq))
    
    # Convert back to BGR after equalization
    image_processed = cv2.cvtColor(image_hsv_eq, cv2.COLOR_HSV2BGR)
    
    # Normalize pixel values
    image_normalized = image_processed / 255.0
    
    return image_normalized

def augment_image(image):
    # Random rotation and flipping
    rotation_angle = np.random.uniform(-20, 20)
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    
    image_rotated = cv2.warpAffine(image, rotation_matrix, (w, h))
    
    # Random horizontal flip
    if np.random.rand() > 0.5:
        image_rotated = cv2.flip(image_rotated, 1)
    
    return image_rotated
