# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 12:31:12 2023

@author: Vasi
"""

import cv2
import numpy as np

# Read the image

image = cv2.imread('motherboard_image.JPEG', cv2.IMREAD_COLOR)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)                  # Gray scale

# Gaussian blur
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

_, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Edge detection

edges = cv2.Canny(thresholded, 50, 150)

# Find contours and filter them out

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


# Thresholds

threshold_area = 1000
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > threshold_area]

# Create a binary mask
mask = np.zeros_like(gray)
cv2.fillPoly(mask, filtered_contours, 255)

# Final Image Extraction 
result = cv2.bitwise_and(image, image, mask=mask)

# Display the results

cv2.imwrite('Thresholded Image.png', thresholded)
cv2.imwrite('Edges.png', edges)
cv2.imwrite('Mask.png', mask)
cv2.imwrite('Result.png', result)

