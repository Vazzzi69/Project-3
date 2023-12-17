# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 12:31:12 2023

@author: Vasi
"""

import cv2
import numpy as np

# Reading the Image

image = cv2.imread('motherboard_image.JPEG', cv2.IMREAD_COLOR)


# Convert Image to Grey Scale

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)                

# Apply Binary Threshold

_, thresholded = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

# Image Colour and Light Processing

blurred = cv2.GaussianBlur(thresholded, (101, 101), 0)

lighting_corrected = cv2.addWeighted(thresholded, 1.5, blurred, -0.5, 0)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

equalized = clahe.apply(lighting_corrected)

smooth_blurred = cv2.GaussianBlur(equalized, (5, 5), 0.5)

bilateral_filtered = cv2.bilateralFilter(smooth_blurred, d=9, sigmaColor=75, sigmaSpace=75)

# Edge detection

edges = cv2.Canny(bilateral_filtered , 50, 150)

kernel = np.ones((2,2), np.uint8)
edges_morphed = cv2.dilate(edges, kernel, iterations=1)  #help the contours more pronunced 

# Find contours and filter them out

contours, _ = cv2.findContours(edges_morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


# Thresholds

threshold_area = 1000
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > threshold_area]

# Binary Masking 

mask = np.zeros_like(gray)

cv2.fillPoly(mask, filtered_contours, 255)

final_mask = cv2.dilate(mask, kernel, iterations=1) # apply the Dialated edges


# Final Image Extraction 

result = cv2.bitwise_and(image, image, mask=final_mask)

# Display the results

cv2.imwrite('binary_threshold.png', thresholded)
cv2.imwrite('bilateral_filtered.png', bilateral_filtered)
cv2.imwrite('Edges.png', edges_morphed)
cv2.imwrite('Mask.png', final_mask)
cv2.imwrite('Result.png', result)

