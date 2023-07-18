import cv2
import numpy as np


def set_contrast(img):
    enhanced_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(enhanced_img)
    enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_GRAY2BGR)
    return enhanced_img


def sharpen_img(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    sharp_img = cv2.filter2D(img, -1, kernel)
    sharp_img = cv2.cvtColor(sharp_img, cv2.COLOR_GRAY2BGR)
    return sharp_img


def add_noise(img):
    stdv = np.random.uniform(0.0, 5.0)
    img = np.clip(img + stdv * np.random.randn(*img.shape), 0.0, 255.0)
    return img
