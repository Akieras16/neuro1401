import cv2 as cv2
import numpy as np
from numpy import random
from matplotlib import pyplot as plt

def import_image_as_greyscale(fp):
    image = cv.imread(fp)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

testim = import_image_as_greyscale('data/monkey.jpg')
plt.imshow(testim, cmap='gray')