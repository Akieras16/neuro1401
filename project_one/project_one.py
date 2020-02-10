import cv2 as cv2
import numpy as np
from numpy import random
from matplotlib import pyplot as plt

def import_image_as_greyscale(fp):
    image = cv2.imread(fp)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def normalize(im, gamma, sigma):
    return gamma *(im**2 /(sigma**2 + np.sum(im**2)))

testim = import_image_as_greyscale('data/monkey.jpg')
#cv2.imshow(testim)
plt.imshow(testim, cmap='gray')
plt.show()

im2 = normalize(testim, 100000, .0001)
plt.imshow(im2)
im2.shape
im2[0, 0, 0]
print("The max value in the normalized image is {0}\n".format(np.max(im2))

tmbw = cv2.cvtColor(testim, cv2.COLOR_BGR2GRAY)
plt.imshow(tmbw)

tmb2 = normalize(tmbw, 100000, .0001)
plt.imshow(tmb2)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = np.linspace(0, testim.shape[0] - 1, testim.shape[0])
y = np.linspace(0, testim.shape[1] - 1, testim.shape[1])

z1 = cv2.cvtColor(testim, cv2.COLOR_BGR2GRAY)

xv, yv = np.meshgrid(x, y)
ax.scatter(xv, yv, z1)

plt.imshow(z1)

fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
ax2.scatter(xv, yv, tmb2)
