import cv2
import numpy as np
from matplotlib import pyplot as plt




def function(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # height and width of image
    height, width = gray.shape
    # use sobel
    Ix = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    Iy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    Ixx = Ix ** 2
    Iyy = Iy ** 2
    Ixy = Iy * Ix
    # harris
    k = 0.03
    window_size = 6
    offset = int(window_size / 2)
    # make an array for harris output and initialize it to zero
    harris_array = np.zeros((height, width));

    #considering formula...
    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            Sxx = np.sum(Ixx[y - offset:y + 1 + offset, x - offset:x + 1 + offset])
            Syy = np.sum(Iyy[y - offset:y + 1 + offset, x - offset:x + 1 + offset])
            Sxy = np.sum(Ixy[y - offset:y + 1 + offset, x - offset:x + 1 + offset])

            # determinant
            det = (Sxx * Syy) - (Sxy ** 2)
            trace = Sxx + Syy
            R = det - k * (trace ** 2)
            harris_array[y, x] = R
    #maximum value
    maximum_val = harris_array.max()
    limitation = 0.1 * maximum_val
    #highlights maximum values only
    img[harris_array > limitation] = [0, 255, 255]
    return  img
# image
img = cv2.imread('right04.jpg')
img =function(img)
cv2.imwrite("rightOutput.jpg " , img)
print("harris output for right04 image")
plt.imshow(img)
plt.show()

img = cv2.imread('left04.jpg')
img =function(img)
cv2.imwrite("leftOutput.jpg " , img)
print("harris output for left04 image")
plt.imshow(img)
plt.show()