import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
	rgb = cv.imread('rgb_img.png')
	# cv.imshow('rgb', rgb)
	# cv.waitKey()
	
	mask = cv.imread('mask.png')

	print(np.unique(mask))

	plt.imshow(mask)
	plt.show()