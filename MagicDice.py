"""
MagicDice.py

Author: Dominick Taylor
Created: February 2019

Computer Vision project used for identifying and counting dice given an input image.
"""

import os
import sys
sys.path.append("/media/removable/NightBox/CS451/MagicDice/MagicDice/")

import cv2
import numpy as np
import argparse
from scipy import signal
from paths import DICE

dice_like = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
             [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
             [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
             [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
             [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
             [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
             [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
             [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

sobel = [[-1, -2, -1], [0,  0,  0],  [1,  2,  1]]


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("image", help="Path to the image to read")
	parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode.")
	argl = parser.parse_args()

	if DICE is not None:
		argl.image = DICE + argl.image

	if not os.path.exists(argl.image):
		print("File not found.")
		sys.exit()

	print("IMAGE PATH: " + argl.image)

	cv2.namedWindow("Magic Dice", cv2.WINDOW_NORMAL)
	cv2.resizeWindow("Magic Dice", 1000, 600)

	# Preprocess the image to obtain desired dimensions
	image = cv2.imread(argl.image)
	image = cv2.resize(image, (1000, 600))

	fixed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# Apply a series of transformations to enhance the image
	# fixed = cv2.GaussianBlur(fixed, (9, 9), 0)
	fixed = cv2.medianBlur(fixed, 5)

	_, fixed = cv2.threshold(fixed, 130, 255, cv2.THRESH_BINARY)
	# kernel = np.ones((3, 3), np.uint8)
	# fixed = cv2.adaptiveThreshold(fixed, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 15)
	# fixed = cv2.morphologyEx(fixed, cv2.MORPH_OPEN, kernel, iterations=1)
	# fixed = cv2.equalizeHist(fixed)

	stuff = cv2.connectedComponentsWithStats(fixed, connectivity=4, ltype=cv2.CV_32S)
	num_labels = stuff[0]
	labels = stuff[1]
	stats = stuff[2]
	centroids = stuff[3]

	_, dots = cv2.threshold(fixed, 0, 255, cv2.THRESH_BINARY_INV)

	for cent in centroids:
		cv2.rectangle(image, (int(cent[0]), int(cent[1])), (int(cent[0])+5, int(cent[1])+5), (0, 0, 255), 3)

	copy = fixed.copy()
	_, contours, _ = cv2.findContours(copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	huh = cv2.filter2D(fixed, 3, dots)

	image = cv2.drawContours(image, contours, -1, (0, 255, 127), 2)
	#for cnt in contours:
		# area = cv2.contourArea(cnt)

	fixed = cv2.cvtColor(fixed, cv2.COLOR_GRAY2BGR)
	dots = cv2.cvtColor(dots, cv2.COLOR_GRAY2BGR)

	both = np.hstack((image, fixed, dots, huh))
	cv2.imshow("Magic Dice", both)
	cv2.waitKey(0)

	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()

