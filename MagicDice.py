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

sobel = [[-1, -2, -1],
		 [ 0,  0,  0],
		 [ 1,  2,  1]]


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
	cv2.resizeWindow("Magic Dice", 1000, 1000)

	# Preprocess the image to obtain desired dimensions
	image = cv2.imread(argl.image)
	image = cv2.resize(image, (1200, 1200))
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	fixed = image
	# Apply a series of transformations to enhance the image
	fixed = cv2.GaussianBlur(fixed, (3, 3), 0)
	# Find the tiny little holes and make them bigger
	fixed = cv2.erode( fixed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) )
	# Further expand the holes
	fixed = cv2.erode( fixed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) )

	fixed = cv2.dilate( fixed, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)) )

	#fixed = cv2.Canny(fixed, 50, 120)

	#sobelx = np.array(sobel)
	#sobely = np.transpose(sobelx)
	#fixed = signal.convolve2d(fixed, sobelx[::-1,::-1])

	#_, fixed = cv2.threshold(fixed, 80, 255, cv2.THRESH_BINARY)
	# fixed = cv2.Canny(fixed, 50, 120)
	# fixed = cv2.applyColorMap(fixed, cv2.COLORMAP_JET)

	both = np.hstack((image, fixed))
	cv2.imshow("Magic Dice", both)
	cv2.waitKey(0)

	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()

