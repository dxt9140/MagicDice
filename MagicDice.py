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
	fixed = cv2.medianBlur(fixed, 5)

	_, fixed = cv2.threshold(fixed, 130, 255, cv2.THRESH_BINARY)

	fixed = cv2.erode(fixed, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
	fixed = cv2.dilate(fixed, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))

	stuff = cv2.connectedComponentsWithStats(fixed, connectivity=4, ltype=cv2.CV_32S)
	num_labels = stuff[0]
	labels = stuff[1]
	stats = stuff[2]
	centroids = stuff[3]

	_, dots = cv2.threshold(fixed, 0, 255, cv2.THRESH_BINARY_INV)
	dots_stuff = cv2.connectedComponentsWithStats(dots, connectivity=4, ltype=cv2.CV_32S)
	dots_num_labels = dots_stuff[0]
	dots_labels = dots_stuff[1]
	dots_stats = dots_stuff[2]
	dots_centroids = dots_stuff[3]

	dots = cv2.dilate(dots, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
	dots = cv2.erode(dots, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

	# Draws the centroids of what is detected as the dice
	for cent in centroids:
		_ = _
		# cv2.rectangle(image, (int(cent[0]), int(cent[1])), (int(cent[0]), int(cent[1])), (0, 0, 255), 3)

	for dots_cent in dots_centroids:
		_ = _
		cv2.circle(image, (int(dots_cent[0]), int(dots_cent[1])), 4, (255, 0, 0), 2)

	copy = fixed.copy()
	dots_copy = dots.copy()
	_, contours, _ = cv2.findContours(copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	_, numbers, _ = cv2.findContours(dots_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	boxes = list()

	for cnt in contours:
		# approx = cv2.approxPolyDP(cnt, 0.01, True)
		rect = cv2.minAreaRect(cnt)
		box = cv2.boxPoints(rect)
		box = np.int0(box)
		boxes.append(box)
		image = cv2.drawContours(image, [box], -1, (0, 127, 255), 2)

	# image = cv2.drawContours(image, contours, 1, (0, 127, 255), 2)
	# image = cv2.drawContours(image, numbers, -1, (0, 255, 127), 2)

	dot_counts = dict()

	# Ignore index 0, it is the background
	for i in range(1, len(boxes)):
		box = boxes[i]
		minx = np.argmin(box, axis=0)[0]
		maxx = np.argmax(box, axis=0)[0]
		miny = np.argmin(box, axis=1)[0]
		maxy = np.argmax(box, axis=1)[0]

		for dot in dots_centroids:
			if ( dot[0] < box[minx][0] or dot[0] < box[maxx][0] ):
				continue
			if ( dot[1] < box[miny][1] or dot[1] > box[maxy][1] ):
				continue

			# Otherwise...
			print( dot, box )
			if not box.tostring() in dot_counts.keys():
				dot_counts[box.tostring()] = 1
			else:
				dot_counts[box.tostring()] += 1

	counts = np.array(np.zeros(6))
	for key in dot_counts.keys():
		counts[dot_counts[key]-1] += 1

	print(counts)

	fixed = cv2.cvtColor(fixed, cv2.COLOR_GRAY2BGR)
	dots = cv2.cvtColor(dots, cv2.COLOR_GRAY2BGR)

	# Print some stuff

	# both = np.hstack((image, fixed, dots))
	both = np.hstack((image, fixed))
	cv2.imshow("Magic Dice", both)
	cv2.waitKey(0)

	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()

