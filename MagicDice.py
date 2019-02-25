"""
MagicDice.py

Author: Dominick Taylor
Created: February 2019

Computer Vision project used for identifying and counting dice given an input image.
"""

import os
import cv2
import numpy as np
import argparse
from scipy import signal


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


def preprocess(image):
    # Preprocess  the image to simplify it and reduce noise
    worker = image
    worker = cv2.cvtColor(worker, cv2.COLOR_BGR2GRAY)
    worker = cv2.medianBlur(worker, 9)
    worker = cv2.GaussianBlur(worker, (5, 5), 0)

    return worker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to the image to read")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode.")
    argl = parser.parse_args()

    if argl.debug:
        print("Debug Mode enabled!")

    cv2.namedWindow("Magic Dice", cv2.WINDOW_NORMAL)

    image = cv2.imread(argl.image)
    image = cv2.resize(image, (1000, 1000))

    fixed = preprocess(image)
    fixed = signal.convolve2d(fixed, dice_like)

    cv2.imshow("Magic Dice", fixed)
    cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

