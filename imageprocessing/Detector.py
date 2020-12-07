import cv2
import imutils
import numpy as np
import os
import pytesseract
import re

from PIL import Image
from .Filterer import Filterer
from .Segmenter import Segmenter

cvImage = np.ndarray


class Detector:
    def __init__(self):
        self.regex: re.Pattern = re.compile("[^a-zA-Z ]")

    def readTextOnCard(self, image: cvImage) -> str:
        image = Filterer.equalizeHistogram(image, cv2.COLOR_BGR2LAB, cv2.COLOR_LAB2BGR, (True, False, False))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        boxes: list = self.__getTextBoundingBoxCandidates(image, True)

        text: str = ""
        (rows, cols) = image.shape[:2]
        for box in boxes:
            (x, y, w, h) = cv2.boundingRect(box)

            bufferX: int = int(w * 0.2)
            bufferY: int = int(h * 0.8)
            (x, y) = (max(0, x - bufferX), max(0, y - bufferY))
            (w, h) = (w + (bufferX * 2), h + (bufferY * 2))
            (maxX, maxY) = (min(cols - 1, x + w), min(rows - 1, y + h))

            textImage: cvImage = image[y:maxY, x:maxX]

            filename: str = "{}.png".format(os.getpid())
            cv2.imwrite(filename, textImage)
            text = pytesseract.image_to_string(Image.open(filename))
            os.remove(filename)

            # Remove excess whitespace and non-alphanumeric characters
            text = (self.regex.sub("", " ".join(text.split()))).upper()
            if text:
                break

        return text

    def __getTextBoundingBoxCandidates(self, image: cvImage, useGradient: bool) -> list:
        # IDEAS
        # - weight areas farther from the expected word to be darker
        # --- side: int = int((5. / 43.) * cols)
        # --- width: int = int((33. / 43.) * cols)
        # --- bottom: int = int((5. / 28.) * rows)
        # --- height: int = int((8. / 28.) * rows)
        # - cv2.Canny(image, 100, 200)

        # TODO: tune rectangle size
        kernel: cvImage = cv2.getStructuringElement(cv2.MORPH_RECT, (32, 8))
        image = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)

        # Mask off side border to remove card edges
        (rows, cols) = image.shape[:2]
        mask: cvImage = np.zeros((rows, cols), dtype="uint8")
        mask[int(0.1 * rows):rows, int(0.1 * cols):int(0.9 * cols)] = 255
        image = cv2.bitwise_and(image, mask)

        if useGradient:
            # Accentuate letter edges (areas with high x-gradient)
            image = cv2.Sobel(image, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
            image = np.absolute(image)
            (minVal, maxVal) = (np.min(image), np.max(image))
            image = (255 * ((image - minVal) / (maxVal - minVal))).astype("uint8")
        else:
            (image, mask) = Segmenter.threshold(image, image, "otsu")

        # Fill box around letters
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        (_, image) = Segmenter.threshold(image, image, "otsu")
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

        # Find contours of box
        boxes: list = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = imutils.grab_contours(boxes)
        boxes = sorted(boxes, key=cv2.contourArea, reverse=True)

        return boxes
