import cv2
import imutils
import numpy as np
import os
import pytesseract

from PIL import Image

cvImage = np.ndarray


class Detector:
    def __init__(self):
        pass

    @staticmethod
    def readWordsOnCard(image: cvImage):
        # TODO: tune this
        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)

        # cv2.Canny(image, 100, 200)
        gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradX = np.absolute(gradX)
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")

        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
        thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        thresh = cv2.erode(thresh, None, iterations=5)

        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)

            pX = int((x + w) * 0.1)
            pY = int((y + h) * 0.1)
            (x, y) = (x - pX, y - pY)
            (w, h) = (w + (pX * 2), h + (pY * 2))

            roi = gray[y:y + h, x:x + w].copy()
            break

        filename = "{}.png".format(os.getpid())
        cv2.imwrite(filename, roi)
        text = pytesseract.image_to_string(Image.open(filename))
        os.remove(filename)

        # cv2.imshow("GRAY", gray)
        # cv2.imshow("BLACKHAT", blackhat)
        # cv2.imshow("THRESH", thresh)
        # cv2.imshow("ROI", roi)
        # cv2.waitKey()

        return text
