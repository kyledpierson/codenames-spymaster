from math import floor, ceil
from typing import Tuple

import cv2
import imutils
from numpy import ndarray as Image


class Preprocessor:
    def __init__(self):
        pass

    @staticmethod
    def resizeToSameSize(image: Image, targetImage: Image, width: int, height: int) -> Tuple:
        resizedImage = imutils.resize(image, width, height)
        resizedTargetImage = imutils.resize(targetImage, width, height)

        height, width = resizedImage.shape[:2]
        targetHeight, targetWidth = resizedTargetImage.shape[:2]
        vertical = height - targetHeight
        horizontal = width - targetWidth

        top = floor(vertical / 2)
        bottom = ceil(vertical / 2)
        left = floor(horizontal / 2)
        right = ceil(horizontal / 2)
        if vertical < 0:
            resizedImage = cv2.copyMakeBorder(resizedImage, borderType=cv2.BORDER_CONSTANT,
                                              top=abs(top), bottom=abs(bottom), left=0, right=0)
        else:
            resizedTargetImage = cv2.copyMakeBorder(resizedTargetImage, borderType=cv2.BORDER_CONSTANT,
                                                    top=top, bottom=bottom, left=0, right=0)
        if horizontal < 0:
            resizedImage = cv2.copyMakeBorder(resizedImage, borderType=cv2.BORDER_CONSTANT,
                                              top=0, bottom=0, left=abs(left), right=abs(right))
        else:
            resizedTargetImage = cv2.copyMakeBorder(resizedTargetImage, borderType=cv2.BORDER_CONSTANT,
                                                    top=0, bottom=0, left=left, right=right)

        return resizedImage, resizedTargetImage
