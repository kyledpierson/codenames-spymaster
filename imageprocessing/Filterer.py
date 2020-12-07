import cv2
import numpy as np

from typing import Tuple

Image = np.ndarray


class Filterer:
    def __init__(self):
        pass

    @staticmethod
    def equalizeHistogram(image: Image, fromColorSpace: int, toColorSpace: int, channelsToEqualize: Tuple) -> Image:
        if toColorSpace > -1:
            image = cv2.cvtColor(image, fromColorSpace)
        (c1, c2, c3) = cv2.split(image)

        # Perform CLAHE
        clahe: cv2.CLAHE = cv2.createCLAHE(clipLimit=1.0)
        c1 = clahe.apply(c1) if channelsToEqualize[0] else c1
        c2 = clahe.apply(c2) if channelsToEqualize[1] else c2
        c3 = clahe.apply(c3) if channelsToEqualize[2] else c3

        image = cv2.merge((c1, c2, c3))
        if fromColorSpace > -1:
            image = cv2.cvtColor(image, toColorSpace)

        return image

    @staticmethod
    def enhanceEdges(image: Image, action: str, kernel: Image = None) -> Image:
        # Alternative: cv2.bilateralFilter(image, 5, 50, 50)
        image = cv2.edgePreservingFilter(image)

        if not kernel:
            if action == "sharpen":
                kernel = np.array([[-1, -1, -1],
                                   [-1, +9, -1],
                                   [-1, -1, -1]], dtype=float)
            elif action == "excessiveSharpen":
                kernel = np.array([[+1, +1, +1],
                                   [+1, -7, +1],
                                   [+1, +1, +1]], dtype=float)
            elif action == "enhanceEdges":
                kernel = np.array([[-1, -1, -1, -1, -1],
                                   [-1, +2, +2, +2, -1],
                                   [-1, +2, +8, +2, -1],
                                   [-1, +2, +2, +2, -1],
                                   [-1, -1, -1, -1, -1]], dtype=float) / 8.0
            else:
                raise ValueError(action + " filter not implemented")

        image = cv2.filter2D(image, -1, kernel)
        return image
