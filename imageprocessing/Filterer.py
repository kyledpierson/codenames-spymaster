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
    def mapHistogram(image: Image, targetImage: Image, fromColorSpace: int, toColorSpace: int) -> Image:
        image = cv2.cvtColor(image, fromColorSpace).astype("float32")
        targetImage = cv2.cvtColor(targetImage, fromColorSpace).astype("float32")

        (c1Mean, c1Std, c2Mean, c2Std, c3Mean, c3Std) = Filterer.__computeImageStats(image)
        (c1MeanTarget, c1StdTarget, c2MeanTarget, c2StdTarget, c3MeanTarget, c3StdTarget) = \
            Filterer.__computeImageStats(targetImage)

        (c1, c2, c3) = cv2.split(image)
        c1 = np.clip((c1Std / c1StdTarget) * (c1 - c1Mean) + c1MeanTarget, 0, 255)
        c2 = np.clip((c2Std / c2StdTarget) * (c2 - c2Mean) + c2MeanTarget, 0, 255)
        c3 = np.clip((c3Std / c3StdTarget) * (c3 - c3Mean) + c3MeanTarget, 0, 255)

        image = cv2.merge((c1, c2, c3))
        image = cv2.cvtColor(image.astype("uint8"), toColorSpace)

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

    # -------------------------------------------------------- #
    # -------------------- Helper Methods -------------------- #
    # -------------------------------------------------------- #
    @staticmethod
    def __computeImageStats(image: Image) -> Tuple:
        (c1, c2, c3) = cv2.split(image)

        c1NoBlack: np.array = np.array([], dtype="uint8")
        c2NoBlack: np.array = np.array([], dtype="uint8")
        c3NoBlack: np.array = np.array([], dtype="uint8")

        for i, row in enumerate(c1):
            for j, col in enumerate(row):
                if c1[i, j] != 0:
                    np.append(c1NoBlack, [c1[i, j]])
                    np.append(c2NoBlack, [c2[i, j]])
                    np.append(c3NoBlack, [c3[i, j]])

        return c1NoBlack.mean(), c1NoBlack.std(), c2NoBlack.mean(), c2NoBlack.std(), c3NoBlack.mean(), c3NoBlack.std()
