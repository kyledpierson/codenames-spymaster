import cv2
import numpy as np

from typing import Tuple

Image = np.ndarray


class Filterer:
    def __init__(self):
        pass

    @staticmethod
    def equalizeHistogram(image: Image, fromColorSpace: int, toColorSpace: int, channelsToEqualize: Tuple) -> Image:
        convertedImage: Image = cv2.cvtColor(image, fromColorSpace)
        c1, c2, c3 = cv2.split(convertedImage)

        # Perform CLAHE
        clahe: cv2.CLAHE = cv2.createCLAHE()
        c1: np.array = clahe.apply(c1) if channelsToEqualize[0] else c1
        c2: np.array = clahe.apply(c2) if channelsToEqualize[1] else c2
        c3: np.array = clahe.apply(c3) if channelsToEqualize[2] else c3

        equalizedConvertedImage: Image = cv2.merge((c1, c2, c3))
        equalizedImage: Image = cv2.cvtColor(equalizedConvertedImage, toColorSpace)

        return equalizedImage

    @staticmethod
    def mapHistogram(image: Image, targetImage: Image, fromColorSpace: int, toColorSpace: int) -> Image:
        convertedImage: Image = cv2.cvtColor(image, fromColorSpace).astype('float32')
        convertedTargetImage: Image = cv2.cvtColor(targetImage, fromColorSpace).astype('float32')

        c1Mean, c1Std, c2Mean, c2Std, c3Mean, c3Std = Filterer.__computeImageStats(convertedImage)
        c1MeanTarget, c1StdTarget, c2MeanTarget, c2StdTarget, c3MeanTarget, c3StdTarget = \
            Filterer.__computeImageStats(convertedTargetImage)

        c1, c2, c3 = cv2.split(convertedImage)
        c1: np.array = np.clip((c1Std / c1StdTarget) * (c1 - c1Mean) + c1MeanTarget, 0, 255)
        c2: np.array = np.clip((c2Std / c2StdTarget) * (c2 - c2Mean) + c2MeanTarget, 0, 255)
        c3: np.array = np.clip((c3Std / c3StdTarget) * (c3 - c3Mean) + c3MeanTarget, 0, 255)

        coloredConvertedImage: Image = cv2.merge([c1, c2, c3])
        coloredImage: Image = cv2.cvtColor(coloredConvertedImage.astype('uint8'), toColorSpace)

        return coloredImage

    @staticmethod
    def enhanceEdges(image: Image, action: str, kernel: Image = None) -> Image:
        enhancedImage: Image = cv2.edgePreservingFilter(image)
        # Alternative: cv2.bilateralFilter(image, 5, 50, 50)

        if not kernel:
            if action == 'sharpen':
                kernel = Image([[-1, -1, -1],
                                [-1, +9, -1],
                                [-1, -1, -1]])
            elif action == 'excessiveSharpen':
                kernel = Image([[+1, +1, +1],
                                [+1, -7, +1],
                                [+1, +1, +1]])
            elif action == 'enhanceEdges':
                kernel = Image([[-1, -1, -1, -1, -1],
                                [-1, +2, +2, +2, -1],
                                [-1, +2, +8, +2, -1],
                                [-1, +2, +2, +2, -1],
                                [-1, -1, -1, -1, -1]]) / 8.0
            else:
                raise ValueError(action + ' filter not implemented')

        enhancedImage = cv2.filter2D(enhancedImage, -1, kernel)
        return enhancedImage

    # -------------------------------------------------------- #
    # -------------------- Helper Methods -------------------- #
    # -------------------------------------------------------- #
    @staticmethod
    def __computeImageStats(image: Image) -> Tuple:
        c1, c2, c3 = cv2.split(image)

        c1NoBlack: np.array = np.array([])
        c2NoBlack: np.array = np.array([])
        c3NoBlack: np.array = np.array([])

        for i, row in enumerate(c1):
            for j, col in enumerate(row):
                if c1[i][j] != 0:
                    np.append(c1NoBlack, [c1[i][j]])
                    np.append(c2NoBlack, [c2[i][j]])
                    np.append(c3NoBlack, [c3[i][j]])

        return c1NoBlack.mean(), c1NoBlack.std(), c2NoBlack.mean(), c2NoBlack.std(), c3NoBlack.mean(), c3NoBlack.std()
