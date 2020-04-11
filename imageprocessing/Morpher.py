import cv2
import numpy as np

Image = np.ndarray


class Morpher:
    def __init__(self):
        pass

    @staticmethod
    def registerWithCorrespondences(image: Image, points: list, targetImage: Image, targetPoints: list) -> Image:
        rows, cols = targetImage.shape[:2]

        # DEBUG: draw circles on correspondences
        for (x, y) in points:
            cv2.circle(image, (x, y), 3, 0, -1)
        for (x, y) in targetPoints:
            cv2.circle(targetImage, (x, y), 3, 0, -1)

        # Alternative: warpMatrix = cv2.getPerspectiveTransform(points, targetPoints)
        warpMatrix, status = cv2.findHomography(points, targetPoints)
        registeredImage = cv2.warpPerspective(image, warpMatrix, (cols, rows))

        return registeredImage

    @staticmethod
    def registerWithEcc(image: Image, targetImage: Image) -> Image:
        rows, cols = targetImage.shape[:2]

        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grayTargetImage = cv2.cvtColor(targetImage, cv2.COLOR_BGR2GRAY)

        warpMatrix = np.eye(3, 3, dtype=np.float32)
        motionType = cv2.MOTION_HOMOGRAPHY
        criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 10000, 0.01)
        inputMask = None

        cc, warpMatrix = cv2.findTransformECC(grayTargetImage, grayImage, warpMatrix, motionType, criteria, inputMask)

        registeredImage = cv2.warpPerspective(image, warpMatrix, (cols, rows),
                                              flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

        return registeredImage
