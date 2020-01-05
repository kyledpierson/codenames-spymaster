import cv2
import numpy as np


class Morpher:
    def __init__(self):
        pass

    def registerWithCorrespondences(self, image, points, targetImage, targetPoints):
        """ Warps image into targetImage, based on correspondences
        :param image: BGR image
        :param points: ordered coordinates of landmarks in image
        :param targetImage: BGR target image
        :param targetPoints: ordered coordinates of landmarks in targetImage
        :return: BGR image warped into targetImage """
        rows, cols = targetImage.shape[:2]

        for (x, y) in points:
            cv2.circle(image, (x, y), 3, 0, -1)
        for (x, y) in targetPoints:
            cv2.circle(targetImage, (x, y), 3, 0, -1)

        # Alternative: warpMatrix = cv2.getPerspectiveTransform(points, targetPoints)
        warpMatrix, status = cv2.findHomography(points, targetPoints)
        registeredImage = cv2.warpPerspective(image, warpMatrix, (cols, rows))

        return registeredImage

    def registerWithEcc(self, image, targetImage):
        """ Performs a perspective transformation of image to the foreground orientation of targetImage
        :param image: BGR image
        :param targetImage: BGR image with which image should be aligned
        :return BGR image aligned to targetImage """
        rows, cols = targetImage.shape[:2]

        grayTargetImage = cv2.cvtColor(targetImage, cv2.COLOR_BGR2GRAY)
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        warpMatrix = np.eye(3, 3, dtype=np.float32)
        motionType = cv2.MOTION_HOMOGRAPHY

        criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 10000, 0.01)

        cc, warpMatrix = cv2.findTransformECC(grayTargetImage, grayImage, warpMatrix, motionType, criteria, None, 5)

        registeredImage = cv2.warpPerspective(image, warpMatrix, (cols, rows),
                                              flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

        return registeredImage
