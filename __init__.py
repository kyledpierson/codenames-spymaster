import cv2
import imutils
import numpy as np

from ImagePreprocessor import ImagePreprocessor
from ImageProcessor import ImageProcessor

# grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ==================================================
inDir = 'inImages/'
outDir = 'outImages/'
imageFilenames = [
    'codenames-key-1.jpg',
    'codenames-key-2.jpg',
    'codenames-key-3.jpg',
    'codenames-key-4.jpg'
]
targetFilename = 'codenames-key.webp'

clipLimit = 40
tileGridSize = (8, 8)
sigma_s = 60
sigma_r = 0.4
# ==================================================
imagePreprocessor = ImagePreprocessor()
imageProcessor = ImageProcessor()

targetImage = cv2.imread(inDir + targetFilename)
targetImage, _ = imageProcessor.segmentGrabcut(targetImage, useRectMask=True)
targetImage = imagePreprocessor.performClahe(targetImage, clipLimit, tileGridSize)

for imageFilename in imageFilenames:
    image = cv2.imread(inDir + imageFilename)
    image, _ = imageProcessor.segmentGrabcut(image, useRectMask=True)
    image = imagePreprocessor.performClahe(image, clipLimit, tileGridSize)

    image, resizedTargetImage = imagePreprocessor.resizeToSameSize(image, targetImage, 500, 500)
    image = imagePreprocessor.equalizeHistogram(image, resizedTargetImage)
    image = imagePreprocessor.registerWithEcc(image, resizedTargetImage, 10000, 0.1)
    image = imagePreprocessor.enhanceEdges(image, sigma_s=sigma_s, sigma_r=sigma_r)

    b = image[:, :, 0].copy()
    g = image[:, :, 1].copy()
    r = image[:, :, 2].copy()

    rows, cols = image.shape[:2]
    for i in range(rows):
        for j in range(cols):
            b[i, j] = max(0, 255 - np.linalg.norm(np.array([255, 0, 0]) - np.array(image[i, j])))
            g[i, j] = max(0, 255 - np.linalg.norm(np.array([0, 255, 0]) - np.array(image[i, j])))
            r[i, j] = max(0, 255 - np.linalg.norm(np.array([0, 0, 255]) - np.array(image[i, j])))

    bThresholded = imageProcessor.segmentThreshold(image, b, 'otsu')
    gThresholded = imageProcessor.segmentThreshold(image, g, 'otsu')
    rThresholded = imageProcessor.segmentThreshold(image, r, 'otsu')

    cv2.imwrite(outDir + str(clipLimit) + '-' + str(tileGridSize) + '-' + str(sigma_s) + '-' + str(
        sigma_r) + '-' + imageFilename, image)

    cv2.imwrite(outDir + "b-" + imageFilename, b)
    cv2.imwrite(outDir + "g-" + imageFilename, g)
    cv2.imwrite(outDir + "r-" + imageFilename, r)
    cv2.imwrite(outDir + "b-thresholded-" + imageFilename, bThresholded)
    cv2.imwrite(outDir + "g-thresholded-" + imageFilename, gThresholded)
    cv2.imwrite(outDir + "r-thresholded-" + imageFilename, rThresholded)
