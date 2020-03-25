import cv2

from KeycardReader import KeycardReader

"""
Ideas
 - Canny

cv2.imshow("CLAHE image", claheImage)
cv2.imshow("Edge image", edgeImage)
cv2.imshow("Color image", colorImage)
cv2.imshow("Shape image", shapeImage)
cv2.imshow("Grabcut image", grabcutImage)
cv2.imshow("Mask Image", maskImage)
cv2.waitKey(0)
"""

# ==================================================
inDir = 'inImages/'
outDir = 'outImages/'
imageFilenames = [
    'keycard-1.jpg',
    'keycard-2.jpg',
    'keycard-3.jpg',
    'keycard-4.jpg'
]
referenceImageFileName = "keycard-reference.webp"
# ==================================================

if __name__ == '__main__':
    keycardReader = KeycardReader(referenceImageFileName)

    for imageFilename in imageFilenames:
        image, b, r, bThresholded, rThresholded = keycardReader.extractKeycardDescriptor(inDir + imageFilename)

        cv2.imwrite(outDir + imageFilename, image)
        cv2.imwrite(outDir + "b-" + imageFilename, b)
        cv2.imwrite(outDir + "r-" + imageFilename, r)
        cv2.imwrite(outDir + "b-thresholded-" + imageFilename, bThresholded)
        cv2.imwrite(outDir + "r-thresholded-" + imageFilename, rThresholded)
