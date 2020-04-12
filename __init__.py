import cv2

from KeycardReader import KeycardReader

"""
Ideas
 - Canny

if colorSpace is cv2.COLOR_BGR2HLS:
    lightRed1 = (0, 100, 150)
    darkRed1 = (40, 150, 255)
    lightRed2 = (215, 100, 150)
    darkRed2 = (255, 150, 255)
    lightBlue = (115, 75, 120)
    darkBlue = (155, 130, 255)

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
    'keycard-4.jpg',
    'keycard-5.jpg',
    # 'keycard-6.jpg',
    # 'keycard-7.jpg',
    # 'keycard-8.jpg',
    # 'keycard-dark-1.jpg',
    # 'keycard-dark-2.jpg'
]
referenceImageFileName = "keycard-reference.webp"
# ==================================================

if __name__ == '__main__':
    keycardReader = KeycardReader(referenceImageFileName)

    for imageFilename in imageFilenames:
        image, bThresholded, rThresholded = keycardReader.extractKeycardDescriptor(inDir + imageFilename)

        cv2.imwrite(outDir + imageFilename, image)
        cv2.imwrite(outDir + "b-thresholded-" + imageFilename, bThresholded)
        cv2.imwrite(outDir + "r-thresholded-" + imageFilename, rThresholded)
