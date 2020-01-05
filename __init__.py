import cv2

from KeycardReader import KeycardReader

# ==================================================
inDir = 'inImages/'
outDir = 'outImages/'
imageFilenames = [
    'keycard-1.jpg',
    'keycard-2.jpg',
    'keycard-3.jpg',
    'keycard-4.jpg'
]
# ==================================================

if __name__ == '__main__':
    keycardReader = KeycardReader()

    for imageFilename in imageFilenames:
        image, b, r, bThresholded, rThresholded = keycardReader.extractKeycardDescriptor(inDir + imageFilename)

        cv2.imwrite(outDir + imageFilename, image)
        cv2.imwrite(outDir + "b-" + imageFilename, b)
        cv2.imwrite(outDir + "r-" + imageFilename, r)
        cv2.imwrite(outDir + "b-thresholded-" + imageFilename, bThresholded)
        cv2.imwrite(outDir + "r-thresholded-" + imageFilename, rThresholded)
