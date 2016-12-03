import cv2
import numpy as np
from sys import exit

#---------------------------------------- Local constants ---------------------------------------------------------#
MIN_CONTOUR_AREA = 0
MAX_CONTOUR_AREA = 400

RESIZED_WIDTH = 10
RESIZED_HEIGHT = 15

CLASSIFICATIONS_PATH = "D:\\Works\\NITK Curriculum Projects\\Semester IV\\POP Mini-project\\ocr\\training_data\\classifications.txt"
FLAT_IMAGES_PATH = "D:\\Works\\NITK Curriculum Projects\\Semester IV\\POP Mini-project\\ocr\\training_data\\flat_images.txt"
TRAIN_IMAGE_PATH = "D:\\Works\\NITK Curriculum Projects\\Semester IV\\POP Mini-project\\ocr\\training_data\\training_ocr_extended_a.png"
#################################################################

def generateData(img):
    # Generate a gray scale version of the image 'img'
    imgGrayScale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Perform thresholding
    _, imgThresh = cv2.threshold(imgGrayScale, 127, 255, cv2.THRESH_BINARY_INV)
    # Find contours
    _, contours, _ = cv2.findContours(imgThresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.imshow("Thresh", imgThresh)
    # Classify the contours
    flattenedImages = np.empty((0, RESIZED_WIDTH*RESIZED_HEIGHT))
    classifications = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if MIN_CONTOUR_AREA < cv2.contourArea(contour) < MAX_CONTOUR_AREA and h > 2:
            # Draw a rectangle around detected image for displaying current contour
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
            imgResized = cv2.resize(imgThresh[y:y+h, x:x+w], (RESIZED_WIDTH, RESIZED_HEIGHT))
            cv2.imshow("Resized ROI", imgResized)
            cv2.imshow("Bounded", img)
            # Get key input from user
            key = cv2.waitKey(0)
            if key == 27:
                # Escape has been pressed; exit immediately
                exit()
            # Add the key to the classifications
            classifications.append(key)
            # Reshape the resized ROI into a 1D array
            imgFlat = imgResized.reshape((1, RESIZED_WIDTH*RESIZED_HEIGHT))
            # Add the image data in the flattened images array
            flattenedImages = np.append(flattenedImages, imgFlat, 0)

    # Finish up
    classifications = np.array(classifications, np.float32)
    finalClassifications = classifications.reshape((classifications.size, 1))
    classFile = open(CLASSIFICATIONS_PATH,'ab')
    flatIFile = open(FLAT_IMAGES_PATH,'ab')
    np.savetxt(classFile, finalClassifications)
    np.savetxt(flatIFile, flattenedImages)

if __name__ == "__main__":
    # Open an image
    img = cv2.imread(TRAIN_IMAGE_PATH)
    if img is None:
        print "Image load failed; exiting..."
        exit(1)
    # Pass the image to the generateData function
    generateData(img)
    # Show the final image
    cv2.imshow("Final Image", img)

    cv2.destroyAllWindows()