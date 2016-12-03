import cv2
import numpy as np
from sys import exit

#---------------------------------------- Local constants ---------------------------------------------------------#
MIN_CONTOUR_AREA        = 0
MAX_CONTOUR_AREA        = 400

RESIZED_WIDTH           = 10
RESIZED_HEIGHT          = 15

CLASSIFICATIONS_PATH    = "D:\\Works\\NITK Curriculum Projects\\Semester IV\\POP Mini-project\\ocr\\training_data\\classifications.txt"
FLAT_IMAGES_PATH        = "D:\\Works\\NITK Curriculum Projects\\Semester IV\\POP Mini-project\\ocr\\training_data\\flat_images.txt"
TRAIN_IMAGE_PATH        = "D:\\Works\\NITK Curriculum Projects\\Semester IV\\POP Mini-project\\ocr\\training_data\\training_ocr_extended_a.png"


#---------------------------------------- Class definitions -------------------------------------------------------#
class DataContour():
    # ----------------------------------- Member variables --------------------------------------------------------#
    contour = None              # Contour
    intRectX = 0                # Bounding rect top left corner x location
    intRectY = 0                # Bounding rect top left corner y location
    intRectWidth = 0            # Bounding rect width
    intRectHeight = 0           # Bounding rect height
    fltArea = 0.0               # Area of contour
    contourValidity = True      # Valdity of the contour

    #---------------------------------------- Constructor ---------------------------------------------------------#
    def __init__(self, contour):
        '''
        Initialize all attributes of the DataContour object created using the passed contour
        :param contour: Simple contour
        '''
        self.contour = contour
        self.intRectX, self.intRectY, self.intRectWidth, self.intRectHeight = cv2.boundingRect(self.contour)
        self.fltArea = cv2.contourArea(self.contour)
        if self.fltArea < MIN_CONTOUR_AREA or self.intRectHeight < 4:
            self.contourValidity = False

    #------------------------------------- Member functions -------------------------------------------------------#
    def isValid(self):
        ''' Return the contour validity: True if valid, False otherwise '''
        return self.contourValidity

    def getRectPoints(self):
        '''
        Return the co-ordinates of the upper-left and lower-right corners of the bounding rectangle
        :return: Two tuples containing co-ordinates corresponding to the two points
        '''
        # Upper left point
        UL = (self.intRectX, self.intRectY)
        # Bottom right point
        BR = (self.intRectX + self.intRectWidth, self.intRectY + self.intRectHeight)

        return UL, BR


#--------------------------------------------- Helper functions ---------------------------------------------------#
def generateData(img):
    '''
    Generate classiification and flattened image data for training the kNN object at a later stage
    :param img: Image to select contours from
    :return: (list_containing_classifications, flattened_images)
    '''
    # Generate a gray scale version of the image 'img'
    imgGrayScale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Perform threshing
    _, imgThresh = cv2.threshold(imgGrayScale, 127, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow("Thresh", imgThresh)
    # Find contours
    _, contours, _ = cv2.findContours(imgThresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # Create a DataContour objects list of valid contours
    validContourObjects = [ contourObject for contourObject in map(DataContour, contours) if contourObject.isValid() ]

    # Initialize the classifications and image lists
    flattenedImages = np.empty((0, RESIZED_WIDTH*RESIZED_HEIGHT))
    classifications = []

    for contourObject in validContourObjects:
        # ul: Upper left point
        # br: Bottom right point
        ul, br = contourObject.getRectPoints()
        # Draw a rectangle around detected image for displaying current contour
        cv2.rectangle(img, ul, br, (0, 0, 255), 1)
        # Resize the image
        imgResized = cv2.resize(imgThresh[ul[1]:br[1], ul[0]:br[0]], (RESIZED_WIDTH, RESIZED_HEIGHT))
        cv2.imshow("Resized ROI", imgResized)
        cv2.imshow("Bounded", img)
        # Get key input from user
        key = cv2.waitKey(0)
        if key == 27:
            # Escape has been pressed; exit
            print "\nExecution aborted as Escape key was been pressed."
            exit(0)
        else:
            print "Key pressed: " + chr(key)
        # Add the key to the classifications
        classifications.append(key)
        # Reshape the resized ROI into a 1D array
        imgFlat = imgResized.reshape((1, RESIZED_WIDTH*RESIZED_HEIGHT))
        # Add the image data in the flattened images array
        flattenedImages = np.append(flattenedImages, imgFlat, 0)

    # Convert the classfication data to floating points and reshape it to a single column
    finalClassifications = np.array(classifications, np.float32)
    finalClassifications = finalClassifications.reshape((finalClassifications.size, 1))

    return finalClassifications, flattenedImages


#------------------------------------------------ Driver function -------------------------------------------------#
def main():
    '''
    Driver functon
    '''
    # Open an image
    img = cv2.imread(TRAIN_IMAGE_PATH)
    if img is None:
        print "ERROR: Image load failed; exiting..."
        exit(1)

    # Pass the image to the generateData function
    finalClassifications, flattenedImages = generateData(img)

    # Open (Or create) the classiifications and flattened images file
    classFile = open(CLASSIFICATIONS_PATH, 'ab')
    flatIFile = open(FLAT_IMAGES_PATH, 'ab')
    # Save the data to the files
    np.savetxt(classFile, finalClassifications)
    np.savetxt(flatIFile, flattenedImages)

    # Exit sequence
    print "\nData has been successfully generated. Exiting..."
    cv2.destroyAllWindows()
    return

#-------------------------------------------------- main ----------------------------------------------------------#
if __name__ == "__main__":
    main()
