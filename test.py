# TrainAndTest.py

import cv2
import numpy as np
import operator as op

#---------------------------------------- Local constants ---------------------------------------------------------#
MIN_CONTOUR_AREA        = 0
RESIZED_IMAGE_WIDTH     = 10
RESIZED_IMAGE_HEIGHT    = 15

CLASSIFICATIONS_PATH    = "D:\\Works\\NITK Curriculum Projects\\Semester IV\\POP Mini-project\\ocr\\training_data\\classifications.txt"
FLAT_IMAGES_PATH        = "D:\\Works\\NITK Curriculum Projects\\Semester IV\\POP Mini-project\\ocr\\training_data\\flat_images.txt"
TEST_IMAGE_PATH         = "D:\\Works\\NITK Curriculum Projects\\Semester IV\\POP Mini-project\\ocr\\training_data\\testing_ocr_extended_a.png"

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
def separateLines(contourList):
    '''
    Segregate the given contours into lists containing contours belonging
    to the same line and yield one list at a time in top to bottom sequence

    :param contourList: Given list of contours
    :return: Generator object that sequentially generates one line list at a time
    '''
    contourList.sort(key = op.attrgetter("intRectY"))  # sort contours from top to bottom

    MAX_OFFSET = max(contourList, key = op.attrgetter("intRectHeight")).intRectHeight - 2
    i = 0
    while i < len(contourList):

        contoursInLine = []
        currentApproxY = contourList[i].intRectY
        contoursInLine.append(contourList[i])  # Add any one of the contours in the line
        if i + 1 < len(contourList):
            i = i + 1
        else:
            break

        # Run a loop that will add all contours of the current line to the list
        while i < len(contourList) and abs(contourList[i].intRectY - currentApproxY) < MAX_OFFSET:
            contoursInLine.append(contourList[i])
            i = i + 1

        yield contoursInLine


def recognizeLine(img, threshImage, contoursInLine, kNN):
    '''
    Recognize the characters in the given contour list and generate a corresponding string
    :param img: Image to draw bounding rectangles on [IMAGE WLL BE MODIFIED]
    :param threshImage: Threshed image
    :param contoursInLine: List of all contours in a single line
    :param kNN: kNearestNeighbour object
    :return: The corresponding character string
    '''
    # Initialize the return string
    currentLineString = ""
    # Sort according to X-co-ordinates so the contours are ordered
    contoursInLine.sort(key = op.attrgetter("intRectX"))
    # Determine the approximate limit between two consecutive characters in the same word
    space_limit = 2 * max(contoursInLine, key = op.attrgetter("intRectWidth")).intRectWidth - 3
    # prevX stores the X co-ordinate of the previous character in the line
    prevX = 0
    # Iterate through all contours
    for dataContourObject in contoursInLine:

        # Draw a bounding rectangle around the current contour
        topLeft, bottomRight = dataContourObject.getRectPoints()
        cv2.rectangle(img, topLeft, bottomRight, (0, 255, 0), 1)


        currentX = dataContourObject.intRectX

        imgROI = threshImage[dataContourObject.intRectY: dataContourObject.intRectY + dataContourObject.intRectHeight,
                 dataContourObject.intRectX: dataContourObject.intRectX + dataContourObject.intRectWidth]

        # Resize image, as the training data used the same size
        resizedROI = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
        # Flatten image into 1D numpy array
        resizedROI = resizedROI.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
        # Convert from 1D numpy array of ints to 1D numpy array of floats, as the kNN object requires floats
        resizedROI = np.float32(resizedROI)
        # Pass resizedROI to the kNN object for detection
        _, results, _, _ = kNN.findNearest(resizedROI, k=1)  # call KNN function find_nearest
        # Get character from results
        detectedChar = str(chr(int(results[0][0])))
        # Show the image with a bounding rectangle
        cv2.imshow("Detected Contours", img)

        # Add a space at an appropriate position
        if prevX != 0 and abs(prevX - currentX) > space_limit:
            currentLineString = currentLineString + " "
        # Append the current character to the final string
        currentLineString = currentLineString + detectedChar

        prevX = currentX

    return currentLineString


#------------------------------------------------ Driver function -------------------------------------------------#
def main():
    '''
    Driver function
    '''
    # Load the required files and the test image; exit with non-zero exit code on any error
    try:
        classfications = np.loadtxt(CLASSIFICATIONS_PATH, np.float32)
    except:
        print "ERROR:\nFailed to open classifcations file. Exiting..."
        exit(1)

    try:
        flattenedImages = np.loadtxt(FLAT_IMAGES_PATH, np.float32)
    except:
        print "ERROR:\nFailed to open dataset file. Exiting..."
        exit(2)

    testImage = cv2.imread(TEST_IMAGE_PATH)
    if testImage is None:
        print "ERROR:\nFailed to read Image. Exiting..."
        exit(3)

    # Instantiate a kNN object
    kNearest = cv2.ml.KNearest_create()
    # Train the kNN object
    kNearest.train(flattenedImages, cv2.ml.ROW_SAMPLE, classfications)

    # Convert the image to gray scale
    imgGray = cv2.cvtColor(testImage, cv2.COLOR_BGR2GRAY)
    # Thresh the gray scale image
    _, imgThresh = cv2.threshold(imgGray, 140, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow("Thresh", imgThresh)
    # Detect all contours in the threshed image
    _, contours, _ = cv2.findContours(imgThresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Generate a list of valid contour Objects using the detected contours
    validContourObjects = [ contourObject for contourObject in map(DataContour, contours) if contourObject.isValid() ]

    # Create a copy of the test image to pass to recognizeLine function as it modifies the passed image
    imageCopy = testImage.copy()

    # Detect and print one line at a time
    for contoursInLine in separateLines(validContourObjects):
        print recognizeLine(imageCopy, imgThresh, contoursInLine, kNearest)

    # Display the original image
    cv2.imshow("Orginal Test Image", testImage)

    # End sequence
    print "\nDetection completed. Press any key to close all images exit..."
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return

#-------------------------------------------------- main ----------------------------------------------------------#
if __name__ == "__main__":
    main()
# end









