# Lab 2, 16-281 General Robotics, Carnegie Mellon University. 
# All rights reserved. Distribution of any material is prohibited.
# author: Qihang Shan (qshan@cmu.edu) since January 2025

# This file implements the "segment_image" AND "calculate_centroids" functions 
# that segment and label the centroid of the image.

# Hint:
# The input image of the function (thresholded image) MUST be a grey scale 
# image with binary values (0 AND 255).
    
# NOTE: YOU NEED TO IMPLEMENT TWO FUNCTIONS HERE!

# IMPORTANT NOTE: THE USE OF FUNCTIONS THAT SIMPLIFIES THE PROBLEM IS NOT 
# ALLOWED AND WILL RESULT IN A ZERO. DO NOT IMPORT ANY OTHER PACKAGE!
# CHANGING FUNCTION INPUTS WILL RESULT IN A ZERO AS WELL!

# NOTE: FILL IN YOUR INFORMATION BELOW

# STUDENT'S NAME: David Alvarez Gomez
# ANDREW ID: davidalv

import os
import cv2
import numpy as np

def segment_image(thresholded_image):
    # -------------------START OF STUDENT IMPLEMENTATION-------------------
    """
    Segments the input binary image into different balls.
    
    --INSERT YOUR ALGORITHM DESCRIPTION HERE--

        This function segments the image using a flood fill type of algorithm,
    First it craetes a new 2D list of zeroes, and also a list of areas that
    starts as empty. Before entering the loop it checks and converts the image
    into a greyscale image with only one value in the pixel which stopped the
    line __ from failing. As soon as the loop is entered the function checks
    if the value of the pixel is white, and that it has not been checked yet,
    by making sure that in the new image the pixel is at value 0. Once that
    happens the value in the new image is set to a "current" integer, and
    then the algorithm fills the rest of the ball with the same integer. This 
    segments the tennis balls and labels them with different values. The while
    loop makes sure that each adjacent pixel is checked and then once that is 
    done it exits and continues searching for other white pixels. Once it
    flood fills it adds the area of whatever was filled into a list, which then
    gets sorted. The first four values in this list are the largest areas, and
    therefore the rest of the areas are discarded as noise.
    
    Args:
    - thresholded_image (numpy.ndarray): Thresholded image.
    
    Returns:
    - output (numpy.ndarray): Labeled image where each region is assigned a 
    unique label (different grey scale values).
    """

    rows = len(thresholded_image)
    cols = len(thresholded_image[0])

    newImg = [[0 for i in range(cols)] for j in range(rows)]
    current = 0
    areas = []
    if len(thresholded_image.shape) == 3:
        thresholded_image = cv2.cvtColor(thresholded_image, cv2.COLOR_BGR2GRAY)
    for row in range(rows):
        for col in range(cols):
            if thresholded_image[row][col] == 255 and newImg[row][col] == 0:
                current += 1
                pixCount = 1
                newImg[row][col] = current
                pixToSearch = [(row,col)]
                while pixToSearch != []:
                    currY, currX = pixToSearch.pop(0)
                    for dy,dx in [(-1,0),(0,-1),(1,0),(0,1)]:
                        newy,newx = currY+dy,currX+dx
                        if 0<=newy<rows and 0<=newx<cols:
                            if (thresholded_image[newy][newx] == 255
                                and newImg[newy][newx] == 0):
                                newImg[newy][newx] = current
                                pixCount += 1
                                pixToSearch.append((newy,newx))
                areas.append((current,pixCount))
    areas.sort(key=lambda x:x[1],reverse=True)
    areas = [areas[0][0],areas[1][0],areas[2][0],areas[3][0]]
    result = [[0 for i in range(cols)] for j in range(rows)]
    shading = [105,155,205,255]
    for row in range(rows):
        for col in range(cols):
            prevVal = newImg[row][col]
            if prevVal in areas:
                index = areas.index(prevVal)
                result[row][col] = shading[index]
    return np.array(result)

    # -------------------END OF STUDENT IMPLEMENTATION-------------------
    
def calculate_centroids(segmented_image):
    # -------------------START OF STUDENT IMPLEMENTATION-------------------
    """
    --INSERT YOUR FUNCTION DESCRIPTION HERE--
    
        First the function creates an empty list that will have the areas, x, y
    coordinates of each different ball and the pixels in the ball. The np.unique
    function returns the unique values in the image and in the loop we skip the 
    ones that are 0, aka that aren't balls. Then the np.where function gives the
    coordinates of the unique value, in two seperate lists. Then the average y
    and x coordinate are calculated and therefore given and appended along 
    with the areas into the list. Then the coordinates are returned in an
    array.

    Args:
    - segmented_image (numpy.ndarray): Segmented image.
    
    Returns:
    - centroids (numpy.ndarray): Array of centroids (x, y) of each segment.
    """
    
    locandAreas = []
    balls = np.unique(segmented_image)
    for ball in balls:
        if ball == 0:
            continue
        y,x = np.where(segmented_image == ball)
        areas = len(y)
        averagey = np.sum(y)/areas
        averagex = np.sum(x)/areas
        locandAreas.append((areas,averagey,averagex))
    locandAreas.sort(key=lambda x:x[0],reverse = True)
    locandAreas = locandAreas[:4]
    finalcentroids = []
    for a,y,x in locandAreas:
        finalcentroids.append((x,y))
    return np.array(finalcentroids)

    # -------------------END OF STUDENT IMPLEMENTATION-------------------

# NOTE: DO NOT CHANGE ANY LINE BELOW!

def segment_file_and_save(filename):
    
    # Read the thresholded image from folder
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    
    # NOTE: Force all values in the image to be either 0 or 255 as imread is
    # NOT always correct in reading the pixel values.
    # DO NOT DELETE THIS!!!
    img[img < 128] = 0
    img[img >= 128] = 255
    
    # Segment the image
    segmented = segment_image(img)

    # YOU COULD COMMENT OUT SOME LINES BELOW IF YOU DO NOT KNOW HOW TO IMPLEMENT
    # THE CALCULATION OF CENTROIDS OR FOR THE EASE OF DEBUGGING.
    
    # Calculate the centroids
    centroids = calculate_centroids(segmented)
    
    radius = 5
    circle_positions = np.column_stack([centroids, np.full((centroids.shape[0], 
        1), radius)])
    
    # Convert the segmented image to RGB for visualization
    labeled = cv2.cvtColor(segmented.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    
    # Draw circles around centroids
    for (x, y, r) in circle_positions:
        cv2.circle(labeled, (int(x), int(y)), int(r), (255, 0, 0), 4)
    
    # Create the output folder if it doesn't exist
    output_folder = "output_images_segmented"
    os.makedirs(output_folder, exist_ok=True)
    
    # Save the labeled image
    filename_without_extension = os.path.splitext(os.path.basename(filename))[0]
    output_filename = os.path.join(output_folder, 
        f"segmented_{filename_without_extension}.jpg")
    cv2.imwrite(output_filename, labeled)
    
def test_segment():

    thresholded_folder = "output_images_thresholded"
    
    filenames = [f for f in os.listdir(thresholded_folder) if f.endswith('.png') or f.endswith('.jpg')]
    
    print('Segmenting images...')
    for filename in filenames:
        print(f'Segmenting {filename}...')

        file_path = os.path.join(thresholded_folder, filename)

        segment_file_and_save(file_path)
        
if __name__ == "__main__":
    test_segment()
