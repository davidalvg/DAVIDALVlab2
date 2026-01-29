# Lab 2, 16-281 General Robotics, Carnegie Mellon University. 
# All rights reserved. Distribution of any material is prohibited.
# author: Qihang Shan (qshan@cmu.edu) since January 2025

# This file implements the "image_to_dist" AND "centroids_to_dist" functions 
# that completes the whole pipeline of Lab 2.
    
# NOTE: YOU NEED TO IMPLEMENT TWO FUNCTIONS HERE!

# IMPORTANT NOTE: THE USE OF FUNCTIONS THAT SIMPLIFIES THE PROBLEM IS NOT 
# ALLOWED AND WILL RESULT IN A ZERO. DO NOT IMPORT ANY OTHER PACKAGE!
# CHANGING FUNCTION INPUTS WILL RESULT IN A ZERO AS WELL!

# NOTE: FILL IN YOUR INFORMATION BELOW

# STUDENT'S NAME: David Alvarez Gomez
# ANDREW ID: davidalv

import numpy as np
import cv2
import os

# IMPORTED FUNCTIONS BELOW ARE ONES YOU HAVE IMPLEMENTED
from threshold_image import threshold_image
from segment_image import segment_image
from segment_image import calculate_centroids

def image_to_dist(input_image, threshold, ball_separation, isTA):
    # -------------------START OF STUDENT IMPLEMENTATION-------------------
    """
    --INSERT YOUR FUNCTION DESCRIPTION HERE--
    
        This function just compiles all of the previous functions and follows
    the steps that are used to calculate the distance from the tennis balls
    to the image. First of all, it intakes the photo as an image file and 
    thresholds it and turns it into a binary image using thresholding based
    on hue, saturation, and value and bounding them in order to make all
    the tennis balls get highlited as white. Next, it uses the segmentation
    function to label the balls and shade them different values of grey. Right
    after the segmentation the calculate centroids function calculates the 
    center of each ball and returns the array of the four centroids of the ball.
    Then the centroids to dist function uses the equations we learned in class,
    h/d = h'/f, in order to calculate d and return that afterwards. To calculate 
    h' we average the calculation of the distances between adjacent balls.

    Args:
    - input_image (numpy.ndarray): Input image.
    - threshold (float array): List of 6 threshold values.
    - ball_separation (int): The physical separation between the balls.
    - isTA (bool): If True, use the TA focal length, otherwise use the student 
    focal length.

    Returns:
    dist (float): The calculated distance.
    """
    
    imgArrayThresholded = threshold_image(input_image,threshold)
    imgArraySegmented = segment_image(imgArrayThresholded)
    centroids = calculate_centroids(imgArraySegmented)
    dist = centroids_to_dist(centroids,ball_separation,isTA)
    return dist

    # -------------------END OF STUDENT IMPLEMENTATION-------------------

def centroids_to_dist(centroids, ball_separation, isTA):
    # -------------------START OF STUDENT IMPLEMENTATION-------------------
    """
    --INSERT YOUR FUNCTION DESCRIPTION HERE--

    Args:
    - centroids (numpy.ndarray): A 4x2 array of centroid positions, 
    where each row is (col, row).
    - ball_separation (int): The physical separation between the balls.
    - isTA (bool): If True, use the TA focal length, otherwise use the student 
    focal length.

    Returns:
    dist (float): The calculated distance.
    """

    # Set the focal length (in pixels) based on calibration for student images
    # ONLY!
    focal_length = 2690
    
    # -------------------END OF STUDENT IMPLEMENTATION-------------------
    
    if isTA:
        # If it's a TA system, use a different focal length
        # NOTE: CHANGE THIS VALUE WILL RESULT IN A ZERO FOR THE LAB
        focal_length = 1450

    # -------------------START OF STUDENT IMPLEMENTATION-------------------
    
    ballpixdists = []
    for ball1 in range(4):
        for ball2 in range(ball1+1,4):
            pixdist = (((centroids[ball1][0]-centroids[ball2][0])**2) + 
                 (centroids[ball1][1]-centroids[ball2][1])**2)**0.5
            ballpixdists.append(pixdist)
    ballpixdists.sort()
    hPrime = sum(ballpixdists[:4])/4
    hnormal = ball_separation * 25.4
    dist = (focal_length*hnormal)/hPrime
    return dist / 25.4

    # -------------------END OF STUDENT IMPLEMENTATION-------------------

# NOTE: DO NOT CHANGE ANY LINE BELOW!

def read_image_data(image_data_file):
    
    image_data = []
    with open(image_data_file, 'r') as file:
        # Skip the first 5 lines in the .dat file due to comments
        for _ in range(5):
            next(file)
        
        # Process the remaining lines
        for line in file:
            parts = line.strip().split()
            filename = parts[0]
            dist = int(parts[1])
            thresholds = list(map(float, parts[2:]))
            image_data.append((filename, dist, thresholds))
    
    return image_data

def run_file_and_grade(filename, threshold, ball_separation, isTA):
    
    img = cv2.imread(filename)
    if 'small' in filename:
        ball_separation = 5
    else:
        ball_separation = 12
    
    dist = round(image_to_dist(img, threshold, ball_separation, isTA))
    return dist

def test_pipeline(student_filenames, student_thresholds, student_distances, 
                  student_folder, ta_folder, ta_filenames, ta_thresholds, 
                  ta_distances):
    student_results = []

    # Processing Student Images
    print("Processing student images...")
    for i in range(len(student_filenames)):
        input_image_path = os.path.join(student_folder, student_filenames[i])
        dist = run_file_and_grade(input_image_path, student_thresholds[i], 
            12, False)
        if student_distances[i] == 0:
            print(f"{student_filenames[i]}: Calculated: {dist}")
            student_results.append(dist)
        else:
            error = round(abs(100 * (student_distances[i] - dist) 
                / student_distances[i]))
            print(f"{student_filenames[i]}: Calculated: {dist}, Expected: {student_distances[i]}, Error: {error}%")
            student_results.append((student_distances[i], dist))
    
    ta_results = []

    # Processing TA Images
    print("Processing TA images...")
    for i in range(len(ta_filenames)):
        input_image_path = os.path.join(ta_folder, ta_filenames[i])
        dist = run_file_and_grade(input_image_path, ta_thresholds[i], 12, True)
        if ta_distances[i] == 0:
            print(f"{ta_filenames[i]}: Calculated: {dist}")
            ta_results.append(dist)
        else:
            error = round(abs(100 * (ta_distances[i] - dist) / ta_distances[i]))
            print(f"{ta_filenames[i]}: Calculated: {dist}, Expected: {ta_distances[i]}, Error: {error}%")
            ta_results.append(dist)

    return student_results, ta_results

# Define image file paths
student_folder = './student_images'
ta_folder = './ta_images'

# Define the paths to the .dat files
student_data_file = './student_images/images.dat'
ta_data_file = './ta_images/images.dat'

# Read image metadata from the .dat files
student_data = read_image_data(student_data_file)
ta_data = read_image_data(ta_data_file)

# Extract student images
student_filenames = [data[0] for data in student_data]
student_distances = [data[1] for data in student_data]
student_thresholds = [data[2] for data in student_data]

# Extract TA images
ta_filenames = [data[0] for data in ta_data]
ta_distances = [data[1] for data in ta_data]
ta_thresholds = [data[2] for data in ta_data]

test_pipeline(student_filenames, student_thresholds, student_distances, 
              student_folder, ta_folder, ta_filenames, ta_thresholds, 
              ta_distances)