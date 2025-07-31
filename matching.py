
import cv2
import numpy as np
import matplotlib.pyplot as plt

import my_cv

# Function to get keypoints and descriptors from an image using the specified descriptor type.
def get_keypoints_and_descriptors(image, descriptor_type='SIFT'):
    if descriptor_type == 'SIFT':
        descriptor = cv2.SIFT_create(contrastThreshold=0.01)
        keypoints, descriptors = descriptor.detectAndCompute(image, None) 
    elif descriptor_type == 'BRIEF':
        fast = cv2.FastFeatureDetector_create()
        keypoints = fast.detect(image, None)
        brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
        keypoints, descriptors = brief.compute(image, keypoints) 
    elif descriptor_type == 'ORB':
        descriptor = cv2.ORB_create()
        keypoints, descriptors = descriptor.detectAndCompute(image, None) 
    else:
        raise ValueError("Unsupported descriptor type. Choose 'SIFT', 'BRIEF', or 'ORB'.")
    
    return keypoints, descriptors

# Receives two images and computes the panoramic resulting image
def match_images(img1, img2, threshold, descriptor, show):
    # Step 1/2: Resize images to the same height while maintaining aspect ratio
    height1, width1, _ = img1.shape
    height2, width2, _ = img2.shape  
    if height1 != height2:
        new_height = min(height1, height2)
        img1 = cv2.resize(img1, (int(width1 * new_height / height1), new_height))
        img2 = cv2.resize(img2, (int(width2 * new_height / height2), new_height)) 
    
    # Step 1: Convert the input color images to grayscale images
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
   
    # Step 2: Find interest points and local invariant descriptors for the pair of images
    keypoints1, descriptors1 = get_keypoints_and_descriptors(gray1, descriptor)
    keypoints2, descriptors2 = get_keypoints_and_descriptors(gray2, descriptor)

    # Step 3: Compute distances (similarities) between each descriptor of the two images
    # Sort them by distance
    bf      = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
     
    # Step 4: Select the best matches for each image descriptor
    num_good_matches = int(len(matches) * threshold)
    good_matches     = matches[:num_good_matches]

    # Step 5: Apply RANSAC technique to estimate the homography matrix
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 2.0)

    print(f"Homography matrix - {descriptor}:")
    print(H, end='\n\n')

    # Step 6: Apply a perspective projection to align the images
    height1, width1, _ = img1.shape
    height2, width2, _ = img2.shape
    
    # Find the dimensions of the resulting panorama 
    panorama_corners = np.array([[0, 0], [0, height1], [width1, height1], [width1, 0]], dtype='float32').reshape(-1, 1, 2)
    warped_corners   = cv2.perspectiveTransform(panorama_corners, H) 
    all_corners      = np.concatenate((panorama_corners, warped_corners), axis=0)
    [x_min, y_min]   = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max]   = np.int32(all_corners.max(axis=0).ravel() + 0.5)
   
    # Warp the img1 to the panoramic view
    translation_dist = [-x_min, -y_min]
    H_translation    = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]]) 
    panorama         = cv2.warpPerspective(img1, H_translation.dot(H), (x_max - x_min, y_max - y_min))
    if show: my_cv.show(panorama, 'Image 1 Warped')
    
    # Step 7: Merge the aligned images and create the panoramic image
    panorama[translation_dist[1]:height2 + translation_dist[1], translation_dist[0]:width2 + translation_dist[0]] = img2  
    if show: my_cv.show(panorama, 'Panoramic Image')   
    
    # Step 8: Draw lines between corresponding points in the pair of match_images
    flags       = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    match_color = (0, 255, 0)
    img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, matchColor=match_color, flags=flags)
    if show: my_cv.show(img_matches, 'Good Matches')

    return panorama, img_matches

def main():
    # Step 0: Read the images
    imageA_path = "data/input_images_match/foto3A.jpg"
    imageB_path = "data/input_images_match/foto3B.jpg"
    imageA      = my_cv.read(imageA_path)
    imageB      = my_cv.read(imageB_path)
    
    # Matching
    good_matches_threshold = 0.60 # percentage
    match_images(imageA, imageB, good_matches_threshold, 'SIFT', True)

if __name__ == '__main__':
    main()
