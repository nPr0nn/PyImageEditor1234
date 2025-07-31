
import os
import cv2
import my_cv
import matplotlib.pyplot as plt

from matching import match_images

def compare_geometric_transforms(input_path, operation, param, inter, output_path):         
    input_image = my_cv.read(input_path)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    
    if operation == 'scale':
        opencv_image  = my_cv.opencv_scale_image(input_image, param, interpolation=inter)
        my_image      = my_cv.my_scale_image(input_image, param, interpolation=inter)
        skimage_image = my_cv.skimage_scale_image(input_image, param, interpolation=inter)
        
    elif operation == 'rotate':
        opencv_image  = my_cv.opencv_rotate_image(input_image, param, interpolation=inter) 
        my_image      = my_cv.my_rotate_image(input_image, param, interpolation=inter)  
        skimage_image = my_cv.skimage_rotate_image(input_image, param, interpolation=inter)  
        
    elif operation == 'resize': 
        opencv_image  = my_cv.opencv_resize_image(input_image, param, interpolation=inter)  
        my_image      = my_cv.my_resize_image(input_image, param, interpolation=inter)
        skimage_image = my_cv.skimage_resize_image(input_image, param, interpolation=inter)  
        
    # Computes metrics
    my_cv.compare(my_image, opencv_image, "My vs OpenCV")
    my_cv.compare(my_image, skimage_image, "My vs Scikit-Image")
    my_cv.compare(opencv_image, skimage_image, "OpenCV vs Scikit-Image")   
    
    # Show the results
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    titles    = ['My Function', 'OpenCV Function', 'Scikit-Image Function']
    images    = [my_image, opencv_image, skimage_image]
    for ax, title, image in zip(axes, titles, images):
        ax.imshow(image)
        ax.set_title(title)
        ax.axis('off')
    plt.margins(0, 0) 
    plt.tight_layout()

    # Save the results
    file_name   = os.path.basename(input_path)
    plt.savefig(os.path.join(output_path, operation + '_' + inter + file_name))
    plt.show()
    
def compare_descriptors(imgA_path, imgB_path, threshold, output_path):
    imgA = my_cv.read(imgA_path) 
    imgB = my_cv.read(imgB_path) 
    imgA = cv2.cvtColor(imgA, cv2.COLOR_BGR2RGB)
    imgB = cv2.cvtColor(imgB, cv2.COLOR_BGR2RGB)

    sift_out, sift_matches   = match_images(imgA, imgB, threshold, 'SIFT', False)
    brief_out, brief_matches = match_images(imgA, imgB, threshold, 'BRIEF', False)
    orb_out, orb_matches     = match_images(imgA, imgB, threshold, 'ORB', False)

    # Show the results 
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    titles    = ['SIFT Keypoints Matching', 'BRIEF Keypoints Matching', 'ORB Keypoints Matching']
    images    = [sift_matches, brief_matches, orb_matches]
    for ax, title, image in zip(axes, titles, images):
        ax.imshow(image)
        ax.set_title(title)
        ax.axis('off')
    plt.margins(0, 0)    
    plt.tight_layout()
    file_name   = os.path.basename(imgA_path)
    plt.savefig(os.path.join(output_path, 'matches_' + str(int(threshold*100)) + '_' + file_name), dpi=200)
    plt.show()
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    titles    = ['SIFT Panoram', 'BRIEF Panoram', 'ORB Panoram']
    images    =  [sift_out, brief_out, orb_out]
    for ax, title, image in zip(axes, titles, images):
        ax.imshow(image)
        ax.set_title(title)
        ax.axis('off') 
    plt.margins(0, 0)
    plt.tight_layout()
    file_name   = os.path.basename(imgA_path)
    plt.savefig(os.path.join(output_path, 'panoram_' + str(int(threshold*100)) + '_' + file_name), dpi=200)
    plt.show()
    
def test_geometric(out):
    out = os.path.join(out, 'geometric') 
    if os.path.exists(out) == False: os.makedirs(out)
    
    compare_geometric_transforms('data/input_images/face.png', 'scale', 0.54, 'nearest', out)
    compare_geometric_transforms('data/input_images/baboon.png', 'resize', (64, 64), 'bilinear', out) 
    compare_geometric_transforms('data/input_images/house.png', 'rotate', 45, 'bicubic', out)  

    
def test_descriptors(out):
    out = os.path.join(out, 'descriptors')
    if os.path.exists(out) == False: os.makedirs(out)
        
    # Various thresholds
    compare_descriptors('data/input_images_match/foto1A.jpg', 'data/input_images_match/foto1B.jpg', 0.20, out)
    compare_descriptors('data/input_images_match/foto1A.jpg', 'data/input_images_match/foto1B.jpg', 0.30, out)
    compare_descriptors('data/input_images_match/foto1A.jpg', 'data/input_images_match/foto1B.jpg', 0.40, out)
    compare_descriptors('data/input_images_match/foto1A.jpg', 'data/input_images_match/foto1B.jpg', 0.60, out)
    compare_descriptors('data/input_images_match/foto1A.jpg', 'data/input_images_match/foto1B.jpg', 0.80, out)
    compare_descriptors('data/input_images_match/foto1A.jpg', 'data/input_images_match/foto1B.jpg', 1.00, out)

    compare_descriptors('data/input_images_match/foto2A.jpg', 'data/input_images_match/foto2B.jpg', 0.60, out)   
    compare_descriptors('data/input_images_match/foto3A.jpg', 'data/input_images_match/foto3B.jpg', 0.60, out) 
    compare_descriptors('data/input_images_match/foto3B.jpg', 'data/input_images_match/foto3A.jpg', 0.60, out) 
    
    compare_descriptors('data/input_images_match/foto4A.jpg', 'data/input_images_match/foto4B.jpg', 0.60, out)  
    compare_descriptors('data/input_images_match/foto5A.jpg', 'data/input_images_match/foto5B.jpg', 0.60, out) 
    pass

if __name__ == '__main__':
    tests_output_folder = "data/tests"
    test_geometric(tests_output_folder)
    test_descriptors(tests_output_folder)    
