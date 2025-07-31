
####################################################
# Computer Vision Module with additional functions #
####################################################

# Works as a layer of abstraction between the OpenCV library and the rest of the code

import os
import cv2
import numpy as np 
from matplotlib import pyplot as plt

from skimage import img_as_float, img_as_ubyte
from skimage.transform import rotate, rescale, resize

from skimage import metrics

import interpolations

#---------------------------------------------------
# Images files i/o functions

# Show the requested image
def show(image, title='Image', scale=1.0):
    cv2.namedWindow(title)
    cv2.moveWindow(title, 40,30)
    image = cv2.resize(image, None, fx=scale, fy=scale)
    cv2.imshow(title, image)
    wait_time = 1000
    while cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) >= 1:
        keyCode = cv2.waitKey(wait_time)
        if(keyCode & 0xFF) == ord("q"):
            break
    cv2.destroyAllWindows()

# Read the requested image 
def read(image_path, scale=1.0, as_gray=False):
    image = cv2.imread(image_path)
    if as_gray:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image

# Write the requested image 
def write(image, image_output_path, scale=1.0):
    cv2.imwrite(image_output_path, image)

#---------------------------------------------------
# File check functions
def check_file_path(file_path):
    if not os.path.exists(file_path):
        print(f"  \033[2;31;43m [ERROR] \033[0;0m The path {file_path} does not exist.")
        return None
    if not os.path.isfile(file_path):
        print(f"  \033[2;31;43m [ERROR] \033[0;0m The path {file_path} exists, but it's not a file.")
        return None
    return file_path

# Checks if the path exists and is a PNG image
def check_file_path_image(file_path, check_exists=True):
    if check_exists:
        file_path = check_file_path(file_path)
    if not file_path:
        return None
    if (not file_path.lower().endswith(".png")) and (not file_path.lower().endswith(".jpg")) and (not file_path.lower().endswith(".jpeg") ):
        print(f"  \033[2;31;43m [ERROR] \033[0;0m The path {file_path} exists, but it's not a PNG or JPG image.")
        return None
    return file_path

# Checks if the path exists and is a folder
def check_folder_path(folder_path):
    if folder_path == "":
        return folder_path
    if not os.path.exists(folder_path):
        print(f"  \033[2;31;43m [ERROR] \033[0;0m The path {folder_path} does not exist.")
        return None
    if not os.path.isdir(folder_path):
        print(f"  \033[2;31;43m [ERROR] \033[0;0m The path {folder_path} exists, but it's not a folder.")
        return None
    return folder_path

#-------------------------------------------------
# Skimage rounds the image dimensions after rotation different from 
# my_functions and openCV, sometimes is needed to crop 1 row or 1 column
# of pixels in order to be able to compare images correctly
def fix_size(image1, image2):
    if image1.shape[:2] != image2.shape[:2]:
        # Determine which image is larger
        if image1.shape[0] > image2.shape[0]:
            larger_image  = image1 
            smaller_image = image2
        else:
            larger_image  = image2 
            smaller_image = image1

        # Crop one pixel line or column from the larger image
        if larger_image.shape[0] > smaller_image.shape[0]:
            larger_image = np.delete(larger_image, 0, axis=0)  # Remove one row
        elif larger_image.shape[1] > smaller_image.shape[1]:
            larger_image = np.delete(larger_image, 0, axis=1)  # Remove one column
    
        return larger_image, smaller_image

    return image1, image2

# Compare two images by using some of skimage metrics
def compare(image1, image2, title):
    print(title)
    image1, image2 = fix_size(image1, image2)

    haussdorff = metrics.hausdorff_distance(image1, image2)
    print("Hausdorff: ", haussdorff)

    nmi = metrics.normalized_mutual_information(image1, image2)
    print("NMI: ", nmi)

    nrmse = metrics.normalized_root_mse(image1, image2)
    print("NRMSE: ", nrmse, end='\n\n')
    
#-------------------------------------------------------------------------------------------------------------------------------
# My Own part A functions

# Transform the image by scaling and rotation using some interpolation
def transform_image(image, transformation_matrix, interpolation='bilinear'):
    height, width = image.shape[:2]
     
    # Compute the new image bounds
    corners = np.array([
        [0, 0, 1],
        [width, 0, 1],
        [0, height, 1],
        [width, height, 1]
    ])

    # Get new image dimensions
    transformed_corners = np.dot(corners, transformation_matrix.T)
    min_x, max_x        = np.min(transformed_corners[:, 0]), np.max(transformed_corners[:, 0])
    min_y, max_y        = np.min(transformed_corners[:, 1]), np.max(transformed_corners[:, 1]) 
 
    # Create the output image with new dimensions
    new_width           = int(np.floor(max_x - min_x))
    new_height          = int(np.floor(max_y - min_y)) 

    # Deal with grayscale and color images
    if image.ndim == 3:
        transformed_image = np.zeros((new_height, new_width, 3), dtype=image.dtype)
    else:
        transformed_image = np.zeros((new_height, new_width), dtype=image.dtype)

    # Update the transformation matrix to account for translation due 
    # to correction of the image bounds
    translation_matrix = np.array([
        [1, 0, -min_x],
        [0, 1, -min_y],
        [0, 0, 1]
    ])

    # Compute the final transformation matrix
    final_transformation_matrix = translation_matrix @ transformation_matrix

    # Create a grid of coordinates in the output image
    xx, yy      = np.meshgrid(np.arange(new_width), np.arange(new_height))
    dest_coords = np.stack([xx, yy, np.ones_like(xx)], axis=-1) # homogeneous coordinates
    
    # Calculate the source coordinates
    src_coords   = np.tensordot(dest_coords, np.linalg.inv(final_transformation_matrix).T, axes=([2], [0]))
    src_x, src_y = src_coords[..., 0], src_coords[..., 1]
    
    # Check if the coordinates are within bounds
    valid_mask = (0 <= src_x) & (src_x < width) & (0 <= src_y) & (src_y < height)
    
    # Perform bilinear interpolation
    if interpolation == 'nearest':
        transformed_image[valid_mask] = interpolations.nearest_neighbor(image, src_x[valid_mask], src_y[valid_mask])
    elif interpolation == 'bilinear':
        transformed_image[valid_mask] = interpolations.bilinear(image, src_x[valid_mask], src_y[valid_mask])
    elif interpolation == 'bicubic':
        transformed_image[valid_mask] = interpolations.bicubic(image, src_x[valid_mask], src_y[valid_mask])        
    elif interpolation == 'lagrangean':
        transformed_image[valid_mask] = interpolations.lagrangean(image, src_x[valid_mask], src_y[valid_mask])      
        
    return transformed_image    

def my_scale_image(image, scale, interpolation='bilinear'):
    scale_matrix = np.array([
        [scale, 0, 0],
        [0, scale, 0],
        [0, 0, 1]
    ]) 
    
    return transform_image(image, scale_matrix, interpolation)

def my_rotate_image(image, angle, interpolation='bilinear'):
    theta     = np.radians(-angle)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    rotation_matrix = np.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta,  cos_theta, 0],
        [0, 0, 1]
    ]) 
    
    return transform_image(image, rotation_matrix, interpolation)

def my_resize_image(image, dim, interpolation='bilinear'):
    height, width         = image.shape[:2]
    new_width, new_height = dim
    x_scale = float(new_width) / float(width)
    y_scale = float(new_height) / float(height)

    resize_matrix = np.array([
        [x_scale, 0, 0],
        [0, y_scale, 0],
        [0, 0, 1]
    ])  
    
    return transform_image(image, resize_matrix, interpolation)

#-------------------------------------------------------------------------------------------------------------------------------
# OpenCV part A functions 

def opencv_rotate_image(image, angle, interpolation='bilinear'):
    # Compute rotation matrix
    (h, w) = image.shape[:2] 
    center = (w // 2, h // 2) 
    M = cv2.getRotationMatrix2D(center, angle, 1.0) 
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    
    # Compute the new bounding dimensions of the image
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # Adjust the rotation matrix to take into account the translation
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    # Define a dictionary to map the interpolation string to OpenCV's interpolation methods
    interpolation_methods = {
        'nearest': cv2.INTER_NEAREST,
        'bilinear': cv2.INTER_LINEAR,
        'bicubic': cv2.INTER_CUBIC,
    }
    
    # Get the interpolation method from the dictionary
    if interpolation not in interpolation_methods:
        raise ValueError(f"Invalid interpolation method: {interpolation}")
                          
    # Perform the rotation
    rotated = cv2.warpAffine(image, M, (new_w, new_h), flags=interpolation_methods[interpolation])
    return rotated


def opencv_scale_image(image, scale_factor, interpolation='bilinear'): 
    # Compute the new dimensions of the image
    (h, w) = image.shape[:2] 
    new_w  = int(w * scale_factor)
    new_h  = int(h * scale_factor)
    
    # Define a dictionary to map the interpolation string to OpenCV's interpolation methods
    interpolation_methods = {
        'nearest': cv2.INTER_NEAREST,
        'bilinear': cv2.INTER_LINEAR,
        'bicubic': cv2.INTER_CUBIC,
    }

    # Get the interpolation method from the dictionary
    if interpolation not in interpolation_methods:
        raise ValueError(f"Invalid interpolation method: {interpolation}") 
     
    # Perform the scaling
    scaled_image = cv2.resize(image, (new_w, new_h), interpolation=interpolation_methods[interpolation]) 
    return scaled_image


def opencv_resize_image(image, dim, interpolation='bilinear'):  
    # Define a dictionary to map the interpolation string to OpenCV's interpolation methods
    interpolation_methods = {
        'nearest': cv2.INTER_NEAREST,
        'bilinear': cv2.INTER_LINEAR_EXACT,
        'bicubic': cv2.INTER_CUBIC,
    }

    # Get the interpolation method from the dictionary
    if interpolation not in interpolation_methods:
        raise ValueError(f"Invalid interpolation method: {interpolation}") 
     
    # Perform the scaling
    resized_image = cv2.resize(image, dim, interpolation=interpolation_methods[interpolation]) 
    return resized_image

#-------------------------------------------------------------------------------------------------------------------------------
# Skimage part A functions 
def skimage_rotate_image(image, angle, interpolation='bilinear'):
    image = img_as_float(image)
    interpolation_methods = {
        'nearest': 0,
        'bilinear': 1,
        'bicubic': 3,
    }
    rotated = rotate(image, angle, resize=True, order=interpolation_methods[interpolation])
    return img_as_ubyte(rotated)

def skimage_scale_image(image, scale_factor, interpolation='bilinear'):
    image = img_as_float(image)
    height, width = image.shape[:2]
    new_dim = (int(height * scale_factor), int(width * scale_factor))
    interpolation_methods = {
        'nearest': 0,
        'bilinear': 1,
        'bicubic': 3,
    }
     
    scaled = resize(image, new_dim, preserve_range=True, anti_aliasing=False, order=interpolation_methods[interpolation])
    return img_as_ubyte(scaled)

def skimage_resize_image(image, dim, interpolation='bilinear'):
    image = img_as_float(image)
    new_h, new_w = dim
    interpolation_methods = {
        'nearest': 0,
        'bilinear': 1,
        'bicubic': 3,
    } 
    resized = resize(image, (new_w, new_h), anti_aliasing=False, order=interpolation_methods[interpolation])
    return img_as_ubyte(resized)
