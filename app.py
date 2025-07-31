

import os
import numpy as np
import my_cv

from matching import match_images

def main():
    print("Welcome! To the image editor 2049!")
    print("What operation do you want to perform ?")
    print("(1) Scale")
    print("(2) Rotate")
    print("(3) Resize")
    print("(4) Panoramic Merge")
    print("(5) Exit")

    operation = int(input("Choose an operation: ")) 
    match operation:
        case 1 | 2 | 3:
            print("Please insert the image path: ", end="")
            input_path = my_cv.check_file_path_image(input())
            if not input_path: exit()
            
            input_img  = my_cv.read(input_path)
            
            print("Please insert the folder where to save the result image: ", end="")
            output_path = my_cv.check_folder_path(input())
            if not output_path: exit()

            # Get the user input for the operation
            match operation:
                case 1:
                    scale = float(input("Choose the scale factor: "))
                case 2:
                    angle = float(input("Choose the rotation angle (counter-clockwise): ")) 
                case 3:
                    dim   = (int(input("Choose the width: ")), int(input("Choose the height: ")))

            print("\nPlease choose which kind of interpolation you would like to use: ")
            print("(1) Nearest Neighbor")
            print("(2) Bilinear")
            print("(3) Bicubic")
            print("(4) Lagrangean")
            interpolation = int(input("Choose an interpolation: ")) 
            if(interpolation < 1 or interpolation > 4): 
                print("Invalid interpolation option. Please choose a number between 1-4")
                exit()

            # Apply the operation 
            interpolations = ['nearest', 'bilinear', 'bicubic', 'lagrangean']
            match operation:
                case 1:
                    out = my_cv.my_scale_image(input_img, scale, interpolation=interpolations[interpolation-1])
                case 2:
                    out = my_cv.my_rotate_image(input_img, angle, interpolation=interpolations[interpolation-1])
                case 3:
                    out = my_cv.my_resize_image(input_img, dim, interpolation=interpolations[interpolation-1])
            
            # Save the result
            my_cv.show(out, 'Result Image')
            file_name   = os.path.basename(input_path)
            output_path = os.path.join(output_path, 'result_' + file_name) 
            my_cv.write(out, output_path)
                    
        case 4:
            print("Please insert the image A path: ", end="") 
            input_path_imgA = my_cv.check_file_path_image(input())
            if not input_path_imgA: exit()
            imgA            = my_cv.read(input_path_imgA)
            
            print("Please insert the image B path: ", end="") 
            input_path_imgB = my_cv.check_file_path_image(input()) 
            if not input_path_imgB: exit()
            
            imgB            = my_cv.read(input_path_imgB)
           
            print("Please insert the folder where to save the result image: ", end="")
            output_path = my_cv.check_folder_path(input())
            if not output_path: exit()
           
            # Create the panoramic image
            good_matches_threshold = float(input("Choose the matches threshold (0-1): "))
            if good_matches_threshold < 0 or good_matches_threshold > 1:
                print("Please choose a number between 0-1")
                exit()

            print("\nPlease choose which descriptor you would like to use: ")
            print("(1) SIFT")
            print("(2) BRIEF")
            print("(3) ORB")
            descriptor = int(input("Choose a descriptor: ")) 
            if(descriptor < 1 or descriptor > 3): 
                print("Invalid descriptor option. Please choose a number between 1-3")
                exit()
               
            descriptors = ['SIFT', 'BRIEF', 'ORB']
            out, _ = match_images(imgA, imgB, good_matches_threshold, descriptors[descriptor-1],show=True)

            # Save the result
            output_path = os.path.join(output_path, 'panoram.png') 
            my_cv.write(out, output_path) 
            
        case 5:
            print("Thanks for using our services :D")
            exit()
        case _:
            print("Invalid operation. Please choose a number between 1-5")
            exit()
 
if __name__ == '__main__':
    main()
