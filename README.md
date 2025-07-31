# PyImage Editor 1234 🖼️

A command-line tool that demonstrates fundamental computer vision algorithms. This project provides functionality for common image manipulations like scaling, rotating, and resizing, with a focus on implementing the underlying geometric transformations from scratch. It also features a panoramic image stitcher.

-----

## 🧠 Core Concepts

This project is built as more than just a utility; it's an educational tool. The geometric transformations (scale, rotate, resize) are implemented from the ground up.

  * **Transformation Matrices**: Operations are handled by creating a 2D affine transformation matrix ($3 \times 3$) for the desired effect (e.g., a rotation matrix, a scaling matrix).
  * **Inverse Mapping**: To generate the output image, the program uses inverse mapping. It iterates through the coordinates of the *target* image, applies the *inverse* transformation matrix to find the corresponding location in the *source* image, and then calculates the pixel value.
  * **Custom Interpolation**: Pixel values for non-integer source coordinates are calculated using custom-built interpolation functions, including:
      * Nearest Neighbor
      * Bilinear
      * Bicubic
      * Lagrangean

The `my_cv.py` file serves as a custom abstraction layer, providing these functions and also its versions from popular libraries by wrapping calls to libraries like OpenCV and Scikit-image, so that the results can be compared.

-----

## ✨ Features

  * **Scale**: Enlarge or shrink an image using a custom transformation implementation.
  * **Rotate**: Rotate an image by a given angle, automatically resizing the image frame to fit the result.
  * **Resize**: Change image dimensions to a new width and height.
  * **Panoramic Merge**: Stitch two overlapping images together using feature matching.
  * **Choice of Interpolation**: Select from multiple high-quality interpolation algorithms for transformations.
  * **Choice of Feature Descriptors**: Use SIFT, BRIEF, or ORB for accurate keypoint matching in panoramas.

-----

## 🚀 Getting Started

### Prerequisites

Make sure you have **Python 3.10+** installed.

### Installation

1.  Clone the repository:
    ```sh
    git clone <your-repository-url>
    ```
2.  Navigate into the project directory:
    ```sh
    cd <repository-name>
    ```
3.  It's highly recommended to create and activate a virtual environment:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
4.  Install the required dependencies:
    ```sh
    pip install opencv-python numpy scikit-image matplotlib
    ```

-----

## 💻 Usage

Run the main application from the project's root directory:

```sh
python main.py
```

You will be greeted with the main menu. Follow the on-screen prompts to select an operation, provide the necessary file paths, and set the required parameters.

### Example Workflow:

1.  Run the script: `python main.py`
2.  Choose an operation from the menu (e.g., `1` for **Scale**).
3.  Enter the path to your source image (e.g., `images/city.jpg`).
4.  Enter the path to the output folder (e.g., `output/`).
5.  Enter the scale factor (e.g., `1.5`).
6.  Select an interpolation method (e.g., `2` for **Bilinear**).

The resulting image will be displayed on screen and saved as `result_city.jpg` in the specified output folder.
