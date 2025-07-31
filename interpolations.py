
import numpy as np

#------------------------------------------------------------------------------
# Nearest Neighbor Interpolation
def nearest_neighbor(image, x, y):
    x0 = np.round(x).astype(np.int32)
    y0 = np.round(y).astype(np.int32)

    # Keep x, y within image
    x0 = np.clip(x0, 0, image.shape[1] - 1)
    y0 = np.clip(y0, 0, image.shape[0] - 1) 
    
    return image[y0, x0]

#------------------------------------------------------------------------------
# Bilinear Interpolation
def bilinear(image, x, y):
    # Get nearest pixel
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32) 

    # Keep x, y within image
    x0 = np.clip(x0, 0, image.shape[1] - 2)
    y0 = np.clip(y0, 0, image.shape[0] - 2) 

    # Get pixel values
    Ia = image[y0, x0]
    Ib = image[y0, x0+1]
    Ic = image[y0+1, x0]
    Id = image[y0+1, x0+1]
     
    # Interpolation weights 
    dx = x - x0
    dy = y - y0
    
    wa = (1 - dx) * (1 - dy)
    wb = dx * (1 - dy) 
    wc = (1 - dx) * dy
    wd = dx * dy

    if image.ndim == 3:
        result = wa[..., None] * Ia + wb[..., None] * Ib + wc[..., None] * Ic + wd[..., None] * Id
    else:
        result = wa * Ia + wb * Ib + wc * Ic + wd * Id
               
    result = np.clip(result, 0, 255) 
    return result  
#------------------------------------------------------------------------------
# Cubic Interpolation 
def P(t):
    t[t < 0] = 0
    return t

def R(s):
    return (1/6) * ( (P(s + 2)**3) - 4*(P(s + 1)**3) + 6*(P(s)**3) - 4*(P(s - 1)**3) )

def bicubic(image, x, y):
    # Get nearest pixel 
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)

    # Keep x, y within image 
    x0 = np.clip(x0, 1, image.shape[1] - 3)
    y0 = np.clip(y0, 1, image.shape[0] - 3)
   
    # Create blank image
    if image.ndim == 3:
        result = np.zeros((x.shape[0], 3))
    else:
        result = np.zeros(x.shape[0])

    # Interpolation weights
    dx = x - x0
    dy = y - y0
    for i in range(4):
        for j in range(4):
            m = i - 1
            n = j - 1
            w = R(m - dx) * R(dy - n)

            if image.ndim == 3:
                result += image[y0 + m, x0 + n] * w[...,None]
            else:
                result += image[y0 + m, x0 + n] * w
               
    result = np.clip(result, 0, 255) 
    return result 

#------------------------------------------------------------------------------
# Lagrangean Interpolation 
def L(image, x0, y0, dx, n):
    # Get pixel values ensuring we stay within bounds
    height, width = image.shape[:2]
    
    # Adjust indices to be within bounds
    x_indices = np.clip([x0-1, x0, x0+1, x0+2], 0, width-1)
    y_index = np.clip(y0 + n - 2, 0, height-1)
    
    Ia = image[y_index, x_indices[0]]
    Ib = image[y_index, x_indices[1]]
    Ic = image[y_index, x_indices[2]]
    Id = image[y_index, x_indices[3]]
    
    wa = (-dx * (dx - 1) * (dx - 2)) / 6
    wb = ((dx + 1) * (dx - 1) * (dx - 2)) / 2
    wc = (-dx * (dx + 1) * (dx - 2)) / 2
    wd = (dx * (dx + 1) * (dx - 1)) / 6     

    if image.ndim == 3:
        result = wa[..., None] * Ia + wb[..., None] * Ib + wc[..., None] * Ic + wd[..., None] * Id
    else:
        result = wa * Ia + wb * Ib + wc * Ic + wd * Id

    result = np.clip(result, 0, 255) 
    return result
        
def lagrangean(image, x, y):
    # Get nearest pixel 
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)

    # Keep x, y within image 
    x0 = np.clip(x0, 0, image.shape[1] - 1)
    y0 = np.clip(y0, 0, image.shape[0] - 1)
    
    # Interpolation weights
    dx = x - x0
    dy = y - y0
   
    Ia = L(image, x0, y0, dx, 1)
    Ib = L(image, x0, y0, dx, 2)
    Ic = L(image, x0, y0, dx, 3)
    Id = L(image, x0, y0, dx, 4)
 
    wa = (-dy * (dy - 1) * (dy - 2)) / 6
    wb = ((dy + 1) * (dy - 1) * (dy - 2)) / 2
    wc = (-dy * (dy + 1) * (dy - 2)) / 2
    wd = (dy * (dy + 1) * (dy - 1)) / 6  
    
    if image.ndim == 3:
        result = wa[..., None] * Ia + wb[..., None] * Ib + wc[..., None] * Ic + wd[..., None] * Id
    else:
        result = wa * Ia + wb * Ib + wc * Ic + wd * Id

    result = np.clip(result, 0, 255) 
    return result
        
