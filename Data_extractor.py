import numpy as np
import Model_fitter as mf
import Image_processor as ip
from scipy import spatial

#Functions which are used to retrieve information from images, stacks
#A stack is just a 3D array of images layered on top of each other
#e.g. stack[:, :, 1] is the 1st 2D image array in a set of images

def find_com(image):
    #Function finds center of reference point using Center of Mass equation
    #Treats each pixel count as a particle with its pixel index as the location, and pixel count as its mass
    #See https://sciencing.com/center-of-mass-definition-equation-how-to-find-w-examples-13725851.html
    img = ip.truncate_image(image, 1/10, np.max(image)) #Truncation aims to remove background    
    Wy = np.sum(image, axis = 1)
    Wx = np.sum(image, axis = 0)
    wsx = 0
    wx = 0
    wsy = 0
    wy = 0
    for x in range(len(Wx)):
        wsx = wsx + x*Wx[x]
        wx = wx + Wx[x]
    for y in range(len(Wy)):
        wsy = wsy + y*Wy[y]
        wy = wy + Wy[y]
    com_x = wsx/wx
    com_y = wsy/wy
    return [com_y, com_x] #Returns COM when results are affected by the values of count error (cerr) and position error (perr)
        

def extract_coords(stack): 
    #all this function does is iterate through the entire stack and perform the COM calculation on each image
    xs = np.zeros(stack.shape[2])
    ys = np.zeros(stack.shape[2])
    for i in range(stack.shape[2]):
        img = stack[:,:,i]
        coord = find_com(stack[:,:,i]) #perr and cerr are the errors due to the increments in the image
        ys[i] = coord[0]
        xs[i] = coord[1]
    return ys, xs


