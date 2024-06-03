import numpy as np
import Camera_Comm as cc
import Data_extractor as de
from astropy.io import fits
  
#This is just a few image processing functions which are occasionally used  

def average_stack(stack):
    avgimg = np.average(stack, axis = 2)
    return avgimg

def crop_stack(stack, center, newdim):
    cstack = np.zeros([newdim[0], newdim[1], stack.shape[2]])
    for i in range(stack.shape[2]):
        cstack[:, :, i] = crop_image(stack[:, :, i], center, newdim)
    return cstack

def poisson_stack(baseimg, num):
    sf = 70
    dim = baseimg.shape
    pstack = np.zeros([dim[0], dim[1], num])
    for i in range(num):
        print(i)
        noise = np.random.poisson((baseimg.astype('float')*sf))/sf
        pstack[:,:,i] = noise
    return pstack

def truncate_image(image, scalefac, maxm):
    #This function truncates images by subtracting the truncation point from all counts and removing all non-positive resulting counts
    t = image.dtype
    image = (image - maxm*scalefac)
    image[np.where(image<0)] = 0
    image = image.astype(t)
    return image

def crop_image(image, center, newdim):
    #Crops an image to a new dimension about a central point
    #newdim must be divisible by 2
    cropimg = np.zeros(newdim)
    for y in range(newdim[0]):
        for x in range(newdim[1]):
            if int(round(y+center[0]-newdim[0]/2)) < image.shape[0] and int(round(x+center[1]-newdim[1]/2)) < image.shape[1]:
                cropimg[y,x] = image[int(round(y+center[0]-newdim[0]/2)), int(round(x+center[1]-newdim[1]/2))]
            else:
                if int(round(y+center[0]-newdim[0]/2)) <= image.shape[0]:
                    yc = cropimg.shape[0]-1
                if int(round(x+center[1]-newdim[1]/2)) <= image.shape[1]:
                    xc = cropimg.shape[1]-1
                cropimg[y,x] = image[yc, xc]
    return cropimg

def align_stack(stack, dim):
    num = stack.shape[2]
    astack = np.zeros([dim, dim, num])
    for i in range(num):
        img = stack[:,:,i]
        y, x = de.find_cog(img)
        crimg = crop_image(img, [y,x], [dim, dim])
        astack[:,:,i] = crimg
    avg = average_stack(astack)
    return avg, astack