import Image_processor as ip
import Data_extractor as de
import zwoasi as asi
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

#https://blog.stevemarple.co.uk/2016/12/aurorawatch-uk-camera-update-1.html
#https://zwoasi.readthedocs.io/en/latest/ 

def setup_camera(itype, roidim, exposure):
    if init(): #Initialise the library
        cam = find_cam()
        if cam == []:
            print('CAM error: No cameras are connected')
            return []
        else:
            maxval = set_type(cam, itype) #Finds the maximum allowed counts per image
            set_speedmode(cam, True) #Set the camera to high speed mode
            if exposure == 0: #This is the condition for us to try and detect the correct exposure
                tval = int(maxval*4/5) #What we want the maximum counts in an image to be
                tol = 0.05 #Termination condition is if result within 1-tol*tval<counts<1+tol*tval
                find_exposure(cam, tval, tol) #Find and set the exposure in microseconds
            else:
                set_exposure(cam, int(abs(exposure))) #Set the exposure in microseconds
            if roidim%8 !=0:
                print('Crop dimensions must be divisible by 8') #Built in camera condition
                roidim = 128 #Default
            image = cap_img(cam) #Take a calibration image
            image = ip.truncate_image(image, 1/2,  np.average(image[np.where(image>maxval/5)])) #This truncation is included in case the background is nonzero - in all my data this was not necessary
            center = de.find_com(image) #Find the center of the calibration image
            #roi format is [top left corner y, top left corner x, width, height]
            roi = [int(center[1]-roidim/2), int(center[0]-roidim/2), roidim, roidim]
            set_roi(roi, cam) #Sets the region of interest of the camera to a roidim x roidim area with the calibration image dot roughly in the center
            #This means that all images taken after setup will only be roidim x roidim image arrays, not the full camera resolution
            return cam
    else:
        return []

def init():
    #On different devices, the path to libASICamera2.so may be different
    path = r"/home/l4proj/ASIStudio/lib/libASICamera2.so"
    #Warns if the path to this file is incorrect
    if not os.path.exists(path):
        print('Camera_Comm: Path to the file libASICamera2.so is incorrect, correct in init() function')
        return False
    else:
        #If path exists, initialise ZWOASI
        if asi.zwolib == None:
            asi.init(r"/home/l4proj/ASIStudio/lib/libASICamera2.so")
            print('Library initialised')
        else:
            print('Library was already initialised')
        return True

def find_cam():
    cams = asi.list_cameras() #Finds all connected cameras
    if cams == []: #Gives a fail state if no cameras are connected
        return []
    else:
        cam = asi.Camera(cams[0]) #Select the first connected camera
        cam.set_control_value(asi.ASI_BANDWIDTHOVERLOAD, cam.get_controls()['BandWidth']['MinValue'])
        return cam #Return zwoasi camera object

def calculate_avgmax(image, tval):
    #Since speckling can cause the maximum value to be significantly different to the maximum 'beam intensity', we try not to rely on only the maximum count
    img2 = image[np.where(image>tval/4)] #Truncate image to above target value/4
    if len(img2) != 0 : #Condition just checks if any part of the image is not removed by truncation
        val = (np.average(img2)*2+np.max(image))/2
    else:
        val = np.max(image)
    return val #Treat this value as roughly the maximum laser intensity, but not maximum speckle intensity
    
def find_exposure(cam, tval, tol):
    #This function is fairly rough and could be improved
    #Attempts to find an exposure which returns results with around an average maximum of maxm
    cam.set_control_value(asi.ASI_GAIN, 0) #Gain is set to lowest allowed
    exposure = 10
    cam.set_control_value(asi.ASI_EXPOSURE, exposure) #Sets exposure
    image = cap_img(cam) #Image is a 2D numpy array
    val = calculate_avgmax(image, tval) #~Max laser intensity, used to check counts
    #This whole following section just varies exposure until val is within (1-tol)*tval < val < (1+tol)*tval
    while val < tval-tol*tval or val > tval+tol*tval:
        sf = tval/val #Scale factor representing distance between desired intensity and actual intensity
        if val<tval-tol*tval:
            exposure = exposure + int(10*sf)
            cam.set_control_value(asi.ASI_EXPOSURE, exposure)
            image1 = cap_img(cam)
            image2 = cap_img(cam)
            val = (calculate_avgmax(image1, tval)+calculate_avgmax(image2, tval))/1.95
        elif val>tval+tol*tval:
            exposure = exposure - int(10*sf)
            cam.set_control_value(asi.ASI_EXPOSURE, exposure)
            image1 = cap_img(cam)
            image2 = cap_img(cam)
            val = (calculate_avgmax(image1, tval)+calculate_avgmax(image2, tval))/1.95
    #This function is not particularly mathematically rigorous but in practice produced the correct results when used

def set_exposure(cam, exposure):
    #Sets the camera exposure in microseconds
    cam.set_control_value(asi.ASI_GAIN, 0)
    cam.set_control_value(asi.ASI_EXPOSURE, exposure)
    
def cap_img(cam):
    #Captures image and returns as 2D integer array with type (8 or 16 bit) defined by camera settings in set_type()
    return cam.capture(initial_sleep = 0.00001) 

def set_roi(roi, cam):
    #Set the region of interest of the camera, so it only returns readings from pixels within the region of interest
    cam.set_roi(start_x = roi[0], start_y = roi[1], width = roi[2], height = roi[3], bins = 1)
    
def set_speedmode(cam, hsp):
    cam.set_control_value(asi.ASI_HIGH_SPEED_MODE, hsp) #High speed mode just speeds up the camera
    
def set_type(cam, itype):
    #This sets the type of image captured by the camera to be either an 8-bit or 16-bit integer array
    if itype == 8:
        cam.set_image_type(asi.ASI_IMG_RAW8)
        maxval = 255
    elif itype == 16:
        #This number of counts is based on an artificial gain, not representative of a real photon count so in practice the only difference between 8-bit and 16-bit images is the file size
        cam.set_image_type(asi.ASI_IMG_RAW16)
        maxval = 65535
    else:
        print(itype, 'is not a valid image bit number')
        maxval = 0
    return maxval #This is used to tell us what number of counts is required to saturate an image