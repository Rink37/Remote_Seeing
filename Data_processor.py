import numpy as np
import Plotter as pt
import Data_extractor as de
import Image_processor as ip
import Model_fitter as mf
from scipy.fft import fft, ifft, fftfreq

#This document defines the set of processes which are done to data to retrieve useful information

def get_raw(stack):
    #This function extracts COM coordinates and their errors from images in a stack
    y_raw, x_raw = de.extract_coords(stack) #COM coordinates directly from the images
    avgimg = ip.average_stack(stack)
    pstack = ip.poisson_stack(avgimg, stack.shape[2])-stack
    py, px = de.extract_coords(pstack)
    ystd = np.std(py)
    xstd = np.std(px)
    y_raw_err = np.array([ystd]*len(y_raw))
    x_raw_err = np.array([xstd]*len(x_raw))
    y_raw, x_raw, oidxs = ignore_outliers(y_raw, x_raw, int(len(y_raw)/50)) #Removes values which deviate significantly from the trends, see 'Data_processor.py' for detail
    y_raw_err[oidxs] = 0 #oidxs is the index of any outlier coordinates, which we have artificially changed so we remove these errors
    x_raw_err[oidxs] = 0
    return y_raw, y_raw_err, x_raw, x_raw_err

def get_NTD(r_raw, r_raw_err, s, f = 0):
    #Here we filter to get only non turbulent drift
    nms = [-1, 0, 1] #Scale of error added
    ras = np.array([r_raw, r_raw, r_raw])
    rfs = np.zeros(3)
    for i in range(len(nms)):
        rarray = r_raw+nms[i]*r_raw_err
        rarray = rarray-np.average(rarray)
        rFFT, rfreqs = FFT(rarray, s)
        rFFT = abs(rFFT[1:len(r_raw)//2]) #Consider only absolute FFT power spectrum at f>0
        rfreqs = rfreqs[1:len(r_raw)//2]
        if np.max(rFFT) > 1:
            rfTM = rfreqs[np.max(np.where(rFFT > np.sqrt(np.max(rFFT)))[0])] #fTM is the maximim frequency with a power in the condition range
        else:
            print('Maximum power of FFT is', np.max(rFFT), ', results may be unpredictable')
            rfTM = rfreqs[np.max(np.where(rFFT*100 > np.sqrt(np.max(rFFT*100)))[0])] #fTM is the maximim frequency with a power in the condition range
        rfs[i] = rfTM
        if f == 0:
            _, rfilter = create_filter(rarray, s, f=rfs[i]) #Creating an array which will produce the desired result when convolved with r_raw
        else:
            _, rfilter = create_filter(rarray, s, f=f)
        ras[i] = filter_array(rarray, rfilter)#Convolve r_raw with the filter function
    r_NTD = ras[1]
    r_NTD_err = np.average([abs(ras[0]-ras[1]), abs(ras[2]-ras[1])], axis = 0)
    rTM = rfs[1]
    rTM_err = np.average([abs(rfs[0]-rfs[1]), abs(rfs[2]-rfs[1])])
    return r_NTD, r_NTD_err, rTM, rTM_err

def get_TM(r_raw, r_raw_err, s, period = 0, fTM = 0, f = 0):
    #Here we filter to get turbulent motion and non turbulent drift (which we subtract after)
    nms = [-1, 0, 1] #Scale of error added
    if period == 0:
        period = len(r_raw)
        periods = np.array([0])
    else:
        periods = np.arange(0, int(len(r_raw)/period)*period+period, period)
    #Periods is an array containing the start and end index of each subsection (e.g size-30 dataset with period 10 would produce periods = [0, 10, 20, 30])
    if periods[-1] != len(r_raw):
        periods = np.append(periods, [len(r_raw)])
    #Above adds the end index of the array if N/period is not an integer
    ras = np.array([r_raw, r_raw, r_raw])
    rfs = np.zeros([3, len(periods)-1])
    for j in range(len(periods)):
        if j > 0:
            for i in range(len(nms)):
                rarray = r_raw[periods[j-1]:periods[j]]+nms[i]*r_raw_err[periods[j-1]:periods[j]]
                rarray = rarray-np.average(rarray)
                #This time the cut off frequency calculation is built into create_filter
                if f == 0:
                    f, rfilter = create_filter(rarray, s) #Creating an array which will produce the desired result when convolved with r_raw
                else:
                    _, rfilter = create_filter(rarray, s, f = f) #Creating an array which will produce the desired result when convolved with r_raw
                rfs[i, j-1] = f
                ras[i][periods[j-1]:periods[j]] = filter_array(rarray, rfilter)#Convolve r_raw with the filter function
    r_TM = ras[1]
    r_TM_err = np.average([abs(ras[0]-ras[1]), abs(ras[2]-ras[1])], axis = 0)
    rfCO = rfs[1]
    rfCO_err = np.average([abs(rfs[0]-rfs[1]), abs(rfs[2]-rfs[1])], axis = 0)
    return r_TM, r_TM_err, rfCO, rfCO_err
    
def ignore_outliers(ys, xs, period = 0):
    #In some data sets, we observed some measurements which were in significantly different positions
    #This is likely due to sudden intense speckle, or the appearance of some other bright spot in the image
    #We define outliers as anything which is not within 4 standard deviations of the mean
    #Since the mean is changing due to NTD, it is better if we separate the dataset into smaller sections which we assume to have a more constant mean
    if period == 0:
        period = len(ys)
        periods = np.array([0])
    else:
        periods = np.arange(0, int(len(ys)/period)*period+period, period)
    #Periods is an array containing the start and end index of each subsection (e.g size-30 dataset with period 10 would produce periods = [0, 10, 20, 30])
    if periods[-1] != len(ys):
        periods = np.append(periods, [len(ys)])
    #Above adds the end index of the array if N/period is not an integer
    idxs = np.array([], dtype = 'int') #array containing the index of any outlier
    for j in range(len(periods)):
        if j > 0:
            xavg = np.average(xs[periods[j-1]:periods[j]])
            xstd = np.std(xs[periods[j-1]:periods[j]])
            yavg = np.average(ys[periods[j-1]:periods[j]])
            ystd = np.std(ys[periods[j-1]:periods[j]])
            #Removing data points outright caused some significant issues later, so setting outliers to be the mean value within a period was chosen as a better option
            for i in range(period):
                if xavg - 4*xstd < xs[(j-1)*period + i] < xavg + 4*xstd:
                    continue
                else:
                    xs[(j-1)*period + i] = xavg
                    np.append(idxs, (j-1)*period + i)
            for i in range(period):
                if yavg - 4*ystd < ys[(j-1)*period + i] < yavg + 4*ystd:
                    continue
                else:
                    ys[(j-1)*period + i] = yavg
                    np.append(idxs, int((j-1)*period + i))
    return ys, xs, idxs

def iFFT(pows):
    cds = ifft(pows)
    return np.real(cds)-np.imag(cds)

def FFT(coords, srate):
    #Calculates the FFT of coords, as well as the frequencies the power spectrum corresponds to 
    coords = np.array(coords)
    cf = (fft(coords-np.average(coords)))
    tf = fftfreq(len(coords), 1/srate)
    return cf, tf

def FFT2D(coords, srate):
    #Calculates the FFT of an array containing multiple sets of coords and returns arrays containing the FFT of each set and its frequencies
    ays = np.zeros([len(coords), int(len(coords[0]))], dtype = np.complex128)#/2)-1], dtype = np.complex128)
    axs = np.zeros([len(coords), int(len(coords[0]))])#/2)-1])
    for v in range(len(coords)):
        val = coords[v]
        ys, xs = FFT(val, srate)
        ays[v] = ys
        axs[v] = xs
    return ays, axs
        
def create_filter(coords, samplerate, f = None, fTM = None):
    #This function creates a filter which has a FFT approximately equivalent to a step function which is 1 between -f<freq<f and 0 otherwise
    #The function was originally designed to only calculate the noise cut off filter function, but manually passing f means we can bypass this calculation to filter for NTD
    if f == None:
        ays, axs = FFT(coords, samplerate)
        ays = np.array([ays])
        axs = np.array([axs])
        avgs = np.average(ays[:, int(len(axs[0])*1/4):int(len(axs[0])*1/2)], axis = 1) #Mean of the horizontal regime
        stds = np.std(ays[:, int(len(axs[0])*1/4):int(len(axs[0])*1/2)], axis = 1) #Standard deviation of the horizontal regime
        for j in range(ays.shape[0]):
            idxs = np.where(abs(ays[j]) > avgs[j]+2.25*stds[j])[0] #Indexes of all results which are not within 2.5 standard deviations of the horizontal regime
            idxs2 = np.where(np.gradient(idxs) < 13)[0] #Indexes which are within 13 steps of each other
            if len(idxs2) > 0:
                f = axs[0][idxs2[len(idxs2)-1]]
            else:
                #If the condition fails we just assume f is the largest frequency with a spacing below average
                print('Failed')
                f = np.mean(axs[0])/2
    fxs = np.array(range(len(coords)))
    fys = np.zeros(len(fxs)) #We are making a function which has equal increments to the input function
    for i in fxs:
        if i > 0:
            fys[i] = np.sin(f*(i)*2*np.pi/samplerate)/(f*(i)*2*np.pi/samplerate) #Just a sinc function with the correct filter frequency to produce an ideal Fourier spectrum
    fys[0] = 1
    fys = fys*1/(2*np.sum(fys)) #Normalising the function
    fys = np.append(fys[::-1], fys) #Making the function symmetric about zero
    return f, fys

def filter_array(array, fys):
    avg = np.average(array) #This is just used to enforce that results are displacements
    center = int(len(fys)/2) #The middle index of the filter array
    farray = np.zeros(len(array)) #Filtered array
    for i in range(len(array)):
        retfys = fys[center-i:center-i+len(array)] #Crop filter array to small range
        tot = np.sum(retfys)
        if tot < 0.99 or tot > 1.01:
            retfys = (1/tot)*retfys #normalisation condition at boundary
        farray[i] = np.sum(np.multiply(array-avg, retfys)) #Convolution eqn.
    return farray #Filtered result is just a convolution of array, fys