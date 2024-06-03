import Image_processor as ip
import Data_extractor as de
import Data_processor as dp
import Model_fitter as mf
import Plotter as pt
import Camera_Comm as cc
#import Temp_Comm as tc
import numpy as np
from astropy.io import fits
import random
import os.path
import time
from pandas import DataFrame, read_excel

timedatloc = '/home/l4proj/Documents/Experiment/'

def setup_camera(itype, roidim, exposure):
    #Sets up camera to suit conditions of experiment
    cc.init()
    cam = cc.find_cam()
    cc.stop_video_mode(cam)
    maxval = cc.set_type(cam, itype)
    cc.set_speedmode(cam, True)
    if exposure == 0:
        #cc.start_video_mode(cam)
        cc.find_exposure(cam, int(maxval*4/5), 0.05)
    elif exposure != 0:
        cc.set_exposure(cam, int(abs(exposure)))
    #cc.start_video_mode(cam)
    if roidim != 0:
        if roidim%8 != 0:
            print('Crop dimensions must be divisible by 8')
        image = cc.cap_img(cam)
        image = ip.truncate_image(image, 1/2,  np.average(image[np.where(image>maxval/5)]))
        center = de.find_cog(image)
        roi = [int(center[1]-roidim/2), int(center[0]-roidim/2), roidim, roidim]
        cc.set_roi(roi, cam)
    print(cam.get_control_values())
    print(cam.get_roi())
    #cc.stop_video_mode()
    return cam, exposure

def retrieve_stack(path, num, skip = 0):
    #Opens images saved as .fits files, converts them into 3D numpy array format
    if skip == 0:
        for i in range(num):
            filename = str(path+str(i)+'.fits')
            hdu = fits.open(filename)
            img = hdu[0].data
            img = ip.truncate_image(img, 1/10, np.max(img))
            if i == 0:
                dim = img.shape
                stack = np.zeros([dim[0], dim[1], num], dtype = img.dtype)
            stack[:,:,i] = img
    else:
        for i in range(int(num/skip)):
            filename = str(path+str(i*skip)+'.fits')
            hdu = fits.open(filename)
            img = hdu[0].data
            img = ip.truncate_image(img, 1/10, np.max(img))
            if i == 0:
                dim = img.shape
                stack = np.zeros([dim[0], dim[1], int(num/skip)], dtype = img.dtype)
            stack[:,:,i] = img
    return stack

def create_stack(path, cam, num, save, sname, spath):
    #Captures a new set of images, saves them as .fits files and returns them in 3D numpy array format
    #cc.start_video_mode(cam)
    stime = time.time()
    if cc.asi.zwolib == None:
        print('Camera must be initialised before capturing images')
    for i in range(num):
        img = cc.cap_img(cam)
        if i == 0:
            dim = img.shape
            stack = np.zeros([dim[0], dim[1], num], dtype = img.dtype)
            times = np.zeros(num)
            t1s = np.zeros(num)
            t2s = np.zeros(num)
        stack[:,:,i] = img
        times[i] = time.time()-stime
        #t1s[i], t2s[i] = tc.get_temps()
        #pt.plot([{'image':img}], [[1,1]])
    #cc.stop_video_mode()
    #etime = time.time()
    #dat = pull_data('Timedat', timedatloc)
    #if dat != 0:
    #    dat['Num'] = dat['Num'][0] + [num]
    #    dat['Time'] = dat['Time'][0] + [etime-stime]
    #else:
    #    dat = {'Time':[etime-stime], 'Num':[num]}
    #print(etime-stime)
    #df = DataFrame(dat)
    #df.to_excel(timedatloc+'Timedat.xlsx', sheet_name = 'vals', index = False)
    df2 = DataFrame({'Times':times, 'T1s':t1s, 'T2s':t2s})
    df2.to_excel(spath+sname+'_tts.xlsx', sheet_name = 'vals', index = False)
    if save:
        for i in range(num):
            filename = str(path+str(i)+'.fits')
            hdu = fits.PrimaryHDU(stack[:,:,i])
            hdu.writeto(filename, overwrite = True)
        extract_and_save(stack, times, t1s, t2s, sname, spath)
    return stack

def poisson_stack(baseimg, num):
    sf = 20
    dim = baseimg.shape
    stack = np.zeros([dim[0], dim[1], num])
    for i in range(num):
        noise = np.random.poisson(baseimg.astype('float')*sf)/sf
        stack[:,:,i] = noise
    #stack = stack.astype('uint8')
    return stack

def extract_and_save(stack, times, t1s, t2s, name, path):
    ops = ['ycoord', 'xcoord', 'yfreq', 'ydisp', 'xfreq',  'xdisp', 'yscroll',  'xscroll', 'yffreq', 'yfdisp', 'xffreq',  'xfdisp', 'times', 't1s', 't2s']
    tdat = pull_data('Timedat', timedatloc)
    tint = np.sum(times)/len(times)
    ycoord, xcoord = de.extract_coords(stack, 'cog')
    ycoord, xcoord = dp.ignore_outliers(ycoord, xcoord, int(len(ycoord)/10))
    _, _, ydisp, xdisp, yfreq, xfreq = dp.find_stds(ycoord, xcoord, 2)
    samplerate = 1/tint
    _, xfilter = dp.create_filter([xcoord], samplerate, 0)
    xscroll = dp.filter_array(xcoord, xfilter)
    _, yfilter = dp.create_filter([ycoord], samplerate, 0)
    yscroll = dp.filter_array(ycoord, yfilter)
    _, _, yfdisp, xfdisp, yffreq, xffreq = dp.find_stds(yscroll, xscroll, 2)
    sd = {}
    for o in ops:
        sd.update({o:locals()[o]})
    df = DataFrame(sd)
    df.to_excel(path+name+'.xlsx', sheet_name='vals', index = False)
    
def pull_data(name, path):
    if os.path.isfile(path+name+'.xlsx'):
        df = read_excel(path+name+'.xlsx', sheet_name = 'vals', index_col = None)
        dat = df.to_dict('list')
        for k in dat.keys():
            dat[k] = [dat[k]]
        return dat
    else:
        print('No file exists with this name and path')
        return 0
        
def append_pull(pulldat, dat):
    ops = ['ycoord', 'xcoord', 'yfreq', 'ydisp', 'xfreq',  'xdisp', 'yscroll',  'xscroll', 'yffreq', 'yfdisp', 'xffreq',  'xfdisp', 'times', 't1s', 't2s']
    for o in ops:
        if o in dat:
            dat[o] = dat[o]+pulldat[o]
        else:
            dat.update({o:pulldat[o]})
    return dat 

def shrink_data(data, skip):
    ops = ['ycoord', 'xcoord', 'yfreq', 'ydisp', 'xfreq',  'xdisp', 'yscroll',  'xscroll', 'yffreq', 'yfdisp', 'xffreq',  'xfdisp', 'times', 't1s', 't2s']
    for o in ops:
        if np.array(data[o]).shape[0] != len(data[o]):
            #print(data[o])
            for i in range(len(data[o])):
                data[o][i] = data[o][i][np.where(np.array(range(len(data[o][i])))%skip == 0)]
        else:
            data[o] = np.array(data[o])[np.where(np.array(range(len(data[o])))%skip == 0)]
    return data

def remove_drift(data):
    yscroll = data['yscroll'].copy()
    xscroll = data['xscroll'].copy()
    ycoord2 = data['ycoord'].copy()
    xcoord2 = data['xcoord'].copy()
    for i in range(len(yscroll)):
        xscroll[i] = xscroll[i] * 1/np.average(np.abs(xscroll[i]))
        yscroll[i] = yscroll[i] * 1/np.average(np.abs(yscroll[i]))
        xcoord2[i] = xcoord2[i] - np.average(xcoord2[i])
        ycoord2[i] = ycoord2[i] - np.average(ycoord2[i])
    xdrift = np.average(xscroll, axis = 0)
    ydrift = np.average(yscroll, axis = 0)
    for i in range(len(yscroll)):
        xcoord2[i] = xcoord2[i]-xdrift*np.average(np.abs(data['xscroll'][i]))
        ycoord2[i] = ycoord2[i]-ydrift*np.average(np.abs(data['yscroll'][i]))
    ops = ['ycoord', 'xcoord', 'yfreq', 'ydisp', 'xfreq',  'xdisp', 'yscroll',  'xscroll', 'yffreq', 'yfdisp', 'xffreq',  'xfdisp', 'times', 't1s', 't2s']
    dat2 = {}
    for i in range(len(yscroll)):
        xcoord = xcoord2[i]
        ycoord = ycoord2[i]
        _, _, ydisp, xdisp, yfreq, xfreq = dp.find_stds(ycoord, xcoord, 2)
        samplerate = 5
        _, xfilter = dp.create_filter([xcoord], samplerate, 0)
        xscroll = dp.filter_array(xcoord, xfilter)
        _, yfilter = dp.create_filter([ycoord], samplerate, 0)
        yscroll = dp.filter_array(ycoord, yfilter)
        _, _, yfdisp, xfdisp, yffreq, xffreq = dp.find_stds(yscroll, xscroll, 2)
        for o in ops:
            if o in dat2:
                dat2[o] = dat2[o]+[locals()[o]]
            else:
                dat2.update({o:[locals()[o]]})
    return dat2

def average_FFT(vals, samplerate):
    ays, axs = dp.FFT2D(vals, samplerate)
    for i in ays:
        i = i * 1/np.average(i)
    avgys = np.average(ays, axis = 0)
    return avgys, axs[0]
    
def model_FFT(data, xs):
    xs2 = xs[1:len(data)//2]
    fps = mf.fit_FFT(data[1:int(len(data)/2)], xs2[:int(len(data)/2)])
    print(fps)
    if fps != []:
        mfft = np.zeros(len(xs))
        for i in range(len(xs)):
            if xs[i] == 0:
                mfft[i] = 0
            else:
                mfft[i] = fps[0]*abs(xs[i])**(-fps[1])
        #print(mfft)
        return mfft
    else:
        print('A model cannot be fit')
        return None

def est_FFT(data, xs):
    rmfft = model_FFT(np.real(data), xs)
    imfft = model_FFT(np.imag(data), xs)
    mfft = rmfft+1j*imfft
    mfft[np.where(xs < 0)] = mfft[np.where(xs < 0)]-2j*imfft[np.where(xs < 0)]
    pt.plot([{'xs':[xs, xs], 'ys':[np.imag(data), np.imag(mfft)]}], [[1,1]])
    return mfft
    
def binbytemp(data, binsize):
    ops = ['ycoord', 'xcoord', 'yfreq', 'ydisp', 'xfreq',  'xdisp', 'yscroll',  'xscroll', 'yffreq', 'yfdisp', 'xffreq',  'xfdisp', 'names', 'times', 't1s', 't2s']
    ptemps = data['t1s'][0]
    bintemps = np.arange(int(np.min(ptemps)), int(np.max(ptemps)), binsize)
    bintemps = bintemps[::-1]
    for i in range(len(bintemps)):
        idxs = np.where(np.logical_and(ptemps <= bintemps[i]+binsize/2, ptemps > bintemps[i]-binsize/2))[0]
        bindat = {}
        for o in ops:
            if o == 'names':
                if i == 0:
                    bindat.update({o:[str(bintemps[i])]})
                else:
                    bindat.update({o:[str(bintemps[i])]})
            else:
                if i == 0:
                    bindat.update({o:[np.array(data[o][0])[idxs]]})
                else:
                    bindat.update({o:[np.array(data[o][0])[idxs]]})
        if i == 0:
            ordat = bindat
        else:
            ordat = append_pull(bindat, ordat)
    return ordat, bintemps

def binbyint(data, interval, bintime = 1):
    ops = ['ycoord', 'xcoord', 'yfreq', 'ydisp', 'xfreq',  'xdisp', 'yscroll',  'xscroll', 'yffreq', 'yfdisp', 'xffreq',  'xfdisp', 't1s', 't2s']
    if not bintime:
        ops = ops + ['times']
    binidxs = np.arange(0, len(data['ycoord'][0]), interval)
    binidxs = np.append(binidxs, [len(data['ycoord'][0])]).astype('uint16')
    for i in range(len(binidxs)-2):
        bindat = {}
        for o in ops:
            #print(np.array(data[o][0]))
            #print(np.array(data[o][0])[binidxs[i]:binidxs[i+1]])
            bindat.update({o:[np.array(data[o][0])[binidxs[i]:binidxs[i+1]]]})
        if bintime:
            bindat.update({'times':[np.array(data['times'][0])[binidxs[i]:binidxs[i+1]]-data['times'][0][binidxs[i]]]})
        if i == 0:
            ordat = bindat
            ordat.update({'names':[np.round(np.average(np.array(data['t1s'][0][binidxs[i]:binidxs[i+1]])), 3)]})
        else:
            ordat = append_pull(bindat, ordat)
            ordat['names'] = ordat['names']+[np.round(np.average(np.array(data['t1s'][0][binidxs[i]:binidxs[i+1]])), 3)]
    return ordat

def slice_data(data, s):
    ops = ['ycoord', 'xcoord', 'yfreq', 'ydisp', 'xfreq',  'xdisp', 'yscroll',  'xscroll', 'yffreq', 'yfdisp', 'xffreq',  'xfdisp', 't1s', 't2s', 'times']
    for o in ops:
        if o in data:
            for i in range(len(data[o])):
                data[o][i] = data[o][i][s[0]:s[1]]
    return data
    
def recalculate_bins(name, path, bs):
    ops = ['ycoord', 'xcoord', 'yfreq', 'ydisp', 'xfreq',  'xdisp', 'yscroll',  'xscroll', 'yffreq', 'yfdisp', 'xffreq',  'xfdisp', 'times', 't1s', 't2s']
    recdat = pull_data(name, path)
    xcoord = np.array(recdat['xcoord'])
    ycoord = np.array(recdat['ycoord'])
    #ycoord, xcoord = dp.ignore_outliers(ycoord, xcoord)
    #print(ycoord, xcoord)
    _, _, ydisp, xdisp, yfreq, xfreq = dp.find_stds(ycoord, xcoord, bs)
    samplerate = 5
    _, xfilter = dp.create_filter([xcoord], samplerate, 0)
    xscroll = dp.filter_array(xcoord, xfilter)
    _, yfilter = dp.create_filter([ycoord], samplerate, 0)
    yscroll = dp.filter_array(ycoord, yfilter)
    _, _, yfdisp, xfdisp, yffreq, xffreq = dp.find_stds(yscroll, xscroll, bs)
    sd = {}
    for o in ops:
        sd.update({o:locals()[o]})
    df = DataFrame(sd)
    df.to_excel(path+name+'.xlsx', sheet_name='vals', index = False)

def refilter_by_bin(ordat, samplerate, num, f = None):
    if len(ordat['xcoord']) == 1:
        scale = 2000
    else:
        scale = 0
    for i in range(len(ordat['xcoord'])):
        print(i)
        xcoord = ordat['xcoord'][i]
        ycoord = ordat['ycoord'][i]
        ycoord, xcoord = dp.ignore_outliers(ycoord, xcoord, scale)
        _, xfilter = dp.create_filter([xcoord], samplerate, 1, f = f)
        ordat['xscroll'][i] = dp.filter_array(xcoord, xfilter)
        _, yfilter = dp.create_filter([ycoord], samplerate, 0, f = f)
        ordat['yscroll'][i] = dp.filter_array(ycoord, yfilter)
        if num != 0:
            ordat = remove_FFTpeaks(ordat, num, i)
        _, _, ordat['yfdisp'][i], ordat['xfdisp'][i], ordat['yffreq'][i], ordat['xffreq'][i] = dp.find_stds(ordat['yscroll'][i], ordat['xscroll'][i], 3)
    return ordat

def remove_FFTpeaks(ordat, num, idx):
    xft, _ = dp.FFT(ordat['xscroll'][idx], 3)
    yft, _ = dp.FFT(ordat['yscroll'][idx], 3)
    txft = xft.copy()
    tyft = yft.copy()
    if num == 1:
        xft[np.where(np.logical_and(abs(np.real(xft)) != np.max(abs(np.real(xft))), abs(np.imag(xft)) != np.max(abs(np.imag(xft)))))] = 0+0j
        yft[np.where(np.logical_and(abs(np.real(yft)) != np.max(abs(np.real(yft))), abs(np.imag(yft)) != np.max(abs(np.imag(yft)))))] = 0+0j
        ordat['xscroll'][idx] = ordat['xscroll'][idx] - dp.ifft(xft)
        ordat['yscroll'][idx] = ordat['yscroll'][idx] - dp.ifft(yft)
    else:
        for i in range(num):
            txft[np.where(abs(np.real(txft)) == np.max(abs(np.real(txft))))] = 0+0j
            txft[np.where(abs(np.imag(txft)) == np.max(abs(np.imag(txft))))] = 0+0j
            tyft[np.where(abs(np.real(tyft)) == np.max(abs(np.real(tyft))))] = 0+0j
            tyft[np.where(abs(np.imag(tyft)) == np.max(abs(np.imag(tyft))))] = 0+0j
        ordat['xscroll'][idx] = ordat['xscroll'][idx] - dp.ifft(xft-txft)
        ordat['yscroll'][idx] = ordat['yscroll'][idx] - dp.ifft(yft-tyft)
    return ordat
        

def query(typename):
    #name query format = [Axislabel, Legendlabel, Description]
    tns = {'ycoord':['Position along image vertical axis (pixels)', 'Detected coordinate', 'ycoord is the y coordinate of the detected centroid of the image'],
           'xcoord':['Position along image horizontal axis (pixels)', 'Detected coordinate', 'xcoord is the x coordinate of the detected centroid of the image'],
           'ystd':['', 'Standard deviation in y axis', 'ystd describes the standard deviation of detected centroids around the average in the image vertical axis'],
           'xstd':['', 'Standard deviation in x axis', 'xstd describes the standard deciation of detected centroids around the average in the image horizontal axis'],
           'yfreq':['Frequency of position along image vertical axis', 'Y posisition distribution', 'yfreq describes the number of times the detected coordinate was found to have a certain position in y'],
           'xfreq':['Frequency of position along image horizontal axis', 'X position distribution', 'xfreq describes the number of times the detected coordinate was found to have a certain position in x'],
           'ydisp':['Position of detected centroid in vertical image axis (pixels)', 'Y position distribution', 'ydisp describes the available positions of detected centroids in the image binned to 2 d.p'],
           'xdisp':['Position of detected centroid in horizontal image axis (pixels', 'X position distribution', 'xdisp describes the available positions of detected centroids in the image binned to 2 d.p'],
           'yscroll':['Scrolling average of position in image vertical axis (pixels)', 'Scrolling average', 'yscroll describes the average position in length n slices of all detected positions across all detected positions'],
           'xscroll':['Scrolling average of position in image horizontal axis (pixels)', 'Scrolling average', 'xscroll describes the average position in length n slices of all detected positions across all detected positions']}
    if typename not in tns:
        print(typename+' is not a queriable entry in current dictionary format. If it should be available, an entry is missing in query.')
    else:
        return tns[typename]
    
    