import Image_processor as ip
import Data_extractor as de
import Data_processor as dp
import Model_fitter as mf
import Plotter as pt
import Camera_Comm as cc
import Temp_Comm as tc
import numpy as np
import CC_calc as corr
import os
import time
from astropy.io import fits
import pandas as pd
from pandas import DataFrame, read_excel

#Find root path i.e. the path to where your Figures, Images, Programs, SaveData folders are
with open('root.txt') as f:
    root = f.readlines()[0]

def Cap_ds(CapName, N, Camsettings = [8, 128, 0]):
    #Function which captures images, time and temperature of probes and stores them appropriately
    #This function has not been tested since being rewritten for further use, so may not work as expected
    #CapName is the name of the dataset (e.g the ith image of 'TestImages' will be found in root\Images\TestImages\TestImages_i.fits)
    #N is the number of samples we want to take 
    #Camsettings format = [image type (8-bit or 16-bit), region of interest side length, exposure]
    if os.path.exists(root+f'\SaveData\{CapName}'):
        print('A data set with this name already exists, type Y if you mean to overwrite')
        overwrite = input("Overwrite?: ")
        if overwrite != 'Y':
            print('Stopping')
            return
    N = int(N) #Enforce that N is an integer
    cam = cc.setup_camera(Camsettings[0], Camsettings[1], Camsettings[2])
    if cam == []: 
        print('Setup has failed') 
        return
    else:
        psuc, ser = tc.find_probes() #Psuc is the success of connecting to serial output ser
        stime = time.time() #Find the time at the start of the data capture
        for i in range(N): #Total number of samples
            img = cc.cap_img(cam) #Takes an image
            if i == 0:
                #Here we create empty numpy arrays which are populated with data
                dim = img.shape #Equivalent to [roidim, roidim]
                stack = np.zeros([dim[0], dim[1], N], dtype = img.dtype) #Images are stored in a 3D numpy array before being saved as it is faster than saving each image now
                times = np.zeros(N)
                if psuc:
                    T1s = np.zeros(N) #Temperatures T read by probes 1 and 2 are only available if connecting to the serial port has not failed
                    T2s = np.zeros(N)
            stack[:,:, i] = img #Store captured image
            times[i] = time.time()-stime #Get time since the start of the data capture
            if psuc:
                T1s[i], T2s[i] = tc.get_temps(ser) #Get probe temperatures if available
        #First we save any data which cannot be stored as images
        #Times, temps are put in an excel file called (CapName)_tts.xlsx, stored in SaveData\Capname\
        if psuc:
            AdDat = DataFrame({'Times':times, 'T1s':T1s, 'T2s':T2s})
        else:
            AdDat = DataFrame({'Times':times})
        if not os.path.exists(root+f'\SaveData\{CapName}'):
            os.makedirs(root+f'\SaveData\{CapName}')
        AdDat.to_excel(root+f'\SaveData\{CapName}\{CapName}_tts.xlsx', sheet_name = 'vals', index = False)
        #Now we save images in the stack as .fits files
        #The ith image in the set is found at root\Images\(CapName)\(CapName)_i.fits
        for i in range(N):
            filename = root+f'\Images\{CapName}\{CapName}_'+str(i)+'.fits'
            hdu = fits.PrimaryHDU(stack[:,:,i])
            hdu.writeto(filename, overwrite = True)
        
def Process_ds(CapName, N = 0, rep = 1, fs = []):
    #Now we want to create a function which processes all our captured data into useable formats
    #Defining N is only necessary if no version of (CapName)_tts.xlsx exists
    if len(fs) != 4:
        CapName2 = CapName
    else:
        CapName2 = CapName+str(fs)
    if os.path.exists(root+f'\SaveData\{CapName}\{CapName2}.xlsx'): #Check if CapName has already been processed and if we wish to do it again
        if rep == 1:
            print(f'{CapName2} appears to have been processed already, type Y if you wish to reprocess')
            Rep = input("Reprocess?: ")
            if Rep != 'Y':
                return
            else:
                print('Continuing')
        elif rep == 0:
            print(f'{CapName2} already processed, skipping (set rep = 0 to disable autoskip)')
            return
    print(CapName, 'initialised')
    if os.path.exists(root+f'\SaveData\{CapName}\{CapName}_tts.xlsx'):
        df = read_excel(root+f'\SaveData\{CapName}\{CapName}_tts.xlsx', sheet_name = 'vals', index_col = None)
        AdDat = df.to_dict('list')
        N = len(AdDat['Times']) #If _tts exists, then AdDat['Time'] must exist - not the case for 'T1s' or 'T2s'
        print(CapName, 'N =', N, 'found')
    elif N == 0:
        if os.path.exists(root+f'\Images\{CapName}\\N.txt'):
            with open(root+f'\Images\{CapName}\\N.txt', 'r') as f:
                lines = f.readlines()
            N = int(lines[0])
            print(CapName, 'N =', N, 'found')
        else:
            print('You must define N if there is no _tts excel file or N.txt file available for', CapName, ' - returning')
            return
    if not os.path.exists(root+f'\SaveData\{CapName}'):
        os.makedirs(root+f'\SaveData\{CapName}')
    Impath = root+f'\Images\{CapName}\{CapName}'
    Savepath = root+f'\SaveData\{CapName}\{CapName2}.xlsx'
    if os.path.exists(root+f'\Images\{CapName}\Format.txt'):
        with open(root+f'\Images\{CapName}\Format.txt', 'r') as f:
            lines = f.readlines()
        if len(lines) != 0:
            ext = lines[0]
        else:
            ext = ''
    else:
        ext = '_'
    if not os.path.exists(Impath+ext+'0.fits'): #Check if we have any images for this set
        print('There do not appear to be any images available for this set')
        return
    else:
        print(f'Processing {CapName2}')
        for i in range(N):
            Fname = Impath+ext+str(i)+'.fits'
            hdu = fits.open(Fname)
            img = hdu[0].data
            if i == 0:
                dim = img.shape
                stack = np.zeros([dim[0], dim[1], N], dtype = img.dtype)
            stack[:, :, i] = img
        #creating the dictionary which will be saved to an excel document
        if N > 100:
            SaveDat = {'x_raw':[], 'x_raw_err':[], 'y_raw':[], 'y_raw_err':[],
                       'x_NTD':[], 'x_NTD_err':[], 'y_NTD':[], 'y_NTD_err':[],
                       'x_TM':[], 'x_TM_err':[], 'y_TM':[], 'y_TM_err':[],
                       'x_noise':[], 'x_noise_err':[], 'y_noise':[], 'y_noise_err':[], 'Times':[]}
            y_raw, y_raw_err, x_raw, x_raw_err = dp.get_raw(stack) #See Data_processor.py for processing details
            y_raw = y_raw-np.average(y_raw) #We make positions into displacements about mean position
            x_raw = x_raw-np.average(x_raw)
            if 'AdDat' in locals():
                Times = AdDat['Times']
                if 'T1s' in AdDat:
                    SaveDat.update({'T1s':AdDat['T1s']})
                if 'T2s' in AdDat:
                    SaveDat.update({'T2s':AdDat['T2s']})
                s = Times[-1]/len(Times) #Assume constant sample rate s
            else:
                s = 3.15
                Times = np.array(range(len(y_raw)))*s #Assume that the samplerate is the same in this as in other data sets
            if len(fs) != 4:
                y_NTD, y_NTD_err, yfTM, yfTM_err = dp.get_NTD(y_raw, y_raw_err, s) #Filter out everything which doesn't meet NTD condition
                x_NTD, x_NTD_err, xfTM, xfTM_err = dp.get_NTD(x_raw, x_raw_err, s)
                y_TM, y_TM_err, yfCO, yfCO_err = dp.get_TM(y_raw, y_raw_err, s) #Filter out noise
                x_TM, x_TM_err, xfCO, xfCO_err = dp.get_TM(x_raw, x_raw_err, s)
            else:
                y_NTD, y_NTD_err, yfTM, yfTM_err = dp.get_NTD(y_raw, y_raw_err, s) #Filter out everything which doesn't meet NTD condition
                x_NTD, x_NTD_err, xfTM, xfTM_err = dp.get_NTD(x_raw, x_raw_err, s)
                y_TM, y_TM_err, yfCO, yfCO_err = dp.get_TM(y_raw, y_raw_err, s, f = fs[2]) #Filter out noise
                x_TM, x_TM_err, xfCO, xfCO_err = dp.get_TM(x_raw, x_raw_err, s, f = fs[3])
            y_noise = y_raw - y_TM 
            x_noise = x_raw - x_TM 
            y_noise_err = y_raw_err+y_TM_err
            x_noise_err = x_raw_err+x_TM_err
            y_TM = y_TM - y_NTD #removing non-turbulent drifts from our turbulent motion
            x_TM = x_TM - x_NTD
            y_TM_err = y_TM_err+y_NTD_err
            x_TM_err = x_TM_err+x_NTD_err
            yfTM = [yfTM] + [None]*(len(yfCO)-1)
            yfTM_err = [yfTM_err] + [None]*(len(yfCO_err)-1)
            xfTM = [xfTM] + [None]*(len(xfCO)-1)
            xfTM_err = [xfTM_err] + [None]*(len(xfCO_err)-1)
            s = [s]+[None]*(len(xfCO_err)-1)
            for o in SaveDat.keys(): #Iterate through all SaveData options and associate them with the defined variables
                if o != 'T1s' and o != 'T2s':
                    SaveDat[o] = locals()[o]
            df = DataFrame(SaveDat)
            df2 = DataFrame({'s':s, 'y f_TM':yfTM, 'y f_TM error':yfTM_err, 'x f_TM':xfTM, 'x f_TM error':xfTM_err, 'y f_CO':yfCO, 'y f_CO error':yfCO_err, 'x f_CO':xfCO, 'x f_CO error':xfCO_err})
        else:
            print('N is too small for filter processes, returning only raw')
            SaveDat = {'x_raw':[], 'x_raw_err':[], 'y_raw':[], 'y_raw_err':[], 'Times':[]}
            y_raw, y_raw_err, x_raw, x_raw_err = dp.get_raw(stack) #See Data_processor.py for processing details
            y_raw = y_raw-np.average(y_raw) #We make positions into displacements about mean position
            x_raw = x_raw-np.average(x_raw)
            if 'AdDat' in locals():
                Times = AdDat['Times']
                if 'T1s' in AdDat:
                    SaveDat.update({'T1s':AdDat['T1s']})
                if 'T2s' in AdDat:
                    SaveDat.update({'T2s':AdDat['T2s']})
                s = Times[-1]/len(Times) #Assume constant sample rate s
            else:
                s = 3.28
                Times = np.array(range(len(y_raw)))*s #Assume that the samplerate is the same in this as in other data sets
            df = DataFrame(SaveDat)
            df2 = DataFrame({'s':[s]})
        if not os.path.exists(root+f'\SaveData\{CapName}'):
            os.makedirs(root+f'\SaveData\{CapName}')
        with pd.ExcelWriter(root+f'\SaveData\{CapName}\{CapName2}.xlsx') as writer:
            df.to_excel(writer, sheet_name ='vals', index = False)
            df2.to_excel(writer, sheet_name ='consts', index = False)
        print('Processed', CapName, f'to {CapName2}.xlsx')
        
def Process_all(rep = 0):
    #This function will perform Process_ds on all image sets named in the Images folder
    #Function autoskips reprocessing, so if one fails repeating wont recalculate sets that didn't fail
    #If _tts.xlsx file unavailable for whatever reason, include an N.txt file in the Images\CapName folder in which the first line is the number of images
    #If image format is non-standard (i.e. not CapName_(image number).fits), you need a Format.txt file which contains whatever replaces '_' otherwise Process_ds() cannot read the images
    #e.g. If you have 1000 images formatted as CapNameNumber(image number).fits with no CapName_tts.xlsx, you need a Format.txt containing the word Number, and a N.txt containing the number 1000 in the Images\CapName folder
    names = os.listdir(root+r'\Images')
    for n in names:
        Process_ds(n, rep = rep) #Set rep = 1 if you don't want to autoskip preprocessed data
    print('All sets of images processed')

def Recover_redundant(ptr, xlname, CapName):
    #You probably won't need this function, it just updates .xlsx files in a redundant format if the images are unavailable
    #In my case I have some save data in an old format, but have lost the images so I can only update from the save file
    #Looks at the {xlname}.xlsx file at ptr (stands for Path To Redundant) then converts it to standard format file named (CapName).xlsx
    #If it cannot recover some of the information, these fields will be left blank in the resulting file
    if not os.path.exists(root+ptr):
        print('Either path not found or this redundant file has already been processed')
        return
    else:
        df = read_excel(root+ptr, sheet_name = 'vals')
        reddat = df.to_dict('list')
        keys = ['ycoord', 'xcoord', 'times', 't1s', 't2s'] #These are the names of any currently useful information in my old format
        curkeys = ['y_raw', 'x_raw', 'Times', 'T1s', 'T2s'] #The current formatting alternatives of these names
        RecDat = {}
        for k in range(len(keys)):
            if keys[k] in reddat:
                RecDat.update({curkeys[k]:reddat[keys[k]]})
                locals()[curkeys[k]] = np.array(reddat[keys[k]])
        if 'Times' in RecDat:
            Times = RecDat['Times']
            s = Times[-1]/len(Times) #Assume constant sample rate s
        else:
            s = 3.28
            Times = np.array(range(len(locals()['y_raw'])))*s #Assume that the samplerate is the same in this as in other data sets
        #For the most part this whole section is just the same as Process_ds()
        y_raw_err = np.zeros(len(locals()['y_raw']))
        x_raw_err = np.zeros(len(locals()['x_raw']))
        y_NTD, y_NTD_err, yfTM, yfTM_err = dp.get_NTD(locals()['y_raw'], y_raw_err, s) #Filter out everything which doesn't meet NTD condition
        x_NTD, x_NTD_err, xfTM, xfTM_err = dp.get_NTD(locals()['x_raw'], x_raw_err, s)
        y_TM, y_TM_err, yfCO, yfCO_err = dp.get_TM(locals()['y_raw'], y_raw_err, s) #Filter out noise
        x_TM, x_TM_err, xfCO, xfCO_err = dp.get_TM(locals()['x_raw'], x_raw_err, s)
        y_noise = locals()['y_raw'] - y_TM
        x_noise = locals()['x_raw'] - x_TM
        y_noise_err = y_raw_err+y_TM_err
        x_noise_err = x_raw_err+x_TM_err
        y_TM = y_TM - y_NTD #removing non-turbulent drifts from our turbulent motion
        x_TM = x_TM - x_NTD
        y_TM_err = y_TM_err+y_NTD_err
        x_TM_err = x_TM_err+x_NTD_err
        yfTM = [yfTM] + [None]*(len(yfCO)-1)
        yfTM_err = [yfTM_err] + [None]*(len(yfCO_err)-1)
        xfTM = [xfTM] + [None]*(len(xfCO)-1)
        xfTM_err = [xfTM_err] + [None]*(len(xfCO_err)-1)
        SaveDat = {'x_raw':[], 'x_raw_err':[], 'y_raw':[], 'y_raw_err':[],
                   'x_NTD':[], 'x_NTD_err':[], 'y_NTD':[], 'y_NTD_err':[],
                   'x_TM':[], 'x_TM_err':[], 'y_TM':[], 'y_TM_err':[],
                   'x_noise':[], 'x_noise_err':[], 'y_noise':[], 'y_noise_err':[], 'Times':[]}
        for o in SaveDat.keys(): #Iterate through all SaveData options and associate them with the defined variables
            if o != 'T1s' and o != 'T2s':
                SaveDat[o] = locals()[o]
        if 'T1s' in RecDat:
            SaveDat.update({'T1s':RecDat['T1s']})
        if 'T2s' in RecDat:
            SaveDat.update({'T2s':RecDat['T2s']})
        df = DataFrame(SaveDat)
        df2 = DataFrame({'s':s, 'y f_TM':yfTM, 'y f_TM error':yfTM_err, 'x f_TM':xfTM, 'x f_TM error':xfTM_err, 'y f_CO':yfCO, 'y f_CO error':yfCO_err, 'x f_CO':xfCO, 'x f_CO error':xfCO_err})
        if not os.path.exists(root+f'\SaveData\{CapName}'):
            os.makedirs(root+f'\SaveData\{CapName}')
        with pd.ExcelWriter(root+f'\SaveData\{CapName}\{CapName}.xlsx') as writer:
            df.to_excel(writer, sheet_name ='vals', index = False)
            df2.to_excel(writer, sheet_name ='consts', index = False)
        print('Processed', ptr, f'to {CapName}.xlsx')
        if not os.path.exists(root+r'\SaveData\Redundant_format'):
            os.makedirs(root+r'\SaveData\Redundant_format')
        os.replace(root+ptr, root+f'\SaveData\Redundant_format\{xlname}.xlsx')
        print(f'Moved {xlname}.xlsx to Redundant_format folder')
    
def Segment(CapName, size):
    if os.path.exists(root+f'\SaveData\{CapName}\{CapName}.xlsx'):
        vals = pd.read_excel(root+f'\SaveData\{CapName}\{CapName}.xlsx', sheet_name = 'vals')
        vals.to_dict('list')
        consts = pd.read_excel(root+f'\SaveData\{CapName}\{CapName}.xlsx', sheet_name = 'consts')
        consts.to_dict('list')
        print('Segmenter: Data found')
    elif os.path.exists(root+f'\Images\{CapName}\{CapName}'+'_0.fits'):
        print('This data set has not previously been processed, processing now')
        Process_ds(CapName)
        vals = pd.read_excel(root+f'\SaveData\{CapName}\{CapName}.xlsx', sheet_name = 'vals')
        vals.to_dict('list')
        consts = pd.read_excel(root+f'\SaveData\{CapName}\{CapName}.xlsx', sheet_name = 'consts')
        consts.to_dict('list')
        print('Segmenter: Data found')
    else:
        print('This data set is not found')
        return
    names = vals.keys()
    for n in names:
        for i in range(int(len(vals['x_raw'])/size)-1):
            locals()[n+'_'+str(i)] = np.array(vals[n])[i*size:(i+1)*size] 
    with pd.ExcelWriter(root+f'\SaveData\{CapName}\{CapName}_seg'+str(size)+'.xlsx') as writer:
        for i in range(int(len(vals['x_raw'])/size)-1):
            SaveDat = {}
            for n in names:
                SaveDat.update({n:locals()[n+'_'+str(i)]})
            df = DataFrame(SaveDat)
            df.to_excel(writer, sheet_name='Seg '+str(i), index = False)
    print(CapName, 'segmented into arrays of length', str(size))
    
def Find_STDs(CapName, size):
    if os.path.exists(root+f'\SaveData\{CapName}\{CapName}_seg'+str(size)+'.xlsx'):
        xl = pd.ExcelFile(root+f'\SaveData\{CapName}\{CapName}_seg'+str(size)+'.xlsx')
    else:
        Segment(CapName, size)
        xl = pd.ExcelFile(root+f'\SaveData\{CapName}\{CapName}_seg'+str(size)+'.xlsx')
    Temps = np.zeros(len(xl.sheet_names))
    Tempserr = np.zeros(len(xl.sheet_names))
    xSTD = np.zeros(len(xl.sheet_names))
    xSTDerr = np.zeros(len(xl.sheet_names))
    ySTD = np.zeros(len(xl.sheet_names))
    ySTDerr = np.zeros(len(xl.sheet_names))
    radSTD = np.zeros(len(xl.sheet_names))
    radSTDerr = np.zeros(len(xl.sheet_names))
    for i in range(len(xl.sheet_names)):
        df = pd.read_excel(root+f'\SaveData\{CapName}\{CapName}_seg{size}.xlsx', sheet_name = xl.sheet_names[i])
        SegData = df.to_dict('list')
        xSTD[i] = np.std(np.array(SegData['x_TM']))
        xSTDerr[i] = abs(np.std(np.array(SegData['x_TM'])+np.array(SegData['x_TM_err']))-xSTD[i])
        ySTD[i] = np.std(np.array(SegData['y_TM']))
        ySTDerr[i] = abs(np.std(np.array(SegData['y_TM'])+np.array(SegData['y_TM_err']))-ySTD[i])
        radSTD[i] = np.sqrt(xSTD[i]**2 + ySTD[i]**2)
        radSTDerr[i] = abs(np.sqrt((xSTD[i]+xSTDerr[i])**2 + (ySTD[i]+ySTDerr[i])**2)-radSTD[i])
        Temps[i] = np.average(np.array(SegData['T1s']))
        Tempserr[i] = np.std(np.array(SegData['T1s']))/np.sqrt(len(np.array(SegData['T1s'])))
    xlf, xlrms = mf.fit_line(Temps, xSTD, 1)
    ylf, ylrms = mf.fit_line(Temps, ySTD, 1)
    rlf, rlrms = mf.fit_line(Temps, radSTD, 1)
    xef, xerms = mf.fit_exp(np.array(Temps), np.array(xSTD))
    yef, yerms = mf.fit_exp(np.array(Temps), np.array(ySTD))
    ref, rerms = mf.fit_exp(np.array(Temps), np.array(radSTD))
    ConstDat = {'y_lin_m':[ylf[0]], 'y_lin_c':[ylf[1]], 'y_lin_rms':[ylrms],
                'x_lin_m':[xlf[0]], 'x_lin_c':[xlf[1]], 'x_lin_rms':[xlrms],
                'rad_lin_m':[rlf[0]], 'rad_lin_c':[rlf[1]], 'r_lin_rms':[rlrms],
                'y_exp_a':[abs(yef[0])], 'y_exp_b':[abs(yef[1])], 'y_exp_c':[abs(yef[2])], 'y_exp_rms':[yerms],
                'x_exp_a':[abs(xef[0])], 'x_exp_b':[abs(xef[1])], 'x_exp_c':[abs(xef[2])], 'x_exp_rms':[xerms],
                'r_exp_a':[abs(ref[0])], 'r_exp_b':[abs(ref[1])], 'r_exp_c':[abs(ref[2])], 'r_exp_rms':[rerms]}    
    SaveDat = {'ySTD':ySTD, 'ySTDerr':ySTDerr, 'xSTD':xSTD, 'xSTDerr':xSTDerr, 'radSTD':radSTD, 'radSTDerr':radSTDerr, 'Temps':Temps, 'Tempserr':Tempserr}
    cdf = DataFrame(ConstDat)
    df = DataFrame(SaveDat)
    df.to_excel(root+f'\SaveData\{CapName}\{CapName}_STD'+str(size)+'.xlsx', sheet_name='vals', index = False)
    with pd.ExcelWriter(root+f'\SaveData\{CapName}\{CapName}_STD'+str(size)+'.xlsx') as writer:
        df.to_excel(writer, sheet_name ='vals', index = False)
        cdf.to_excel(writer, sheet_name ='consts', index = False)
    print(CapName, 'STDs found')
                          
def Plot_single(CapName, fn, **kwargs):
    #This function is a single catch all which should be able to auto format any plot using one set of data
    #fn should just redirect to a function in Plotter.py
    if fn in ['plot_df', 'plot_FFT', 'plot_Temps', 'plot_comps', 'plot_comp_FFT']:
        if os.path.exists(root+f'\SaveData\{CapName}\{CapName}.xlsx'):
            vals = pd.read_excel(root+f'\SaveData\{CapName}\{CapName}.xlsx', sheet_name = 'vals')
            vals.to_dict('list')
            consts = pd.read_excel(root+f'\SaveData\{CapName}\{CapName}.xlsx', sheet_name = 'consts')
            consts.to_dict('list')
            print('Plot: Data found')
        elif os.path.exists(root+f'\Images\{CapName}\{CapName}'+'_0.fits'):
            print('This data set has not previously been processed, processing now')
            Process_ds(CapName)
            vals = pd.read_excel(root+f'\SaveData\{CapName}\{CapName}.xlsx', sheet_name = 'vals')
            vals.to_dict('list')
            consts = pd.read_excel(root+f'\SaveData\{CapName}\{CapName}.xlsx', sheet_name = 'consts')
            consts.to_dict('list')
            print('Plot: Data found')
        else:
            print('This data set is not found')
            return
        if hasattr(pt, fn):
            getattr(pt, fn)(vals, consts, **kwargs)
    elif fn in ['plot_SE']:
        if 'segsize' in kwargs:
            segsize = kwargs['segsize']
        else:
            segsize = 100
        if os.path.exists(root+f'\SaveData\{CapName}\{CapName}_seg{segsize}.xlsx'):
            xl = pd.ExcelFile(root+f'\SaveData\{CapName}\{CapName}_seg{segsize}.xlsx')
            vals = [pd.read_excel(root+f'\SaveData\{CapName}\{CapName}_seg{segsize}.xlsx', sheet_name = xl.sheet_names[1]).to_dict('list'), 
                    pd.read_excel(root+f'\SaveData\{CapName}\{CapName}_seg{segsize}.xlsx', sheet_name = xl.sheet_names[-1]).to_dict('list')]
            xl.close()
            print('Plot: Data found')
        elif os.path.exists(root+f'\SaveData\{CapName}\{CapName}.xlsx'):
            Segment(CapName, segsize)
            xl = pd.ExcelFile(root+f'\SaveData\{CapName}\{CapName}_seg{segsize}.xlsx')
            vals = [pd.read_excel(root+f'\SaveData\{CapName}\{CapName}_seg{segsize}.xlsx', sheet_name = xl.sheet_names[1]).to_dict('list'), 
                    pd.read_excel(root+f'\SaveData\{CapName}\{CapName}_seg{segsize}.xlsx', sheet_name = xl.sheet_names[-1]).to_dict('list')]
            xl.close()
            print('Plot: Data found')
        elif os.path.exists(root+f'\Images\{CapName}\{CapName}'+'_0.fits'):
            print('This data set has not previously been processed, processing now')
            Process_ds(CapName)
            Segment(CapName, segsize)
            xl = pd.ExcelFile(root+f'\SaveData\{CapName}\{CapName}_seg{segsize}.xlsx')
            vals = [pd.read_excel(root+f'\SaveData\{CapName}\{CapName}_seg{segsize}.xlsx', sheet_name = xl.sheet_names[1]).to_dict('list'), 
                    pd.read_excel(root+f'\SaveData\{CapName}\{CapName}_seg{segsize}.xlsx', sheet_name = xl.sheet_names[-1]).to_dict('list')]
            xl.close()
            print('Plot: Data found')
        else:
            print('This data set is not found')
            return
        if hasattr(pt, fn):
            getattr(pt, fn)(vals, **kwargs)
    else:
        print('This does not appear to be a valid plot type')
        
def Plot_STDs(CapNames, size, err = 1, colours = [], ls = [], pscale = [1, 'pixels', 0]):
    if isinstance(CapNames, str):
        CapNames = [CapNames]
    TotDat = {'xSTD':[], 'xSTDerr':[], 'ySTD':[], 'ySTDerr':[], 'radSTD':[], 'radSTDerr':[], 'Temps':[], 'Tempserr':[]}
    TotConsts = {'y_lin_m':[], 'y_lin_c':[], 'y_lin_rms':[],
                'x_lin_m':[], 'x_lin_c':[], 'x_lin_rms':[],
                'rad_lin_m':[], 'rad_lin_c':[], 'r_lin_rms':[],
                'y_exp_a':[], 'y_exp_b':[], 'y_exp_c':[], 'y_exp_rms':[],
                'x_exp_a':[], 'x_exp_b':[], 'x_exp_c':[], 'x_exp_rms':[],
                'r_exp_a':[], 'r_exp_b':[], 'r_exp_c':[], 'r_exp_rms':[]}   
    for n in range(len(CapNames)):
        CapName = CapNames[n]
        if os.path.exists(root+f'\SaveData\{CapName}\{CapName}_STD'+str(size)+'.xlsx'):
            df = read_excel(root+f'\SaveData\{CapName}\{CapName}_STD'+str(size)+'.xlsx', sheet_name = 'vals')
            df2 = read_excel(root+f'\SaveData\{CapName}\{CapName}_STD'+str(size)+'.xlsx', sheet_name = 'consts')
            STDdat = df.to_dict('list')
            STDconsts = df2.to_dict('list')
        else:
            Find_STDs(CapName, size)
            df = read_excel(root+f'\SaveData\{CapName}\{CapName}_STD'+str(size)+'.xlsx', sheet_name = 'vals')
            df2 = read_excel(root+f'\SaveData\{CapName}\{CapName}_STD'+str(size)+'.xlsx', sheet_name = 'consts')
            STDdat = df.to_dict('list')
            STDconsts = df2.to_dict('list')
        for k in TotDat.keys():
            TotDat[k] = TotDat[k]+[STDdat[k][1:]]
        for k in TotConsts.keys():
            TotConsts[k] = TotConsts[k]+[STDconsts[k]]
    for n in range(len(CapNames)):
        CapNames[n] = CapNames[n][:-3]
    Capnames = ['C', 'FS', 'NS']
    pt.plot_STD(TotDat, TotConsts, CapNames, err = err, colours = colours, linestyle = ls, pixscale = pscale)
    
def Plot_images(impaths, cropsize = 0, calc = 0, names = []):
    #Plot is configured for up to 4 images
    if isinstance(impaths, str):
        impaths = [impaths]
    if names == []:
        names = ['a', 'c', 'b', 'd']
    if isinstance(cropsize, float) or isinstance(cropsize, int):
        cropsize = [cropsize]*len(impaths)
    imgdat = []
    sizedat = []
    for i in range(len(impaths)):
        img = fits.open(root+impaths[i])[0].data
        if cropsize != 0:
            dim = img.shape
            if calc == 0:
                center = [dim[0]/2, dim[1]/2]
            else:
                print(img.shape)
                center = de.find_com(img)
            newdim = [cropsize[i], cropsize[i]]
            img = ip.crop_image(img, center, newdim)
        if len(impaths) > 1:
            imgdat = imgdat+[{'image':img, 'title':names[i], 'xax':False, 'yax':False}]
        else:
            imgdat = imgdat+[{'image':img, 'xax':False, 'yax':False}]
        if len(impaths) == 3 and i == 0:
            sizedat = sizedat+[[2,2]]
        else:
            sizedat = sizedat + [[1,1]]
    #print(sizedat)
    pt.plot(imgdat, sizedat)
    
def Plot_CC(CapName, rad = 0):
    if os.path.exists(root+f'\SaveData\{CapName}\{CapName}_tts.xlsx'):
        df = read_excel(root+f'\SaveData\{CapName}\{CapName}_tts.xlsx', sheet_name = 'vals', index_col = None)
        AdDat = df.to_dict('list')
        N = len(AdDat['Times']) #If _tts exists, then AdDat['Time'] must exist - not the case for 'T1s' or 'T2s'
        print(CapName, 'N =', N, 'found')
    elif N == 0:
        if os.path.exists(root+f'\Images\{CapName}\\N.txt'):
            with open(root+f'\Images\{CapName}\\N.txt', 'r') as f:
                lines = f.readlines()
            N = int(lines[0])
            print(CapName, 'N =', N, 'found')
        else:
            print('You must define N if there is no _tts excel file or N.txt file available for', CapName, ' - returning')
            return
    if not os.path.exists(root+f'\SaveData\{CapName}'):
        os.makedirs(root+f'\SaveData\{CapName}')
    Impath = root+f'\Images\{CapName}\{CapName}'
    Savepath = root+f'\SaveData\{CapName}\{CapName}.xlsx'
    if os.path.exists(root+f'\Images\{CapName}\Format.txt'):
        with open(root+f'\Images\{CapName}\Format.txt', 'r') as f:
            lines = f.readlines()
        if len(lines) != 0:
            ext = lines[0]
        else:
            ext = ''
    else:
        ext = '_'
    if not os.path.exists(Impath+ext+'0.fits'): #Check if we have any images for this set
        print('There do not appear to be any images available for this set')
        return
    else:
        print(f'Processing {CapName}')
        for i in range(N):
            Fname = Impath+ext+str(i)+'.fits'
            hdu = fits.open(Fname)
            img = hdu[0].data
            if i == 0:
                dim = img.shape
                stack = np.zeros([dim[0], dim[1], N], dtype = img.dtype)
            stack[:, :, i] = img
    if os.path.exists(root+f'\SaveData\{CapName}\{CapName}.xlsx'):
        vals = pd.read_excel(root+f'\SaveData\{CapName}\{CapName}.xlsx', sheet_name = 'vals')
        vals.to_dict('list')
        consts = pd.read_excel(root+f'\SaveData\{CapName}\{CapName}.xlsx', sheet_name = 'consts')
        consts.to_dict('list')
        print('Plot: Data found')
    elif os.path.exists(root+f'\Images\{CapName}\{CapName}'+'_0.fits'):
        print('This data set has not previously been processed, processing now')
        Process_ds(CapName)
        vals = pd.read_excel(root+f'\SaveData\{CapName}\{CapName}.xlsx', sheet_name = 'vals')
        vals.to_dict('list')
        consts = pd.read_excel(root+f'\SaveData\{CapName}\{CapName}.xlsx', sheet_name = 'consts')
        consts.to_dict('list')
        print('Plot: Data found')
    else:
        print('This data set is not found')
        return
    if os.path.exists(root+f'\SaveData\{CapName}\{CapName}_corr.xlsx'):
        cvals = pd.read_excel(root+f'\SaveData\{CapName}\{CapName}_corr.xlsx', sheet_name = 'vals')
        cvals.to_dict('list')
        crosscorr = cvals['corr']
        print('Correlation found')
    else:
        print('Calculating correlation')
        crosscorr = corr.find_stack_CC_slow(stack)
        cordat = {'corr':crosscorr}
        df = DataFrame(cordat)
        df.to_excel(root+f'\SaveData\{CapName}\{CapName}_corr.xlsx', sheet_name = 'vals', index = False)
        print('Correlation found')
    pt.plot_CC(vals, crosscorr, rad = rad)
    
    
def Plot_multiple(names, ext = [], **kwargs):
    if isinstance(names, str):
        print('There is only one set of data, perhaps you meant Plot_single()?')
    else:
        if ext == []:
            ext = ['']*len(names)
        vals = []
        consts = []
        for n in range(len(names)):
            CapName = names[n]
            if os.path.exists(root+f'\SaveData\{CapName}\{CapName}'+ext[n]+'.xlsx'):
                tvals = pd.read_excel(root+f'\SaveData\{CapName}\{CapName}'+ext[n]+'.xlsx', sheet_name = 'vals')
                tvals.to_dict('list')
                tconsts = pd.read_excel(root+f'\SaveData\{CapName}\{CapName}'+ext[n]+'.xlsx', sheet_name = 'consts')
                tconsts.to_dict('list')
                vals += [tvals]
                consts += [tconsts]
                print('Plot: Data found')
            elif os.path.exists(root+f'\Images\{CapName}\{CapName}'+'_0.fits'):
                print('This data set has not previously been processed, processing now')
                Process_ds(CapName)
                tvals = pd.read_excel(root+f'\SaveData\{CapName}\{CapName}'+ext[n]+'.xlsx', sheet_name = 'vals')
                tvals.to_dict('list')
                tconsts = pd.read_excel(root+f'\SaveData\{CapName}\{CapName}'+ext[n]+'.xlsx', sheet_name = 'consts')
                tconsts.to_dict('list')
                vals += [tvals]
                consts += [tconsts]
                print('Plot: Data found')
            else:
                print(f'{CapName} is not found')
                return
        pt.plot_multidf(vals, consts, **kwargs)

def Find_SN(names, nname, cf = 0):
    if cf == 0:
        if os.path.exists(root+f'\SaveData\{nname}\{nname}.xlsx'):
            tvals = pd.read_excel(root+f'\SaveData\{nname}\{nname}.xlsx', sheet_name = 'vals')
            tvals.to_dict('list')
            tconsts = pd.read_excel(root+f'\SaveData\{nname}\{nname}.xlsx', sheet_name = 'consts')
            tconsts.to_dict('list')
            nvals = tvals
            nconsts = tconsts
            print('Plot: Data found')
        elif os.path.exists(root+f'\Images\{nname}\{nname}'+'_0.fits'):
            print('This data set has not previously been processed, processing now')
            Process_ds(nname)
            tvals = pd.read_excel(root+f'\SaveData\{nname}\{nname}.xlsx', sheet_name = 'vals')
            tvals.to_dict('list')
            tconsts = pd.read_excel(root+f'\SaveData\{nname}\{nname}.xlsx', sheet_name = 'consts')
            tconsts.to_dict('list')
            nvals = tvals
            nconsts = tconsts
            print('Plot: Data found')
        else:
            print(f'{nname} is not found')
            return
    if isinstance(names, str):
        print('There is only one set of data, perhaps you meant Plot_single()?')
    else:
        vals = []
        consts = []
        for n in names:
            CapName = n
            if os.path.exists(root+f'\SaveData\{CapName}\{CapName}.xlsx'):
                tvals = pd.read_excel(root+f'\SaveData\{CapName}\{CapName}.xlsx', sheet_name = 'vals')
                tvals.to_dict('list')
                tconsts = pd.read_excel(root+f'\SaveData\{CapName}\{CapName}.xlsx', sheet_name = 'consts')
                tconsts.to_dict('list')
                vals += [tvals]
                consts += [tconsts]
                print('Plot: Data found')
            elif os.path.exists(root+f'\Images\{CapName}\{CapName}'+'_0.fits'):
                print('This data set has not previously been processed, processing now')
                Process_ds(CapName)
                tvals = pd.read_excel(root+f'\SaveData\{CapName}\{CapName}.xlsx', sheet_name = 'vals')
                tvals.to_dict('list')
                tconsts = pd.read_excel(root+f'\SaveData\{CapName}\{CapName}.xlsx', sheet_name = 'consts')
                tconsts.to_dict('list')
                vals += [tvals]
                consts += [tconsts]
                print('Plot: Data found')
            else:
                print(f'{CapName} is not found')
                return
            if cf == 0:
                SN = np.average(np.sqrt(np.array(tvals['y_TM'])**2+np.array(tvals['x_TM'])**2))/np.average(np.sqrt(np.array(nvals['y_TM'])**2+np.array(nvals['x_TM'])**2))
                print(CapName, 'S/N =', SN)
            else:
                fs = [tconsts['y f_TM'][0], tconsts['x f_TM'][0], tconsts['y f_CO'][0], tconsts['x f_CO'][0]]
                Process_ds(nname, fs=fs, rep = 0)
                nname2= nname+str(fs)
                nvals = pd.read_excel(root+f'\SaveData\{nname}\{nname2}.xlsx', sheet_name = 'vals')
                nvals.to_dict('list')
                SN = np.average(np.sqrt(np.array(tvals['y_TM'])**2+np.array(tvals['x_TM'])**2))/np.average(np.sqrt(np.array(nvals['y_TM'])**2+np.array(nvals['x_TM'])**2))
                print(CapName, 'S/N =', SN)
                

def Find_best_size(CapName):
    if os.path.exists(root+f'\SaveData\{CapName}\{CapName}.xlsx'):
        vals = pd.read_excel(root+f'\SaveData\{CapName}\{CapName}.xlsx', sheet_name = 'vals')
        vals.to_dict('list')
        print('Size finder: Data found')
    elif os.path.exists(root+f'\Images\{CapName}\{CapName}'+'_0.fits'):
        print('This data set has not previously been processed, processing now')
        Process_ds(CapName)
        vals = pd.read_excel(root+f'\SaveData\{CapName}\{CapName}.xlsx', sheet_name = 'vals')
        vals.to_dict('list')
        print('Size finder: Data found')
    testarray = np.array(vals['y_TM'])
    sliceSTD = np.zeros(5000)
    for i in range(5000):
        sliceSTD[i] = np.std(testarray[int(len(testarray)/2)-i:int(len(testarray)/2)+i])
    pt.plot([{'xs':[np.array(range(5000))[25:]*2], 'ys':[sliceSTD[25:]], 'ylabel':'Standard deviation of slice', 'xlabel':'Number of items in slice'}], [[1,1]])
    
def plot_Poisson(CapName, N=0):
    if os.path.exists(root+f'\SaveData\{CapName}\{CapName}_tts.xlsx') and N == 0:
        df = read_excel(root+f'\SaveData\{CapName}\{CapName}_tts.xlsx', sheet_name = 'vals', index_col = None)
        AdDat = df.to_dict('list')
        N = len(AdDat['Times']) #If _tts exists, then AdDat['Time'] must exist - not the case for 'T1s' or 'T2s'
        print(CapName, 'N =', N, 'found')
    elif N == 0:
        if os.path.exists(root+f'\Images\{CapName}\\N.txt'):
            with open(root+f'\Images\{CapName}\\N.txt', 'r') as f:
                lines = f.readlines()
            N = int(lines[0])
            print(CapName, 'N =', N, 'found')
        else:
            print('You must define N if there is no _tts excel file or N.txt file available for', CapName, ' - returning')
            return
    Impath = root+f'\Images\{CapName}\{CapName}'
    if os.path.exists(root+f'\Images\{CapName}\Format.txt'):
        with open(root+f'\Images\{CapName}\Format.txt', 'r') as f:
            lines = f.readlines()
        if len(lines) != 0:
            ext = lines[0]
        else:
            ext = ''
    else:
        ext = '_'
    if not os.path.exists(Impath+ext+'0.fits'): #Check if we have any images for this set
        print('There do not appear to be any images available for this set')
        return
    else:
        print(f'Processing {CapName}')
        for i in range(N):
            Fname = Impath+ext+str(i)+'.fits'
            hdu = fits.open(Fname)
            img = hdu[0].data
            if i == 0:
                dim = img.shape
                stack = np.zeros([dim[0], dim[1], N], dtype = img.dtype)
            stack[:, :, i] = img
        #creating the dictionary which will be saved to an excel document
        if N > 100:
            SaveDat = {'x_raw':[], 'x_raw_err':[], 'y_raw':[], 'y_raw_err':[],
                       'x_NTD':[], 'x_NTD_err':[], 'y_NTD':[], 'y_NTD_err':[],
                       'x_TM':[], 'x_TM_err':[], 'y_TM':[], 'y_TM_err':[],
                       'x_noise':[], 'x_noise_err':[], 'y_noise':[], 'y_noise_err':[], 'Times':[]}
            y_raw, y_raw_err, x_raw, x_raw_err = dp.get_raw(stack) #See Data_processor.py for processing details
            y_raw = y_raw-np.average(y_raw) #We make positions into displacements about mean position
            x_raw = x_raw-np.average(x_raw)
            avgimg = ip.average_stack(stack)
            pstack = ip.poisson_stack(avgimg, stack.shape[2])#-stack
            py, px = de.extract_coords(pstack)
            py = py-np.average(py) #We make positions into displacements about mean position
            px = px-np.average(px)
            dn = 10
            xFFT, xfreqs = dp.FFT(x_raw, srate = 3.15)
            xFFT = abs(xFFT[1:len(x_raw)//2])
            xFFT = xFFT[:int(len(xFFT)/dn)]
            xfreqs = xfreqs[1:len(x_raw)//2]
            xfreqs = xfreqs[:int(len(xfreqs)/dn)]
            yFFT, yfreqs = dp.FFT(y_raw, srate = 3.15)
            yFFT = abs(yFFT[1:len(y_raw)//2])
            yFFT = yFFT[:int(len(yFFT)/dn)]
            yfreqs = yfreqs[1:len(y_raw)//2]
            yfreqs = yfreqs[:int(len(yfreqs)/dn)]
            pxFFT, pxfreqs = dp.FFT(px, srate = 3.15)
            pxFFT = abs(pxFFT[1:len(px)//2])
            pxFFT = pxFFT[:int(len(pxFFT)/dn)]
            pxfreqs = pxfreqs[1:len(px)//2]
            pxfreqs = pxfreqs[:int(len(pxfreqs)/dn)]
            pyFFT, pyfreqs = dp.FFT(py, srate = 3.15)
            pyFFT = abs(pyFFT[1:len(py)//2])
            pyFFT = pyFFT[:int(len(pyFFT)/dn)]
            pyfreqs = pyfreqs[1:len(py)//2]
            pyfreqs = pyfreqs[:int(len(pyfreqs)/dn)]
            yplotdata = {'xs':[np.array(range(N))/3.15, np.array(range(N))/3.15], 'ys':[x_raw+0.1, px-0.1], 'xlabel':'Time (s)', 'labels':['DCS C', 'Poisson simulation'], 'ylabel':'Displacement\n(pixels)'}
            xplotdata = {'xs':[xfreqs, pxfreqs], 'ys':[xFFT, pxFFT], 'xlabel':'Frequency (Hz)', 'ylabel':'Power', 'ysc':[0, 30]}
            pt.plot([yplotdata, xplotdata], [[1,2],[1,2]])