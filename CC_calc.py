import numpy as np

def FFT2d(image):
    return np.fft.fft2(image)

def IFFT2d(image):
    return np.fft.ifft2(image)

def conjugate(fftimg):
    return np.real(fftimg)-np.imag(fftimg)

def CC(rimg, cimg):
    rfft = FFT2d(rimg)
    cfft = FFT2d(cimg)
    cfft = conjugate(cfft)
    h = rfft*cfft
    corrres = IFFT2d(h)
    return corrres, np.max(np.real(corrres)), np.where(corrres == np.max(corrres))[0]

def find_stack_CC(stack):
    N = stack.shape[2]
    corr = np.zeros(int(N))
    for i in range(stack.shape[2]):
        _, corr[i], _ = CC(stack[:,:,0], stack[:,:,i])
        if i > 0:
            corr[i] = corr[i]/corr[0]
    corr[0] = 1
    print(np.max(corr))
    return corr

def find_stack_CC_slow(stack):
    #ZNCC https://en.wikipedia.org/wiki/Cross-correlation
    corr = np.zeros(int(stack.shape[2]))
    n = stack.shape[0]*stack.shape[1]
    templ = stack[:, :, 0]
    STDt = np.std(templ)
    mut = np.average(templ)
    for i in range(stack.shape[2]):
        img = stack[:, :, i]
        STDi = np.std(img)
        mui = np.average(img)
        corr[i] = 0
        for y in range(stack.shape[0]):
            for x in range(stack.shape[1]):
                corr[i] += (1/(STDi*STDt))*(img[y,x]-mui)*(templ[y,x]-mut)
        corr[i] = (1/n)*corr[i]
        print(i, corr[i])
    return corr
    