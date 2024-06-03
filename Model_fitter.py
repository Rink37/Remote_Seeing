import numpy as np
from numpy.polynomial.polynomial import polyfit
from scipy.stats import chisquare
from scipy import optimize
import matplotlib.pyplot as plt
import Image_processor as ip
import Plotter as pt
import matplotlib.pyplot as plt

def fit_line(xs, dat, pol):
    c, m = polyfit(xs, dat, pol)
    expec = m*xs+c
    return [m, c], np.mean((dat-expec)**2)

def exp_model(params, xs, dat):
    params[0] = abs(params[0])
    params[1] = abs(params[1])
    params[2] = abs(params[2])
    expec = params[0]*np.exp(xs*params[1])+params[2]
    return np.mean((dat-expec)**2)

def fit_exp(xs, dat):
    #powell
    params = optimize.minimize(exp_model, x0 = [0.0005, 0.15 ,0], args = (xs, dat), method = 'Powell')
    print(params.x)
    print(dat, abs(params.x[0])*np.exp(xs*abs(params.x[1]))+abs(params.x[2]))
    return params.x, params.fun