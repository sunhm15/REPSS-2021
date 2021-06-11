from scipy import interpolate
import numpy as np
from scipy.stats import pearsonr

def get_ibi(peak, sig_fps):
    peak_loc = np.where(peak==1)[0]
    IPI = (peak_loc[1:]-peak_loc[0:-1])/sig_fps
    T = np.zeros(len(IPI));
    for i in range(len(IPI)):
        T[i] = np.sum(IPI[:i])

    IBI = np.array([T, IPI]) # x-axis is time, y-axis is ibi
    return IBI

def ibi_error(ibi, ibi_gt):
    f = interpolate.interp1d(ibi[0], 1000*ibi[1]) # convert to ms
    f_gt = interpolate.interp1d(ibi_gt[0], 1000*ibi_gt[1])
    
    t_min = np.max([ibi[0].min(), ibi_gt[0].min()])
    t_max = np.min([ibi[0].max(), ibi_gt[0].max()]) # find the interpolation x-axis range
    
    inter_ibi = f(np.linspace(t_min,t_max,100))
    inter_ibi_gt = f_gt(np.linspace(t_min,t_max,100)) #interpolation
    
    ibi_err = np.mean(np.abs(inter_ibi-inter_ibi_gt)) # compare absolute error between the 2 interpolated IBI curves 
    return ibi_err

def hr_error(ibi, ibi_gt):
    hr = 60 / np.mean(ibi[1])
    hr_gt = 60 / np.mean(ibi_gt[1]) # compute heart rate from ibi
    hr_err = np.abs(hr - hr_gt)
    return hr, hr_gt, hr_err