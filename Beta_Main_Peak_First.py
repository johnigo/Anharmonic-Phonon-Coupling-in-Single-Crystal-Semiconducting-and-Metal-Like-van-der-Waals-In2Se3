# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 13:30:13 2018

@author: John

Intended to be called by other program, all it wants is 
This code 

Do 
1 fits one Pseudo voigt to the Low wave number of the main In2Se3 peak
2 subtract 1, fit secondary peak with pseudo voigt
3 subtract 2, fit main peak with pseudo voigt 
4 report 2&3

While 

"""

# how much extra to fit beyond x peak
x_add = 2

import numpy as np
# for smoothing the curves
import scipy.interpolate as interp #import splev 

from lmfit.models import PseudoVoigtModel, VoigtModel, ConstantModel
#from lmfit import Parameters, Parameter

import matplotlib.pyplot as plt

def fit_two_Voigt(x_lst,y_lst,x_min_flt,x_max_flt,print_all_fits_bool,place_to_save_str):    
    # this returns 
    # x region of fit
    # y region of fit
    # smoothed y prime
    # smoothed y prime prime
    x_fit,y_fit, y_p, y_pp = smooth_the_data(x_lst, y_lst, x_min_flt, x_max_flt)
    
    # fit main peak side
    x_max = x_fit[np.argmax(y_pp)]  
    
    if not (108 < x_max < 111):
        x_max = 109
        
    x_holder = []
    y_holder = []
    for x,y in zip(x_lst, y_lst):
        if x < x_max+x_add:                   # The + const is to get the down part also 
            x_holder.append(x)
            y_holder.append(y)
    
    peak_one = fit_one_Voigt(x_holder,y_holder, 'one')
    
    
    '''
    First try was fit peak to rest of region
    Second try is to fit peak left of x_max a little
    '''
    
    # FIT SECOND PEAK
    x_holder = []
    y_holder = []
    for f,g in zip(x_fit,y_fit):
        if f > x_max + 3 * x_add:
            y_holder.append(g-peak_one.eval(x=f))
            x_holder.append(f)
    
    peak_two = fit_one_Voigt(x_holder,y_holder, 'two')
    
    # Refit main peak over peak 1
    x_holder = []
    y_holder = []
    for f,g in zip(x_fit,y_fit):
        if x < x_max + x_add:
            x_holder.append(f)
            y_holder.append(g-peak_two.eval(x=f))
    
    peak_one = fit_one_Voigt(x_holder,y_holder,'one')

    if print_all_fits_bool:
        x_dense = np.arange(x_min_flt,x_max_flt,(x_max_flt-x_min_flt)/300.0).tolist()
        
        # First peak
        y_one = []
        for f in x_dense:
            y_one.append(peak_one.eval(x=f))

        # Second peak
        y_two = []
        for f in x_dense:
            y_two.append(peak_two.eval(x=f))
        
        # All together
        y_together = []
        for f in x_dense:
            y_together.append(peak_one.eval(x=f) + peak_two.eval(x=f))
        
        # Data - second peak
        x_m_2nd = []
        y_m_2nd = []
        for f,g in zip(x_fit,y_fit):
            if f < x_max + x_add:
                y_m_2nd.append(g-peak_two.eval(x=f))
                x_m_2nd.append(f)

        # data-second peak
        y_hh =[]
        for f,g in zip(x_fit,y_fit):
            y_hh.append(g-peak_two.eval(x=f))
        
        # fit, each peak and data
        if True:
            plt.plot(x_fit,y_fit,'bx', label ="data")
            plt.plot(x_dense, y_together,'r', label = 'Together fit')
            plt.plot([x_fit[np.argmax(y_pp)]],[y_fit[np.argmax(y_pp)]], 'ko', ms = 10, label ="MaxGuess")
        
            # both peaks on their own
            plt.plot(x_dense, y_one, 'g')
            plt.plot(x_dense, y_two, 'g')
            
            # axis label
            plt.xlabel("Inv Cm")
            plt.ylabel("counts")
            plt.title("The two fits summed together")
            
            plt.legend()
            plt.savefig(place_to_save_str+"_AllTogether")
            plt.clf()
        
        # print data & data-peak_two & Fit peak 1
        if True: 
            plt.plot(x_fit,y_fit,'bx', label ="data")
            plt.plot(x_m_2nd,y_m_2nd,'ko', label = "y minus 2nd")
            plt.plot(x_dense,y_one,'g', label = "fit on y - 2nd peak")
            plt.plot([x_fit[np.argmax(y_pp)]],[y_fit[np.argmax(y_pp)]], 'ko', ms = 10, label ="MaxGuess")

            
            plt.xlabel("Inv Cm")
            plt.ylabel("counts")
            plt.title("Fitting preformed on black circles")
            
            plt.legend()
            plt.savefig(place_to_save_str+"_m2nd")
            plt.clf()
        
        
        # Print main peak & Data
        if False:
            plt.plot(x_fit,y_fit,'bx', label ="data")
            plt.plot(x_dense, y_one,'r', label = 'Main peak')
            plt.plot([x_fit[np.argmax(y_pp)]],[y_fit[np.argmax(y_pp)]], 'ko', ms = 10, label ="MaxGuess")
            
            plt.legend()
            plt.savefig(place_to_save_str+"_Mainpeak")
            plt.clf()
        
        # Second peak & Data-Main Peak
        if False:
            plt.plot(x_fit,y_holder,'bx', label ="data")
            plt.plot(x_dense, y_two,'r', label = 'Second peak')
            
            plt.legend()
            plt.savefig(place_to_save_str+"_Secondpeak")
            plt.clf()
        
        
    # Taken from 
    # https://stackoverflow.com/questions/1781571/how-to-concatenate-two-dictionaries-to-create-a-new-one-in-python?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
    return reduce(lambda x,y: dict(x,**y), (peak_one.best_values, peak_two.best_values))

def fit_one_Voigt(x_lst,y_lst, pre):
    '''
    Fits one Pseudo Voigt returns the 
    results object
    '''
    x_lst = np.asarray(x_lst)
    y_lst = np.asarray(y_lst)
    
    mod = VoigtModel(prefix = pre, independent_vars=['x'],nan_policy='raise')
    
    # here we set up the peak fitting guess. Then the peak fitter will make a parameter object out of them
    mod.set_param_hint(pre+'amplitude', value = 4 * np.max(y_lst), min = 3*np.max(y_lst), max = 7*np.max(y_lst), vary=True)
    # mod.set_param_hint(prefp+'center', value = x_max, min = x_max*(1-wiggle_room), max = x_max*(1+wiggle_room),vary=True)
    mod.set_param_hint(pre+'center', value = x_lst[np.argmax(y_lst)], vary=True)
    # Basically FWHM/3.6
    w_guess = 2
    mod.set_param_hint(pre+'sigma', value = w_guess, min = 0, max = 5*w_guess,vary=True)
    
    result = mod.fit(y_lst, x = x_lst, params = mod.make_params())
    
    return result

def fit_two_Psudo_Voigt(x_lst,y_lst,x_min_flt,x_max_flt,print_all_fits_bool,place_to_save_str):
    
    # this returns 
    # x region of fit
    # y region of fit
    # smoothed y prime
    # smoothed y prime prime
    x_fit,y_fit, y_p, y_pp = smooth_the_data(x_lst, y_lst, x_min_flt, x_max_flt)
    
    # fit main peak side
    x_max = x_fit[np.argmax(y_pp)]    
    x_holder = []
    y_holder = []
    for x,y in zip(x_lst, y_lst):
        if x < x_max:
            x_holder.append(x)
            y_holder.append(y)
    
    peak_one = fit_one_Psudo_Voigt(x_holder,y_holder)
    print "vvv Hot Holy HANNAH" 
    print peak_one.best_values
    print "^^^ Bill Boyvoycoy" 
    
    if print_all_fits_bool:
        x_dense = np.arange(x_min_flt,x_max_flt,(x_max_flt-x_min_flt)/300.0).tolist()
        
        y_dense = []
        for f in x_dense:
            y_dense.append(peak_one.eval(x=f))
        
        print x_fit
        print y_fit
        plt.plot(x_fit,y_fit,'bx', label ="data")
        plt.plot(x_dense, y_dense,'r', label = 'left fit')
        
        plt.legend()
        plt.savefig(place_to_save_str)
        plt.clf()
    
    return 0


def fit_one_Psudo_Voigt(x_lst,y_lst):
    '''
    Fits one Pseudo Voigt returns the 
    results object
    '''
    mod = PseudoVoigtModel(independent_vars=['x'],nan_policy='raise')
    
    x_lst = np.asarray(x_lst)
    y_lst = np.asarray(y_lst)
    # Guess good values computer
    mod.guess(y_lst,x = x_lst)
    
    result = mod.fit(y_lst, x = x_lst)
    
    return result

def smooth_the_data(x_lst, y_lst, x_min_flt, x_max_flt):
    '''
    Takes entire data set, x,y 
    cuts it down to the fitting region and 
    returns y's smoothed derivatives
    Return x_fit, y_fit, y_p, y_pp
    All as nparrays
    '''
    
    # Restrict the fit
    x_fit = []
    y_fit = []
    
    for x,y in zip(x_lst, y_lst):
        if x_min_flt < x < x_max_flt:
            x_fit.append(float(x))
            y_fit.append(float(y))
    
    x_fit = np.asarray(x_fit)
    y_fit = np.asarray(y_fit)   
    
    # now we find the parameters using the - d^2/dx^2
    ysmooth = interp.interp1d(x_fit, y_fit, kind='cubic')
    # differentiate x 2
    yp = np.gradient(ysmooth(x_fit))
    ypp = np.gradient(yp)
    # we want the peaks of -d2/dx2 
    ypp = np.asarray([-x for x in ypp])
    
    return x_fit, y_fit, yp, ypp
