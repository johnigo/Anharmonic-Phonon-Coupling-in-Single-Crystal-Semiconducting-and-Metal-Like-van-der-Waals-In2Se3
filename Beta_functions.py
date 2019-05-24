# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 10:12:09 2018

@author: John

This is for hte final version of hte beta code
it has the functions that are used by the final version of hte 
beta code

It runs much faster than the crap I'd written before

Runs wiyth Beta_main_3
"""

'''
Terrible code...
'''

'''
Parameters for 
'''
# how much extra to fit beyond x peak
x_add = 1
# max width of each case
width_one_peak = 3
width_two_peak = .4

'''
Parameters for fit_one_Voigt
'''
# the guess and the bounds on the maximum of the voigt
value_max = 4
value_max_min = .5
value_max_max = 5.0
# how many multiples of the guess can the width be
max_width = 3

width_1 = 4
width_2 = 5


peak_1_min = -25
peak_1_max = 6

peak_2_min = 5
peak_2_max = 20


step_at = 95
step_width = 10
prefs = "stp"
prefc = 'c'    
w_guess = 3 # sigma


import numpy as np
# for smoothing the curves
import scipy.interpolate as interp #import splev 

# for fit two voigts and fit voigt and step respectivly 
from lmfit.models import VoigtModel, ConstantModel, StepModel
from lmfit import Parameters
#from lmfit import Parameters, Parameter

import matplotlib.pyplot as plt

def compare_two_fits(x_lst,y_lst,x_min_flt,x_max_flt,print_all_fits_bool,place_to_save_str,temp):
    '''
    Taks as input
    x & y lists to preform the fit on
    bounds on the x axis
    weather or not to print fits
    if print fits, where to save
    '''
    if temp > 180:
        #one_V = fit_Voigt_and_step(x_lst,y_lst,x_min_flt,x_max_flt, 'one',width_1, width_2, print_all_fits_bool,place_to_save_str)
        one_V = fit_Voigt_and_Const(x_lst,y_lst,x_min_flt,x_max_flt, 'one',
                                    width_1, width_2, 4,
                                    print_all_fits_bool,place_to_save_str)
                
        chi_one = one_V.chisqr
        chi_two = chi_one + 7
    elif temp < 140:
        two_V = fit_main_peak_first(x_lst,y_lst,x_min_flt,x_max_flt,width_1, width_2,print_all_fits_bool,place_to_save_str)
        
        chi_two = two_V[0].chisqr + two_V[1].chisqr
        chi_one = chi_two + 7
    else:
        #one_V = fit_Voigt_and_step(x_lst,y_lst,x_min_flt,x_max_flt, 'one', width_1, width_2, print_all_fits_bool,place_to_save_str)
        one_V = fit_Voigt_and_Const(x_lst,y_lst,x_min_flt,x_max_flt, 'one',
                                    width_1, width_2, 4,
                                    print_all_fits_bool,place_to_save_str)

        two_V = fit_main_peak_first(x_lst,y_lst,x_min_flt,x_max_flt,width_1, width_2, print_all_fits_bool,place_to_save_str)
        
        chi_two = two_V[0].chisqr + two_V[1].chisqr
        chi_one = one_V.chisqr
    '''
    # list of Result object
    # This is really gross, but go fuck yourself
    # [peak_one, peak_two]
    two_V = fit_main_peak_first(x_lst,y_lst,x_min_flt,x_max_flt,print_all_fits_bool,place_to_save_str)
    # result object
    # one_V = fit_Voigt_and_Const(x_lst,y_lst,x_min_flt,x_max_flt,'one', width_one_peak, print_all_fits_bool,place_to_save_str)
    one_V = fit_Voigt_and_step(x_lst,y_lst,x_min_flt,x_max_flt, 'one', print_all_fits_bool,place_to_save_str)
    #                                       X MIN     X_MAX     prefix, 
    # compare results
    chi_two = two_V[0].chisqr + two_V[1].chisqr
    chi_one = one_V.chisqr
    '''
    #print 
    #print "\t chi_one = ", chi_one
    #print "\t chi_two = ", chi_two
    
    if chi_one > chi_two:           # return chi_two
        # Taken from 
        # https://stackoverflow.com/questions/1781571/how-to-concatenate-two-dictionaries-to-create-a-new-one-in-python?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
        rtn = reduce(lambda x,y: dict(x,**y), (two_V[0].best_values, two_V[1].best_values))
        rtn['Winner'] = 'Two'
        print "Peak one ", rtn['onecenter'], '\t', rtn['oneamplitude'], '\t', rtn['onesigma']
        print "Peak two ", rtn['twocenter'], '\t', rtn['twoamplitude'], '\t', rtn['twosigma']
    else:                           # return chi_one
        rtn = one_V.best_values
        rtn['Winner'] = 'One'
    
    # return the parameters and which one of the fits won! 
    # will preform AWPA of "Two" and simply write "One"
    return rtn

def fit_Voigt_and_Const(x_lst,y_lst, x_min_flt,x_max_flt, pre, width_1, width_2, w_guess, print_all_fits_bool, place_to_save_str):
    '''
    Fits one Voigt to y on x_min < x < x_max returns the result object
    pre = prefix you'd like the peak to have
    w_guess 
        make 2 for thin peaks
        make 5 for fat peak
    '''
    x_bkp = x_lst
    y_bkp = y_lst
    
    x_lst, y_lst, y_p, ypp = smooth_the_data(x_lst, y_lst, x_min_flt, x_max_flt)
    # x_lst = np.asarray(x_lst)
    # y_lst = np.asarray(y_lst)
    
    v = VoigtModel(prefix = pre, independent_vars=['x'],nan_policy='raise')
    c = ConstantModel(prefix=pre, independent_vars=['x'], nan_policy ='raise')
    mod = v + c
    
    # Guess parameters for voigt function
    # https://lmfit.github.io/lmfit-py/builtin_models.html
    mod.set_param_hint(pre+'amplitude', value = np.max(y_lst), min = value_max_min*np.max(y_lst), max = 5.0*value_max_max*np.max(y_lst), vary=True)
    
    if 109.5 < x_lst[np.argmax(y_lst)] < 111.5: # this is a fat peak
        mod.set_param_hint(pre+'center', value = x_lst[np.argmax(y_lst)], min = x_lst[np.argmax(y_lst)]*.9, max = x_lst[np.argmax(y_lst)]*1.1,vary=True)
        # Fwhm ish
        mod.set_param_hint(pre+'sigma', value = width_2*1.5, min = .5*width_2, max = 3*width_2,vary=True)
    elif x_lst[np.argmax(y_lst)] < 80:
        mod.set_param_hint(pre+'center', value = x_lst[np.argmax(y_lst)], vary=True)
        # Fwhm ish
        mod.set_param_hint(pre+'sigma', value = width_2*1.5, min = .5*width_2, max = 3*width_2,vary=True)
    else: # this is a thin peak
        mod.set_param_hint(pre+'center', value = x_lst[np.argmax(y_lst)], min = x_lst[np.argmax(y_lst)]*.98 , max = x_lst[np.argmax(y_lst)]*1.02, vary=True)
        # fwhm Ish
        mod.set_param_hint(pre+'sigma', value = width_1, min = .5, max = 1.5*width_1,vary=True)
        
    # mod.set_param_hint(pre+'center', value = x_lst[np.argmax(y_lst)], vary=True)
    #mod.set_param_hint(pre+'sigma', value = w_guess, min = 0, max = max_width*w_guess,vary=True)
    # guess constant offset, varry
    vvv = max([abs(np.min(y_lst)),20])
    mod.set_param_hint(pre+'c', value = vvv, min = -7*vvv, max = 7*vvv,vary=True)
    
    # let ratio varry
    # mod.set_param_hint(pre+'gamma', value = .5, min = .1, max = 6,vary=True)
    
    if pre == 'one':
        # restrict the fit again
        x_f = []
        y_f = []
        
        for a,b in zip(x_lst.tolist(),y_lst.tolist()):
            if x_lst[np.argmax(y_lst)]+peak_1_min <= a <= x_lst[np.argmax(y_lst)]+peak_1_max:
                x_f.append(a)
                y_f.append(b)
        
        x_f = np.asarray(x_f)
        y_f = np.asarray(y_f)
    else:
        x_f = x_lst
        y_f = y_lst
    
    result = mod.fit(y_f, x = x_f, params = mod.make_params())
    
    if print_all_fits_bool:
        # make dense plot of function
        x_dense = np.arange(x_min_flt,x_max_flt,(x_max_flt-x_min_flt)/300.0).tolist()
        y_fit = []
        for f in x_dense:
            y_fit.append(result.eval(x=f))
        
        # get data to fit
        x_dat = []
        y_dat = []
        for a,b in zip(x_bkp,y_bkp):
            if x_min_flt < a < x_max_flt:
                x_dat.append(a)
                y_dat.append(b)
        
        plt.plot(x_dat, y_dat, 'bx', label = 'Data')
        plt.plot(x_f,y_f,'ro', label = 'Fit Points')
        plt.plot(x_dense, y_fit, 'g', label = 'Fit')
        
        # axis label
        plt.xlabel("Inv Cm")
        plt.ylabel("counts")
        plt.title("Fit one Voigt")
        
        plt.legend()
        plt.savefig(place_to_save_str+"_oneVoigt")
        plt.clf()
    
    return result

def fit_Voigt_and_step(x_lst,y_lst,x_min_flt,x_max_flt, pre, width_1, width_2, print_all_fits_bool,place_to_save_str):
    '''
    x_lst = x axis
    y_lst = spectra to fit
    first = beginning of fitting regions
    last = end of fitting region
    print_all_fits = Bool, do you want to save all plots
    place_to_save = string that is the filename where we're saving the data
    
    returns result object
    
    '''

    # Restrict the fit
    x_bkp = x_lst
    y_bkp = y_lst
    
    x_lst, y_lst, y_p, ypp = smooth_and_remove_step(x_lst, y_lst, x_min_flt, x_max_flt,True)
    
    '''
    *******************************************************
    Section of bad code that it'd take too long to do right
    *******************************************************
    '''
    step_at = 95
    step_width = 10
    prefp = pre
    prefs = "stp"
    prefc = 'c'    
    w_guess = 3 # sigma
    '''
    *******************************************************
    Section of bad code that it'd take too long to do right
    *******************************************************
    '''
    
    # this is the money
    # defines the model that'll be fit
    peak = VoigtModel(prefix = prefp, independent_vars=['x'],nan_policy='raise')
    step = StepModel(prefix = prefs, independent_vars=['x'],form='logistic')
    const = ConstantModel(prefix = prefc,independent_vars=['x'], nan_policy='raise', form ='logistic')
    
    mod = peak + step + const
    #mod = peak + const
    
    # guess parameters
    x_max = x_lst[np.argmax(y_lst)]
    y_max = y_lst[np.argmax(y_lst)]
    
    # Peak
    # here we set up the peak fitting guess. Then the peak fitter will make a parameter object out of them
    mod.set_param_hint(prefp+'amplitude', value = value_max*y_max, min = .6*value_max_min*y_max,max = 4*value_max_max*y_max, vary=True)
    # mod.set_param_hint(prefp+'center', value = x_max, min = x_max*(1-wiggle_room), max = x_max*(1+wiggle_room),vary=True)
    mod.set_param_hint(prefp+'center', value = x_max,min = x_max*.97, max = x_max*1.03, vary=True)
    
     # Basically FWHM/3.6
    if pre =='one':     # fitting with only one peak
        mod.set_param_hint(prefp+'sigma', value = width_1, min = .25*width_2, max = 2*width_1,vary=True)
    else:               # fitting with two peaks
        mod.set_param_hint(prefp+'sigma', value = width_2, min = 0, max = width_1,vary=True)
    
    # Constant
    top = []
    bottom = []
    for a,b in zip(x_lst,y_lst):
        if a > 135:
            top.append(b)
        elif a < 93:
            bottom.append(b)
    top = np.mean(np.asarray(top))
    bottom = np.mean(np.asarray(bottom))
            
    mod.set_param_hint(prefc+'c', value = bottom, min = -3*bottom, max = 3*bottom,vary=True)
    
    # restrict the fit again
    x_fit = []
    y_fit = []
    for a,b in zip(x_lst,y_lst):
        if 80 < a < 135:
            x_fit.append(a)
            y_fit.append(b)
    top = y_fit[0]
    bottom = y_fit[-1]
    
    # Step
    # Step height
    delta = 2*abs(top - bottom)
    if delta == 0:
        delta = 1
    mod.set_param_hint(prefs+'amplitude', value = delta, min = -3*delta, max = 3*delta, vary=True)
    # Charastic width
    mod.set_param_hint(prefs+'sigma', value = 3,min = 1, max = 3, vary=False)
    # The half way point... 
    mod.set_param_hint(prefs+'center', value = step_at, min = step_at-step_width, max = step_at+step_width, vary = False)
    
    result = mod.fit(y_fit, x=x_fit, params = mod.make_params())
    
    # If print all fits ... 
    if print_all_fits_bool:
        x_dense = np.arange(x_min_flt,x_max_flt,(x_max_flt-x_min_flt)/300.0).tolist()
        
        # each component
        for x in result.best_values:
            if prefp in x:      # Get peak
                peak.set_param_hint(x, value = result.best_values[str(x)])
            elif prefs in x:    # Get step
                step.set_param_hint(x, value = result.best_values[str(x)])
        
        # Data - 'background' 
        y_m_background = []
        for a,b in zip(x_lst,y_lst):
            y_m_background.append(b - result.eval(x=a) + peak.eval(x=a,  params=peak.make_params()))
        
        peak_only = [peak.eval(x=yy, params=peak.make_params()) for yy in x_dense]
        #stp_only = [result.best_values['stpamplitude'] + result.best_values['cc']]*len(x_dense)
        # sum of them
        #y_fit = [a+b for a,b in zip(peak_only,stp_only)]
        y_fit = [a+b for a in peak_only]
        
        plt.plot(x_dense,peak_only, 'g', label = 'Peak Only')
        #plt.plot(x_dense,stp_only, 'g--', label = None)
        #plt.plot(x_dense, y_fit, 'g', label = "Fit Result")        
        
        plt.plot(x_lst,y_lst,'bx', label= "Data")
        plt.plot(x_lst,y_m_background,'ko', label= "Data-Background")
        
        plt.title("Fit vs Data")
        plt.xlabel("Inv Cm")
        plt.ylabel("counts")
        plt.legend()
        plt.savefig(place_to_save_str+"Voigt&Step")
        plt.clf()    
    
    return result

def fit_main_peak_first(x_lst,y_lst,x_min_flt,x_max_flt,width_1,width_2,print_all_fits_bool,place_to_save_str):    
    '''
    takes as input
    x and y lists, bounds (max and min) of fit region in x
    weather or not to save the fits, and where to save the fits
    
    Main peak -> second peak -> main peak
    returns a dictionary of fit results for the two peaks
                + 'chisqr' = chisquared value of the fit
    '''
    
    # this returns 
    # x region of fit
    # y region of fit
    # smoothed y prime
    # smoothed y prime prime
    x_fit, y_fit, y_smooth, y_p, y_pp = smooth_and_remove_step(x_lst, y_lst, x_min_flt, x_max_flt,False)
    
    # fit main peak side
    x_max = x_fit[np.argmax(y_smooth)]  
    
    if not (108 < x_max < 111):
        x_max = 109
    
    x_1st = []
    y_1st = []
    for a,b in zip(x_fit,y_fit):
        if x_max + peak_1_min < a < x_max+peak_1_max:
            x_1st.append(a)
            y_1st.append(b)
    peak_one = fit_Voigt_and_Const(x_1st, y_1st, x_max + peak_1_min, x_max+peak_1_max, \
                                   'one', width_1, width_2, width_two_peak, False, place_to_save_str)
    
    '''
    First try was fit peak to rest of region
    Second try is to fit peak left of x_max a little
    '''
    
    # FIT SECOND PEAK
    x_2nd = []
    y_2nd = []
    two_fit_min = x_max + peak_2_min
    two_fit_max = x_max + peak_2_max
    for f,g in zip(x_fit,y_fit):
        if two_fit_min < f < two_fit_max:
            y_2nd.append(g-peak_one.eval(x=f)+peak_one.best_values['onec'])
            x_2nd.append(f)
            
    peak_two = fit_Voigt_and_Const(x_2nd, y_2nd, two_fit_min, two_fit_max, \
                                   'two', width_1, width_2, width_two_peak, False, place_to_save_str)

    # Refit main peak over data - peak 2
    y_holder = []
    for f,g in zip(x_fit,y_fit):
        y_holder.append(g-peak_two.eval(x=f))
    # same region as before
    x_1st = []
    y_1st = []
    for a,b in zip(x_fit,y_fit):
        if x_max + peak_1_min < a < x_max+peak_1_max:
            x_1st.append(a)
            y_1st.append(b-peak_two.eval(x=f))
    
    #peak_one = fit_Voigt_and_Const(x_holder,y_holder,'one')
    peak_one = fit_Voigt_and_Const(x_1st, y_1st, np.min(x_fit), x_max+x_add, \
                                   'one', width_1, width_2, width_two_peak, False, place_to_save_str)

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
        if False:
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
        
        # data, data - peak1, data-peak2, fit
        if True:
            # data - peak 1
            plt.plot(x_2nd, y_2nd, 'ro')
            plt.plot(x_1st, y_1st, 'ro')
            # data - peak 2
            #plt.plot(x_m_2nd,y_m_2nd,'r.')
            # all data
            plt.plot(x_fit,y_fit,'b+', label = 'data')
            
            # both peaks on their own
            plt.plot(x_dense, y_one, 'g--')
            plt.plot(x_dense, y_two, 'g--')
            # resulting fit
            plt.plot(x_dense, y_together,'g', label = 'Together fit')
            
            # axis label
            plt.xlabel("Inv Cm")
            plt.ylabel("counts")
            plt.title("The two fits summed together")
            
            plt.xlim((100,125))
            
            plt.legend()
            plt.savefig(place_to_save_str+"_AllTogether")
            plt.clf()
        
        # print data & data-peak_two & Fit peak 1
        if False: 
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
    
    return [peak_one, peak_two]
    
    '''
    # Taken from 
    # https://stackoverflow.com/questions/1781571/how-to-concatenate-two-dictionaries-to-create-a-new-one-in-python?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
    rtn = reduce(lambda x,y: dict(x,**y), (peak_one.best_values, peak_two.best_values))
    rtn['chisqr'] = peak_one.chisqr + peak_two.chisqr
    return rtn
    '''


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

def smooth_and_remove_step(x_lst, y_lst, x_min_flt,x_max_flt,rmv_step_bool):
    ''' 
    Takes entire data set, x and y
    cuts down the spectra s.t x_min < x < x_max
    THEN
    Removes a step function from y_lst
    '''
    
    # Restrict the fit
    x_fit = []
    y_fit = []
    
    top_lst = []
    bottom_lst = []
    
    for x,y in zip(x_lst, y_lst):
        # Restrict the fitting region
        if x_min_flt < x < x_max_flt:
            x_fit.append(float(x))
            y_fit.append(float(y))
        
        # Find top and bottom of step 
        if x < x_min_flt + 7:
            bottom_lst.append(float(y))
        elif x > x_max_flt - 7:
            top_lst.append(float(y))
    
    x_fit = np.asarray(x_fit)
    y_fit = np.asarray(y_fit)   
  
    top = np.mean(np.asarray(top_lst))
    bottom = np.mean(np.asarray(bottom_lst))
    delta = top-bottom
    
    if (rmv_step_bool):
        # Step Parameters
        step_at = 100
        step_width = 1    
        pp = Parameters()
        pp.add_many(
                ('amplitude',delta),
                    ('sigma',step_width),
                    ('center',step_at)
                    )
        step = StepModel(form = 'erf', prefix='', independent_vars=['x'])
        
        y_fit = np.asarray([yy-bottom-step.eval(x=xx, params=pp) for xx,yy in zip(x_fit,y_fit)])
    
    # rest is the same as smooth_the_data 
    
    # now we find the parameters using the - d^2/dx^2
    ysmooth = interp.interp1d(x_fit, y_fit, kind='cubic')
    # differentiate x 2
    yp = np.gradient(ysmooth(x_fit))
    ypp = np.gradient(yp)
    # we want the peaks of -d2/dx2 
    ypp = np.asarray([-x for x in ypp])
    
    return x_fit, y_fit, ysmooth, yp, ypp










