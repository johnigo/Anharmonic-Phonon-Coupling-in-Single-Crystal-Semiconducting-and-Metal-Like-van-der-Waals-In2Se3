# -*- coding: utf-8 -*-
"""
Created on Fri Mar 09 11:01:36 2018

@author: John
"""

# helps read data
import glob
# To interface with Pandas adn pyplot
import numpy as np
# to fuck with the data
import pandas as pd
# for plotting
import matplotlib.pyplot as plt

class Beta_data():
    '''
    '''
    
    def __init__(self,data_location,xunits,yunits):
        self.xunits = xunits
        self.yunits = yunits
        
        self.data = _get_data(self,data_location,xunits,yunits)
        ''' Returns a dictionary with the DataFrame
        Key:value = filename:ydata + xunits:xdata''' 
        #self.peakpos = _peakpos(self, data_location,xunits,yunits)
        #
        ''' Now does nothing
        Will Return a dictionary with normalized y data
        Key:Value = filename:normalized(ydata) + xunits:xdata'''
        
        #self.fit_one_lz = fit_one_lz(self, xunits, p_guess, w_guess)
        
def _get_data(self, data_location, xunits,yunits):
    '''
    reads in the data, returns a dataframe whose headers are 
    the filenames 
    '''
    
    all_files = glob.glob(data_location+"/*.txt")
    
    d = {}
    
    for f in all_files:
        # get file name
        fname = f.split('\\')[-1].split(".")[0]
        # read the files
        holder = np.genfromtxt(f,delimiter='\t',names=[xunits,fname])
        # put in dictionary
        d[fname] = holder[fname]
    # append x axis
    d[xunits] = holder[xunits]
    # concatinate l and return that
    df = pd.DataFrame(d)
    
    return df

def _normalized(self,data_location,xunits,yunits):
    
    
    return 0

# Taken from 
#https://stackoverflow.com/questions/10365225/extract-digits-in-a-simple-way-from-a-python-string
def get_num(x):
    return int(''.join(ele for ele in x if ele.isdigit()))

def get_temps(lst, temp_strs):
    ''' 
    takes list of all DataFrame columns, extracts each unique temperature
    '''
    tmp_lst = []
    
    # Loop over all filenames 
    for x in lst: 
        # split x 
        xsplit = x.split("_")
        # reset temp
        temp = ""
        # 
        for y in temp_strs:
            try:
                temp = get_num(xsplit[xsplit.index(y)+1])
            except ValueError:
                temp = temp
        
        if temp in tmp_lst:
            next
        elif temp =='':
            print "bad temp list", x, " not in list" 
        else: 
            tmp_lst.append(temp)
    
    return sorted(tmp_lst)

def get_pwrs(lst, pwr_strs):
    ''' 
    takes list of all DataFrame columns, extracts each unique temperature
    Same as get_temps
    '''
    pwr_lst = []
    
    # Loop over all filenames 
    for x in lst: 
        # split x 
        xsplit = x.split("_")
        # reset temp
        temp = ""
        # 
        for y in pwr_strs:
            try:
                temp = get_num(xsplit[xsplit.index(y)+1])
            except ValueError:
                temp = temp
        
        if temp in pwr_lst:
            next
        elif temp =='':
            print "bad pwr list", x, " not in list" 
        else: 
            pwr_lst.append(temp)
    
    return sorted(pwr_lst)

def get_temp_pwr_other(lst,temp_flags,pwr_flags):
    '''
    takes list of filenames
    returns 
    [[temperature],[power],[othershit]]
    for each filename, so
    {filename:[temp,power,other],filename:[temp,pwr,other]}
    temp is whatever comes after temperature
    power is whatever comes after power
    other is whatever is left - usually layer name, is the same length
        as power list
    '''
    rtn = {}
    
    for name in lst:
        
        holder = name.split("_")
        
        # get temp
        temp = ''
        for x in range(0,len(holder)):
            if holder[x] in temp_flags:
                temp = holder[x+1]
        # get power
        pwr = ''
        for x in range(0,len(holder)):
            if holder[x] in pwr_flags:
                pwr = holder[x+1]
        
        # whatever is left
        other = ''
        for x in range(0,len(holder)):
            if not(holder[x] == temp or holder[x] == pwr or \
                   holder[x] in temp_flags or holder[x] in pwr_flags):
                other += str(holder[x])
        
        rtn[str(name)] = [str(temp), str(pwr),str(other)]
    
    return rtn

def get_other(lst,pwr_flags,temp_flags):
    '''
    Get all the shit out of the filename that's not temp or pwr
    if nothing else, returns "none" 
    '''
    rtn = []
    
    for x in lst:
        
        name = ''
        # split x
        xsplit = x.split("_")
        
        # This tags all the entrys 
        # that are power or temp related
        kill_lst = [0]*len(xsplit)
        for x in range(0,len(xsplit)):
            if xsplit[x] in temp_flags or xsplit[x] in pwr_flags:
                if x == len(xsplit)-1:
                    kill_lst[x] = 1
                else:
                    kill_lst[x] = 1
                    kill_lst[x+1] = 1
        
        # if whatever is untagged, add it to name
        for kl,xp in zip(kill_lst, xsplit):
            if kl == 0:
                name+=xp
        
        rtn.append(name)
    
    return rtn


def DoubleLorentzian(params,x,data):
        PI = 3.14159265358979323846264338327950288419716
        # LZ one
        center1 = params['center1']
        width1 = params['width1']
        scale1 = params['scaling1']
        const1 = params['const1']
        
        # LZ two
        center2 = params['center2']
        width2 = params['width2']
        scale2 = params['scaling2']
        const2 = params['const2']
    
        model = const1 + scale1/PI*(.5 *width1)/((x-center1)**2+(.5*width1)**2) +\
                                   const2 + scale2/PI*(.5*width2)/((x-center2)**2+(.5*width2)**2)
        
        return model - data

def johnLorentzian(params,x,data): #p[0]==center , p[1] == width, p[2] == scalingfactor, p[3] == constant offset
    PI = 3.14159265358979323846264338327950288419716
    # LZ one
    center1 = float(params['center1'])
    width1 = float(params['width1'])
    scale1 = float(params['scaling1'])
    const1 = float(params['const1'])
    
    model = const1 + scale1/PI*(.5 *width1)/((x-center1)**2+(.5*width1)**2)
    
    return model - data 

def print_all_fits(x, y, first, last, p, place_to_save):
    '''
    Saves x, y and the fit in place to save
    '''
    
    # restrict the plotting range
    '''
    *********************
    This is terrible code
    *********************
    '''
        # Restrict fitting region
    x_plt = []
    y_plt = []
    for x,y in zip(x,y):
        if first <= x and x <= last:
            x_plt.append(x)
            y_plt.append(y)
    
    x_vs_y = plt.plot(x_plt,y_plt,'rx', label ="spectra")
    # make fit have more points than the data
    '''
    *********************
    This is terrible code
    *********************
    '''
    num_pts = 200
    # This plots the fit resulst
    fff = np.arange(min(x_plt),max(x_plt),(max(x_plt)-min(x_plt))/num_pts).tolist()
    ggg = [DoubleLorentzian(p,xx,0) for xx in fff]
    fit_plt = plt.plot(fff,ggg,color = 'blue', label = 'fit')
    
    # this plots each LZ on it's own
    p_one = {}
    p_two = {}
    for f in p.keys():
        if "1" in f:
            # Keys need to have a "1" in them
            p_one[f] = p[f]
        if "2" in f:
            # Keys need to have a "1" in them
            key = str(f[:-1] + "1")
            p_two[key] = p[f]
    
    l1_plt = plt.plot(fff,[johnLorentzian(p_one,xx,0) for xx in fff],color = 'green', label = 'One')
    l2_plt = plt.plot(fff,[johnLorentzian(p_two,xx,0) for xx in fff],color = 'green', label = 'Two')
    
    plt.xlabel("Wavelength")
    plt.ylabel("Peak pos")
    plt.title(place_to_save.split('\\')[-1])
    
    plt.savefig(place_to_save)
    plt.clf()
    
    return 0

def johnLorentzian(params,x,data): #p[0]==center , p[1] == width, p[2] == scalingfactor, p[3] == constant offset
    PI = 3.14159265358979323846264338327950288419716
    # LZ one
    center1 = float(params['center1'])
    width1 = float(params['width1'])
    scale1 = float(params['scaling1'])
    const1 = float(params['const1'])
    
    model = const1 + scale1/PI*(.5 *width1)/((x-center1)**2+(.5*width1)**2)
    
    return model - data 

def Voigt_fwhm(param_dict):
    '''
    Takes parameter dictionary from Voigt fit, gets fwhm based one ... 
    https://en.wikipedia.org/wiki/Voigt_profile#The_width_of_the_Voigt_profile
    
    If gamme is tied to sigma
    https://lmfit.github.io/lmfit-py/builtin_models.html
    '''
    f_l = 1
    f_g = 1
    f_v = .5346* f_l + pow(.2166* f_l**2 + f_g**2,1/2)
    
    
    # Unfinished! 
    
    return f_v

def fit_Voigt_and_step(x_lst,y_lst,x_min_flt,x_max_flt,print_all_fits_bool,place_to_save_str):
    '''
    x_lst = x axis
    y_lst = spectra to fit
    first = beginning of fitting regions
    last = end of fitting region
    print_all_fits = Bool, do you want to save all plots
    place_to_save = string that is the filename where we're saving the data
    
    '''
    import numpy as np
    # for smoothing the curves
    import scipy.interpolate as interp #import splev 
    
    from lmfit.models import VoigtModel, StepModel, ConstantModel
    from lmfit import CompositeModel
    
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
    
    '''
    *******************************************************
    Section of bad code that it'd take too long to do right
    *******************************************************
    '''
    step_at = 100
    step_width = 3
    prefp = "one"
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
    step = StepModel(prefix = prefs, independent_vars=['x'], nan_policy='raise')
    const = ConstantModel(prefix = prefc,independent_vars=['x'], nan_policy='raise', form ='logistic')
    
    mod = peak + step + const
    
    # guess parameters
    x_max = x_fit[np.argmax(y_fit)]
    y_max = y_fit[np.argmax(y_fit)]
    
    # Peak
    # here we set up the peak fitting guess. Then the peak fitter will make a parameter object out of them
    mod.set_param_hint(prefp+'amplitude', value = 4*y_max, min = y_max,max = 30*y_max, vary=True)
    # mod.set_param_hint(prefp+'center', value = x_max, min = x_max*(1-wiggle_room), max = x_max*(1+wiggle_room),vary=True)
    mod.set_param_hint(prefp+'center', value = x_max, vary=True)
    # Basically FWHM/3.6
    mod.set_param_hint(prefp+'sigma', value = w_guess, min = 0, max = 5*w_guess,vary=True)
    
    # Step
    # Step height
    delta = abs(y_fit[-1]-y_fit[0])
    mod.set_param_hint(prefs+'amplitude', value = delta, min = delta*.9, max = delta*1.1, vary=True)
    # Charastic width
    mod.set_param_hint(prefs+'sigma', value = 2,min = 1, max = 3, vary=True)
    # The half way point... 
    mod.set_param_hint(prefs+'center', value = step_at, min = step_at-step_width, max = step_at+step_width, vary = True)
    
    # Constant
    mod.set_param_hint(prefc+'c', value = y_fit[-1], min = 0, max = 2*y_fit[0],vary=True)    
    
    result = mod.fit(y_fit, x=x_fit, params = mod.make_params())
    
    # If print all fits ... 
    if print_all_fits_bool:
        x_dense = np.arange(x_min_flt,x_max_flt,(x_max_flt-x_min_flt)/300.0).tolist()
        
        result.plot_fit(xlabel='Inv Cm', ylabel='counts',datafmt = 'xb', numpoints=len(x_fit)*10)
        
        for x in result.best_values:
            if prefp in x:      # Get peak
                peak.set_param_hint(x, value = result.best_values[str(x)])
            elif prefs in x:    # Get step
                step.set_param_hint(x, value = result.best_values[str(x)])
        
        comp = [result.best_values['cc'] + peak.eval(x=yy, params=peak.make_params()) for yy in x_dense]
        plt.plot(x_dense,comp, 'green', label = None)
        
        comp = [result.best_values['stpamplitude'] + result.best_values['cc']]*len(x_dense)
        plt.plot(x_dense, comp, 'green', label= None)
        
        # comp = [result.best_values['cc'] + step.eval(x=yy, params=step.make_params()) for yy in x_dense]
        # plt.plot(x_dense, comp, 'green', label= None)
        
        plt.title("Fit vs Data")
        plt.legend()
        plt.savefig(place_to_save_str)
        plt.clf()    
    
    return result.best_values
    
def fit_two_Psudo_Voigt(x_lst,y_lst,x_min_flt,x_max_flt,print_all_fits_bool,place_to_save_str):
    '''
    x_lst = x axis
    y_lst = spectra to fit
    first = beginning of fitting regions
    last = end of fitting region
    print_all_fits = Bool, do you want to save all plots
    place_to_save = string that is the filename where we're saving the data
    
    This takes the spectra and fits two Lorentzian curves to it. 
    Returns dictionary of fit values 
    Parameters have prefixes "one" for first V, "two" for second V, "c" for constant
    '''
    
    import numpy as np
    # for smoothing the curves
    import scipy.interpolate as interp #import splev 
    
    from lmfit.models import PseudoVoigtModel, ConstantModel
    
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
    
    '''
    *******************************************************
    Section of bad code that it'd take too long to do right
    *******************************************************
    '''
    # % of wavelength you want the peak centers to move 
    wiggle_room = .05
    w_guess = 3 # sigma
    pref1 = 'one'
    pref2 = 'two'
    prefo = 'off'
    
    # if the fancy shit doesn't work, this is how far in index
    # we shift the 2nd peak and max over
    doesnt_work_shift = 10
    '''
    *******************************************************
    Section of bad code that it'd take too long to do right
    *******************************************************
    '''
    
    # this is the money
    # defines the model that'll be fit
    peak1 = PseudoVoigtModel(prefix = pref1, independent_vars=['x'],nan_policy='raise')
    peak2 = PseudoVoigtModel(prefix = pref2, independent_vars=['x'],nan_policy='raise')
    offset = ConstantModel(prefix=prefo, independent_vars=['x'],nan_policy='raise')
    
    mod = peak1 + peak2 + offset
    
    # guess parameters
    x_max = x_fit[np.argmax(ypp)]
    y_max = y_fit[np.argmax(ypp)]
    
    # peak #1 
    # here we set up the peak fitting guess. Then the peak fitter will make a parameter object out of them
    mod.set_param_hint(pref1+'amplitude', value = 5*y_max, min=y_max*.8,max = y_max*6,vary=True)
    
    mod.set_param_hint(pref1+'center', value = x_max, min = x_max*(1-wiggle_room), max = x_max*(1+wiggle_room),vary=True)
    
    mod.set_param_hint(pref1+'sigma', value = w_guess, min = 0, max = 5*w_guess,vary=True)
    
    # Set Fraction
    mod.set_param_hint(pref1+'fraction', value = .5, min = 0, max = 1,vary=True)

    # Change gama maybe
    # mod.set_param_hint(pref1+'gamma', value = 1, vary=True)
    
    # peak #2
    x_trunk = []
    y_trunk = []
    ypp_trunk = []
    try:
        for a,b,c in zip(x_fit.tolist(),y_fit.tolist(),ypp.tolist()):
            '''
            BAD CODE MAKE THIS BETTER
            '''
            if x_max + 8 < a < x_max + 12:
                x_trunk.append(a)
                y_trunk.append(b)
                ypp_trunk.append(c)
        x_trunk = np.asarray(x_trunk)
        y_trunk = np.asarray(y_trunk)
        ypp_trunk = np.asarray(ypp_trunk)
        
        x_max_2 = x_trunk[np.argmax(ypp_trunk)]
        y_max_2 = y_trunk[np.argmax(ypp_trunk)]
        
    except ValueError:
        x_max_2 = x_trunk[np.argmax(ypp) + doesnt_work_shift]
        y_max_2 = y_trunk[np.argmax(ypp) + doesnt_work_shift]
        
    # add peak 2 paramaters 
    mod.set_param_hint(pref2+'amplitude', value = 4*y_max_2, min=y_max_2*.8,max = y_max_2*6,vary=True)
    # changed the bounds to be near other peak
    mod.set_param_hint(pref2+'center', value = x_max_2, min = x_max+8, max = x_max+14,vary=True)
    
    mod.set_param_hint(pref2+'sigma', value = w_guess/2, min = 0, max = w_guess ,vary=True)
    #mod.set_param_hint(pref2+'sigma', pref1 + 'sigma' < expr < pref1 + 'sigma')
    
    # Set Fraction
    mod.set_param_hint(pref2+'fraction', value = .5, min = 0, max = 1,vary=True)
    
    # Change gama maybe
    # mod.set_param_hint(pref2+'gamma', value = 1, vary=False)
    
    # constant offest
    mod.set_param_hint(prefo+'c', value = y_fit[-1], min = 0, max = 5*y_fit[-1],vary=False)
    
    # this does the fitting
    # the params = mod.ma... is what initializes the parameters
    result = mod.fit(y_fit, x=x_fit, params = mod.make_params())
    
    # If print all fits ... 
    if print_all_fits_bool:
        x_dense = np.arange(x_min_flt,x_max_flt,(x_max_flt-x_min_flt)/300.0).tolist()
        
        result.plot_fit(xlabel='Inv Cm', ylabel='counts',datafmt = 'xb', numpoints=len(x_fit)*10)
        
        '''
        Here we make paramaters for peak 1 and 2
        '''
        for x in result.best_values:
            if pref1 in x:
                peak1.set_param_hint(x, value = result.best_values[str(x)])
            elif pref2 in x:
                peak2.set_param_hint(x, value = result.best_values[str(x)])
            else:
                peak1.set_param_hint(x, value = result.best_values[str(x)])
                peak2.set_param_hint(x, value = result.best_values[str(x)])
        
        comp = [peak1.eval(x=yy, params=peak1.make_params()) for yy in x_dense]
        plt.plot(x_dense,comp, 'green', label = None)
        
        comp = [peak2.eval(x=yy, params=peak2.make_params()) for yy in x_dense]
        plt.plot(x_dense, comp, 'green', label= None)
        plt.title("Fit vs Data")
        plt.ylim(0, 1.1*np.max(y_fit))
        plt.legend()
        plt.savefig(place_to_save_str)
        plt.clf()
        
    return result.best_values
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

def fit_two_Voigt(x_lst,y_lst,x_min_flt,x_max_flt,print_all_fits_bool,place_to_save_str):
    '''
    x_lst = x axis
    y_lst = spectra to fit
    first = beginning of fitting regions
    last = end of fitting region
    print_all_fits = Bool, do you want to save all plots
    place_to_save = string that is the filename where we're saving the data
    
    This takes the spectra and fits two Lorentzian curves to it. 
    Returns dictionary of fit values 
    Parameters have prefixes "one" for first V, "two" for second V, "c" for constant
    '''
    
    # Follows
    # http://lmfit.github.io/lmfit-py/builtin_models.html#example-1-fit-peaked-data-to-gaussian-lorentzian-and-voigt-profiles
    # and 
    # http://cars9.uchicago.edu/software/python/lmfit_MinimizerResult/builtin_models.html
    # and 
    # the alpha version of this code
    # Figure out Composit Model
    
    import numpy as np
    # for smoothing the curves
    import scipy.interpolate as interp #import splev 
    
    from lmfit.models import VoigtModel, ConstantModel
    
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
    
    '''
    *******************************************************
    Section of bad code that it'd take too long to do right
    *******************************************************
    '''
    # % of wavelength you want the peak centers to move 
    wiggle_room = .05
    w_guess = 1 # sigma
    pref1 = 'one'
    pref2 = 'two'
    prefo = 'off'
    
    # if the fancy shit doesn't work, this is how far in index
    # we shift the 2nd peak and max over
    doesnt_work_shift = 10
    '''
    *******************************************************
    Section of bad code that it'd take too long to do right
    *******************************************************
    '''
    
    # this is the money
    # defines the model that'll be fit
    peak1 = VoigtModel(prefix = pref1, independent_vars=['x'],nan_policy='raise')
    peak2 = VoigtModel(prefix = pref2, independent_vars=['x'],nan_policy='raise')
    offset = ConstantModel(prefix=prefo, independent_vars=['x'],nan_policy='raise')
    
    mod = peak1 + peak2 + offset
    
    # guess parameters
    x_max = x_fit[np.argmax(ypp)]
    y_max = y_fit[np.argmax(ypp)]
    
    # peak #1 
    # here we set up the peak fitting guess. Then the peak fitter will make a parameter object out of them
    mod.set_param_hint(pref1+'amplitude', value = 4*y_max, min=y_max*.8,max = y_max*9,vary=True)
    
    mod.set_param_hint(pref1+'center', value = x_max, min = x_max*(1-wiggle_room), max = x_max*(1+wiggle_room),vary=True)
    
    mod.set_param_hint(pref1+'sigma', value = w_guess, min = 0, max = 5*w_guess,vary=True)
    
    # Change gama maybe
    mod.set_param_hint(pref1+'gamma', value = 1, vary=True)
    
    # peak #2
    x_trunk = []
    y_trunk = []
    ypp_trunk = []
    try:
        for a,b,c in zip(x_fit.tolist(),y_fit.tolist(),ypp.tolist()):
            '''
            BAD CODE MAKE THIS BETTER
            '''
            if x_max + 8 < a < x_max + 12:
                x_trunk.append(a)
                y_trunk.append(b)
                ypp_trunk.append(c)
        x_trunk = np.asarray(x_trunk)
        y_trunk = np.asarray(y_trunk)
        ypp_trunk = np.asarray(ypp_trunk)
        
        x_max_2 = x_trunk[np.argmax(ypp_trunk)]
        y_max_2 = y_trunk[np.argmax(ypp_trunk)]
        
    except ValueError:
        x_max_2 = x_trunk[np.argmax(ypp) + doesnt_work_shift]
        y_max_2 = y_trunk[np.argmax(ypp) + doesnt_work_shift]
    
    # add peak 2 paramaters 
    mod.set_param_hint(pref2+'amplitude', value = 4*y_max_2, min=y_max_2*.8,max = y_max_2*9,vary=True)
    # changed the bounds to be near other peak
    mod.set_param_hint(pref2+'center', value = x_max_2, min = x_max+8, max = x_max+14,vary=True)
    mod.set_param_hint(pref2+'sigma', value = w_guess, min = 0, max = 5*w_guess,vary=True)
    
    # Change gama maybe
    mod.set_param_hint(pref2+'gamma', value = 1, vary=False)
    
    # constant offest
    mod.set_param_hint(prefo+'c', value = y_fit[-1], min = 0, max = 5*y_fit[-1],vary=False)
    
    # this does the fitting
    # the params = mod.ma... is what initializes the parameters
    result = mod.fit(y_fit, x=x_fit, params = mod.make_params())
    
    # If print all fits ... 
    if print_all_fits:
        x_dense = np.arange(x_min_flt,x_max_flt,(x_max_flt-x_min_flt)/300.0).tolist()
        
        result.plot_fit(xlabel='Inv Cm', ylabel='counts',datafmt = 'xb', numpoints=len(x_fit)*10)
        
        '''
        Here we make paramaters for peak 1 and 2
        '''
        for x in result.best_values:
            if pref1 in x:
                peak1.set_param_hint(x, value = result.best_values[str(x)])
            elif pref2 in x:
                peak2.set_param_hint(x, value = result.best_values[str(x)])
            else:
                peak1.set_param_hint(x, value = result.best_values[str(x)])
                peak2.set_param_hint(x, value = result.best_values[str(x)])
        
        comp = [peak1.eval(x=yy, params=peak1.make_params()) for yy in x_dense]
        plt.plot(x_dense,comp, 'green', label = None)
        
        comp = [peak2.eval(x=yy, params=peak2.make_params()) for yy in x_dense]
        plt.plot(x_dense, comp, 'green', label= None)
        plt.title("Fit vs Data")
        plt.ylim(0, 1.1*np.max(y_fit))
        plt.legend()
        plt.savefig(place_to_save_str)
        plt.clf()
        
    return result.best_values
    
def fit_two_lz(x_lst, spectra_lst, first, last):
    '''
    Give it a list x spectra and y spectra
    where to begin the fit and where to end it
    '''
    
    # for gradient
    import numpy as np
    # for smoothing the curves
    import scipy.interpolate as interp #import splev 
    # for non linear fitting
    from lmfit import minimize, Minimizer, Parameters, Parameter, report_fit
    
    # Restrict fitting region
    x_fit = []
    y_fit = []
    for x,y in zip(x_lst,spectra_lst):
        if first <= x and x <= last:
            x_fit.append(x)
            y_fit.append(y)
    
    # The length of the fitting region
    len_fit = len(x_fit)
    '''
    FUCK ME THIS IS BAD CODE
    '''
    wiggle_room = .2
    w_guess = 5
    const = y_fit[0]
    ysmooth = interp.interp1d(x_fit, y_fit, kind='cubic')
    # normalize d/dx of smooth spectra
    # yp = list_normalizer(np.gradient(ysmooth(x_lst)).tolist(),1,0)
    yp = np.gradient(ysmooth(x_fit)).tolist()
    # normalize d2/dx2 of smooth spectra
    #ypp = list_normalizer(np.gradient(yp).tolist(),1,0)
    ypp = np.gradient(yp).tolist()
    # we want the peaks of -d2/dx2 
    ypp = [-x for x in ypp]
    '''
    FUCK ME THIS IS BAD CODE
    '''
    
    # normalize spectra
    # don't normalize it
    # spectra_lst = list_normalizer(spectra_lst,1,0)
    # interpolate spectra to smooth
    
    
    '''
    Start nailing down parameters 
    '''
    
    max_value = max(ypp)
    max_index = ypp.index(max_value)
    
    max_value = y_fit[max_index]
    peak1_guess = x_fit[max_index]
    
    '''
    First peak is easy to find, now we have two regeims for the rest
    Are the peaks distinguishable, Yes or not
    '''
    params = Parameters()
    '''
    If one wanted to put in something very tricky, like say, 
    fitting a Lorentzian to the LHS of the first peak and subtracting that
    LZ from the spectra, this is where you would do it
    ''' 
    second_index = len_fit - 5
    try: 
        # Find maximum in ypp on the range restricted to what's left of the spectra
        second_value = 0
        for x,y in zip(x_fit,ypp):
            if x > x_fit[max_index]:
                if y > second_value:
                    second_value = y
                    second_index = x_fit.index(x)
                    
    except ValueError:
        second_value = y_fit[max_index - 4]
        print "Value Error in finding second index"
        
    # Nail down the index and value for second peak
    peak2_guess = x_fit[second_index]
    second_value = y_fit[second_index]
    
    # This checks that the centers are *actually* distinguishable
    if not (9 < abs(peak1_guess - peak2_guess) < 11):
        peak2_guess = peak1_guess + 10
    
    '''
    Now we make the parameters dictionary for lmfit
    '''
    params.add('center1',value=peak1_guess,min=peak1_guess-wiggle_room*len_fit,max=peak1_guess+wiggle_room*len_fit, vary=True)
    params.add('center2',value=peak2_guess,min=peak2_guess-wiggle_room*len_fit,max=peak2_guess+wiggle_room*len_fit, vary=True)
    
    params.add('width1', value = w_guess, min = 0, max =5*w_guess, vary=True)
    params.add('width2', value = w_guess, min = 0, max =5*w_guess, vary=True)
    
    params.add('const1', value = const, min = -.1, max = .1, vary=True)
    params.add('const2', value = const, expr='const1',min = -.1, max = .1, vary=False)
    
    # scaling is given by 
    # Height max = scaling*2/(Pi* width)
    s1 = max_value*(2.0/3.14159*(params['width1'])**-1)**-1
    s2 = second_value*(2.0/3.14159*(params['width2'])**-1)**-1
                      
    # Functionally the same as what's above
    params['scaling1'] = Parameter(name='scaling1', value = s1,min = 0, max = 2*s1, vary=True)
    params['scaling2'] = Parameter(name='scaling2', value = s2,min = 0, max = 2*s2, vary=True)
    
    '''
    We have the parameters! 
    Time to do actual fitting
    '''
    
    # Work horse
    minner = Minimizer(DoubleLorentzian, params, fcn_args=(x_fit, y_fit))
    result = minner.minimize(method='least_squares')
    
    return_dict = {}
    
    for x in sorted(result.params):
        return_dict[x] = result.params[x].value
        
    return return_dict

    


