# -*- coding: utf-8 -*-
"""
Created on Fri Mar 09 11:01:36 2018

@author: John


Alpha Data returns a DataFrame of

"""

# helps read data
import glob
# To interface with Pandas adn pyplot
import numpy as np
# to fuck with the data
import pandas as pd
# for plotting
import matplotlib.pyplot as plt

class Alpha_data():
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

def johnLorentzian(params,x,data): #p[0]==center , p[1] == width, p[2] == scalingfactor, p[3] == constant offset
        PI = 3.14159265358979323846264338327950288419716
        # LZ one
        center1 = params['center1']
        width1 = params['width1']
        scale1 = params['scaling1']
        const1 = params['const1']
        
        model = const1 + scale1/PI*(.5 *width1)/((x-center1)**2+(.5*width1)**2)
        
        return model - data

def print_all_fits(x, y, p, place_to_save):
    '''
    Saves x, y and the fit in place to save
    '''
    
    x_plt = []
    y_plt = []
    # restrict the plotting range
    '''
    *********************
    This is terrible code
    *********************
    '''
    for xx,yy in zip(x,y):
        if abs(xx - p['center1']) <= p['width1']*5:
            x_plt.append(xx)
            y_plt.append(yy)
    
    x_vs_y = plt.plot(x_plt,y_plt,'rx', label ="spectra")
    # make fit have more points than the data
    '''
    *********************
    This is terrible code
    *********************
    '''
    num_pts = 200
    fff = np.arange(min(x_plt),max(x_plt),(max(x_plt)-min(x_plt))/num_pts).tolist()
    ggg = [johnLorentzian(p,xx,0) for xx in fff]
    fit_plt = plt.plot(fff,ggg,color = 'blue', label = 'fit')
    
    plt.xlabel("Wavelength")
    plt.ylabel("Peak pos")
    plt.legend()
    plt.title(place_to_save.split('\\')[-1])
    
    plt.savefig(place_to_save)
    plt.clf()
    
    return 0

def fit_one_Voigt(y_spectra, x_axis,x_min,x_max,print_all_fits,place_to_save):
    ''' 
    Return a results object of the parameter values from fitting one 
    Voigt 103 peak, or whatever you put as the peak guess
    the width fit is ~2x the width guess
    if print all fits is true it prints a fit whwere it's told to
    if no it just returns results object
    '''
    
    # from 
    # http://lmfit.github.io/lmfit-py/builtin_models.html#example-1-fit-peaked-data-to-gaussian-lorentzian-and-voigt-profiles
    # and 
    # http://cars9.uchicago.edu/software/python/lmfit_MinimizerResult/builtin_models.html
    
    import numpy as np
    from lmfit.models import VoigtModel, ConstantModel
    from lmfit import Parameters
    
    #this is to supress consol output later on
    import contextlib2
    
    # Restrict the fit
    x_fit = []
    y_fit = []
    
    for x,y in zip(x_axis, y_spectra):
        if x_min < x < x_max:
            x_fit.append(float(x))
            y_fit.append(float(y))
    
    x_fit = np.asarray(x_fit)
    y_fit = np.asarray(y_fit)
    
    #this is the money
    peak = VoigtModel(independent_vars=['x'],nan_policy='raise')
    offset = ConstantModel(independent_vars=['x'],nan_policy='raise')
    
    mod = peak + offset
    # hopefully this supresses consol output
    with contextlib2.redirect_stdout(None):
        # guess parameters
        pars = Parameters()
        pars.add('amplitude', value = np.max(y_fit), min=0,max = np.max(y_fit)*1.5,vary=True)
        # pars.add('center', value = x_fit[np.argmax(y_fit)], min = .99*x_fit[np.argmax(y_fit)], max = 1.01*x_fit[np.argmax(y_fit)],vary=True)        
        pars.add('center', value = 103, min = 102, max = 104.2,vary=True)
        pars.add('fwhm', value = len(x_fit)/10, min = 0, max = len(x_fit),vary=True)
        #pars.add('c', value = x_fit[-1], min = 0, max = 5*x_fit[-1],vary=True)
    
        # pars = mod.make_params(amplitude=np.amax(y_fit), center=x_fit[np.argmax(y_fit)], fwhm=len(x_fit)/10, c=x_fit[-1])
        pars = mod.make_params(pars)
    
    # guess parameters
    pars = Parameters()
    pars.add('amplitude', value = np.max(y_fit), min=0,max = np.max(y_fit)*1.5,vary=True)
    # pars.add('center', value = x_fit[np.argmax(y_fit)], min = .99*x_fit[np.argmax(y_fit)], max = 1.01*x_fit[np.argmax(y_fit)],vary=True)        
    pars.add('center', value = 103, min = 102, max = 104.2,vary=True)
    pars.add('fwhm', value = len(x_fit)/10, min = 0, max = len(x_fit),vary=True)
    #pars.add('c', value = x_fit[-1], min = 0, max = 5*x_fit[-1],vary=True)

    # pars = mod.make_params(amplitude=np.amax(y_fit), center=x_fit[np.argmax(y_fit)], fwhm=len(x_fit)/10, c=x_fit[-1])
    pars = mod.make_params(pars)
    
    
    
    result = mod.fit(y_fit, params = pars, x=x_fit,verbose=False)
    
    if print_all_fits: 
        #x_h_res = np.arange(x_min,x_max, (x_max-x_min)/(10.0*len(x_fit)))
        #print result
        #y_h_res = mod.eval(x = x_h_res,params= result.best_fit())
        
        #for z,b in zip(x_h_res,y_h_res):
        #    print z,b
        result.plot_fit(xlabel='Inv Cm', ylabel='counts',datafmt = 'xb', numpoints=len(x_fit)*10)
        
        plt.savefig(place_to_save)
        plt.clf()
    
    
    
    
    if result.best_values['amplitude'] < 0:
        print "bad fit"
    
    return result










def fit_one_lz(y_spectra, x_axis, p_guess, w_guess):
    ''' 
    Return a dictionary of the parameter values from fitting one 
    LZ near the 103 peak, or whatever you put as the peak guess
    the width fit is ~2x the width guess
    '''
    
    from lmfit import minimize, Minimizer, Parameters, report_fit
    
    if p_guess > max(x_axis) or p_guess < min(x_axis):
        print "p_guess is bad"
        p_guess = 103
    
    # Start of fit
    xfit = []
    yfit = []
    
    for x,y in zip(x_axis, y_spectra):
        if abs(x-p_guess) < 5*w_guess:
            xfit.append(x)
            yfit.append(y)
    
    # Construct guess parameters  
    height_guess = max(yfit)

    p_guess = xfit[yfit.index(height_guess)]

    # just average the last 10 points, no peak out there
    holder = y_spectra[-10:-1]
    offset_guess = sum(holder)/len(holder)          # just a guess for the constant offset
    
    # now put into params dictionary
    params = Parameters()
    params.add('center1', value = p_guess, min=p_guess*.95,max = p_guess*1.05,vary=True)
    params.add('width1', value = w_guess, min = 0, max = w_guess*6,vary=True)
    params.add('scaling1', value = 1*height_guess, min = 0, max = 1.5*height_guess,vary=True)
    params.add('const1', value = offset_guess, min = 0, max = 2*offset_guess,vary=True)
    
    # Work horse
    minner = Minimizer(johnLorentzian, params, fcn_args=(xfit, yfit))
    result = minner.minimize(method='least_squares')
    
    returnDict = {}

    for x in sorted(result.params):
        returnDict[x] = result.params[x].value
    
    return returnDict


    























