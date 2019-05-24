# -*- coding: utf-8 -*-
"""
Created on Fri Mar 09 10:52:25 2018
@author: John

There is a class Alpha_Class
    .data returns a DataFrame with all the data
    
The rest of the probgram runs off this dataframe

This fits the spectra and writes the parameters in various ways

"""

file_location = r'C:\Users\John\Desktop\ReHe\159_CrI3 bulk_633 nm_1% power_hole50_180s x2_-150~1400cm-1_no pol_10K.txt'
place_to_save_FWHM = r'C:\Users\John\Desktop\FWHM_fixed_temp.txt'
place_to_save_peak = r'C:\Users\John\Desktop\Peak_fixed_temp.txt'
place_to_dense = r'C:\Users\John\Desktop\dense.txt'
x_units = 'wavelength'
y_units = 'counts'

# Do you want to print all foto's
print_all_fits = True
place_to_dump_foto = r'C:\Users\John\Desktop\cccc'

# where you want the fit to happen
x_max = 110
x_min = 100

# The flag in the file name preceding the temperature
temp_flags = ["temp", 'TEMP', 'Temp',"Temperature"]
# The flag in the file name preceding the power
pwr_flags = ["pwr", "PWR", "Pwr","Power","power"]

'''
Toolboxes
'''

import Alpha_class 

import pandas as pd
import numpy as np
from lmfit.models import VoigtModel, ConstantModel
from lmfit import Parameters

import lmfit

'''
Start of code
'''

# Now I have a DataFrame
data_df = Alpha_class.Alpha_data(file_location, x_units, y_units)

# makes a list of all temps that appear
tmp_lst = Alpha_class.get_temps(list(data_df.data),temp_flags)
# makes a list of all powers that apper
pwr_lst = Alpha_class.get_pwrs(list(data_df.data),pwr_flags)

# Make dataframes for the resulting data
# these are basically tables
peak_df = pd.DataFrame(columns=sorted(tmp_lst),index=sorted(pwr_lst))
fwhm_df = pd.DataFrame(columns=sorted(tmp_lst),index=sorted(pwr_lst))

peak_df['names'] = [0]*len(pwr_lst)
fwhm_df['names'] = [0]*len(pwr_lst)


# Gets full picture of spectra for yi
dense_function_df = pd.DataFrame()


for x in list(data_df.data):
    print x
    #p = Alpha_class.fit_one_lz(data_df.data.loc[:,x],data_df.data.loc[:,x_units],103,5)
    
    try:
        names = Alpha_class.get_temp_pwr_other([x],temp_flags,pwr_flags)
        
        # Find the temp
        tmp = Alpha_class.get_num(names[x][0])
        # Find the power
        pwr = Alpha_class.get_num(names[x][1])
        # Everything else
        layer = names[x][2]
        
        if print_all_fits:
            place_to_save = "".join([place_to_dump_foto,"\\", "_".join(["Temp",str(tmp),"Pwr", str(pwr),str(layer),"alpha"])])
            V_p = Alpha_class.fit_one_Voigt(data_df.data.loc[:,x],data_df.data.loc[:,x_units],x_min,x_max,True, place_to_save)
            
        else:
            V_p = Alpha_class.fit_one_Voigt(data_df.data.loc[:,x],data_df.data.loc[:,x_units],x_min,x_max,True, place_to_save)
        
        
        peak_df.loc[pwr,tmp] =  V_p.params['center']
        fwhm_df.loc[pwr,tmp] = V_p.params['fwhm']
        
        print 'FWHM', V_p.params['fwhm']
        print 'peak', V_p.params['center']
                   
        peak_df.loc[pwr,'names'] =  layer
        fwhm_df.loc[pwr,'names'] = layer
        
        
        # Mar 7
        if True:
            # this is to print out the results of the fit in high deff
    
            bb = np.arange(x_min, x_max, (x_max-x_min)/2000.0).tolist()
            #for x in V_p.best_values:
            #    print V_p.best_values[x]
        
            dense_function_df[str(pwr)+"uW Dense"] = [V_p.eval(params=V_p.params,x = f) for f in bb]
        

        #print pwr
        #print lmfit.report_fit(V_p.params)
        
                   
                   
        # print all data??? 
        #if print_all_fits: 
        #    layer = "1"
            
            #Alpha_class.print_all_fits(data_df.data.loc[:,x_units],data_df.data.loc[:,x],p,place_to_save)
            #data_df.data.loc[:,x],data_df.data.loc[:,x_units]
        
    except IndexError:
        print "xaxix not included"
    except ValueError:
        print x, " can't be passed to 'get num'"
    
peak_df.index.name = "Power"
fwhm_df.index.name = "Power"
print list(peak_df)
    
peak_df.to_csv(place_to_save_peak, sep=',')    
fwhm_df.to_csv(place_to_save_FWHM, sep=',')  

dense_function_df['xdense'] = bb
dense_function_df.set_index('xdense', inplace = True)
dense_function_df.to_csv(place_to_dense, sep =',') 
    
        

