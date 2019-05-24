# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 09:57:41 2018

@author: John

'Same' as Beta_main_2, ****BUT ****
1 Uses only the smarter fitting (main, second, main) and 1 Voigt fitting 

Compares the results and decides which is better for each layer/temp combination

*** Only *** saves awpa 1 and peak 1

"""

file_location = r'C:\Users\John\Desktop\mmm'
file_location = r'C:\Users\John\Desktop\data'
#file_location = r'C:\Users\John\Desktop\Raman(Power,Temp) data\Beta_fixed_power_minus_bkg'
                                             
# AWPA, and main -> second -> main fitting
place_to_save_FWHM = r'C:\Users\John\Desktop\FWHM.txt'
#place_to_save_peak = r'C:\Users\John\Desktop\Mar 30\AAAAA_JOHNCLICKME_Beta_fixed_temp_peak.txt'
place_to_save_peak = r'C:\Users\John\Desktop\Peak.txt'

# Do you want to print all foto's
print_all_fits = True
#place_to_dump_foto = r'C:\Users\John\Desktop\Mar 30\dataSet'
place_to_dump_foto = r'C:\Users\John\Desktop\bbbb'

# Fitting region
first = 80
last = 150

x_units = 'wavelength'
y_units = 'counts'

# The flag in the file name preceding the temperature
temp_flags = ["temp", 'TEMP', 'Temp',"Temperature"]
# The flag in the file name preceding the power
pwr_flags = ["pwr", "PWR", "Pwr","Power","power"]

import Beta_class 
import pandas as pd
import Beta_functions

from sets import Set

'''
*****************SOP********************
'''


# Now I have a DataFrame
data_df = Beta_class.Beta_data(file_location, x_units, y_units)

# make peak data frame

# makes a list of all temps that appear
tmp_lst = Beta_class.get_temps(list(data_df.data),temp_flags)
# makes a list of all powers that apper
pwr_lst = Beta_class.get_pwrs(list(data_df.data),pwr_flags)
# makes a list of all names taht appear
name_lst = Beta_class.get_temp_pwr_other(list(data_df.data),temp_flags,pwr_flags)

# makes list of tupels for multi index later on
pwr_set = Set()
name_set = Set()
for tpn in zip(Beta_class.get_temp_pwr_other(list(data_df.data),temp_flags,pwr_flags).values()):
    # this loop is over all pulled out by ^^
    # [temp, power, other]
    # append a tuple that'll be the index
    try:
        pwr_set.add(Beta_class.get_num(tpn[0][1]))
        name_set.add(tpn[0][2])
    except:
        print tpn
        
# Make dataframes for the resulting data
# these are basically tables
# initialized the tables 
peak_df = pd.DataFrame(columns = sorted(tmp_lst), index = pd.MultiIndex.from_product([pwr_set,name_set], names=['Power','name']))

fwhm_df = pd.DataFrame(columns = sorted(tmp_lst), index = pd.MultiIndex.from_product([pwr_set,name_set], names=['Power','name']))


'''
******************
Start of loop
******************
'''
max_lst = []

for x in list(data_df.data):
    
    # Get the name for the indexing of the rest of the program
    # if doesn't work, just skip to next loop 
    try:
        # is a dict with three elements
        # filename: [temp,power,other]
        names = Beta_class.get_temp_pwr_other([x],temp_flags,pwr_flags)
        # Find the temp
        tmp = Beta_class.get_num(names[x][0])
        # Find the power
        pwr = Beta_class.get_num(names[x][1])
        # Everything else
        layer = names[x][2]
        #print x
        #print names
        #print layer, tmp, pwr
    except:
        print x, "Something went wrong in name extraction"
        continue
    
    # The name is g0000000000000000000000000000d!
    if print_all_fits:
        # Do both fits
        # returns dictionary of the paramaters
        # oneamplutude,twoamplitude...
        # "Winner' : 'One' if one Voigt + const fits better
        # 'Winner' : "Two" it two Voigt + const fits better
        # Print all the fits
        place_to_save = "".join([place_to_dump_foto,"\\", "_".join(["Temp",str(tmp),"Pwr", str(pwr),str(layer),"Beta"])])
        winner_dict = Beta_functions.compare_two_fits(data_df.data.loc[:,x_units],data_df.data.loc[:,x], first,last,print_all_fits, place_to_save, tmp)
        max_lst.append(max(data_df.data.loc[:,x]))
    else:
        # Do both fits
        # returns dictionary of the paramaters
        # oneamplutude,twoamplitude...
        # "Winner' : 'One' if one Voigt + const fits better
        # 'Winner' : "Two" it two Voigt + const fits better
        # Print none of the fits
        winner_dict = Beta_functions.compare_two_fits(data_df.data.loc[:,x_units],data_df.data.loc[:,x], first,last,print_all_fits, place_to_save,tmp)
    
    # Now Winner is what we need, so put Peak 1 and FWHM 1 into dataframe
    
    if winner_dict['Winner'] == 'One':
        peak_df.loc[(pwr,layer),tmp] = winner_dict['onecenter']
        fwhm_df.loc[(pwr,layer),tmp] = 3.6013*winner_dict['onesigma']
                        
    else:
        peak_df.loc[(pwr,layer),tmp] = winner_dict['onecenter']*winner_dict['oneamplitude']/(winner_dict['oneamplitude']+winner_dict['twoamplitude']) + \
                                winner_dict['twocenter']*winner_dict['twoamplitude']/(winner_dict['oneamplitude']+winner_dict['twoamplitude'])
        # new way April 4
        fwhm_df.loc[(pwr,layer),tmp] = 3.6013 * \
                    winner_dict['onesigma'] + winner_dict['twosigma'] * winner_dict['twoamplitude']/winner_dict['oneamplitude']
        # old way Before April 4
        '''
        fwhm_df.loc[(pwr,layer),tmp] = 3.6013 * \
                                winner_dict['onesigma']*winner_dict['oneamplitude']/(winner_dict['oneamplitude']+winner_dict['twoamplitude']) + \
                                winner_dict['twosigma']*winner_dict['twoamplitude']/(winner_dict['oneamplitude']+winner_dict['twoamplitude'])          
        '''
    print list(data_df.data).index(x) , " of " , len(list(data_df.data)), \
                  '\t', x, '\t', winner_dict['Winner']
    
avg = 0.0
for x in max_lst:
    avg += x
print avg/len(max_lst)
               
peak_df.to_csv(place_to_save_peak, sep='\t')
fwhm_df.to_csv(place_to_save_FWHM, sep = '\t')

        
    







































