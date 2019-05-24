# Anharmonic-Phonon-Coupling-in-Single-Crystal-Semiconducting-and-Metal-Like-van-der-Waals-In2Se3
This code was used to fit Raman data that varied with Temperature and Power

First off, this code is terrible, it runs but smells terrible.
There are almost no comments, may god have mercy on my soul. 
I apologize to whomever tries to use or read it, if you have trouble, contact me and I'll be willing to help. 

Dependencies 
os			https://docs.python.org/3/library/os.html
glob		https://docs.python.org/2/library/glob.html
numpy		https://www.numpy.org/
pandas		https://pandas.pydata.org/
scipy		https://www.scipy.org/
lmfit 		https://lmfit.github.io/lmfit-py/
peakutils	https://pypi.org/project/PeakUtils/

Files 	
JohnFunctions.py
	a set of helper functions called by other code to keep everything pretty
Beta_Functions
	a set of helper functions used to fit Beta data
Beta_class
	class used to store Beta data... Because I didn't understand OOP at all when I wrote this. 
Beta_Main_Peak_First.py
	Used to fit two pseudo Voigt profiles to BEta data, done this way so it'd be easier to debug. 
Alpha_Class
	Read data, make dataframe... I didn't understand OOP
	
Alpha_Main_2.py
	Fits one Lorentzian to the alpha peak at ~104 inv cm
	Saves fit parameters for later
	
Beta_main_3.py
	Fits two Pseudo-Voigt to the two peaks in Beta's spectrum, for as long as it can
	Saves fit parameters and makes a nice plot

	
	


	



