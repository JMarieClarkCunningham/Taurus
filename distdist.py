import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import matplotlib.patches as mpatches
from PyAstronomy import pyasl
from matplotlib import colors
from astropy import units as u
from astropy import constants as const
from astropy.coordinates import SkyCoord

###  'distdist.py' generates a distance distribution histogram for the targets in the  ###
###  overlap between Fleming et al 2019 and Rebull et al. 2004, i.e. finding the stellr###
###  rotation of the stars in the Taurus cluster, to determine if they are indeed part ###
###  of two two main associations centred at 130.6 ± 0.7 pc and 160.2 ± 0.9            ###

'''
#Cluster    Star              SpTyp      Radius    vsini      Period   Gaia DR2 ##################   SSTtau ID           R.A.      Dec.      Parall    Distance   PMRA     PMDec    µ
#-[0]-------[1]---------------[2]--------[3]-------[4]--------[5]-----------------[6]----------------[7]-----------------[8]-------[9]-------[10]------[11]-------[12]-----[13]----[14]-
Tau-Aur     AATau            K7-M0      1.5       11.4       8.22     Gaia DR2 147818450613367424   043455.4 + 242853   68.7309   24.4813   7.29      137.20     3.48    −20.99   21.27   
Tau-Aur     CITau            K7         1.6       10.4       np.nan   Gaia DR2 145203159127518336   043352.0 + 225030   68.4668   22.8416   6.30      158.71     8.90    −17.07   19.25   
Tau-Aur     CX Tau           M2         1.8       18.2       np.nan   Gaia DR2 162758236656524416   041447.8 + 264811   63.6995   26.8030   7.82      127.93     9.02    −22.45   24.20
Tau-Aur     CYTau            M1         1.5       10.0       7.50     Gaia DR2 164698634160139264   041733.7 + 282046   64.3906   28.3462   7.76      128.88     9.18    −25.55   27.15     
Tau-Aur     DETau            M1         2.2       10.0       7.60     Gaia DR2 152511475478780416   042155.6 + 275506   65.4819   27.9183   7.85      127.37     10.75   −27.22   29.27   
Tau-Aur     DGTau            M0         2.3       21.7       6.30     Gaia DR2 151262700852297728   042704.6 + 260616   66.7696   26.1044   8.25      121.18     6.16    −19.30   20.26   
Tau-Aur     DITau            M0         2.1       10.5       7.70     Gaia DR2 151374198202645376   042942.4 + 263249   67.4270   26.5469   7.40      135.12     6.90    −21.21   22.30   
Tau-Aur     DLTau            M1         1.4       16.0       9.40     Gaia DR2 148010281032823552   043339.0 + 252038   68.4129   25.3438   6.28      159.34     9.33    −18.29   20.53   
Tau-Aur     DNTau            M0         2.0        8.1       6.30     Gaia DR2 147606657186323712   043527.3 + 241458   68.8641   24.2496   7.80      128.22     6.05    −20.77   21.64   
Tau-Aur     DOTau            K7         2.2       11.0       np.nan   Gaia DR2 148449913884294528   043828.5 + 261049   69.6191   26.1803   7.17      139.38     6.13    −21.34   22.20   
Tau-Aur     IPTau            M0         1.5 <=    11.0       3.25     Gaia DR2 152226491513195648   042457.0 + 271156   66.2379   27.1989   7.66      130.57     8.36    −26.75   28.02   
Tau-Aur     V1070Tau         K7         2.1       12.9       5.66     Gaia DR2 164422961683000320   041941.2 + 274948   64.9220   27.8299   7.91      126.37     9.94    −24.96   26.87   
Tau-Aur     V1115Tau         M0         1.5       21.9       3.35     Gaia DR2 148037764527442944   043619.0 + 254258   69.0796   25.7163   7.81      128.02     8.91    −27.47   28.88   
'''

###  CITau [Gaia DR2 145203159127518336], CX Tau [Gaia DR2 162758236656524416], and DOTau [Gaia DR2 148449913884294528] are omitted from the input list because rotation period  ###
###  analysis (Rebull et al 2004) did not yield results for this target. And numpy don't like no stinkin' nans when it makes a histogram. ###
###  Without these two targets, we have a total of ten members of the Taurus system with distances (Fleming et al 2019) and rotation per. ###


names =     ['AATau','CYTau','DETau','DGTau','DITau','DLTau','DNTau','IPTau','V1070Tau','V1115Tau']        #(Rebull et al. 2004)
spectrals = ['K7','M1','M1','M0','M0','M1','M0','K7','M0','K7','M0']
radiuss =   [1.5,1.5,2.2,2.3,2.1,1.4,2.0,1.5,2.1,1.5] * u.Rsun
periods =   [8.22,7.50,7.60,6.30,7.70,9.40,6.30,3.25,5.66,3.35]                                            #(Rebull et al. 2004)
RAs =       [68.7309,64.3906,65.4819,66.7696,67.4270,68.4129,68.8641,66.2379,64.9220,69.0796]              #(Fleming et al. 2019)
DECs =      [24.4813,28.3462,27.9183,26.1044,26.5469,25.3438,24.2496,27.1989,27.8299,25.7163] * u.degree   #(Fleming et al. 2019)
paralaxs =  [7.29,7.76,7.85,8.25,7.40,6.28,7.80,7.66,7.91,7.81]                                            #(Fleming et al. 2019)
distances = [137.20,128.88,127.37,121.18,135.12,159.34,128.22,130.57,126.37,128.02]                        #(Fleming et al. 2019)
PMRAs =     [3.48,9.18,10.7,6.16,6.90,9.33,6.05,8.36,9.94,8.91]                                            #(Fleming et al. 2019)
#PMDECs =[−20.99,−17.07,−25.55,−27.22,−19.30,−21.21,−18.29,−20.77,−21.34,−26.75,−24.96,−27.47]

###  Make a bunch of empty lists used for calculation in the following analysis
cols=[]
near=[]
far=[]
nperiod=[]

print('These are the systems that are close (d = 110 - 150 pcs):')
print('Name', 'Spectral', 'Radius', 'Period', 'RA', 'DEC', 'Paralax', 'Distance')
for name, spectral, radius, period, RA, DEC, paralax, distance in zip(
    names, spectrals, radiuss, periods, RAs, DECs, paralaxs, distances):
    if distance < 150:
        print(name, spectral, radius, period, RA, DEC, paralax, distance)
        near.append(distance)
        nperiod.append(period)
        nrotperiod_mean = np.mean(nperiod)
print('The average of their rotational periods is', nrotperiod_mean)
print('------------------------------------------------------------------------')
print('These are the systems that are far (d = 150 - 180 pcs)')
print('Name', 'Spectral', 'Radius', 'Period', 'RA', 'DEC', 'Paralax', 'Distance')
for name, spectral, radius, period, RA, DEC, paralax, distance in zip(
    names, spectrals, radiuss, periods, RAs, DECs, paralaxs, distances):
    if distance > 150:
        print(name, spectral, radius, period, RA, DEC, paralax, distance)
        far.append(distance)

### Function to map the colors as a list from the input distances 
### red points: d = 110 - 150 pcs
### blu points: d = 150 - 180 pcs

for idx in enumerate (zip(distances)):
    if distance < 150:          ### Nearer Systems 
        cols.append('red')
    elif distance > 150:        ### Farer Systems
        cols.append('blue')

############################ Period Distribution Histogram ###############################
### Global Plot Properties
fig, axs = plt.subplots(1, 2, sharey = True, tight_layout = True)
mid = 150
N_points = 10

### Period Distribution (Left Panel)
perin_bins = 11
periind = [10, 10, 10, 8.84]
perivals = [0,0,0,1]
pericolors = ['b','b','b','b']
perihist = axs[0].hist(periods, perin_bins, color = 'red')
axs[0].set_ylabel ('Frequency', size = 12)
axs[0].set_xlabel ('Period [days]', size = 12)
axs[0].locator_params(nbins=10, axis='x')      #Increase number of tics on period (x) axis 
#axs[0].set_xticklabels(tick_labels.astype(int))#Make period (x) axis tic marks integers
axs[0].set_xlabel ('Period [days]', size = 12)
peribar = axs[0].bar(periind, perivals, width=0.7, align='edge', color=pericolors) 

### Distance Distribution (Right Panel)
distn_bins = 10
distind = [130,130,130,155.5]
distvals = [0,0,0,1]
distcolors = ['r','r','r','b']
axs[1].hist(distances, distn_bins, color = 'red') 
axs[1].set_xlabel ('Distance [pcs]', size = 12)
distbar = axs[1].bar(distind, distvals, width=4, align='edge', color=distcolors)

### Color Key
far_patch = mpatches.Patch(color='blue', label='150 - 180 pc')
near_patch = mpatches.Patch(color='red', label='110 - 150 pc')

plt.legend(handles=[near_patch, far_patch])
plt.show()
fig.savefig('DistRotDist.jpeg')

############################ (RA, DEC, distance) Polar Plot  #############################
RADECdist = plt.figure(figsize=(5, 10))

'''for RA, DEC, distance in zip(RAs, DECs, distances):
    plt.polar((RA,), (DEC,), distance)
    plt.text(x, y, '%d, %d' % (int(x), int(y)),
             transform=trans_offset,
             horizontalalignment='center',
             verticalalignment='bottom')

#plt.show()'''