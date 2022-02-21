import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
from scipy import interpolate
from astropy.io import ascii
from astropy.table import Table

#8/24/18
#Analyse mode output from GYRE

#Some Plotting Stuff
mpl.rcParams['xtick.major.pad'] = 8
mpl.rcParams['ytick.major.pad'] = 8

mpl.rcParams['axes.linewidth']=3
mpl.rcParams['lines.markeredgewidth'] = 0.5

mpl.rcParams['xtick.major.pad'] = 12
mpl.rcParams['ytick.major.pad'] = 10

plt.rcParams['font.family'] = 'stixgeneral'

plt.rcParams['mathtext.fontset'] = 'stix'

############
#Parameters#
############

#Constants
gravConst = 6.67259e-8
solRad = 6.963e10
solLum = 3.9e33
solMass = 1.99e33
sectoyear = 3.16888e-8
c = 2.99792458e10 #speed of light
a = 7.5657e-15 #Radiative transfer const

#Model Specific
#Open and read the file
with open('hse_profile_mc01_g135_gad135_R1_orig.dat.fgong','r') as f:
    lines=f.readlines()

profile = list(map(lambda x: x.replace('-', ' -'),lines[8:]))
profile = list(map(lambda x: x.replace('E -','E-'),profile))
profile = list(map(lambda x: x.split(),profile ))
header = lines[5].split()
header = np.array(header)
header = header.astype(float)

m0 = header[0]
r0 = header[1]


profile = np.array(profile)
profile = profile.astype(float)

profile1 = profile[::8]
profile2 = profile[1::8]
profile3 = profile[2::8]
profile = np.append(profile1,profile2,axis=1)
profile = np.append(profile,profile3,axis=1)

#Organize profile data
radiusCol = profile[:,0] #radius coordinate of cell (Rsun)
qCol = 10.**profile[:,1] #fraction of star mass interior
rhoCol = profile[:,4] # in cgs (need to check units)
pressureCol = profile[:,3] #pressure (cgs)

freqList = []
lList = []
CnlList = []
QList = []
MMassList = []

q = 0.1

for l in np.arange(2,11,1):
    m=0

    lList.append(l)
    with open(r'fmodel'+str(l)+'.txt','r') as f:
        lines=f.readlines()

    attributes = lines[3].split()
    freq = float(attributes[2])
    freqIm = float(attributes[3])
    npg = float(attributes[1])
    print( 'omega = ' + str(freq) )
    
    freqList.append(freq)
    
    eigenInfo = list(map(lambda x: x.split(),lines[6:]))
    x = np.array(eigenInfo)
    eigenInfo = x.astype(float)
        
    r = eigenInfo[:,0]
    xir = eigenInfo[:,1]  * np.sqrt(4*np.pi)
    xirIm= eigenInfo[:,2] * np.sqrt(4*np.pi)
    xih= eigenInfo[:,3]   * np.sqrt(4*np.pi)
    xihIm= eigenInfo[:,4] * np.sqrt(4*np.pi)
    rho = eigenInfo[:,5]

    integrand = l*(r**(l+1))*rho*(xir+(l+1)*xih)
    integrand2 = r**2*rho*(xir**2+xirIm**2 + l*(l+1)*(xih**2+xihIm**2))
    integrand_C = rho*r**2*(2*xir*xih+xih**2)
        
    tck = interpolate.splrep(r,integrand,s=0)
    tck2 = interpolate.splrep(r,integrand2,s=0)
    tckC = interpolate.splrep(r,integrand_C,s=0)
    #yint = interpolate.splint(0,1., tck)
    #norm = interpolate.splint(0,1., tck2)
    #Cnl = interpolate.splint(0,1.,tckC)
    yint = interpolate.splint(r[0],r[-1], tck)
    norm = interpolate.splint(r[0],r[-1], tck2)
    Cnl = interpolate.splint(r[0],r[-1],tckC)

    N = 1/math.sqrt(norm)
    MMass = norm/(xir[-1]**2)
    Cnl = Cnl*N**2.
    CnlList.append(Cnl)
    MMassList.append(MMass)
    QList.append(N*yint)
    print('l: '+str(l))
    print('Qcalc:' + str(N*yint))
    print('Mode Mass: '+str(MMass))
    print('Cnl: '+str(Cnl))


    # write the Edist file
    ascii.write(Table([r,xir,xih,rho,integrand2*N*N],names=["r","xir","xih","rho","dE_dr(normalized)"]),
                output='fmodel'+str(l)+'_Edist.txt',overwrite=True)

    
    #plt.figure(2,figsize=(5,5))
    plt.plot(r,xir/xir[-1],label = '$l=$ '+str(l),linewidth=3)

plt.xlim(0.,1.0)
plt.tick_params(labelsize=10,length=10, width = 3)
plt.tick_params(which = 'minor', length=7, width = 2)
plt.xlabel('$r/R_*$',fontsize=15)
plt.ylabel(r'$\xi_r/\xi_r[R]$',fontsize=15)
plt.legend(loc='best',fontsize=15)
plt.tight_layout()
plt.show()
    

## Now write the modeproperties file
ascii.write(Table([[2,3,4,5,6,7,8,9,10],freqList,QList,CnlList,MMassList],
            names=['l', 'omega','Q','C','MMass']),
            output='ModeProperties.txt',overwrite=True)
