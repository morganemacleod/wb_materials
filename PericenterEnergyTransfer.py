import sys
import numpy as np
import math
from scipy import interpolate
import numpy.fft as fft
from astropy.table import Table
from astropy.io import ascii
from scipy.special import sph_harm
import matplotlib.pyplot as plt

class LinearTheory:
    def __init__(self,files_directory):
        mp = ascii.read(files_directory+"ModeProperties.txt")
        self.l_Array = np.array(mp['l'])
        self.omega_Array = np.array(mp['omega'])
        self.Q_Array = np.array(mp['Q'])
        self.C_Array = np.array(mp['C'])
        self.MMass_Array = np.array(mp['MMass'])
        self.eigenfuncs = {}
        for l in range(2,11):
            self.eigenfuncs[str(int(l))] = ascii.read(files_directory+'fmodel'+str(int(l))+'_Edist.txt')
        
        
    def FindOrbit(self,ecc):
        '''Calculate the orbtial separatation (D) and true anomaly (Phi) evenly spaced in time in 
        preparation for calculating the Hansen coefficients, F_Nm, as a Fourier Transform
        Input: ecc 
        Output: D_Array (orbital separation scalled to the semi-major axis), Phi_Array (orbital true anomaly),
             and Nmax (related to the sampling rate of D_Array and Phi_Array)
        '''

        Nmax = 2.*int(np.ceil(1./(1-ecc)*np.log(0.001)/np.log(ecc))+2) # Maximum necessary N (number of samples) for accuracy
        Nmax = min(Nmax,100000.) #Makes sure that very high eccentricity (e>~ 0.95) doesn't take forever. 
        EList = np.arange(-np.pi,np.pi,2.*np.pi/(2.*Nmax)) #Eccentric anomaly
        MList = EList - ecc*np.sin(EList) #Mean Anomaly
        PhiList = np.sign(EList)*np.arccos((np.cos(EList)-ecc)/(1.-ecc*np.cos(EList))) #True Anomaly
        DList = (1.-ecc*np.cos(EList)) #Binary Separation scaled to semimajor axis

        #Evenly sample True Anomaly and Binary Separation in Mean Anomaly (evenly spaced in time)
        tckD = interpolate.splrep(MList,DList,s=0.)
        tckPhi = interpolate.splrep(MList,PhiList,s=0.)

        MListNew = np.arange(-np.pi,np.pi,2.*np.pi/(2.*Nmax))

        D_Array = interpolate.splev(MListNew,tckD)
        Phi_Array = interpolate.splev(MListNew,tckPhi)

        return D_Array, Phi_Array, Nmax

    def Hansen(self,D_Array,Phi_Array,Nmax, l, m):
        ''' Calculate the Hansen Coefficients, F_Nm for a given orbital eccentricity
            and azimuthal mode number m.
            Inputs: D_Array, Phi_Array, Nmax ,l, m
            Output: a list of positive integers N, a list of Hansen Coefficients F_Nm.
        '''
        FFT = fft.fft(1./(D_Array)**(l+1)*np.exp(-1.j*m*Phi_Array))/Nmax/2.

        #Re-orering output from the FFT, leave out n=0 term. It is always 0.
        NList = np.arange(1,Nmax/2.+1) 
        #These are defined in equation 6 of Vick & Lai 2020, but are fastest to calculate using FFT
        FNmList = list(map(lambda x: abs(x), FFT[-1:-np.int(Nmax/2.+1):-1]))
        return NList, FNmList

    def FindW_lm(self,l,m):
        ''' Calculate the numerical coefficient W_lm
            Inputs: integer l, m
            Output: float W_lm
        '''
        return (-1.)**((l+m)/2.)*(4.*np.pi/(2.*l+1.)*math.factorial(l-m)*math.factorial(l+m))**0.5 \
                /(2**l*math.factorial((l-m)/2.)*math.factorial((l+m)/2.))

    def FindK_lm(self,q,r_p, ecc, omega_S, D_Array, Phi_Array, Nmax, l, m):
        '''
            Calculate the integral K_lm efficiently by taking advantage of Hansen Coefficients
            Inputs:(mass ratio, pericenter distance (R*), eccentricity, stellar spin rate (GM_*/R_*^3)^(1/2),
                    mode number l, mode number m)
            Output: K_lm 
        '''
        #Calculate a few intermediate quantities
        W_lm = self.FindW_lm(l,m) #numerical coefficient
        Omega = ((1.+q)/r_p**3.*(1.-ecc)**3.)**(0.5) #Caluclate the orbital frequency (G=M*=R*=1)
        P = 2.*np.pi/Omega #Orbital period (G=M*=R*=1)

        #Pick out the relevant mode properties
        omega = self.omega_Array[l-2] #mode frequency in absence of rotation (G=M*=R*=1)
        C = self.C_Array[l-2] #1st order frequency correction for rotation (dimensionless)
        Q = self.Q_Array[l-2] #Overlap integral Q (dimensionless)
        sigma = omega+ m*(1.-C)*omega_S  #mode frequency in the inertial frame

        #Calculate relevant Hansen Coefficients

        N_Array, F_Nm_pos_Array = self.Hansen(D_Array, Phi_Array, Nmax, l, m) #Hansen coeffs for positive N
        N_Array, F_Nm_neg_Array = self.Hansen(D_Array, Phi_Array, Nmax, l, -m) #Hansen coeffs for negative N

        #Defined in equation 65 of Vick and Lai, but written in terms of the F_Nm
        K_lm = W_lm/np.pi*(1.-ecc)**(l+1.)*np.sin(sigma*P/2.)*np.sum((-1.)**N_Array/(sigma - N_Array*Omega)*F_Nm_pos_Array
                + (-1.)**N_Array/(sigma + N_Array*Omega)*F_Nm_neg_Array) #includes positive and negative N

        return K_lm

    def FindDeltaE_l(self,q,r_p,ecc,omega_S, D_Array, Phi_Array, Nmax,l):
        '''Return the energy transfer and angular momentum transfer at pericenter for a given
           orbit, stellar spin rate, and mode degree l.
           Inputs: (mass ratio, pericenter distance (R_*), eccentricity, spin rate [(GM*/R*^3)^(1/2)], l)
           Outputs: [Mode Energy, Mode Angular Momentum] (G=M_*=R_*=1) 
        '''

        #Pick out the right mode properties
        omega = self.omega_Array[l-2] #mode frequency in absence of rotation (G=M*=R*=1)
        eps = omega
        C = self.C_Array[l-2] #1st order frequency correction for rotation (dimensionless)
        Q = self.Q_Array[l-2] #Overlap integral Q (dimensionless)

        m_Array = np.arange(-1.*l, l+1, 2) #Array of possible azimuthal mode numbers
        sigma_Array = omega+ m_Array*(1.-C)*omega_S  #mode frequency in the inertial frame 
        Klm_Array = np.array(list(map(lambda x: self.FindK_lm(q,r_p,ecc,omega_S,D_Array,Phi_Array,Nmax,l,x),m_Array)))

        #Equation 64 of Vick & Lai 2020
        DeltaE_lm_Array = 2.*np.pi**2.*q**2./r_p**(2*l+2)*(sigma_Array/eps)*Q**2.*Klm_Array**2.
        #Relationship between mode energy and mode angular momentum
        DeltaL_lm_Array = DeltaE_lm_Array*m_Array/sigma_Array

        return np.sum(DeltaE_lm_Array), np.sum(DeltaL_lm_Array)
        
    def FindDeltaE_lm(self,q,r_p,ecc,omega_S, D_Array, Phi_Array, Nmax,l,m):
        '''Return the energy transfer at pericenter for a given
           orbit, stellar spin rate, and mode numbers l, m.
           Inputs: (mass ratio, pericenter distance (R_*), eccentricity, spin rate [(GM*/R*^3)^(1/2)], l, m)
           Outputs: Mode Energy (G=M_*=R_*=1) 
        '''

        #Pick out the right mode properties
        omega = self.omega_Array[l-2] #mode frequency in absence of rotation (G=M*=R*=1)
        eps = omega
        C = self.C_Array[l-2] #1st order frequency correction for rotation (dimensionless)
        Q = self.Q_Array[l-2] #Overlap integral Q (dimensionless)

        sigma = omega+ m*(1.-C)*omega_S  #mode frequency in the inertial frame 
        Klm = self.FindK_lm(q,r_p,ecc,omega_S,D_Array,Phi_Array,Nmax,l,m)

        #Equation 64 of Vick & Lai 2020
        DeltaE_lm = 2.*np.pi**2.*q**2./r_p**(2*l+2)*(sigma/eps)*Q**2.*Klm**2.
        #Relationship between mode energy and mode angular momentum

        return DeltaE_lm
        
    def MakeDeltaE_lmArray(self,q,r_p,ecc,omega_S, D_Array, Phi_Array, Nmax, num_l):
        '''Return an array with E_lm for all modes. Rows correspond to different l, columns to different m
           forbidden combinations of m and l (e.g. m>l are set to 0)
        '''
        DeltaE_lm_Array = np.zeros((num_l-1,2*num_l+1))
        for l in self.l_Array[:num_l-1]:
            l=int(l)
            m_Array = np.arange(-1.*l, l+1, 2)
            for m in m_Array:
                m=int(m)
                DeltaE_lm_Array[l-2][num_l+m] = self.FindDeltaE_lm(q,r_p,ecc,omega_S,D_Array,Phi_Array,Nmax,l,m)
        return DeltaE_lm_Array

    def VectorComponents(self,l,m,theta,phi):
        '''Calculates the angular dependence on the lagrangian displacement vector.
         xi(r,theta,phi) = xi_r Y_lm e_r + xi_h r nabla Y_lm. This function gives the
         radial component Y_lm e_r and the polar and azimuthal components r nabla Y_lm. 
         inputs: l,m, polar angle theta, azimuthal angle phi
         outputs: an array of components, [radial, polar ,azimuthal]'''
        if theta == 0.:
            theta = 1.e-20 #Avoid division by zero

        r_comp = sph_harm(m,l,phi,theta) #radial component

        phi_comp = 1.j*m*sph_harm(m,l,phi,theta)/np.sin(theta) #azimuthal component

        #polar component
        if l-m >= 1: #Make sure m i<= l in derivative term
            theta_comp = m*np.cos(theta)/np.sin(theta)*sph_harm(m,l,phi,theta) + np.sqrt((l-m)*(l+m+1))*np.exp(-1.j*phi)*sph_harm(m+1,l,phi,theta)
        else:
            theta_comp = m*np.cos(theta)/np.sin(theta)*sph_harm(m,l,phi,theta)
        return np.array([r_comp,theta_comp,phi_comp])  

    def xi(self,r,theta,phi,DeltaE_lm_Array,t_p):
        '''Calculates the lagrangian displacement at time t_p after the pericenter passage as 
        a function of r, theta, and phi
        inputs: r, theta, phi, DeltaE_lm_Array (array of energy transferred for each mode l, m), time since
        pericenter passage t_p (R_*^3/G/M_*)(1/2)
        output:an array where each row corresponds to a degree l and the columns to r, theta, phi. i.e. summing
        over column 0 would give the total displacement in the radial direction.'''
        xiComp_Array = []
        num_l = int(len(DeltaE_lm_Array)+1)
        for l in self.l_Array[:num_l-1]: #Loop over l
            
            l_string = str(int(l))        
            m_Array = np.arange(-1.*l, l+1, 2) #Array of possible azimuthal mode numbers
            sphHarm_Array = np.array([0,0,0])


            #set up interpolation over radial coordinate r
            #tck_xir = interpolate.splrep(self.eigenfuncs[l_string]['r'],self.eigenfuncs[l_string]['xir'],s=0) 
            #tck_xih = interpolate.splrep(self.eigenfuncs[l_string]['r'],self.eigenfuncs[l_string]['xih'],s=0)
            #generate xi_r(r) and xi_h(r)
            #xir = lambda x: interpolate.splev(x,tck_xir,der=0)
            #xih = lambda x: interpolate.splev(x,tck_xih,der=0)

            ## interpolate xir, xih and provide reasonable extrapolation beyond R1 for nonlinear case
            xir = np.where(r<=self.eigenfuncs[l_string]['r'][-1],
                           np.interp(r,self.eigenfuncs[l_string]['r'],self.eigenfuncs[l_string]['xir']),
                           self.eigenfuncs[l_string]['xir'][-1]*(r/self.eigenfuncs[l_string]['r'][-1])**(l+2))

            xih = np.where(r<=self.eigenfuncs[l_string]['r'][-1],
                           np.interp(r,self.eigenfuncs[l_string]['r'],self.eigenfuncs[l_string]['xih']),
                           self.eigenfuncs[l_string]['xih'][-1]*(r/self.eigenfuncs[l_string]['r'][-1])**(l+2))

            

            xiComp=np.array([0.,0.,0.])
            
            for m in list(map(int, m_Array)): #loop over m
                sphHarm_Array = self.VectorComponents(l,m,theta,phi)
                DeltaE_lm = DeltaE_lm_Array[int(l)-2][num_l+int(m)]
            
                c = np.sqrt(DeltaE_lm)/self.omega_Array[int(l)-2]*np.exp(-1.j*self.omega_Array[int(l)-2]*t_p) #time dependence and mode amplitude
                xiComp = xiComp + [c*xir*sphHarm_Array[0],c*xih*sphHarm_Array[1],c*xih*sphHarm_Array[2]] #scale eigenmode to match energy transfer in pericenter passage
            xiComp_Array.append(list(xiComp))
        xiComp_Array = np.array(xiComp_Array)
        return xiComp_Array

    def epsilon(self,r,theta,phi,DeltaE_lm_Array,t_p):
        '''Gives the energy density from the oscillations as a function of position, energy transfer at pericenter, and time
        since a pericenter passage. 
        inputs: r, theta, phi,DeltaE_Array (array of energy transferred for each mode l calculated in the main function), time since
        pericenter passage t_p (R_*^3/G/M_*)(1/2)
        output: energy density'''
        xi_Array = np.transpose(self.xi(r,theta,phi,DeltaE_lm_Array,t_p)) #the transposition helps w/ matrix multiplication
        xi_vec = np.sum(xi_Array,axis=1) #Sum over all l to get displacement vector (radial, polar, azimuthal components)
        num_l = int(len(DeltaE_lm_Array)+1)  
        xidot_vec = np.sum(-1.j*xi_Array*self.omega_Array[:num_l-1],axis=1) #Sum_l -i omega_l xi_l (time derivative of xi)
        Cdotxi_vec =np.sum(xi_Array*self.omega_Array[:num_l-1]**2.,axis=1) #Sum_l omega_l^2 xi_l

        #The expression below is from the first equation in the Mode Energy Distribution Notes. The sum is essentially 
        #taking the dot product.
        return 0.5*np.sum(np.real(xidot_vec*np.conjugate(xidot_vec) + xi_vec*np.conjugate(Cdotxi_vec)))

    def test(self):

        '''Requests the orbital parameters, stellar spin rate, and number of modes.
        Ouputs the energy and angular momentum transferred to each mode and the total
        energy and angular momentum transferred. Works best for ecc <~ 0.95'''


        #Get User Input

        #q = input('Enter a mass ratio (M2/M1): ')
        #r_p = input('Enter the pericenter distance [R*] (Works best for r_p < 3.5 R*): ')
        #ecc = input('Enter the eccentricity (Works best for e > 0.7): ')
        #omega_S = input('Enter the stellar spin rate [(GM*/R*^3)^(1/2)] (Works best for omega_S < 0.4): ')
        #omega_S = 0.0
        q = 0.1
        r_p = 1.1
        ecc = 0.989
        omega_S = 0
        num_l = int( input('Enter the maximum mode degree (largest l, can go up to 10): ') )
        t_p = 1.0 #float( input('Enter the time since pericenter (in units sqrt(R_*^3/(GM_*)): ') )

        D_Array, Phi_Array, Nmax = self.FindOrbit(ecc)

        DeltaE_DeltaL_Array = np.array(list(map(lambda x: self.FindDeltaE_l(q,r_p,ecc,omega_S,D_Array,Phi_Array,Nmax,int(x)), self.l_Array[:num_l-1])))
        DeltaE_Array = DeltaE_DeltaL_Array[:,0]
        DeltaL_Array = DeltaE_DeltaL_Array[:,1] 
        
        DeltaE_lm_Array = self.MakeDeltaE_lmArray(q,r_p,ecc,omega_S,D_Array,Phi_Array,Nmax,num_l)
        print (DeltaE_lm_Array)

        print('Total Energy Transfer (G=M*=R*=1): '+str(np.sum(DeltaE_Array)))
        for l in self.l_Array[:num_l-1]:
            print ('l='+str(int(l)), 'DeltaE_l= '+str(DeltaE_Array[int(l)-2]))
        print('Total Angular Momentum Transfer (G=M*=R*=1): '+str(np.sum(DeltaL_Array)))
        for l in self.l_Array[:num_l-1]:
            print ('l='+str(int(l)), 'DeltaL_l= '+str(DeltaL_Array[int(l)-2]))
            
            
        print("-----\n epsilon test:", self.epsilon(1.0,np.pi/2,np.pi/2,DeltaE_lm_Array,t_p))


