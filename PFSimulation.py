import math
import numpy as np
import matplotlib.pyplot as plt
import pdb
import time
from scipy import constants
#Partion Function
#Some Constants
#
#TODO: Implement algorithm with the precision
#TODO: Range of the input. Limit the range or Work on float overflow
#
#Kb = dm(constants.k)    #Unit:m2*kg*s-2*K-1
#Heta = dm(constants.h)#Unit:Js
Kb = 1.38064e-23
HETA =1.054571e-34
DMPREC = 40
Na = 6.022e23
C = 3e8
class SimpleHarmonicOscillator:
    #freq is a list of frequency
    def __init__(self,freq=3e12, interval = 50, maxn = 1e5):
        self.freq = freq #Default Natural Frequency which will make the simulation converge at fa faster rate
        self.interval = interval #number of data points in the plot
        self.maxn = maxn
    #Produce a graph with respect to beta
    #x: a serie of temperatures to be plotted against
    #def PF(self,tmpRange,freq, interval = 50,maxn = 100,graph = True):
     #   T = np.linspace( tmpRange[0], tmpRange[1], interval)
      #  Freq: frequency of SHO. Note: due to accuracy of the small number, the summation is not accurate under lower frequency.The default freq is 300GHz at infra-red range
    def PF_Theory(self,tmpRange,freq=3e12,interval = 50 ,graph = False):
        T = np.linspace( tmpRange[0], tmpRange[1], interval)
        omega = freq*2*np.pi
        #k1 = beta* H_bar = 1.43878e-11 * 1/T
        hv_k = 1.05457e-34*freq*2*np.pi/(1.38064e-23)
        hv_kT = hv_k/T
        P = np.exp(-0.5*hv_kT)/(1-np.exp(-hv_kT))
        if graph:
            plt.plot(hv_kT,P)
            plt.title('Theoretical PF for SMO at Freq:%f' % freq)
            plt.show()
        return [T,P]
    #
    #Input: tmpRange = [lowtemp,hightemp]
    #       interval = # of data points between these two intervals
    #       freq = natural oscillation frequency
    #       
    #Unit: All energy is in unit of eV
    #TODO: This function is fast buy not accurate under current maxn. Require further look
    def PF_Fast(self,tmpRange,freq=3e12, interval = 50,maxn = 1e5,graph = True):
        #simple profiling:
        #To-Do:
        #Plot against:y= h_bar*freq/(K_b*T)
        #calculate PF use y, PF=f(y)
        #But plot it in the equivalent T value:PF VS T
        start_time = time.time()
        self.freq = freq
        parallelRowCnt = int(1e4)
        rowInterval = int(maxn/parallelRowCnt)
        T = np.linspace( tmpRange[0], tmpRange[1], interval)
        omega = freq*2*np.pi
        #k1 = /beta* H_bar = 1.43878e-11 * 1/T
        k1 = 7.6382e-12/T
        C1 = np.exp(-k1*omega/2)
        C2 = np.exp(-k1*omega)
        #Vectorize the operation:
        C1 = np.tile(C1,(parallelRowCnt,1))
        C2 = np.tile(C2,(parallelRowCnt,1))
        n  = np.arange(0,int(maxn),rowInterval).reshape(parallelRowCnt,1) #create basis for power of n
        n = np.tile(n,(1,interval))

        #P = C1*Sum(C2^n)
        P = np.zeros((parallelRowCnt,interval))
        for adder in range(rowInterval):
            P = P + np.power(C2,n)   
        P = P * C1 
        
        #Sum All Rows in P
        P = np.sum(P,axis=0)         
        #Plot Partition Function VS Temeprature
        if graph:
            plt.plot(T,P)
            #simple profiling:
            print('PF takes seconds:'+str(time.time()-start_time))
            plt.title('Fast PF for SMO')
            plt.show()
        return [T,P]

    def PF(self,tmpRange,freq=3e12, interval = 50,maxn = 1e5,graph = True):
        start_Time = time.time()
        T = np.linspace( tmpRange[0], tmpRange[1], interval)
        self.T = T
        omega = freq*2*np.pi
        #k1 = /beta* H_bar = 1.43878e-11 * 1/T
         #Generate ratioX
        hv_k = 1.05457e-34*freq*2*np.pi/(1.38064e-23)
        hv_kT = hv_k/T
        self.hv_kT = hv_kT
        C1 = np.exp(-hv_kT/2)
        C2 = np.exp(-hv_kT)
        #P = C1*Sum(C2^n)
        P = np.zeros(interval)
        for n in range(int(maxn)):
            P = P + np.power(C2,n)   
        P = P * C1 
        
        
        #Generic Experiment
        def SMHEnergyGenerator(n):
            return (n+1/2)*HETA*freq*2*np.pi
        [T_GE,PF_GE] = PF_Generic(tmpRange,SMHEnergyGenerator,interval = interval,maxn = maxn)
        
        #Experimentt Ends
        
        [T_theory,P_theory] = self.PF_Theory(tmpRange)
        #Plot Partition Function VS Temeprature
        if graph:
           print('Base PF takes second:'+str(time.time()-start_Time))
           fig = plt.figure()
           ax = fig.add_subplot(2,2,1)
           #ax.plot(hv_kT,P,'g',label='Approximated')
           ax.plot(hv_kT,PF_GE,'g',label='Approximated')
           ax.legend()
           ax = fig.add_subplot(2,2,2)
           ax.plot(hv_kT,P_theory,'r',label='Theory')
           ax.legend()
           ax = fig.add_subplot(2,2,3)
           ax.plot(hv_kT,PF_GE-P_theory,'b',label='Diff(A-T)')
           ax.legend()
           fig.suptitle('Average Partion Function VS hV/K_b*T')
           plt.savefig('Parition Function For Simple Harmonic Oscillator.png')
           plt.show();
        self.PF = [T,P]
        return [T,P,hv_kT,omega]
    
    def PF_WithGenericPF(self,tmpRange,freq=3e12, interval = 50,maxn = 1e5,graph = True):
        def SMHEnergyGenerator(n):
            return (n+1/2)*HETA*freq*2*np.pi
        [T_GE,PF_GE] = self.computeEngine(tmpRange,SMHEnergyGenerator,maxn = maxn)
        return True

    #
    #TODO:Consider Make FE,AE,S to be a generic function. Here, we use analytical Solution
    #TODO:Maybe create a parent class to umbrella these two
    #TODO:Add plot captions and group them into figures
    #

    #Free Energy
    #For Now: use analytic solutions
    def FE_Theory(self,PF):
        [T,_,hv_kT,omega] = PF
        #A = HETA*omega/2 + (Kb*T)*np.log(1 - np.exp(-hv_kT))
        A = HETA*omega/(2*Kb) + T*np.log(1 - np.exp(-hv_kT))
        return [T,A]
    def FE(self,PF,graph = True):
        T = PF[0]
        P = PF[1]
        hv_kT = PF[2]
        #A = -(Kb*T)*np.log(P)
        A = -T*np.log(P)
        self.FEvalue = A*Kb
        [_,A_theory] = self.FE_Theory(PF) 
        if graph:
           fig = plt.figure()
           ax = fig.add_subplot(2,2,1)
           ax.plot(hv_kT,A,'g',label='Approximated')
           ax.legend()
           ax = fig.add_subplot(2,2,2)
           ax.plot(hv_kT,A_theory,'r',label='Theory')
           ax.legend()
           ax = fig.add_subplot(2,2,3)
           ax.plot(hv_kT,A-A_theory,'b',label='Diff(A-T)')
           ax.legend()
           fig.suptitle('Average Free Energy/kT VS Temperature hv/kT')
           plt.savefig('Free Energy For Simple Harmonic Oscillator.png')
           plt.show()
        return [T,A]
        

    #Average Energy
    def AE_Theory(self,PF, graph = False):
        [T,_,_,_]= PF
        omega = self.freq*2*np.pi
        #k1 = /beta* H_bar = 1.43878e-11 * 1/T
        k1 = 7.6382e-12/T
        E = HETA*omega*( 1/2+1/(np.exp(k1*omega)-1) )/Kb
        #All energy is expressed as E
        if graph:
            plt.plot(T,E)
            plt.show()
        return [T,E]
    #PF = [T,P] : an array of two elements. 0th Element: Temeprature, 1st Element: Partition Function
    def AE(self,PF,graph = True):
       #Average Energy/Kb = T^2* d(ln(Z))/d(T)
       #As Numpy has accuracy limit, the energy will be expressed as E

       T = PF[0]
       P = PF[1]
       hv_kT = PF[2]
       In_Z = np.log(P)
       deltaT = T[1]-T[0]
       #Eavg = Kb*T*np.gradient(In_Z,deltaT)
       Eavg = -1/(T*T)*np.gradient(In_Z,deltaT)
       [T_theory,E_theory] = self.AE_Theory(PF)
       if graph:
           fig = plt.figure()
           ax = fig.add_subplot(2,2,1)
           ax.plot(hv_kT,Eavg,'g',label='Approximated')
           ax.legend()
           ax = fig.add_subplot(2,2,2)
           ax.plot(hv_kT,E_theory,'r',label='Theory')
           ax.legend()
           ax = fig.add_subplot(2,2,3)
           ax.plot(hv_kT,Eavg-E_theory,'b',label='Diff(A-T)')
           ax.legend()
           fig.suptitle('Average Energy/kT VS Temeprature(hv/kT)')
           plt.savefig('Mean Energy For Simple Harmonic Oscillator.png')
           plt.show()

       return [T,Eavg]
         
       
            
    def Entropy2(self,x, graph = False):
        beta = [1 / Kb * v for v in x]
        #k*ln(1-e^(-b*h*v))
        c1 =[ - Kb*np.log( 1-np.exp( -b*Heta*self.freq) ) for b in beta]
        #1/T
        c2 = [ 1/t for t in x]
        #hv*e^(-bhv)/ (1-e^(-bhv))
        c3 = [ (Heta * self.freq * np.exp(-b * Heta * self.freq)) / (
                1 - np.exp(-b * Heta * self.freq)) for b in beta ]
        S = np.add( c1, np.multiply( c2, c3 ) )
        if graph:
            plt.plot(beta, S)
            plt.show()
        return S
    def Entropy(self,graph = True):
        #-dF/dT
        #http://www.nyu.edu/classes/tuckerman/stat.mech/lectures/lecture_13/node8.html
        FEv = self.FEvalue
        T = self.T
        deltaT = T[1]-T[0]
        #Eavg = Kb*T*np.gradient(In_Z,deltaT)
        S = -np.gradient(FEv,deltaT)
        S = S/Kb
        newX = 1/self.hv_kT
        S_Theory = self.Entropy_Theory()/Kb
        if graph:
           fig = plt.figure()
           ax = fig.add_subplot(2,2,1)
           ax.plot(newX,S,'g',label='Approximated')
           ax.legend()
           ax = fig.add_subplot(2,2,2)
           ax.plot(newX,S_Theory,'r',label='Theory')
           ax.legend()
           ax = fig.add_subplot(2,2,3)
           ax.plot(newX,S-S_Theory,'r',label='Diff(A-T)')
           ax.legend()
           fig.suptitle('S/Kb VS Temperature kT/hv')
           plt.savefig('Entropy for SHM.png')
           plt.show()
    def Entropy_Theory(self):
        beta = 1/(Kb*self.T)
        left = -Kb*np.log(1 - np.exp(-beta*HETA*2*np.pi*self.freq))
        right = (HETA*2*np.pi*self.freq/self.T)*(np.exp(-beta*2*np.pi*self.freq*HETA)/(1-np.exp(-beta*2*np.pi*self.freq*HETA)))
        return left+right
        
        

#Sturge, Qn 5.5, Part F.Treat it as an ideal gas
class diatomicPF():
    #TODO: set to real values of E0 and R0
    #Rotation Constant of Carbon Oxide Used was cited from http://adsabs.harvard.edu/abs/1965JMoSp..18..418R
    #The value is represented in wavenumber, i.e:1/cm, therefore, it needs to be converted to energy unit
    H_planck = 6.6204e-34
    B0_CO = 1.922521
    E0_CO = B0_CO*100*H_planck*C
    #R0= Na*Kb
    R0 = Na*Kb
    def __init__(self,E0 = 0,maxJ = 100,):
        self.maxJ = maxJ
        self.E0 = E0 if E0 != 0 else self.E0_CO

    def PF(self,tmpRange,interval = 100,maxj = 1e5,graph = True):
        T = np.linspace( tmpRange[0], tmpRange[1], interval)
        def g(j):
            return 2*j+1
        def E(j):
            return self.E0*j*(j+1)
        #Y = [ self.PFonT(g,E,t) for t in T  ]
        P = np.zeros(interval)
        C1 = self.E0/Kb
        beta = 1/(Kb*T)
        for j in range(int(maxj)):
            tmp1 = np.exp(-1/T*C1)
            P = P + g(j)*np.power(tmp1,j*(j+1))
        
        #calculate internal energy U
        #Here we assume N = Na
        ln_P = np.log(P)
        U = - np.gradient(ln_P,beta)
        Crot = np.gradient(U,T)
        Y = Crot/Kb
        Y = Y[:-5]
        X = T*Kb/self.E0
        X = X[:-5]
        
        
        
        #Crot = -N*d(ln(P))/d(beta)
        if graph:
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            ax.plot(X,Y,'g')
            plt.xlabel('T*Kb/E0')
            plt.ylabel('Crot/R0')
            fig.suptitle('Diatomic Molecule Heat Capacity Vs Temperature, Sturge P101')
            plt.savefig('Diatomic Molecule Crot VS T.png')
            plt.show()
        return [T,Y]

    #TODO:Vectorize the Calculation for Better efficiency
    def PFonT(self,g , E, T):
        acc = 0
        for j in range( 1, self.maxJ+1 ):
            acc = acc + g(j)*np.exp(-1/(Kb*T)*E(j))
        return acc

    #pf : [T,Y] contains numerical points on the partition function
    def AE(self, pf ):
        T = pf[0]
        beta = [ 1/t*Kb for t in T]
        Z = pf[1]
        Z = np.log(Z)
        dZ = centerDifferenceMethod(T,Y)
        return [T,-1*dZ]


    #Rotational contribution to molar heat capacity
    def Cv_rot(self, AE,graph = False ):
        T = AE[0]
        Ea = AE[1]
        Cv = centerDifferenceMethod(T,Ea)
        if graph:
            x = [t*Kb/self.E0 for t in T]
            y = [ c/R0 for C in Cv]
            plt.plot(x,y)
            plt.show()
        return [T,Cv]

#
#General Numerical Approximation Functions
#
def PF_Generic(tmpRange,energyGenerator,interval = 100,maxn = 1e5,xValue = None, nlm = None):
    nlmMode = False
    if nlm != None:
        nlmMode = True
        print("n,l,m quantum number mode is on.Please make sure energyGenerator function could take in three variables")
    T = np.linspace( tmpRange[0], tmpRange[1], interval)     
    beta = 1/(Kb*T)
    PF = [0]*len(T)
    for n in range(int(maxn)):
        if nlmMode:
            En = energyGenerator(nlm[0],nlm[1],nlm[2])
        else:
            En = energyGenerator(n)
            PF = PF + np.exp(-beta*En)
    return [T,PF]

#2nd Order Method
def centerDifferenceMethod(self,x=[],y=[]):
    #head and tail
    dY = np.zeros( len(x) )
    dY[0] = (y[1]-y[0])/(x[1]-x[0])
    dY[-1] = (y[-1]-y[-2])/(x[-1]-x[-2])
    for i in range(1,len(x)):
        dY[i] = (y[i+1]-y[i-1])/(x[i+1]-x[i-1])
    return []
#Method 2: Use 1D guassian filter
#Source: https://stackoverflow.com/questions/18991408/python-finite-difference-functions
#Method 3: Use more level of forward difference
#Source:https://www.geometrictools.com/Documentation/FiniteDifferences.pdf

def SHO_Demonstration():
    graph = True
    print("Usage Demonstration")
    sho1d = SimpleHarmonicOscillator()
    PFsho = sho1d.PF([10,500],graph=graph)
    #sho1d.AE(PF=PFsho,graph=graph)
    #sho1d.FE(PF=PFsho,graph=graph)
   # diatomicCO = diatomicPF()
    #diatomicCO.PF([1,200])

if __name__ == "__main__":
    SHO_Demonstration()
    
