import decimal as DM
import math
from decimal import Decimal as dm
import numpy as np
import mpmath
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
class SimpleHarmonicOscillatorPF:
    #freq is a list of frequency
    def __init__(self):
        self.freq = 3e12 #Default Natural Frequency which will make the simulation converge at fa faster rate
        self.interval = 50 #number of data points in the plot
        #Set decimal precision
        DM.getcontext().prec= DMPREC

    #Produce a graph with respect to beta
    #x: a serie of temperatures to be plotted against
    #def PF(self,tmpRange,freq, interval = 50,maxn = 100,graph = True):
     #   T = np.linspace( tmpRange[0], tmpRange[1], interval)
      #  Freq: frequency of SHO. Note: due to accuracy of the small number, the summation is not accurate under lower frequency.The default freq is 300GHz at infra-red range
    def PF_Theory(self,tmpRange,freq=3e12,interval = 50 ,graph = True):
        T = np.linspace( tmpRange[0], tmpRange[1], interval)
        omega = freq*2*np.pi
        #k1 = beta* H_bar = 1.43878e-11 * 1/T
        k1 = 7.6382e-12/T
        P = np.exp(-0.5*k1*omega)/(1-np.exp(-k1*omega))
        if graph:
            plt.plot(T,P)
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
        start_time = time.time()

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
            #pdb.set_trace()
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
        omega = freq*2*np.pi
        #k1 = /beta* H_bar = 1.43878e-11 * 1/T
        k1 = 7.6382e-12/T
        C1 = np.exp(-k1*omega/2)
        C2 = np.exp(-k1*omega)

        #P = C1*Sum(C2^n)
        P = np.zeros(interval)
        for n in range(int(maxn)):
            P = P + np.power(C2,n)   
        P = P * C1 
         
        #Plot Partition Function VS Temeprature
        if graph:
            plt.plot(T,P)
            print('Base PF takes second:'+str(time.time()-start_Time))
            plt.title('Base PF for SMO')
            plt.show()
        self.PF = [T,P]
        return [T,P]    

    #
    #TODO:Consider Make FE,AE,S to be a generic function. Here, we use analytical Solution
    #TODO:Maybe create a parent class to umbrella these two
    #TODO:Add plot captions and group them into figures
    #

    #Free Energy
    #For Now: use analytic solutions
    def FE_Theory(self,x,graph = False):
        beta = [1 / Kb * v for v in x]
        y = [Heta*self.freq/2 + (1/b)*np.log(1-np.exp(-b*Heta*self.freq)) for b in beta]
        if graph:
            plt.plt(beta,y)
            plt.show()
        return y
    def FE(self,tmpRange,graph = True):
        pass

        

    #Average Energy
    def AE_Theory(self,PF, graph = False):
        [T,_]= PF
        omega = self.freq*2*np.pi
        #k1 = /beta* H_bar = 1.43878e-11 * 1/T
        k1 = 7.6382e-12/T
        E = HETA*omega*( 1/2+1/(np.exp(k1*omega)-1) )
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
       In_Z = np.log(P)
       deltaT = T[1]-T[0]
       Eavg = Kb*T*T*np.gradient(In_Z,deltaT)
       [T_theory,E_theory] = self.AE_Theory(PF)
       if graph:
           plt.plot(T,Eavg,label='Approximated')
           plt.plot(T_theory,E_theory,label='Theory')
           plt.ylabel("<E> (J)")
           plt.xlabel("Temeprate (K)")
           plt.title("Average Energy VS Temperature for SHO")
           plt.legend()

           plt.show()

       return [T,Eavg]
         
       
            
    def Entropy(self,x, graph = False):
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

#Sturge, Qn 5.5, Part F.Treat it as an ideal gas
class diatomicPF():
    #TODO: set to real values of E0 and R0
    E0 = 1.1
    R0 = 1
    def __init__(self, maxJ = 100, tRange = [1,3000]):
        self.maxJ = maxJ
        self.tRange = tRange

    def PF(self, graph = False):
        def g(j):
            return 2*j+1
        def E(j):
            return E0*j*(j+1)
        T = np.linspace( self.tRange[0], self.tRange[1], 100)
        Y = [ self.PFonT(g,E,t) for t in T  ]
        if graph:
            plt.plot(T,Y)
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
#TODO:Verify the accuracy of this method.Implement more if needed
#
#2nd Order Method
def centerDifferenceMethod(self,x=[],y=[]):
    #head and tail
    dY = np.zeros( len(x) )
    dY[0] = (y[1]-y[0])/(x[1]-x[0])
    dY[-1] = (y[-1]-y[-2])/(x[-1]-x[-2])
    for i in range(1,len(x)):
        dY[i] = (y[i+1]-y[i-1])/(x[i+1]-x[i-1])
    return []

def demonstration():
    print("Usage Demonstration")
    sho1d = SimpleHarmonicOscillatorPF()
    sho1d.PF([100,373])
    PFsho=sho1d.PF_Theory([100,373])
    sho1d.AE(PF=PFsho)


if __name__ == "__main__":
    DM.getcontext().prec = DMPREC
    demonstration()
    
