import decimal as DM
import math
from decimal import Decimal as dm
import numpy as np
import mpmath
import matplotlib.pyplot as plt
from scipy import constants
#Partion Function
#Some Constants
#
#TODO: Implement algorithm with the precision
#TODO: Range of the input. Limit the range or Work on float overflow
#
#Kb = dm(constants.k)    #Unit:m2*kg*s-2*K-1
#Heta = dm(constants.h)#Unit:Js
Kb = 1e-5
Heta =2e-5
DMPREC = 40
class SimpleHarmonicOscillatorPF:
    #freq is a list of frequency
    def __init__(self,freq):
        self.freq = freq
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
        k1 = 1.43878e-11/T
        P = np.exp(-0.5*k1*omega)/(1-np.exp(-k1*omega))
        if graph:
            plt.plot(T,P)
            plt.show()
        return P
    #
    #Input: tmpRange = [lowtemp,hightemp]
    #       interval = # of data points between these two intervals
    #       freq = natural oscillation frequency
    #       
    #Unit: All energy is in unit of eV
    def PF(self,tmpRange,freq=3e12, interval = 50,maxn = 1000000,graph = True):
        T = np.linspace( tmpRange[0], tmpRange[1], interval)
        omega = freq*2*np.pi
        #k1 = /beta* H_bar = 1.43878e-11 * 1/T
        k1 = 1.43878e-11/T
        C1 = np.exp(-k1*omega/2)
        C2 = np.exp(-k1*omega)

        #P = C1*Sum(C2^n)
        P = np.zeros(interval)
        for n in range(interval):
            P = P + np.power(C2,n)   
        P = P * C1 
         
        #Plot Partition Function VS Temeprature
        if graph:
            plt.plot(T,P)
            plt.show()
        return P

    #
    #TODO:Consider Make FE,AE,S to be a generic function. Here, we use analytical Solution
    #TODO:Maybe create a parent class to umbrella these two
    #TODO:Add plot captions and group them into figures
    #

    #Free Energy
    #For Now: use analytic solutions
    def FE(self,x,graph = False):
        beta = [1 / Kb * v for v in x]
        y = [Heta*self.freq/2 + (1/b)*np.log(1-np.exp(-b*Heta*self.freq)) for b in beta]
        if graph:
            plt.plt(beta,y)
            plt.show()
        return y

    #Average Energy
    def AE(self,x, graph = False):
        beta = [1 / Kb * v for v in x]
        y = [Heta * self.freq / 2 + (Heta*self.freq*np.exp(-b*Heta*self.freq)) / (1 - np.exp(-b * Heta * self.freq)) for b in beta]
        if graph:
            plt.plot(beta,y)
            plt.show()
        return y

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
    freq = 100
    nu = np.linspace(1, 10, num=100)
    sho1d = SimpleHarmonicOscillatorPF(freq)
    sho1d.PF([100,373],1e12)
    sho1d.PF_Theory([100,373],1e12)
   #sho1d.PF_Theory(nu)
    #sho1d.Entropy(nu)
    #sho1d.AE(nu)


if __name__ == "__main__":
    DM.getcontext().prec = DMPREC
    demonstration()
    
