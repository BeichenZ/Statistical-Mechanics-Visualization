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
    def PF(self,x,graph = False):
        beta = [ 1/Kb*v for v in x ]
        y = [ 1/(np.exp(b*Heta*self.freq/2)-np.exp(-b*Heta*self.freq/2)) for b in beta]
        if graph:
            plt.plot(beta,y)
            plt.show()
        return y

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




def demonstration():
    print("Usage Demonstration")
    freq = 100
    nu = np.linspace(1, 10, num=100)
    sho1d = SimpleHarmonicOscillatorPF(freq)
    sho1d.PF(nu)
    sho1d.Entropy(nu)
    sho1d.AE(nu)


if __name__ == "__main__":
    DM.getcontext().prec = DMPREC
    demonstration()
    
