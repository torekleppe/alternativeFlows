

import sys
import numpy as np

import specFuns as sf 
import copy
import targets as td
import matplotlib.pyplot as plt
import pandas as pd

LOG_ZERO_THRESH = -700.0

class SNHtuningPars:
    
    def __init__(self, hMacro=0.1, delta=0.1, oSigmaSq = 1.0, Cmin=0,Cmax=10):
        
        self.hMacro = hMacro
        self.delta = delta
        self.oSigmaSq = oSigmaSq
        self.Cmin = Cmin
        self.Cmax = Cmax



class SNHState:
    def firstEval(self,lpFun,q,tp):
        [f,g] = lpFun(q)
        self.q = q
        self.f = f
        self.g = g
        self.p = np.repeat(np.nan,len(q))
        self.omega = np.repeat(np.nan,len(q))
        self.Ham = np.nan


    def momentumRefresh(self,lpFun,tp):
        self.p = np.random.normal(size=len(self.q))
        self.omega = np.sqrt(tp.oSigmaSq)*np.random.normal()
        self.Ham = -self.f + 0.5*np.dot(self.p,self.p) + (0.5/tp.oSigmaSq)*(self.omega**2)
    
    def momentumFlip(self):
        self.p = -self.p
        self.omega = -self.omega

    def velocity(self):
        return(self.p)
    
    def __str__(self):
        return ("SNHState class:\nq: " + str(self.q) + "\np: " + str(self.p) + "\nomega: " + str(self.omega) + "\nHamiltonian: " + str(self.Ham))

    def __repr__(self):
        return ("SNHState class:\nq: " + str(self.q) + "\np: " + str(self.p) + "\nomega: " + str(self.omega) + "\nHamiltonian: " + str(self.Ham))

        







class NHscalarIntergrator:
    
    def NHTBTATBT(self,s,lpFun,oSigmaSq=1.0,h=0.1,nstep=1):
        q = copy.deepcopy(s.q)
        p = copy.deepcopy(s.p)
        g = copy.deepcopy(s.g)
        o = copy.deepcopy(s.omega)
        d = len(q)
        
        ljac = 0.0
        
        #qs = np.zeros((d,nstep+1))
        #qs[:,0] = q
        #HH = np.ones(nstep+1)
        #HH[0] = -s.Ham
        
        for i in range(nstep):
            o = o + 0.25*h*oSigmaSq*(np.dot(p,p)-d)
            
            p = p*np.exp(-0.5*h*o) + g*sf.linOdefacScalar(-0.5*h,o) #- g*((np.expm1(-0.5*h*o))/o)
            ljac -= 0.5*h*d*o
            
            o = o + 0.25*h*oSigmaSq*(np.dot(p,p)-d)
            
            q = q + h*p
            [f,g] = lpFun(q)
            
            o = o + 0.25*h*oSigmaSq*(np.dot(p,p)-d)
            
            p = p*np.exp(-0.5*h*o) + g*sf.linOdefacScalar(-0.5*h,o) #- g*((np.expm1(-0.5*h*o))/o)
            ljac -= 0.5*h*d*o
            
            o = o + 0.25*h*oSigmaSq*(np.dot(p,p)-d)
            
            #qs[:,i+1] = q
            #HH[i+1] = f -0.5*np.dot(p,p) - (0.5/oSigmaSq)*(o**2) + ljac

        #plt.subplot(1,2,1)
        #plt.plot(qs[0,:],qs[1,:])
        #plt.subplot(1,2,2)
        #plt.plot(HH)
        
        sOut = SNHState()
        sOut.q = q
        sOut.p = p
        sOut.g = g
        sOut.f = f
        sOut.omega = o
        sOut.Ham = -f + 0.5*np.dot(p,p) + (0.5/oSigmaSq)*(o**2)
        
        return(sOut,ljac)
        


class adaptSNHstepE(NHscalarIntergrator):
    
    def __init__(self):
        self.gradEval = 0
        self.HamErrs = []
        self.logJacs = []
        self.Ifs = []
        self.Ibs = []
        self.basic = []
        self.Cobs = []
    
    def name(self):
        return("adaptSNHstepE")
    
    def reset(self):
        self.gradEval = 0
        self.HamErrs = []
        self.logJacs = []
        self.Ifs = []
        self.Ibs = []
        self.basic = []
        self.Cobs = []

    def diagnostics(self):
        
        return (pd.Series([self.gradEval, np.max(np.abs(self.HamErrs)),
                          np.min(self.Ifs),
                          np.max(self.Ifs),
                          np.mean(self.basic)],
                         index=['gradEvals', 'energyErr',
                                'minIf', 'maxIf', 'propBasic']))


    
    def getState(self):
        return(SNHState())
    
    def propBasic(self):
        return(np.mean(self.basic))
    
    def Cobs(self):
        return(np.mean(self.Cobs))
    
    def __call__(self,s,lpFun,tp):
        
        
        
        If = tp.Cmax
        for c in range(tp.Cmin,tp.Cmax+1):
            nstep = 2**c
            (sOut,Wout) = self.NHTBTATBT(s,lpFun,tp.oSigmaSq,h=(tp.hMacro/nstep),nstep=nstep)
            self.gradEval += nstep
            
            
            
            
            
            locAcc = -sOut.Ham + s.Ham + Wout
            Eerr = np.abs(locAcc)
            
            Cest = np.abs(Eerr)*nstep**2/(tp.hMacro**3)
            
            
            if(np.abs(locAcc) < tp.delta):
                If = c
                break
        
        self.HamErrs.append(locAcc)
        self.Cobs.append(Cest)
        Ib = If
        if(If>tp.Cmin):
            
            self.basic.append(0)
            
            sF = copy.deepcopy(sOut)
            sF.momentumFlip()
            
            for c in range(tp.Cmin,If):
                nstep = 2**c
                (sTmp,Wtmp) = self.NHTBTATBT(sF,lpFun,tp.oSigmaSq,h=(tp.hMacro/nstep),nstep=nstep)
                self.gradEval += nstep
                
                locAcc = -sTmp.Ham + sF.Ham + Wtmp
                #print(locAcc)
                
                if(np.abs(locAcc) < tp.delta):
                    Ib = c
                    break
        else:
            
            self.basic.append(1)

        
        self.logJacs.append(Wout)
        self.Ifs.append(If)
        self.Ibs.append(Ib)
        
        if(Ib < If): Wout += LOG_ZERO_THRESH 
        
        return(sOut,Wout)
        



        


#lp = td.modFunnel
#q0 = np.array([-0.4,-0.5])
#tp = SNHtuningPars()


#s = SNHState()
#s.firstEval(lp, q0, tp)
#s.momentumRefresh(lp, tp)

# step = adaptSNHstepE()

# n = 100

# qs = np.zeros((len(s.q),n+1))
# qs[:,0] = s.q
# HH = np.zeros(n+1)
# HH[0] = -s.Ham

# ljsum = 0.0

# for i in range(n):
#     (s,lj) = step(s,lp,tp)
#     ljsum += lj
#     qs[:,i+1] = s.q
#     HH[i+1] = -s.Ham + ljsum

# plt.subplot(1,2,1)
# plt.plot(qs[0,:],qs[1,:])

# plt.subplot(1,2,2)
# plt.plot(HH)

#igr = NHscalarIntergrator()
#(s1,ljac)=igr.NHTBTATBT(s,lp,nstep=100)


#s1.momentumFlip()
#(s0b,ljacb)=igr.NHTBTATBT(s1,lp,nstep=100)

