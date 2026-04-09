import numpy as np
import targets as td
import copy
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
        self.omega = np.nan
        self.Ham = np.nan


    def momentumRefresh(self,lpFun,tp):
        self.p = np.random.normal(size=len(self.q))
        self.omega = np.sqrt(tp.oSigmaSq)*np.random.normal()
        self.Ham = -self.f + 0.5*np.dot(self.p,self.p) + (0.5/tp.oSigmaSq)*(self.omega)**2
    
    def momentumFlip(self):
        self.p = -self.p
        self.omega = -self.omega

    def velocity(self):
        return(self.p)
    
    def __str__(self):
        return ("SNHState class:\nq: " + str(self.q) + "\np: " + str(self.p) + "\nomega: " + str(self.omega) + "\nHamiltonian: " + str(self.Ham))

    def __repr__(self):
        return ("SNHState class:\nq: " + str(self.q) + "\np: " + str(self.p) + "\nomega: " + str(self.omega) + "\nHamiltonian: " + str(self.Ham))

        



class SNHintegrators:
    
    def LFint(self,s,lpFun,oSigmaSq,h=0.01,nstep=10):
        q = s.q
        p = s.p
        g = s.g
        o = s.omega
        d = len(q)
        
        HamOld = s.Ham
        
        #eta = np.repeat(1.0, len(q))
        
        ljac = 0.0
        errEst = 0.0
        
        for i in range(nstep):
            ph = p*np.exp(-0.5*h*o) + g*((1.0-np.exp(-0.5*h*o))/o)
            #etah = eta*np.exp(0.5*h*o)
            
            q = q + h*ph
            
            [f,g] = lpFun(q)
            oOld = o
            o = o + h*oSigmaSq*(np.dot(ph,ph)-d)
            
            stepLjac = -0.5*h*d*(o+oOld)
            ljac += stepLjac 
            
            
            p = ph*np.exp(-0.5*h*o) + g*((1.0-np.exp(-0.5*h*o))/o)
            #eta = etah*np.exp(0.5*h*o)
            
            Ham = -f + 0.5*np.dot(p,p) + (0.5/oSigmaSq)*o**2
            
            stepErr = HamOld - Ham + stepLjac
            errEst += np.abs(stepErr)/nstep
            
            HamOld = Ham
            
            
            
            
        
        sOut = SNHState()
        sOut.f = f
        sOut.q = q
        sOut.p = p
        sOut.g = g
        sOut.omega = o
        sOut.Ham = Ham
        
        Cest = errEst*h**(-3.0)
        
        return(sOut,ljac,Cest)
            
class adaptSNHstepE(SNHintegrators):
    
    def __init__(self):
        self.gradEval = 0
        self.HamErrs = []
        self.logJacs = []
        self.Ifs = []
        self.Ibs = []
        self.basic = []
        self.Cobs = []
    
    
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
            (sOut,Wout,Cest) = self.LFint(s,lpFun,tp.oSigmaSq,h=(tp.hMacro/nstep),nstep=nstep)
            self.gradEval += nstep
            locAcc = -sOut.Ham + s.Ham + Wout
            
            
            
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
                (sTmp,Wtmp,_) = self.LFint(sF,lpFun,tp.oSigmaSq,h=(tp.hMacro/nstep),nstep=nstep)
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
        




lp = td.stdGauss
tp = SNHtuningPars()
tp.hMacro = 0.5
s = SNHState()

q0 = np.array([1.0,-0.5])

s.firstEval(lp, q0, tp)
s.momentumRefresh(lp, tp)

step = adaptSNHstepE()

(s1,w) = step(s,lp,tp)

s1.momentumFlip()

(s0b,wb) = step(s1,lp,tp)




#ii = NHintegrators()
#(s1,ljac,Cest) = ii.LFint(s,lp, tp.oSigmaSq)
#s1.momentumFlip()
#(s0b,ljacb,Cestb) = ii.LFint(s1,lp, tp.oSigmaSq)
