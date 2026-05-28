
import numpy as np
import targets as td
import copy
import pandas as pd


LOG_ZERO_THRESH = -700.0
DELTA_THRESH = 100.0




class BIKState:
    
    def __init__(self):
        self.Ham = np.nan
    
    def firstEval(self,lpFun,q,tp):
        [f,g] = lpFun(q)
        self.q = q
        self.f = f
        self.g = g
        self.u = np.repeat(np.nan,len(q))
        self.Ham = -f
    
    def momentumRefresh(self,lpFun,tp):
        p = np.random.normal(size=len(self.q))
        for i in range(len(tp.inds)):
            self.u[tp.inds[i]] = (1.0/np.linalg.norm(p[tp.inds[i]]))*p[tp.inds[i]] 
        
        
    def velocity(self):
        return(self.u)
        
    
    def momentumFlip(self):
        self.u = -self.u
    
    def __str__(self):
        return ("IKState class:\nq: " + str(self.q) + "\nu: " + str(self.u) + "\nHamiltonian: " + str(self.Ham))

    def __repr__(self):
        return ("IKState class:\nq: " + str(self.q) + "\nu: " + str(self.u) + "\nHamiltonian: " + str(self.Ham))





class BIKtuningPars:
    
    def __init__(self, hMacro=0.1, delta=0.1, Cmin=0,Cmax=16):
        self.hMacro = hMacro
        self.delta = delta
        self.Cmin = Cmin
        self.Cmax = Cmax
        self.inds = None
    
    def blockTwoC(self,d):
        self.inds = []
        
        even = d%2==0
        nb = d//2 if even else (d-1)//2
        if(even):
            for i in range(nb):
                self.inds.append(np.array([2*i,2*i+1]))
        else:
            for i in range(nb-1):
                self.inds.append(np.array([2*i,2*i+1]))
            self.inds.append(np.array([d-3,d-2,d-1]))
        
        
        

class BIKintegrators: #(ABC):

    def integrate(self,s,lp,inds,h=0.1,nstep=10):
        
        q = copy.deepcopy(s.q)
        p = copy.deepcopy(s.u)
        g = copy.deepcopy(s.g)
        
        ph = np.zeros_like(p)
        ljac = 0.0
        
        errEst = 0.0
        Hold = -s.f
        
        for i in range(nstep):
            ljacStep = 0.0
            for j in range(len(inds)):
                gnorm = np.linalg.norm(g[inds[j]])
                delta = 0.5*h/(len(inds[j])-1)*gnorm
                ee = (1.0/gnorm)*g[inds[j]]
                
                sh = np.sinh(delta)
                ch = np.cosh(delta)
                ip = np.dot(ee,p[inds[j]])
                
                fac1 = 1.0/(ch+sh*ip)
                
                ph[inds[j]] = fac1*p[inds[j]] + fac1*(sh+ip*(ch-1.0))*ee
                ljacStep += (len(inds[j])-1)*np.log(fac1)
                
                
            
            q += h*ph
            [f,g] = lp(q)
            
            for j in range(len(inds)):
                gnorm = np.linalg.norm(g[inds[j]])
                delta = 0.5*h/(len(inds[j])-1)*gnorm
                ee = (1.0/gnorm)*g[inds[j]]
                
                sh = np.sinh(delta)
                ch = np.cosh(delta)
                ip = np.dot(ee,ph[inds[j]])
                
                fac1 = 1.0/(ch+sh*ip)
                
                p[inds[j]] = fac1*ph[inds[j]] + fac1*(sh+ip*(ch-1.0))*ee
                ljacStep += (len(inds[j])-1)*np.log(fac1)
                
            stepErr = Hold + f + ljacStep
            
            errEst += np.abs(stepErr)/nstep   
            Hold = -f
            ljac += ljacStep
             
             
        sOut = BIKState()
        sOut.q = q
        sOut.u = p
        sOut.f = f
        sOut.Ham = -f
        sOut.g = g
        
        Cest = errEst*h**(-3.0)
        
        return(sOut,ljac,Cest)
    
    
class adaptBIKstepE(BIKintegrators):
    
    def __init__(self):
        self.gradEval = 0
        self.HamErrs = []
        self.logJacs = []
        self.Ifs = []
        self.Ibs = []
        self.basic = []
        self.Cobs = []
    
    def name(self):
        return("adaptBIKstepE")
    
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
        return(BIKState())
    
    def propBasic(self):
        return(np.mean(self.basic))
    
    def Cobs(self):
        return(np.mean(self.Cobs))
    
    def __call__(self,s,lpFun,tp):
        
        
        
        If = tp.Cmax
        for c in range(tp.Cmin,tp.Cmax+1):
            nstep = 2**c
            (sOut,Wout,Cest) = self.integrate(s,lpFun,tp.inds,h=(tp.hMacro/nstep),nstep=nstep)
            self.gradEval += nstep
            locAcc = -sOut.Ham + s.Ham + Wout
            
            
            #print(locAcc)
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
                (sTmp,Wtmp,_) = self.integrate(sF,lpFun,tp.inds,h=(tp.hMacro/nstep),nstep=nstep)
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
        

    
    



        
# d = 4
# lp = td.stdGauss
# q0 = np.random.normal(size=d)

# tp = BIKtuningPars()
# tp.blockTwoC(len(q0))

# s = BIKState()
# s.firstEval(lp, q0, tp)
# s.momentumRefresh(lp, tp)

# step = adaptBIKstepE()

# (s1,ljac) = step(s,lp,tp)

# s1.momentumFlip()

# (s0b,ljacb) = step(s1,lp,tp)




# igr = BIKintegrators()
# (s1,jac,C) = igr.integrate(lp, s, tp.inds,h=0.001)

# s1.momentumFlip()
# (s0b,jacb,_) = igr.integrate(lp, s1, tp.inds,h=0.001)




