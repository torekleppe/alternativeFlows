import numpy as np
import scipy as sp
import targets as td
import copy
import pandas as pd

LOG_ZERO_THRESH = -700.0


class expPnorm:
    def __init__(self,alpha=0.1,oStd=0.1):
        self.alpha = alpha
        self.oStd = oStd
        self.ov = oStd**2
        
        
    def ell(self,p):
        return(-0.1*(1.0/2.0)*(np.log(self.alpha+p**2)-np.log(self.alpha)))
        #return(-(self.alpha*p**2))
    
    def d_ell(self,p):
        return(-0.1*p/(self.alpha + p**2))
        #return(-2.0*self.alpha*p)

    def dotQ(self,p,omega):
        return(np.exp(omega)*p)
    
    def dotP(self,g,omega):
        return(np.exp(omega)*g)
    
    def velocity(self,p,omega):
        return(np.exp(omega)*p)
    
    def dotOmega(self,g,p,omega):
        return(self.d_ell(p)*np.exp(omega)*g)
    
    def Ham(self,f,p,omega):
        return(-f + 0.5*np.dot(p,p) + (0.5/self.ov)*np.sum((omega-self.ell(p))**2))
    
    def momentumRefresh(self,dim):
        p = np.random.normal(size=dim)
        omega = self.ell(p) + self.oStd*np.random.normal(size=dim)
        return(p,omega)
    
    
    def implicitOmegaStep(self,h,g0,p0,o0):
        a = 0.5*h*self.d_ell(p0)*g0
        arg = -a*np.exp(o0)
        if(all(arg > -0.36787944117144233)):
            W = sp.special.lambertw(arg,tol=1.0e-14).real
            return(o0 - W,-np.sum(np.log(1.0+W)))
        else:
            return(np.repeat(np.nan,len(g0)),np.nan)
        
class VSState:
    
    def __init__(self):
        self.Ham = np.nan
    
    def firstEval(self,lpFun,q,tp):
        [f,g] = lpFun(q)
        self.q = q
        self.f = f
        self.g = g
        self.p = np.repeat(np.nan,len(q))
        self.omega = np.repeat(np.nan,len(q))
        self.v = np.repeat(np.nan,len(q))
        self.Ham = np.nan
    
    def momentumRefresh(self,lpFun,tp):
        (p,o) = tp.pd.momentumRefresh(len(self.q))
        self.p = p
        self.omega = o
        self.v = tp.pd.velocity(p, o)
        self.Ham = tp.pd.Ham(self.f, p, o)
        
        
    def velocity(self):
        return(self.p)
        
    
    def momentumFlip(self):
        self.p = -self.p
    
    def __str__(self):
        return ("VSState class:\nq: " + str(self.q) + "\np: " + str(self.p) + "\nomega: " + str(self.omega) + "\nHamiltonian: " + str(self.Ham))

    def __repr__(self):
        return ("VSState class:\nq: " + str(self.q) + "\np: " + str(self.p) + "\nomega: " + str(self.omega) + "\nHamiltonian: " + str(self.Ham))





class VStuningPars:
    
    
    
    def __init__(self, hMacro=0.1, delta=0.1, Cmin=0,Cmax=16,
                 procDef=expPnorm()):
        
        self.hMacro = hMacro
        self.delta = delta
        self.Cmin = Cmin
        self.Cmax = Cmax
        self.pd = procDef
        



class VSintegrators: #(ABC):
    
    def vecSundmanInt(self,s,lpFun,pd,h=0.1,nstep=1):
        q = s.q
        p = s.p
        o = s.omega
        g = s.g
        lJac = 0.0
        nev = 0
        for i in range(nstep):
            (oh,lJac1) = pd.implicitOmegaStep(h,g,p,o)
            if(any(np.isnan(oh))):
                return(VSState(),np.nan,nev)
            lJac += lJac1
            ph = p + 0.5*h*pd.dotP(g,oh)
            q = q + h*pd.dotQ(ph,oh)
            [f,g] = lpFun(q)
            nev+=1
            p = ph + 0.5*h*pd.dotP(g,oh)
            hG = 0.5*h*pd.dotOmega(g,p,oh)
            if(any(hG< -1.0)):
                return(VSState(),np.nan,nev)
            o = oh + hG
            lJac += np.sum(np.log(1.0+hG))
            
            
        s1 = VSState()
        s1.f = f
        s1.g = g
        s1.q = q
        s1.p = p
        s1.omega = o
        s1.Ham = pd.Ham(f,p,o)
        s1.v = pd.velocity(p,o)
        
        return(s1,lJac,nev)
        
        
        
        
        
        

class adaptVSstepE(VSintegrators):

    def __init__(self):
        self.gradEval = 0
        self.Hams = []
        self.Ifs = []
        self.Ibs = []
        self.basic = []
        self.Cobs = []
    
    
    def reset(self):
        self.gradEval = 0
        self.Hams = []
        self.Ifs = []
        self.Ibs = []
        self.basic = []
        self.Cobs = []

    def diagnostics(self):
        
        return (pd.Series([self.gradEval, np.max(self.Hams)-np.min(self.Hams),
                          np.min(self.Ifs),
                          np.max(self.Ifs),
                          np.mean(self.basic)],
                         index=['gradEvals', 'energyErr',
                                'minIf', 'maxIf', 'propBasic']))


    
    def getState(self):
        return(VSState())
    
    def propBasic(self):
        return(np.mean(self.basic))
    
    def Cobs(self):
        return(np.median(self.Cobs))
    
    
    def __call__(self,s,lpFun,tp):
        
        if not self.Hams:
            self.Hams.append(s.Ham)
            
        If = tp.Cmax
        
        for c in range(tp.Cmin,tp.Cmax+1):
            nstep = 2**c
            
            (sOut,lJac,nev) = self.vecSundmanInt(s,lpFun,tp.pd,tp.hMacro/nstep,nstep)
            self.gradEval += nev
            
            Eerr = np.abs(sOut.Ham-s.Ham)
            
            
            Cobs = np.abs(Eerr)*nstep**2/(tp.hMacro**3)
            
            if(Eerr < tp.delta):
                If = c
                break
            
        Ib = If
        self.Cobs.append(Cobs)
        
        if(If>tp.Cmin):
            self.basic.append(0)
            sB = copy.deepcopy(sOut)
            sB.momentumFlip()
            
            for c in range(tp.Cmin,If):
                nstep = 2**c
                
                (sTmp,_,nev) = self.vecSundmanInt(sB,lpFun,tp.pd,tp.hMacro/nstep,nstep)
                self.gradEval += nev
                
                Eerr = np.abs(sTmp.Ham-sB.Ham)
                
                
                if(Eerr < tp.delta):
                    Ib = c
                    break
        
        else:
            self.basic.append(1)

        self.Ifs.append(If)
        self.Ibs.append(Ib)
        self.Hams.append(sOut.Ham)
        
        lwt = lJac if Ib==If else LOG_ZERO_THRESH
        
        
        return(sOut,lwt)    

   
        

        


       
# import matplotlib.pyplot as plt
# #np.random.seed(1)
# lp = td.modFunnel
# tp = VStuningPars(Cmax=6)
# tp.hMacro = 0.01
# s = VSState()
# s.firstEval(lp,np.array([-1.0,-0.01]),tp)
# s.momentumRefresh(lp,tp)

# nstep = 1000
# step = adaptVSstepE()
# qs = np.zeros((2,nstep+1))
# qs[:,0] = s.q

# os = np.zeros((2,nstep+1))
# os[:,0] = s.omega

# Hams = np.zeros(nstep+1)
# Hams[0] = s.Ham
# ljac = 0.0
# ljacs = np.zeros(nstep+1)


# for i in range(nstep):
#     (s,lwt) = step(s,lp,tp)
#     qs[:,i+1] = s.q
#     os[:,i+1] = s.omega
#     Hams[i+1] = s.Ham
#     ljac += lwt
#     ljacs[i+1] = ljac    

# plt.subplot(1,4,1)    
# plt.plot(qs[0,:],qs[1,:])
# plt.subplot(1,4,2)
# plt.plot(Hams)
# plt.subplot(1,4,3)
# plt.plot(ljacs)
# plt.subplot(1,4,4)
# plt.plot(os[0,:],os[1,:])







# (s1,lwt) = step(s,lp,tp)


# s1.momentumFlip()
# (s0b,lwtb) = step(s1,lp,tp)


