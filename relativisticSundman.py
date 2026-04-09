import numpy as np
import scipy as sp
import targets as td
import relativistic as rel
import pandas as pd
import copy

LOG_ZERO_THRESH = -700.0

class RELISstate:
    
        
    
    
    def __init__(self):
        self.Ham = np.nan
    
    
    
    
    def firstEval(self,lpFun,q,tp):
        [f,g] = lpFun(q)
        self.q = q
        self.f = f
        self.g = g
        self.Hv = np.repeat(np.nan,len(q))
        self.p = np.repeat(np.nan,len(q))
        self.v = np.repeat(np.nan,len(q))
        self.o = np.nan
        self.Ham = np.nan
        
        
    def momentumRefresh(self,lpFun,tp):
        [f,g] = lpFun(self.q) # already computed 
        self.p = rel.RELunivarMomentum().rng(size=len(self.q),c=tp.c)
        self.v = self.p/np.sqrt(1.0 + (self.p/tp.c)**2)
        [_,_,Hv] = lpFun(self.q,v=self.v)
        self.Hv = Hv
        omean = -tp.alpha*np.log(tp.eta + np.dot(g,g))
        self.o =  omean + tp.oSigma*np.random.normal()
        self.Ham = -self.f + (tp.c**2)*np.sum(np.sqrt(1.0 + (self.p/tp.c)**2)) + 0.5/(tp.oSigma**2)*(self.o-omean)**2
        
        
        
    def velocity(self):
        
        return(self.v)
        
    
    def momentumFlip(self):
        self.p = -self.p
        self.v = -self.v
        self.Hv = -self.Hv
    
    def __str__(self):
        return ("RELIState class:\nq: " + str(self.q) + "\np: " + str(self.p) + "\no: " + str(self.o) + "\nHamiltonian: " + str(self.Ham))

    def __repr__(self):
        return ("RELIState class:\nq: " + str(self.q) + "\np: " + str(self.p) + "\no: " + str(self.o) + "\nHamiltonian: " + str(self.Ham))


class RELIStuningPars:
    
    def __init__(self, c=1.0, alpha=0.5, eta=0.1, oSigma=0.2, hMacro=0.2, delta=0.3, Cmin=0,Cmax=16):
        self.c = c
        self.alpha = alpha
        self.eta = eta
        self.oSigma = oSigma
        self.hMacro = hMacro
        self.delta = delta
        self.Cmin = Cmin
        self.Cmax = Cmax
        


class RELISintegrator:
    
    def integrate(self,s,lpFun,c,alpha,eta,oSigma,h=0.025,nstep=1):
        
        q = s.q
        p = s.p
        g = s.g
        o = s.o
        Hv = s.Hv
        
        G = -2.0*alpha/(eta+np.dot(g,g))*np.dot(g,Hv)
        
        a0 = -0.5*h*G
        
        ljac = 0.0
        
        for i in range(nstep):
            lwarg = a0*np.exp(o)
            if(lwarg< -0.3678794411714423):
                return(RELISstate(),np.nan)
            oh = o - sp.special.lambertw(lwarg,tol=1.0e-14).real
            ljac -= np.log(1.0+o-oh)
            eoh = np.exp(oh)
            ph = p + 0.5*h*eoh*g
            vh = ph/np.sqrt(1+(ph/c)**2)
            q = q + h*eoh*vh
            
            [f,g] = lpFun(q)
            p = ph + 0.5*h*eoh*g
            v = p/np.sqrt(1+(p/c)**2)
            [_,_,Hv] = lpFun(q,v=v)
            G = -2.0*alpha/(eta+np.dot(g,g))*np.dot(g,Hv)
            o = oh + 0.5*h*G*eoh
            jac2 = 1.0+o-oh
            if(jac2 < 1.0e-14):
                return(RELISstate(),np.nan)
            ljac += np.log(1.0+o-oh)
            a0 = -0.5*h*G
        
        
        sOut = RELISstate()
        sOut.f = f
        sOut.q = q
        sOut.p = p
        sOut.o = o
        sOut.g = g
        sOut.Hv = Hv
        sOut.v = v
        
        omean = -alpha*np.log(eta + np.dot(g,g))
        
        sOut.Ham = -f + (c**2)*np.sum(np.sqrt(1.0 + (p/c)**2)) + 0.5/(oSigma**2)*(o-omean)**2
        
        
        return(sOut,ljac)
        


        
class adaptRELISstepE(RELISintegrator):
    
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
        return(RELISstate())
    
    def propBasic(self):
        return(np.mean(self.basic))
    
    def Cobs(self):
        return(np.median(self.Cobs))
    
    def conservedRange(self):
        return(np.max(self.Hams)-np.min(self.Hams))

    
    def __call__(self,s,lpFun,tp):
        
        if not self.Hams:
            self.Hams.append(s.Ham)
            
        If = tp.Cmax
        
        for c in range(tp.Cmin,tp.Cmax+1):
            nstep = 2**c
            
            (sOut,ljac) = self.integrate(s=s,lpFun=lpFun,
                                  c=tp.c,
                                  alpha=tp.alpha,
                                  eta=tp.eta,
                                  oSigma=tp.oSigma,
                                  h=tp.hMacro/nstep,nstep=nstep)
            self.gradEval += nstep
            
            
            
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
                
                (sTmp,_) = self.integrate(sB,lpFun,c=tp.c,
                                      alpha=tp.alpha,
                                      eta=tp.eta,
                                      oSigma=tp.oSigma,
                                      h=tp.hMacro/nstep,nstep=nstep)
                self.gradEval += nstep
                
                Eerr = np.abs(sTmp.Ham-sB.Ham)
                
                if(Eerr < tp.delta):
                    Ib = c
                    break
        
        else:
            self.basic.append(1)

        self.Ifs.append(If)
        self.Ibs.append(Ib)
        self.Hams.append(sOut.Ham)
        
        lwt = ljac if Ib==If else LOG_ZERO_THRESH
        
        
        return(sOut,lwt)
        




# lp = td.stdGauss

# tp = RELIStuningPars()
# tp.c = 1.0
# tp.alpha = 1.0/6.0
# tp.eta = 1.0
# s = RELISstate()
# q0 = np.array([-1.0,-1.1])
# s.firstEval(lp, q0, tp)
# s.momentumRefresh(lp, tp)

# ig = RELISintegrator()
# step = adaptRELISstepE()

# (s1,ljf) = ig.integrate(s, lp, tp.c, tp.alpha, tp.eta, tp.oSigma)
# s1.momentumFlip()
# (s0b,ljb) = ig.integrate(s1, lp, tp.c, tp.alpha, tp.eta, tp.oSigma)

# import matplotlib.pyplot as plt

# nInt = 100
# qq = np.zeros((2,nInt+1))
# ljac = np.zeros(nInt+1)
# qq[:,0] = q0

# for i in range(nInt):
#     (s1,lj) = step(s, lp, tp)
#     qq[:,i+1] = s1.q
#     ljac[i+1] = ljac[i] + lj
#     s = s1
    
# plt.subplot(1,2,1)  
# plt.scatter(qq[0,:],qq[1,:],c=np.exp(ljac),cmap='grey',s=10)
# plt.subplot(1,2,2)
# plt.plot(np.exp(ljac))  
