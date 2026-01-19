import targets as td
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import copy
import pandas as pd
from abc import ABC
from relativisticKineticDistribution import globalRelativisticKineticDistribution

LOG_ZERO_THRESH = -700.0



def RELI_GLF_solveB(v0,g,h,c):
    cSq = c**2
    cCube = cSq*c
    a = np.repeat(-c*(1.0 - 1.0e-14),len(v0))
    b = np.repeat(-a[0],len(v0))
    fa = v0 + (0.5*h/cCube)*g*(cSq-a**2)**1.5 - a
    fb = v0 + (0.5*h/cCube)*g*(cSq-b**2)**1.5 - b
    print(fa)
    print(fb)
    if(any(fa*fb>0.0)):
        pass
        
    
    for it in range(20):
        t = 0.5*(a+b)
        ft = v0 + (0.5*h/cCube)*g*(cSq-t**2)**1.5 - t
        print(ft)
        l = ft*fa > 0.0
        a[l] = t[l]
        b[~l] = t[~l]
        
        print((a,b,b-a))
    
    
    t = 0.5*(a+b)
    conv = False
    for it in range(4):
        ft = v0 + (0.5*h/cCube)*g*(cSq-t**2)**1.5 - t
        print(ft)
        if(np.max(np.abs(ft))<1.0e-13):
            conv = True
            break
        dft = -1.0 - (1.5*h/cCube)*g*t*np.sqrt(cSq-t**2)
        t -= ft/dft
        t = np.minimum(b,np.maximum(a,t))
    if(not conv):
        print("convergence problems in RELI_GLF_solveB, max error:")
        print(np.max(np.abs(ft)))
    return(t)
        
        

class RELunivarMomentum:
    
    
    def logpdf(self,x,c=4.0):
  
        return(sp.stats.genhyperbolic.logpdf(x,p=1,a=c**2,b=0.0,scale=c))
    
    def rng(self,size=1,c=4.0):
        return(sp.stats.genhyperbolic.rvs(p=1,a=c**2,b=0.0,scale=c,size=size))
        
    
    def testPars(self,c=4.0):
        xx = np.linspace(start=-4.0,stop=4.0,num=1000)
        ff = self.logpdf(xx,c)
        plt.plot(xx,ff-(-c**2*np.sqrt(1.0 + xx**2/(c**2))+c**2))
        


class RELtuningPars:
    
    def __init__(self, c=1.0, hMacro=0.1, delta=0.1, Cmin=0,Cmax=16):
        self.c = c
        self.cSq = c**2
        self.hMacro = hMacro
        self.delta = delta
        self.Cmin = Cmin
        self.Cmax = Cmax





class RELstate:
    
    def __init__(self):
        self.Ham = np.nan
        self.simMod = globalRelativisticKineticDistribution
    
    def firstEval(self,lpFun,q,tp):
        [f,g] = lpFun(q)
        self.q = q
        self.f = f
        self.g = g
        self.p = np.repeat(np.nan,len(q))
        self.Ham = np.nan
        
        
    def momentumRefresh(self,lpFun,tp):
        self.simMod.checkCurrent(d=len(self.q),c=tp.c,ngrid=2048)
        self.p = self.simMod.sample()
        self.vfac = np.sqrt(1.0+np.dot(self.p,self.p)/(tp.cSq))
        self.Ham = -self.f + tp.cSq*self.vfac
        
        
    def velocity(self):
        return((1.0/self.vfac)*self.p)
        
    
    def momentumFlip(self):
        self.p = -self.p
    
    def __str__(self):
        return ("RELState class:\nq: " + str(self.q) + "\np: " + str(self.p) + "\nHamiltonian: " + str(self.Ham))

    def __repr__(self):
        return ("RELState class:\nq: " + str(self.q) + "\np: " + str(self.p) + "\nHamiltonian: " + str(self.Ham))


class RELIstate:
    
    def __init__(self):
        self.Ham = np.nan
    
    def firstEval(self,lpFun,q,tp):
        [f,g] = lpFun(q)
        self.q = q
        self.f = f
        self.g = g
        self.p = np.repeat(np.nan,len(q))
        self.v = np.repeat(np.nan,len(q))
        self.Ham = np.nan
        
        
    def momentumRefresh(self,lpFun,tp):
        
        self.p = RELunivarMomentum().rng(size=len(self.q),c=tp.c)
        self.v = self.p/np.sqrt(1.0 + (self.p/tp.c)**2)
        self.Ham = -self.f + (tp.cSq)*np.sum(np.sqrt(1.0 + (self.p/tp.c)**2))
        
        
    def velocity(self):
        
        return(self.v)
        
    
    def momentumFlip(self):
        self.p = -self.p
        self.v = -self.v
    
    def __str__(self):
        return ("RELIState class:\nq: " + str(self.q) + "\np: " + str(self.p) + "\nHamiltonian: " + str(self.Ham))

    def __repr__(self):
        return ("RELIState class:\nq: " + str(self.q) + "\np: " + str(self.p) + "\nHamiltonian: " + str(self.Ham))



class RELintegrators(ABC):
    def RELI_GLF_rapidity_solveB(self,w0,g,h,c):
        # initial bracketing
        a = w0-1.0
        b = w0+1.0
        fac = 2.0
        notdone = np.repeat(True, len(w0))
        fa = np.zeros(len(w0))
        fb = np.zeros(len(w0))
        for it in range(10):
            fa[notdone] = w0[notdone] + h*g[notdone]*np.exp(a[notdone])/(c*(1.0+np.exp(2.0*a[notdone]))) - a[notdone]
            fb[notdone] = w0[notdone] + h*g[notdone]*np.exp(b[notdone])/(c*(1.0+np.exp(2.0*b[notdone]))) - b[notdone]
            notdone = fa*fb>0.0
            if(all(~notdone)): break
            a[notdone] -= fac
            b[notdone] += fac
            fac *= 2.0
        if(not all(~notdone)):
            print("bracket problems in RELI_GLF_rapidity_solveB")
            return(w0)
        
        # bisection seach
        for it in range(20):
            t = 0.5*(a+b)
            ft = w0 + h*g*np.exp(t)/(c*(1.0+np.exp(2.0*t))) - t
            l = ft*fa > 0.0
            a[l] = t[l]
            fa[l] = ft[l]
            b[~l] = t[~l]
            if(all(np.abs(b-a)<1.0e-4)): break

        # Newton refinement
        w = 0.5*(a+b)
        notdone = True
        for it in range(10):
            ew = np.exp(w)
            sechw = 2.0*ew/(1.0+np.exp(2.0*w))
            ft = w0 + (0.5/c)*h*g*sechw - w
            dft = (0.5*h/c)*g*sechw - (0.5*h/c)*g*ew*sechw**2 - 1.0
            delta = ft/dft
            w -= delta
            w = np.maximum(a,np.minimum(b,w))
            if(all(np.abs(delta)<1.0e-14)): 
                notdone = False
                break
        
        if(notdone): 
            print("Newton iterations problems in RELI_GLF_rapidity_solveB, max error:")
            print(np.max(ft))
        
            
        
        return(w,dft)
        

    # regular Leap frog integrator
    def LFp(self,s,lpFun,c,h,nstep=1):
        g = s.g
        q = s.q
        p = s.p
        for i in range(nstep):
            ph = p + 0.5*h*g
            M = np.sqrt(1.0 + np.dot(ph,ph)/(c**2))
            q = q + h*ph/M
            [f,g] = lpFun(q)
            p = ph + 0.5*h*g
        
        sOut = RELstate()
        sOut.q = q
        sOut.p = p
        sOut.f = f
        sOut.g = g
        sOut.vfac = np.sqrt(1.0+np.dot(p,p)/(c**2))
        sOut.Ham = -f + (c**2)*sOut.vfac
        return(sOut)
    
    
    def LFpI(self,s,lpFun,c,h,nstep=1):
        g = s.g
        q = s.q
        p = s.p
        for i in range(nstep):
            ph = p + 0.5*h*g
            v = ph/np.sqrt((ph/c)**2 + 1.0)
            q = q + h*v
            [f,g] = lpFun(q)
            p = ph + 0.5*h*g
        
        sOut = RELIstate()
        sOut.q = q
        sOut.p = p
        sOut.f = f
        sOut.g = g
        sOut.v = p/np.sqrt((p/c)**2 + 1.0)
        sOut.Ham = -f + (c**2)*np.sum(np.sqrt(1.0 + (p/c)**2))
        return(sOut)
    
    
            
            
    def LFrI(self,s,lpFun,c,h,nstep=1):
        g = s.g
        q = s.q
        p = s.p
        tmp = np.sqrt(c**2 + p**2)
        w = 0.5*np.log((tmp+p)/(tmp-p))
        wlJac = -np.sum(np.log(tmp)) 
        
        wjac = np.repeat(1.0, len(q))
        for i in range(nstep):
            (wh,dft) = self.RELI_GLF_rapidity_solveB(w,g,h,c)
            wjac *= dft
            e2w = np.exp(2.0*wh)
            q = q+h*c*(e2w-1.0)/(e2w+1.0)
            [f,g] = lpFun(q)
            w = wh + (h/c)*g*np.exp(wh)/(1.0+e2w)
            
            sechwh = 1.0/np.cosh(wh)
            wjac *= 1.0  - (0.5*h/c)*g*sechwh**2*np.exp(wh) + (0.5*h/c)*g*sechwh
            
        
        p = 0.5*c*(np.exp(2.0*w)-1.0)/np.exp(w)
        
        plJac = np.sum(np.log(0.5*c*(np.exp(2.0*w)+1.0)/np.exp(w)))
        
        sOut = RELIstate()
        sOut.q = q
        sOut.p = p
        sOut.f = f
        sOut.g = g
        sOut.v = p/np.sqrt((p/c)**2 + 1.0)
        sOut.Ham = -f + (c**2)*np.sum(np.sqrt(1.0 + (p/c)**2))
        lJac = 0.0
        print((-wjac))
        print((wlJac,plJac))
        return(sOut,lJac)
        


  
 
class adaptRELstepE(RELintegrators):
    
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
        return(RELstate())
    
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
            
            sOut = self.LFp(s=s,lpFun=lpFun,c=tp.c,h=tp.hMacro/nstep,nstep=nstep)
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
                
                sTmp = self.LFp(sB,lpFun,c=tp.c,h=tp.hMacro/nstep,nstep=nstep)
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
        
        lwt = 0.0 if Ib==If else LOG_ZERO_THRESH
        
        
        return(sOut,lwt)
        
        
        
class adaptRELIstepE(RELintegrators):
    
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
        return(RELIstate())
    
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
            
            sOut = self.LFpI(s=s,lpFun=lpFun,c=tp.c,h=tp.hMacro/nstep,nstep=nstep)
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
                
                sTmp = self.LFpI(sB,lpFun,c=tp.c,h=tp.hMacro/nstep,nstep=nstep)
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
        
        lwt = 0.0 if Ib==If else LOG_ZERO_THRESH
        
        
        return(sOut,lwt)
        

        
class adaptRELIstepRapidityE(RELintegrators):
    
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
        return(RELIstate())
    
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
            
            (sOut,lJac) = self.LFrI(s=s,lpFun=lpFun,c=tp.c,h=tp.hMacro/nstep,nstep=nstep)
            self.gradEval += nstep
            
            
            
            Eerr = np.abs(sOut.Ham-s.Ham+lJac)
            
            
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
                
                (sTmp,lJacB) = self.LFrI(sB,lpFun,c=tp.c,h=tp.hMacro/nstep,nstep=nstep)
                self.gradEval += nstep
                
                Eerr = np.abs(sTmp.Ham-sB.Ham+lJacB)
                
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
        

        
# class foo(RELintegrators):
#     def __call__(self,s,lpFun,tp,h=0.1,nstep=1):
#         return(self.LFvI(s=s,lpFun=lpFun,c=tp.c,h=h,nstep=nstep))



#tp = RELtuningPars(c=4.0,delta=1.0e-6)
#f = RELIstate()
#f.firstEval(td.stdGauss,np.array([-1.0,10.2]),tp)
#f.momentumRefresh(td.stdGauss, tp)

#s = adaptRELIstepRapidityE()

#lp = td.stdGauss
#(f1,foo) = s(f,lp,tp)

#f1.momentumFlip()

#(f0b,foo) = s(f1,lp,tp)











