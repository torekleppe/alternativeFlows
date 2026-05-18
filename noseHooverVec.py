
import sys
import numpy as np
import cppimport.import_hook
import specFuns as sf
import noseHooverVecCore
import copy
import targets as td
import matplotlib.pyplot as plt
import pandas as pd

LOG_ZERO_THRESH = -700.0

class NHtuningPars:
    
    def __init__(self, hMacro=0.1, delta=0.1, oSigmaSq = 4.0, Cmin=0,Cmax=10):
        
        self.hMacro = hMacro
        self.delta = delta
        self.oSigmaSq = oSigmaSq
        self.Cmin = Cmin
        self.Cmax = Cmax



class NHState:
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
        self.omega = np.sqrt(tp.oSigmaSq)*np.random.normal(size=len(self.q))
        self.Ham = -self.f + 0.5*np.dot(self.p,self.p) + (0.5/tp.oSigmaSq)*np.dot(self.omega,self.omega)
    
    def momentumFlip(self):
        self.p = -self.p
        self.omega = -self.omega

    def velocity(self):
        return(self.p)
    
    def __str__(self):
        return ("NHState class:\nq: " + str(self.q) + "\np: " + str(self.p) + "\nomega: " + str(self.omega) + "\nHamiltonian: " + str(self.Ham))

    def __repr__(self):
        return ("NHState class:\nq: " + str(self.q) + "\np: " + str(self.p) + "\nomega: " + str(self.omega) + "\nHamiltonian: " + str(self.Ham))

        



class NHintegrators:


    def NH_BSstep(self,p0,o0,g,oSigmaSq=1.0,h=0.1,eps=1.0e-13):
        d = len(p0)
        if(len(o0) != d or len(g) != d):
            sys.exit("bad inputs to NH_Bstep")
        r = noseHooverVecCore.vecNH_Bstep(p0,o0,g,oSigmaSq,h,eps)
        if(len(r)<3*d):
            sys.exit("NH_Bstep failed")
        p = r[0:d]
        o = r[d:(2*d)]
        lj = np.sum(r[(2*d):(3*d)])
        return(p,o,lj)



    def integrateBS(self,s,lpFun,oSigmaSq=4.0,h=0.1,nstep=1):
        
        q = copy.deepcopy(s.q)
        p = copy.deepcopy(s.p)
        omega = copy.deepcopy(s.omega)
        g = copy.deepcopy(s.g)
        
        hhalf = 0.5*h
        ljac = 0.0
        
        #qs_ = np.zeros((len(q),nstep+1))
        #qs_[:,0] = q
        
        for i in range(nstep):
            ph,oh,ljac1 = self.NH_BSstep(p, omega, g,oSigmaSq=oSigmaSq,h=hhalf)
            q += h*ph
            [f,g] = lpFun(q)
            p,omega,ljac2 = self.NH_BSstep(ph, oh, g,oSigmaSq=oSigmaSq,h=hhalf)
            ljac += ljac1 + ljac2
            #qs_[:,i+1] = q
        
        
        #plt.plot(qs_[0,:],qs_[1,:])
                
        sOut = NHState()
        sOut.q = q
        sOut.p = p
        sOut.omega = omega
        sOut.g = g
        sOut.f = f
        sOut.Ham = -f + 0.5*np.dot(p,p) + (0.5/oSigmaSq)*np.dot(omega,omega)
        
        return(sOut,ljac)
    
    def integrateTBTATBT(self,s,lpFun,oSigmaSq=4.0,h=0.1,nstep=1):
       q = copy.deepcopy(s.q)
       p = copy.deepcopy(s.p)
       o = copy.deepcopy(s.omega)
       g = copy.deepcopy(s.g)
       
       hhalf = 0.5*h
       hquart = 0.25*h
       ljac = 0.0
       
       #qs_ = np.zeros((len(q),nstep+1))
       #qs_[:,0] = q
       #Hs = np.zeros(nstep+1)
       #Hs[0] = -s.Ham
       for i in range(nstep):
           o = o + hquart*oSigmaSq*(p*p - 1.0)
           p = p*np.exp(-0.5*h*o) + g*sf.linOdefacVector(-0.5*h,o) #- g*((np.expm1(-0.5*h*o))/o)
           
           ljac -= 0.5*h*np.sum(o)
           q += h*p
           [f,g] = lpFun(q)
           o = o + hhalf*oSigmaSq*(p*p - 1.0)
           
           p = p*np.exp(-0.5*h*o) + g*sf.linOdefacVector(-0.5*h,o) #- g*((np.expm1(-0.5*h*o))/o)
           ljac -= 0.5*h*np.sum(o)
           
           o = o + hquart*oSigmaSq*(p*p - 1.0)
           
           #qs_[:,i+1] = q
           #Hs[i+1] = -(-f + 0.5*np.dot(p,p) + (0.5/oSigmaSq)*np.dot(o,o)) + ljac
           
           
       #plt.figure()
       #plt.subplot(1,2,1)
       #plt.plot(qs_[0,:],qs_[1,:])
       
       #plt.subplot(1,2,2)
       #plt.plot(Hs)
       
       
       sOut = NHState()
       sOut.q = q
       sOut.p = p
       sOut.omega = o
       sOut.g = g
       sOut.f = f
       sOut.Ham = -f + 0.5*np.dot(p,p) + (0.5/oSigmaSq)*np.dot(o,o)
       return(sOut,ljac)

class adaptNHstepE(NHintegrators):
    
    def __init__(self):
        self.gradEval = 0
        self.HamErrs = []
        self.logJacs = []
        self.Ifs = []
        self.Ibs = []
        self.basic = []
        self.Cobs = []
    
    def name(self):
        return("adaptNHstepE")
    
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
        return(NHState())
    
    def propBasic(self):
        return(np.mean(self.basic))
    
    def Cobs(self):
        return(np.mean(self.Cobs))
    
    def __call__(self,s,lpFun,tp):
        
        
        
        If = tp.Cmax
        for c in range(tp.Cmin,tp.Cmax+1):
            nstep = 2**c
            (sOut,Wout) = self.integrateTBTATBT(s,lpFun,tp.oSigmaSq,h=(tp.hMacro/nstep),nstep=nstep)
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
                (sTmp,Wtmp) = self.integrateTBTATBT(sF,lpFun,tp.oSigmaSq,h=(tp.hMacro/nstep),nstep=nstep)
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
        



              


# step = adaptNHstepE()

#lp = td.smileDistr # td.corrGauss # td.modFunnel # td.funnel1 # td.stdGauss #
#s = NHState()
#q0=np.random.normal(size=2)
# q0[0] = -0.0
# q0[1] = 0.0001
#tp = NHtuningPars(hMacro=0.2)

#s.firstEval(lp, q0, tp)
#s.momentumRefresh(lp, tp)


#igr = NHintegrators()

#igr.integrateTBTATBT(s, lp,oSigmaSq=tp.oSigmaSq,h=0.01,nstep=1000)

# nn=100
# qs = np.zeros((2,nn+1))
# ll = np.zeros(nn+1)
# qs[:,0] = s.q
# ll[0] = -s.Ham

# ljacsum = 0.0

# for i in range(nn):
#     (s,ljac) = step(s,lp,tp)
#     ljacsum += ljac
#     ll[i+1] = -s.Ham + ljacsum
#     qs[:,i+1] = s.q


# plt.subplot(1,2,1)
# plt.plot(qs[0,:],qs[1,:])
# plt.subplot(1,2,2)
# plt.plot(ll)



# tests from here on
# d = 1
# p0 = np.random.normal(size=d)
# o0 = np.random.normal(size=d)
# g = np.random.normal(size=d)



# h = 0.2
# oSigmaSq = 3.0

# ff = NH_integrators()

# p1,o1,lj1=ff.NH_BSstep(p0,o0,g,oSigmaSq=oSigmaSq,h=h)



# p0b,o0b,ljb = ff.NH_BSstep(-p1,-o1,g,oSigmaSq=oSigmaSq,h=h)


# print(p0+p0b)
# print(o0+o0b)


# import scipy as sp
# p1t = 0*p1
# o1t = 0*o1
# lj = 0.0
# for i in range(len(p0)):
#     def odeS(t,y):
#         return(np.array([g[i]-y[0]*y[1],oSigmaSq*(y[0]**2-1.0),-y[1]]))
    
#     y0 = np.array([p0[i],o0[i],0])

#     outS = sp.integrate.solve_ivp(odeS,t_span=(0,h),y0=y0,rtol=1.0e-13,atol=1.0e-13)
#     p1t[i] = outS.y[0,-1]
#     o1t[i] = outS.y[1,-1]    
#     lj += outS.y[2,-1]
    

# print(p1-p1t)
# print(o1-o1t)
# print(lj-lj1)
