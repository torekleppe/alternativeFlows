import numpy as np
import copy

class AJSstate:
    
    def firstEval(self,lpFun,q,tp):
        [f,g] = lpFun(q)
        self.q = q
        self.f = f
        self.g = g
        self.p = np.repeat(np.nan,len(q))
        self.Ham = np.nan
        
        
    def momentumRefresh(self,lpFun,tp):
        self.p = np.random.normal(size=len(self.q))
        self.Ham = -self.f + 0.5*np.dot(self.p,self.p)
        
    def momentumFlip(self):
        self.p = -self.p
    
    def kineticLangevinRefresh(self,gamma,h):
        rho = np.exp(-gamma*h)
        self.p = rho*self.p + np.sqrt(1.0-rho**2)*np.random.normal(size=len(self.p))
        self.Ham = -self.f + 0.5*np.dot(self.p,self.p)
        
    def partialRefresh(self,rho):
        
        self.p = rho*self.p + np.sqrt(1.0-rho**2)*np.random.normal(size=len(self.p))
        self.Ham = -self.f + 0.5*np.dot(self.p,self.p)
    
    
    def velocity(self):
        return(self.p)
    
    def __str__(self):
        return ("AJSstate class:\nq: " + str(self.q) + "\np: " + str(self.p) + "\nHamiltonian: " + str(self.Ham))

    def __repr__(self):
        return ("AJSstate class:\nq: " + str(self.q) + "\np: " + str(self.p) + "\nHamiltonian: " + str(self.Ham))

class AJStuningPars:
    
    def __init__(self, hMacro=0.1, delta=0.1, Cmin=0,Cmax=10,ss=[0],fs=[1]):
        
        self.hMacro = hMacro
        self.delta = delta
        
        self.Cmin = Cmin
        self.Cmax = Cmax
        
        self.ss = ss
        self.fs = fs


class adaptJSstepE:
    
    def slowStep(self,lp,q0,p0,f0,g0,h,tp):
        
        ngrad = 0
        H0 = -f0 + 0.5*np.dot(p0,p0)
        cc = tp.Cmax
        for c in range(tp.Cmin,tp.Cmax+1):
            nstep = 2**c
            hh = h/nstep
            q = copy.deepcopy(q0)
            p = copy.deepcopy(p0)
            g = copy.deepcopy(g0)
            
            
            
            for i in range(nstep):
                ph = p[tp.ss] + 0.5*hh*g[tp.ss]
                q[tp.ss] = q[tp.ss] + hh*ph
                [f,g] = lp(q)
                ngrad += 1
                p[tp.ss] = ph + 0.5*hh*g[tp.ss]
            
            H1 = -f + 0.5*np.dot(p,p)
            
            if(np.abs(H0-H1)<tp.delta):
                cc = c
                break
            
        
        qOut = copy.deepcopy(q)
        pOut = copy.deepcopy(p)
        gOut = copy.deepcopy(g)
        fOut = copy.deepcopy(f)
        lwt = 0.0
            
        if(cc>tp.Cmin):
            for c in range(tp.Cmin,cc):
                nstep = 2**c
                hh = h/nstep
                q = copy.deepcopy(qOut)
                p = -copy.deepcopy(pOut)
                g = copy.deepcopy(gOut)
            
                
                for i in range(nstep):
                    ph = p[tp.ss] + 0.5*hh*g[tp.ss]
                    q[tp.ss] = q[tp.ss] + hh*ph
                    [f,g] = lp(q)
                    ngrad += 1
                    p[tp.ss] = ph + 0.5*hh*g[tp.ss]
                
                Hob = -f + 0.5*np.dot(p,p)
                
                if(np.abs(H0b-H1)<tp.delta):
                    rev = -700.0
                    break
        
        return(qOut,pOut,fOut,gOut,lwt,ngrad)



    def fastStep(self,lp,q0,p0,f0,g0,h,tp):
        
        ngrad = 0
        H0 = -f0 + 0.5*np.dot(p0,p0)
        cc = tp.Cmax
        
        
        
        for c in range(tp.Cmin,tp.Cmax+1):
            q = copy.deepcopy(q0)
            p = copy.deepcopy(p0)
            g = copy.deepcopy(g0)
            hh = h/(2**c)
            #print("forward h: " + str(hh) )
            for i in range(2**tp.Cmin):
                ph = p[tp.fs] + 0.5*hh*g[tp.fs]
                q[tp.fs] = q0[tp.fs] + hh*ph
                [f,g] = lp(q)
                ngrad += 1
                p[tp.fs] = ph + 0.5*hh*g[tp.fs]
            
            H1 = -f + 0.5*np.dot(p,p)
            
            if(np.abs(H0-H1)<tp.delta):
                cc = c
                break
            
        qOut = copy.deepcopy(q)
        pOut = copy.deepcopy(p)
        gOut = copy.deepcopy(g)
        fOut = copy.deepcopy(f)
        lwt = 0.0
        
            
        if(cc>tp.Cmin):
            print("reduced")
            for c in range(tp.Cmin,cc):
                
                q = copy.deepcopy(qOut)
                p = -copy.deepcopy(pOut)
                g = copy.deepcopy(gOut)
                hh = h/(2**c)
                #print("backward h: " + str(hh) )
                for i in range(2**tp.Cmin):
                    ph = p[tp.fs] + 0.5*hh*g[tp.fs]
                    q[tp.fs] = q0[tp.fs] + hh*ph
                    [f,g] = lp(q)
                    ngrad += 1
                    p[tp.fs] = ph + 0.5*hh*g[tp.fs]
                
                H0b = -f + 0.5*np.dot(p,p)
                
                if(np.abs(H1-H0b)<tp.delta):
                    lwt = -700.0
                    break
            
        
        return(qOut,pOut,fOut,gOut,lwt,ngrad)
        
        

    def __call__(self,s,lpFun,tp):
        (qh,ph,fh,gh,lwtS1,ngradS1) = self.slowStep(lpFun, s.q, s.p, s.f, s.g, 0.5*tp.hMacro, tp)
        (qm,pm,fm,gm,lwtF,ngradF) = self.fastStep(lpFun, qh, ph, fh, gh, tp.hMacro, tp)
        (qf,pf,ff,gf,lwtS2,ngradS2) = self.slowStep(lpFun, qm, pm, fm, gm, 0.5*tp.hMacro, tp)
        sOut = AJSstate()
        sOut.q = qf
        sOut.p = pf
        sOut.f = ff
        sOut.g = gf
        sOut.Ham = -ff + 0.5*np.dot(pf,pf)
        print((lwtS1,lwtF,lwtS2))
        return(sOut,lwtS1+lwtS2+lwtF)
        
        
        
        


import targets as td
import matplotlib.pyplot as plt

tp = AJStuningPars(hMacro=0.01)

s = AJSstate()
q0 = np.array([-3.0,-0.005])
lp = td.modFunnel
s.firstEval(lp, q0, tp)
s.momentumRefresh(lp, tp)

step = adaptJSstepE()

nn = 1000


qs = np.zeros((2,nn+1))
qs[:,0] = s.q

lwtSum = 0.0
Hs = np.zeros(nn+1)
Hs[0] = -s.Ham

for i in range(nn):
    (s,lwt) = step(s,lp,tp)
    lwtSum += lwt
    qs[:,i+1] =s.q
    Hs[i+1] = -s.Ham + lwtSum

plt.subplot(1,2,1)
plt.plot(qs[0,:],qs[1,:])
plt.subplot(1,2,2)
plt.plot(Hs)

#(s1,lwt) = step(s,lp,tp)

#s1.momentumFlip()

#(s0b,lwtb) = step(s1,lp,tp)

#print(s.Ham-s1.Ham)



