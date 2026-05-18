import numpy as np
import targets as td
import copy
import pandas as pd
import matplotlib.pyplot as plt
import copy
import MCMCutils as ut




def JsplitAdaptSlow(lp,q0,p0,indSlow,h,deltaSlow):
    
    [f0,g0] = lp(q0)
    
    H0 = -f0 + 0.5*np.sum(p0[indSlow]**2)
    
    cMax = 11
    cc = cMax
    for c in range(cMax+1):
        nstep = 2**c
        hh = h/nstep
        g = copy.deepcopy(g0)
        q = copy.deepcopy(q0)
        p = copy.deepcopy(p0)
        
        for i in range(nstep):
            ph = p[indSlow] + 0.5*hh*g[indSlow]
            q[indSlow] = q[indSlow] + hh*ph
            [f,g] = lp(q)
            p[indSlow] = ph + 0.5*hh*g[indSlow]
            
        H1 = -f +  0.5*np.sum(p[indSlow]**2)

        if(np.abs(H0-H1)<deltaSlow):
            print(c)
            cc = c
            break
    
    
    return q,p
        
        
def JsplitAdaptFast(lp,q0,p0,indFast,h,deltaFast):
    
    [f0,g0] = lp(q0)
    
    
    
    H0 = -f0 + 0.5*np.sum(p0**2)
    Cmax = 10
    cc = Cmax
    
    for c in range(Cmax+1):

        hc = h/(4**c)        
        
        hh = 0.5*hc
        q = copy.deepcopy(q0)
        p = copy.deepcopy(p0)
        g = copy.deepcopy(g0)
        
        
    
        ph = p[indFast] + 0.5*hh*g[indFast]
        q[indFast] = q[indFast] + hh*ph
    
        [f,g] = lp(q)
        
        g1 = copy.deepcopy(g)
        
        p[indFast] = ph + 0.5*hh*g[indFast]
    
        ph = p[indFast] + 0.5*hh*g[indFast]
        q[indFast] = q[indFast] + hh*ph
    
        [f,g] = lp(q)
        
        g2 = copy.deepcopy(g)
        
        p[indFast] = ph + 0.5*hh*g[indFast]
        
        
        H1 = -f + 0.5*np.sum(p**2)
        
        Ih = 0.5*hc*(g0 + g2)
        I2h = hh*(0.5*g0 + g1 + 0.5*g2)
        Isimp = (4.0/3.0)*I2h - (1.0/3.0)*Ih
        
        err = np.abs(np.linalg.norm(Isimp)-np.linalg.norm(I2h))
        
        print(err)
        if(err<deltaFast) : # and np.abs(H0-H1)<0.1):
            cc = c
            print("fast: " + str(c))
            break
        
        
        
    
    
    qout = copy.deepcopy(q)
    pout = copy.deepcopy(p)
    gout = copy.deepcopy(g)
    
    
    if(cc>0):
        
        for c in range(cc+1):
            hc = h/(4**c)        
            
            hh = 0.5*hc
            q = copy.deepcopy(qout)
            p = -copy.deepcopy(pout)
            g = copy.deepcopy(gout)
            
            
        
            ph = p[indFast] + 0.5*hh*g[indFast]
            q[indFast] = q[indFast] + hh*ph
        
            [f,g] = lp(q)
            
            g1 = copy.deepcopy(g)
            
            p[indFast] = ph + 0.5*hh*g[indFast]
        
            ph = p[indFast] + 0.5*hh*g[indFast]
            q[indFast] = q[indFast] + hh*ph
        
            [f,g] = lp(q)
            
            g2 = copy.deepcopy(g)
            
            p[indFast] = ph + 0.5*hh*g[indFast]
            
            
            H0b = -f + 0.5*np.sum(p**2)
            
            Ih = 0.5*hc*(gout + g2)
            I2h = hh*(0.5*gout + g1 + 0.5*g2)
            Isimp = (4.0/3.0)*I2h - (1.0/3.0)*Ih
            
            
            err = np.abs(np.linalg.norm(Isimp)-np.linalg.norm(I2h))
            
            print(err)
            
            if(err<deltaFast) : # and np.abs(H0-H1)<0.1):
                cc = c
                print("fast: " + str(c))
                break
    
    
    
    return(qout,pout)




    
def JsplitAdapt(lp,q0,p0,indFast,indSlow,hfast=0.15,hslow=0.3,nstep=1,deltaFast=0.001,deltaSlow=0.1):
    
    
    q = copy.deepcopy(q0)
    p = copy.deepcopy(p0)
    
    [f,g] = lp(q)
    H0 = -f + 0.5*np.sum(p**2)
    
    hslowHalf = 0.5*hslow
    
    qs = np.zeros((len(q),nstep+1))
    qs[:,0] = q
    
    Hs = np.zeros(nstep+1)
    Hs[0] = H0
    
    
    for s in range(nstep):
        (q,p) = JsplitAdaptSlow(lp, q, p, indSlow, hslowHalf, deltaSlow)
        (q,p) = JsplitAdaptFast(lp, q, p, indFast, hfast, deltaFast)
        (q,p) = JsplitAdaptSlow(lp, q, p, indSlow, hslowHalf, deltaSlow)
        
        qs[:,s+1] = q
        [f,g] = lp(q)
        
        Hs[s+1] = -f + 0.5*np.sum(p**2)
    
    [f,g] = lp(q)
    H1 = -f + 0.5*np.sum(p**2)
    
    print(H0-H1)
    
    plt.subplot(1,2,1)
    plt.plot(qs[0,:],qs[1,:])
    plt.subplot(1,2,2)
    plt.plot(Hs)
    






def Jsplit(lp,q0,p0,indFast,indSlow,hfast=0.25,hslow=0.25,nstep=32):
    hh = 0.5*hslow
    
    q = copy.deepcopy(q0)
    p = copy.deepcopy(p0)
    
    [f0,g] = lp(q)
    
    qs = np.zeros((len(q),nstep+1))
    qs[:,0] = q
    
    for i in range(nstep):
        
        ph = p[indSlow] + 0.5*hh*g[indSlow]
        q[indSlow] = q[indSlow] + hh*ph
        
        [f,g] = lp(q)
        p[indSlow] = ph + 0.5*hh*g[indSlow]
    
        ph = p[indFast] + 0.5*hfast*g[indFast]
        q[indFast] = q[indFast] + hfast*ph
        
        [f,g] = lp(q)
        p[indFast] = ph + 0.5*hfast*g[indFast]
        
        ph = p[indSlow] + 0.5*hh*g[indSlow]
        q[indSlow] = q[indSlow] + hh*ph
        
        [f,g] = lp(q)
        p[indSlow] = ph + 0.5*hh*g[indSlow]
        
        qs[:,i+1] = q
        
    la = -f0 + 0.5*np.dot(p0,p0) - (-f + 0.5*np.dot(p,p))
    plt.plot(qs[0,:],qs[1,:])   
    
    return(q,la)


lp = td.modFunnel # td.corrGauss #td.smileDistr # td.funnel1 # td.funnel10 #td.modFunnel #td.stdGauss #

q0 = np.array([-2.2,0.0]) #np.random.normal(size=2) # n  #
p0 = np.random.normal(size=2) # np.array([-0.5,-1.0,0.])

indFast = np.array([1])
indSlow = np.array([0])

JsplitAdapt(lp , q0, p0, indFast, indSlow, nstep=100)

#JsplitAdapt(lp , q0, p0, indFast, indSlow, nstep=100, deltaFast=1e100)



#niter = 20000
#qs = np.zeros((len(q0),niter))

#q = q0

#for i in range(niter):
#    p = np.random.normal(size=len(q))
#    (qp,la) = Jsplit(lp,q,p,indFast,indSlow)
#    
#    if(np.random.uniform()< np.exp(min(0,la))):
#        q = qp
#    
#    qs[:,i] = q