import numpy as np
import targets as td
import copy
import pandas as pd
import matplotlib.pyplot as plt
import copy
import MCMCutils as ut






def Jsplit(lp,q0,p0,indFast,indSlow,hfast=0.15,hslow=0.3,nstep=32):
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


lp = td.smileDistr

q0 = np.array([-0.5,0.3])
#p0 = np.array([-0.5,-1.0,0.])

indFast = np.array([1])
indSlow = np.array([0])


niter = 20000
qs = np.zeros((len(q0),niter))

q = q0

for i in range(niter):
    p = np.random.normal(size=len(q))
    (qp,la) = Jsplit(lp,q,p,indFast,indSlow)
    
    if(np.random.uniform()< np.exp(min(0,la))):
        q = qp
    
    qs[:,i] = q