
import numpy as np
import targets as td
import copy
import matplotlib.pyplot as plt
import arviz as az




def integrate(lp,q0,p0,inds,h=0.01,nstep=10):
    
    q = copy.deepcopy(q0)
    p = copy.deepcopy(p0)
    [f,g] = lp(q)
    
    E = np.zeros(nstep+1)
    E[0] = f
    qs = np.zeros((len(q),nstep+1))
    qs[:,0] = q
    ph = np.zeros_like(p)
    ljac = 0.0
    #print(inds)
    for i in range(nstep):
        for j in range(len(inds)):
            gnorm = np.linalg.norm(g[inds[j]])
            delta = 0.5*h/(len(inds[j])-1)*gnorm
            ee = (1.0/gnorm)*g[inds[j]]
            
            sh = np.sinh(delta)
            ch = np.cosh(delta)
            ip = np.dot(ee,p[inds[j]])
            
            fac1 = 1.0/(ch+sh*ip)
            
            ph[inds[j]] = fac1*p[inds[j]] + fac1*(sh+ip*(ch-1.0))*ee
            ljac += (len(inds[j])-1)*np.log(fac1)
            
            
        
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
            ljac += (len(inds[j])-1)*np.log(fac1)
            
            
        
        E[i+1] = f + ljac
        qs[:,i+1] = q
        alpha = np.exp(min(0.0,E[-1]-E[0]))
    #plt.figure()
    #plt.subplot(1,2,1)
    #plt.plot(E)
    #plt.subplot(1,2,2)
    #plt.plot(qs[0,:],qs[1,:])
    return((q,p,ljac,alpha))
    
    
lp = td.zeroMeanAR1

s = 9
inds = [np.array([0,1,2,3])]
#np.random.seed(s)
q0 = np.random.normal(size=4) #np.array([0.3,0.35,0.33,0.36])
q = copy.deepcopy(q0)
niter = 10000
qs = np.zeros((len(q0),niter))
qsb = np.zeros((len(q0),niter))

for i in range(niter):
    p0 = np.random.normal(size=len(q0))
    for j in range(len(inds)):
        p0[inds[j]] = (1.0/np.linalg.norm(p0[inds[j]]))*p0[inds[j]]
        
    (q1,p1,lj1,alpha)=integrate(lp,q,p0,inds,h=0.1,nstep=30)
    
    if(np.random.uniform()<alpha):
        q = copy.deepcopy(q1)
    qs[:,i] = q
        
        
        


inds = [np.array([0,2]),np.array([1,3])]

q = copy.deepcopy(q0)
for i in range(niter):
    p0 = np.random.normal(size=len(q0))
    for j in range(len(inds)):
        p0[inds[j]] = (1.0/np.linalg.norm(p0[inds[j]]))*p0[inds[j]]
        
    (q1,p1,lj1,alpha)=integrate(lp,q,p0,inds,h=0.1,nstep=30)
    if(np.random.uniform()<alpha):
        q = copy.deepcopy(q1)
    qsb[:,i] = q
        
print(az.ess(qs[0,:]))
print(az.ess(qsb[0,:]))


