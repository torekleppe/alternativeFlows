import numpy as np
import targets as td
import matplotlib.pyplot as plt



b = 0.1
#a = np.exp(0.5*b)
c = 1.0
alpha = 0.9

def Q(q,p,g):
    #return(np.exp((0.5/alpha)*p**2))
    #return((1.0+(p/c)**2)**alpha)
    return((1.0+b*np.dot(g,g))**alpha)
    
def G(lpFun,q,p,g):
    #return(p*g/alpha)
    #return(2.0*alpha*p*g/(c**2+p**2))
    qdot = p/np.sqrt(1.0+(p/c)**2)
    [_,_,Hp] = lpFun(q,v=qdot)
    return(2.0*alpha*b*np.dot(Hp,g)/(1.0+b*np.dot(g,g)))




def integrate(lpFun,q0,p0,rho0,h=0.1,nstep=100):
    
    q = q0
    p = p0
    rho = rho0
    [f,g] = lpFun(q)
    
    H0 = -f + 0.5*np.dot(p,p)
    Hr0 = -rho/Q(q,p,g)
    rhos = np.zeros((d,nstep+1))
    Qs = np.zeros((d,nstep+1))
    ps = np.zeros((d,nstep+1))
    qs = np.zeros((d,nstep+1))
    lwts = np.zeros(nstep+1)
    rhos[:,0] = rho
    Qs[:,0] = Q(q,p,g)
    ps[:,0] = p
    qs[:,0] = q
    lwts[0] = -np.sum(np.log(Q(q,p,g)))
    for i in range(nstep):
        rhoh = rho + 0.5*h*G(lpFun,q,p,g)
        ph = p + 0.5*h/rhoh*g
        q = q + (h/rhoh)*ph
        [f,g] = lpFun(q)
        p = ph + 0.5*h/rhoh*g
        rho = rhoh + 0.5*h*G(lpFun,q,p,g)
        rhos[:,i+1] = rho
        Qs[:,i+1] = Q(q,p,g)
        ps[:,i+1] = p
        qs[:,i+1] = q
        lwts[i+1] = -np.sum(np.log(Q(q,p,g)))
        
        
    plt.subplot(2,2,1)
    plt.plot(1/rhos[0,:])    
    plt.plot(1/rhos[1,:])
    plt.subplot(2,2,2)
    plt.plot(Qs[0,:]/rhos[0,:])
    plt.subplot(2,2,3)
    plt.plot(qs[0,:],qs[1,:],'.-')
    plt.subplot(2,2,4)
    plt.plot(lwts,'.-')
    H1 = -f + + 0.5*np.dot(p,p)
    Hr1 = -rho/Q(q,p,g)
    print(H0-H1)
    print(np.max(np.abs(Hr0-Hr1)))
    
    return(q,p,rho)
    
    return(q,p,rho)

def integrateREL(lpFun,q0,p0,rho0,h=0.01,nstep=1000):
    
    q = q0
    p = p0
    rho = rho0
    [f,g] = lpFun(q)
    
    H0 = -f + np.sum(c**2*np.sqrt(1.0+(p/c)**2))
    Hr0 = -rho/Q(q,p,g)
    rhos = np.zeros((d,nstep+1))
    Qs = np.zeros((d,nstep+1))
    ps = np.zeros((d,nstep+1))
    qs = np.zeros((d,nstep+1))
    lwts = np.zeros(nstep+1)
    rhos[:,0] = rho
    Qs[:,0] = Q(q,p,g)
    ps[:,0] = p
    qs[:,0] = q
    lwts[0] = -np.sum(np.log(Q(q,p,g)))
    for i in range(nstep):
        rhoh = rho + 0.5*h*G(lpFun,q,p,g)
        ph = p + 0.5*h/rhoh*g
        q = q + (h/rhoh)*ph/np.sqrt(1.0+(ph/c)**2)
        [f,g] = lpFun(q)
        p = ph + 0.5*h/rhoh*g
        rho = rhoh + 0.5*h*G(lpFun,q,p,g)
        rhos[:,i+1] = rho
        Qs[:,i+1] = Q(q,p,g)
        ps[:,i+1] = p
        qs[:,i+1] = q
        lwts[i+1] = -np.sum(np.log(Q(q,p,g)))
        
        
    plt.subplot(2,2,1)
    plt.plot(1/rhos[0,:])    
    plt.plot(1/rhos[1,:])
    plt.subplot(2,2,2)
    plt.plot(Qs[1,:]/rhos[1,:])
    
    plt.subplot(2,2,3)
    plt.plot(qs[0,:],qs[1,:],'.-')
    plt.subplot(2,2,4)
    plt.plot(lwts,'.-')
    H1 = -f + np.sum(c**2*np.sqrt(1.0+(p/c)**2))
    Hr1 = -rho/Q(q,p,g)
    print(H0-H1)
    print(np.max(np.abs(Hr0-Hr1)))
    
    return(q,p,rho)
    
    
        
d = 2
lp = td.modFunnel # td.corrGauss #      td.funnel1 # 

q0 = np.array([-2.0,0.01]) # np.random.normal(size=d) # 

p0 = np.random.normal(size=d)

[f0,g0] = lp(q0)

rho0 = Q(q0,p0,g0) # -Q(q0,p0)*np.log(np.random.uniform(size=d))




(q1,p1,rho1) = integrateREL(lp,q0,p0,rho0)

(q0b,p0b,rho0b) = integrateREL(lp,q1,-p1,rho1)


