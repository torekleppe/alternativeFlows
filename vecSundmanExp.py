



import numpy as np
import scipy as sp
import targets as td


alpha = 0.5
ostd = 0.1



def rr(p):
    return -alpha*np.log(1.0+p**2)

def dr(p):
    return -2.0*alpha*p/(1.0+p**2)

def G(p,g,omega):
    return dr(p)*np.exp(omega)*g

def Q(p,omega):
    return (0.5/ostd**2)*(omega + alpha*np.log(1.0+p**2))**2


def lambertW(x):
    pass
    


def intStep(lpFun,q0,p0,o0,h=0.1):
    
    [f,g] = lpFun(q0)
    
    H0 = -f + 0.5*np.dot(p0,p0) + np.sum(Q(p0,o0))
    
    aa = 0.5*h*g*dr(p0)
    print(-aa*np.exp(o0)< -1.0/np.exp(1.0))
    W = sp.special.lambertw(-aa*np.exp(o0),tol=1.0e-14).real
    oh = o0 - W
    
    ljac1 = -np.sum(np.log(1.0+W))
    
    Eo = np.exp(oh)
    ph = p0 + 0.5*h*Eo*g
    q = q0 + h*Eo*ph
    [f,g] = lpFun(q)
    p = ph + 0.5*h*Eo*g
    GG = G(p,g,oh)
    ljac2 = np.sum(np.log(1.0+0.5*h*GG))
    o = oh + 0.5*h*GG
    
    H1 = -f + 0.5*np.dot(p,p) + np.sum(Q(p,o))
    print(H0-H1)
    print(ljac1+ljac2)
    return (q,p,o)


lp = td.modFunnel

q0 = np.random.normal(size=2)

p0 = np.random.normal(size=2)

o0 = rr(p0) + ostd*np.random.normal(size=2)

(q,p,o) = intStep(lp, q0, p0, o0)


(q0b,p0b,o0b) = intStep(lp, q, -p, o)

print(q0-q0b)
print(p0+p0b)
print(o0-o0b)