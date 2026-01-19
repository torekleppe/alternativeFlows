import numpy as np
import scipy as sp
import targets as td
import matplotlib.pyplot as plt

lp = td.stdGauss
alpha = 1.0
Qsd = 1.0

def G(p,omega,g):
    return -alpha*2.0*p/(1.0+p**2)*np.exp(omega)*g

def Q(p,omega):
    return (0.5/(Qsd**2))*(omega + alpha*np.log(1.0+p**2))**2


def ode(t,z):
    d = len(z)//3

    q = z[0:d]
    p = z[d:(2*d)]
    omega = z[(2*d):(3*d)]
    
    [f,g] = lp(q)
    eOmega = np.exp(omega)
    GG = G(p,omega,g)
    
    return(np.concat((eOmega*p,eOmega*g,GG)))


def Ham(z):
    d = len(z)//3

    q = z[0:d]
    p = z[d:(2*d)]
    omega = z[(2*d):(3*d)]
    
    [f,g] = lp(q)
    return( - f + 0.5*np.dot(p,p) + np.sum(Q(p,omega)))
    

Tmax = 1.6
niter = 10000
d = 2
q = np.random.normal(size=d)

samples = np.zeros((d,niter))

for i in range(niter):
    p = np.random.normal(size=d)
    omega = ( -alpha*np.log(1.0+p**2) + Qsd*np.random.normal(size=d))
    y0 = np.concat( (q,p,omega))
    out = sp.integrate.solve_ivp(ode, t_span=(0.0,Tmax), y0=y0,atol=1.0e-6,rtol=1.0e-6)
    
    q = out.y[0:d,-1]
    samples[:,i] = q
    
