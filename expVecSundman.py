import numpy as np
import scipy as sp
import targets as td
import matplotlib.pyplot as plt

lp = td.stdGauss
alpha = 1.0
Qsd = 0.1

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
    #return( - f + 0.5*np.dot(p,p) + np.sum(Q(p,omega)))
    
    return(np.sum(Q(p,omega)))
    

Tmax = 100.0


q0 = np.random.normal(size=2)
p0 = np.random.normal(size=2)

omega0 = -alpha*np.log(1.0+p0**2) + Qsd*np.random.normal(size=2)

y0 = np.concat( (q0,p0,omega0))

out = sp.integrate.solve_ivp(ode, t_span=(0.0,Tmax), y0=y0, t_eval=np.linspace(0, Tmax,num=100),atol=1.0e-10,rtol=1.0e-10)

plt.subplot(1,3,1)
plt.plot(out.y[0,:],out.y[1,:],'.-')


y1 = out.y[:,-1]
y1[2:4] = y1[2:4]
out2 = sp.integrate.solve_ivp(ode, t_span=(0.0,Tmax), y0=y0, t_eval=np.linspace(0, Tmax,num=100),atol=1.0e-10,rtol=1.0e-10)
plt.plot(out2.y[0,:],out2.y[1,:],'.-')

plt.subplot(1,3,2)
plt.plot(out.y[4,:],out.y[5,:],'.-')


ham = np.zeros_like(out.t)
for i in range(len(out.t)):
    ham[i] = Ham(out.y[:,i]) 

plt.subplot(1,3,3)
plt.plot(ham-ham[0])
