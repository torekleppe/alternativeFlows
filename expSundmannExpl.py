import scipy as sp
import numpy as np
import targets as td
import matplotlib.pyplot as plt


lp =  td.funnel10
sigma = 0.2
alpha = 1.0
beta = 0.5


def ell(p):
    return(beta*(-0.5*np.log(alpha+p**2)))

def dell(p):
    return(beta*(-p/(p**2+alpha)))


def odeSys(t,y):
    d = (len(y)-1)//3
    
    q = y[0:d]
    p = y[d:(2*d)]
    omega = y[(2*d):(3*d)]
    
    
    [f,g] = lp(q)
    Rv = np.exp(omega)
    L = dell(p)
    
    
    dq = Rv*p
    dp = Rv*g
    domega = Rv*L*g
    
    return(np.concatenate((dq, dp,domega,np.array([np.sum(domega)]))))


def conserved(y):
    d = (len(y)-1)//3
    
    q = y[0:d]
    p = y[d:(2*d)]
    omega = y[(2*d):(3*d)]
    
    [f,g] = lp(q)
    
    Q = omega - ell(p)
    
    return((-f + 0.5*np.dot(p,p) + 0.5/sigma**2*np.dot(Q,Q),Q))


q0 = np.random.normal(size=11)
q0[0] = 0.0
p0 = np.random.normal(size=len(q0))
o0 = ell(p0) + sigma*np.random.normal(size=len(q0))
y0 = np.concatenate((q0,p0,o0,np.zeros(1)))

T = 10.0

out = sp.integrate.solve_ivp(odeSys, [0,T], y0,t_eval=np.linspace(0, T, num=100),atol=1.e-12,rtol=1.0e-12)

cc = np.zeros_like(out.t)
QQ = np.zeros((len(q0),len(out.t)))

for i in range(len(out.t)):
    (cc[i],QQ[:,i]) = conserved(out.y[:,i])
d = len(q0)

plt.subplot(2,2,1)
plt.plot(out.y[0,:],out.y[1,:],'.')
plt.subplot(2,2,2)
plt.plot(out.y[-1,:])
plt.subplot(2,2,3)
plt.plot(cc)
plt.subplot(2,2,4)
plt.plot(out.y[2*d,:],out.y[2*d+1,:],'.-')





y0b = out.y[:,-1].copy()
y0b[-1] = 0.0
y0b[d:(2*d)] = -y0b[d:(2*d)]

print(out.y[-1,-1])

outb = sp.integrate.solve_ivp(odeSys, [0,T], y0b,t_eval=np.linspace(0, T, num=100),atol=1.e-12,rtol=1.0e-12)

plt.subplot(2,2,1)
plt.plot(outb.y[0,:],outb.y[1,:],'r')
print(outb.y[-1,-1])