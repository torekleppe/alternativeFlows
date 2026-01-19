import numpy as np
import scipy as sp
import targets as td
import relativistic as rel
import matplotlib.pyplot as plt

eta = 0.1
c = 1.5

lp = td.smileDistr #td.corrGauss #td.funnel1



def ode(t,y):
    d = (len(y)-1)//2
    q = y[0:d]
    p = y[d:(2*d)]
    o = y[-1]
    
    v = p/np.sqrt(1.0 + (p/c)**2)
    
    [f,g,Hv] = lp(q,v=v)
    
    return(np.concat((np.exp(o)*v,
                      np.exp(o)*g,
                      np.array([-np.exp(o)*np.dot(g,Hv)/(eta+np.dot(g,g))]))))


def Ham(y):
    d = (len(y)-1)//2
    q = y[0:d]
    p = y[d:(2*d)]
    o = y[-1]
    [f,g] = lp(q)
    return(-f + c**2*np.sum(np.sqrt(1.0+(p/c)**2)) + 0.5*(o + 0.5*np.log(eta + np.dot(g,g)))**2)
    

d = 2
Tmax = 100.0
q0 = np.random.normal(size=d)
[f,g] = lp(q0)
p0 = rel.RELunivarMomentum().rng(size=d,c=c)
o0 = -0.5*np.log(eta + np.dot(g,g)) + 0.1*np.random.normal()
y0 = np.concat((q0,p0,np.array([o0])))
               

out = sp.integrate.solve_ivp(ode, (0.0,Tmax), y0, t_eval=np.linspace(0, Tmax,num=100),atol=1e-10,rtol=1e-10)

Hams = np.zeros_like(out.t)
for i in range(len(out.t)):
    Hams[i] = Ham(out.y[:,i])

plt.subplot(1,3,1)
plt.plot(out.y[0,:],out.y[1,:],'.-')
plt.plot(out.y[0,0],out.y[1,0],'.g')

plt.subplot(1,3,2)
plt.plot(out.y[-1,:])

plt.subplot(1,3,3)
plt.plot(Hams)