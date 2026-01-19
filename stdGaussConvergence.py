import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import targets as td


d = 10000

def odeIK(t,y):
    dd = len(y)//2
    x = y[0:dd]
    u = y[dd:(2*dd)]
    g = -x
    
    F = (1.0/(dd-1))*(g - np.dot(g, u)*u)
    return(np.concat((u,F)))


def odeH(t,y):
    dd = len(y)//2
    x = y[0:dd]
    v = y[dd:(2*dd)]
    g = -x
    return(np.concat((v,g)))

np.random.seed(1)

q0 = np.random.normal(size=d)
v0 = np.random.normal(size=d)
Tmax = 3*2.0*np.pi


dd = 10

y0 = np.concat((q0[0:dd],v0[0:dd]))
Hout10 = sp.integrate.solve_ivp(odeH, t_span=(0.0,Tmax), y0=y0, rtol=1.0e-6,atol=1.0e-6)

u0 = v0[0:dd]/np.linalg.norm(v0[0:dd])

IKy0 = np.concat((q0[0:dd],u0))
IKout10 = sp.integrate.solve_ivp(odeIK, t_span=(0.0,Tmax*np.sqrt(dd-1)), y0=IKy0, rtol=1.0e-6,atol=1.0e-6)

plt.subplot(1,4,1)
plt.plot(Hout10.t,Hout10.y[0,:],'.-',label="Hamiltonian")
plt.plot(IKout10.t/np.sqrt(dd-1),IKout10.y[0,:],label="Isokinetic")
plt.legend()
plt.title("d=10")
plt.axis([0,Tmax,-3,3])

dd = 100

y0 = np.concat((q0[0:dd],v0[0:dd]))
Hout10 = sp.integrate.solve_ivp(odeH, t_span=(0.0,Tmax), y0=y0, rtol=1.0e-6,atol=1.0e-6)

u0 = v0[0:dd]/np.linalg.norm(v0[0:dd])

IKy0 = np.concat((q0[0:dd],u0))
IKout10 = sp.integrate.solve_ivp(odeIK, t_span=(0.0,Tmax*np.sqrt(dd-1)), y0=IKy0, rtol=1.0e-6,atol=1.0e-6)

plt.subplot(1,4,2)
plt.plot(Hout10.t,Hout10.y[0,:],'.-',label="Hamiltonian")
plt.plot(IKout10.t/np.sqrt(dd-1),IKout10.y[0,:],label="Isokinetic")
plt.axis([0,Tmax,-3,3])



plt.title("d=100")



dd = 1000

y0 = np.concat((q0[0:dd],v0[0:dd]))
Hout10 = sp.integrate.solve_ivp(odeH, t_span=(0.0,Tmax), y0=y0, rtol=1.0e-6,atol=1.0e-6)

u0 = v0[0:dd]/np.linalg.norm(v0[0:dd])

IKy0 = np.concat((q0[0:dd],u0))
IKout10 = sp.integrate.solve_ivp(odeIK, t_span=(0.0,Tmax*np.sqrt(dd-1)), y0=IKy0, rtol=1.0e-6,atol=1.0e-6)

plt.subplot(1,4,3)
plt.plot(Hout10.t,Hout10.y[0,:],'.-',label="Hamiltonian")
plt.plot(IKout10.t/np.sqrt(dd-1),IKout10.y[0,:],label="Isokinetic")

plt.title("d=1000")
plt.axis([0,Tmax,-3,3])

dd = 10000

y0 = np.concat((q0[0:dd],v0[0:dd]))
Hout10 = sp.integrate.solve_ivp(odeH, t_span=(0.0,Tmax), y0=y0, rtol=1.0e-6,atol=1.0e-6)

u0 = v0[0:dd]/np.linalg.norm(v0[0:dd])

IKy0 = np.concat((q0[0:dd],u0))
IKout10 = sp.integrate.solve_ivp(odeIK, t_span=(0.0,Tmax*np.sqrt(dd-1)), y0=IKy0, rtol=1.0e-6,atol=1.0e-6)

plt.subplot(1,4,4)
plt.plot(Hout10.t,Hout10.y[0,:],'.-',label="Hamiltonian")
plt.plot(IKout10.t/np.sqrt(dd-1),IKout10.y[0,:],label="Isokinetic")

plt.title("d=10000")
plt.axis([0,Tmax,-3,3])




