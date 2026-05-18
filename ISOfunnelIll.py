import numpy as np
import isokinetic as iso 
import WASPSampler as wasps
import NUTSsampler as nuts
import WALNUTSP as walnutsp
import targets as td
import matplotlib.pyplot as plt
import pandas as pd
import arviz as az
lp = td.funnel10

np.random.seed(1)


q0 = np.random.normal(size=11)
q0[0] = 0.0


if(1==0):
    Nr = nuts.NUTSampler(debug=False,orbitStats=True)
    Wr = walnutsp.WALNUTSP(debug=False,orbitStats=True)
    Nr.run(lp,q0=q0,step=iso.adaptIKstepE(),tp0=iso.IKtuningPars(),niter=101000)
    np.save("ISOfunnelIll_NS",Nr.samples)
    np.save("ISOfunnelIll_NMAX",Nr.orbitMax)
    np.save("ISOfunnelIll_NMIN",Nr.orbitMin)
    np.save("ISOfunnelIll_NSMAX",Nr.sorbitMax)
    np.save("ISOfunnelIll_NSMIN",Nr.sorbitMin)
    Nr.diagnostics.to_csv("ISOfunnelIll_ND")
    Wr.run(lp,q0=q0,step=iso.adaptIKstepE(),tp0=iso.IKtuningPars(),niter=101000)
    np.save("ISOfunnelIll_WS",Wr.samples)
    np.save("ISOfunnelIll_WMAX",Wr.orbitMax)
    np.save("ISOfunnelIll_WMIN",Wr.orbitMin)
    np.save("ISOfunnelIll_WSMAX",Wr.orbitMax)
    np.save("ISOfunnelIll_WSMIN",Wr.orbitMin)
    Wr.diagnostics.to_csv("ISOfunnelIll_WD")


Ns = np.load("ISOfunnelIll_NS.npy")
Ws = np.load("ISOfunnelIll_WS.npy")

Nd = pd.read_csv("ISOfunnelIll_ND")
Wd = pd.read_csv("ISOfunnelIll_WD")

Nd = Nd[1000:101000]
Wd = Wd[1000:101000]


Nma = np.load("ISOfunnelIll_NMAX.npy")
Nmi = np.load("ISOfunnelIll_NMIN.npy")
Nma = Nma[:,1000:101000]
Nmi = Nmi[:,1000:101000]

Wma = np.load("ISOfunnelIll_WMAX.npy")
Wmi = np.load("ISOfunnelIll_WMIN.npy")
Wma = Wma[:,1000:101000]
Wmi = Wmi[:,1000:101000]

Nsma = np.load("ISOfunnelIll_NSMAX.npy")
Nsmi = np.load("ISOfunnelIll_NSMIN.npy")
Nsma = Nsma[:,1000:101000]
Nsmi = Nsmi[:,1000:101000]

Wsma = np.load("ISOfunnelIll_WSMAX.npy")
Wsmi = np.load("ISOfunnelIll_WSMIN.npy")
Wsma = Wsma[:,1000:101000]
Wsmi = Wsmi[:,1000:101000]


plt.figure(1,figsize=(14,4))
ax=plt.subplot(1,2,1)
plt.title("traceplot of $\\omega$, WALNUTS-I")
plt.plot(Ns[0,1000:11000])
plt.axis([0,10000,-12,12])
plt.xlabel("iteration #")
plt.ylabel("$\\omega$")
plt.subplot(1,2,2)
plt.title("traceplot of $\\omega$, WALNUTS-PI")
plt.plot(Ws[0,1000:11000])
plt.axis([0,10000,-12,12])
plt.xlabel("iteration #")
plt.ylabel("$\\omega$")

plt.savefig("ISOfunnelIll_trace.pdf")


plt.figure(2,figsize=(14,4))
ng = 64
grCenter = np.linspace(start=np.quantile(Wma[0,:],0.01), stop=np.quantile(Wma[0,:],0.99),num=ng)
hh = grCenter[1]-grCenter[0]
Wlow = 0.0*grCenter
Whigh = 0.0*grCenter
Wmed = 0.0*grCenter
Nlow = 0.0*grCenter
Nhigh = 0.0*grCenter
Nmed = 0.0*grCenter

Wotlow = 0.0*grCenter
Wothigh = 0.0*grCenter
Wotmed = 0.0*grCenter
Notlow = 0.0*grCenter
Nothigh = 0.0*grCenter
Notmed = 0.0*grCenter


Wol = Wd.h*(Wd.b-Wd.a)
Nol = Nd.h*(Nd.b-Nd.a)


Wot = Wd.h*np.abs(Wd['j'])
Not = Nd.h*np.abs(Nd.L)


for i in range(ng):
    inds = np.logical_and(Wma[0,:]<grCenter[i]+0.5*hh, Wma[0,:]>grCenter[i]-0.5*hh)
    
    Wmed[i] = np.quantile(Wol[inds], 0.5)
    Wlow[i] = np.quantile(Wol[inds], 0.25)
    Whigh[i] = np.quantile(Wol[inds], 0.75)
    
    Wotmed[i] = np.quantile(Wot[inds], 0.5)
    Wotlow[i] = np.quantile(Wot[inds], 0.25)
    Wothigh[i] = np.quantile(Wot[inds], 0.75)
    
    inds = np.logical_and(Nma[0,:]<grCenter[i]+0.5*hh, Nma[0,:]>grCenter[i]-0.5*hh)
    
    Nmed[i] = np.quantile(Nol[inds], 0.5)
    Nlow[i] = np.quantile(Nol[inds], 0.25)
    Nhigh[i] = np.quantile(Nol[inds], 0.75)
    
    Notmed[i] = np.quantile(Not[inds], 0.5)
    Notlow[i] = np.quantile(Not[inds], 0.25)
    Nothigh[i] = np.quantile(Not[inds], 0.75)
    
plt.subplot(1,4,3)
#plt.semilogy(Wma[0,:],Wd.b-Nd.a,'.r',ms=0.1)
#plt.semilogy(Nma[0,:],Nd.b-Nd.a,'.k',ms=0.1)
plt.semilogy(grCenter, Wmed,'r',label="median WALNUTS-PI")
plt.semilogy(grCenter, Wlow,'--r',lw=0.5)
plt.semilogy(grCenter, Whigh,'--r',lw=0.5)

plt.semilogy(grCenter, Nmed,'k',label="median WALNUTS-I")
plt.semilogy(grCenter, Nhigh,'--k',lw=0.5)
plt.semilogy(grCenter, Nlow,'--k',lw=0.5)
plt.title("samplable orbit time-length")
plt.xlabel("largest $\\omega$ in orbit")
plt.legend()

plt.subplot(1,4,4)
plt.semilogy(grCenter, Wotmed,'r',label="median WALNUTS-PI")
plt.semilogy(grCenter, Wotlow,'--r',lw=0.5)
plt.semilogy(grCenter, Wothigh,'--r',lw=0.5)

plt.semilogy(grCenter, Notmed,'k',label="median WALNUTS-I")
plt.semilogy(grCenter, Nothigh,'--k',lw=0.5)
plt.semilogy(grCenter, Notlow,'--k',lw=0.5)
plt.title("time-length move")
plt.xlabel("largest $\\omega$ in orbit")
plt.legend()


plt.subplot(1,4,1)
plt.hist(Wma[0,:]-Wmi[0,:],30,density=True,label="WALNUTS-PI")
plt.hist(Nma[0,:]-Nmi[0,:],30,density=True,histtype='step',lw=2,label="WALNUTS-I")
plt.axis([0,20,0,0.4])
plt.title("all int. range in $\\omega$-direction")
plt.xlabel("orbit $\\omega_{\\max}-\\omega_{\\min}$")
plt.legend()



plt.subplot(1,4,2)
plt.hist(Wsma[0,:]-Wsmi[0,:],30,density=True,label="WALNUTS-PI")
plt.hist(Nsma[0,:]-Nsmi[0,:],30,density=True,histtype='step',lw=2,label="WALNUTS-I")
plt.axis([0,20,0,0.4])
plt.title("samplable orbit range $\\omega$-direction")
plt.xlabel("orbit $\\omega_{\\max}-\\omega_{\\min}$")
plt.legend()


plt.savefig("ISOfunnelIll_orbit.pdf")


print(az.ess(Ws[0,1000:101000]))
print(az.ess(Ns[0,1000:101000]))
print(az.ess(Ws[0,1000:101000])/az.ess(Ns[0,1000:101000]))
print("----")
print(np.mean(Wd.gradEvals))
print(np.mean(Nd.gradEvals))
print(np.mean(Wd.gradEvals)/np.mean(Nd.gradEvals))

print("----")
print(np.mean(np.maximum(-Wd.a,Wd.b)/(-Wd.a+Wd.b)))

print("----")
print((az.ess(Ws[0,1000:101000])/np.mean(Wd.gradEvals))/(az.ess(Ns[0,1000:101000])/np.mean(Nd.gradEvals)))


