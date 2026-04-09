import numpy as np
import hamiltonian as hmc
import isokinetic as iso
import noseHoover as nh
import scalarNoseHoover as snh
import kineticLangevinSampler as kls
import WASPSampler as wasp
import targets as td


lp = td.funnel10 #rescaled

q0 = np.random.normal( size = 11)
q0[0] = 0.0

def gen(q):
    return(np.array([q[0],q[0]**2,np.sum((np.exp(9/4)*q[1:11])**2)/10.0]))
    



samIK = kls.kineticLangevinSampler()
samIK.run(lp, q0,generated=gen,step=hmc.adaptHMCstepE(),tp0=hmc.HMCtuningPars(), niter=2000,nwarmup=1000,nthin=128,gamma=0.5)


samIKI = kls.kineticLangevinSampler()
samIKI.run(lp, q0,generated=gen,step=iso.adaptIKstepE(),tp0=iso.IKtuningPars(), niter=2000,nwarmup=1000,nthin=128,gamma=0.5)


samW = wasp.WASPSampler()
samW.run(lp, q0,generated=gen,step=hmc.adaptHMCstepE(),tp0=hmc.HMCtuningPars(Cmin=2), niter=2000,nwarmup=1000)


samWI = wasp.WASPSampler()
samWI.run(lp, q0,generated=gen,step=iso.adaptIKstepE(),tp0=iso.IKtuningPars(Cmin=2), niter=2000,nwarmup=1000)


samNH = wasp.WASPSampler()
samNH.run(lp, q0,generated=gen,step=nh.adaptNHstepE(),tp0=nh.NHtuningPars(Cmin=2,oSigmaSq = 2.0**2), niter=2000,nwarmup=1000)



samSNH = wasp.WASPSampler()
samSNH.run(lp, q0,generated=gen,step=snh.adaptSNHstepE(),tp0=snh.SNHtuningPars(Cmin=2,oSigmaSq = 4.0**2), niter=2000,nwarmup=1000)



plt.semilogx(np.cumsum(samWI.samples[2,1000:101000])/np.linspace(1,100000,num=100000))

vr = 6.0/5.0*np.exp(18.0) - np.exp(9)
mn = np.exp(9.0/2.0)

plt.semilogx(mn+2*np.sqrt(vr/np.linspace(1,100000,num=100000)),'b')
plt.semilogx(mn-2*np.sqrt(vr/np.linspace(1,100000,num=100000)),'b')
plt.semilogx(np.repeat(mn,100000),'--')

plt.ylim(-100,250)
plt.xlim(1000,100000)




