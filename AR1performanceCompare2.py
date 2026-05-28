# script showing difference in performance for AR(1) model


import targets as td
import numpy as np

import NUTSsampler as nuts
import WALNUTSP as wp

import isokinetic as iso
import hamiltonian as hmc

import arviz as az

import matplotlib.pyplot as plt


np.random.seed(2)
q0 = np.random.normal(size=500)
for i in range(499):
    q0[i+1] = 0.95*q0[i] + np.sqrt(1.0-0.95**2)*q0[i+1]

# generated quantities
def gen(q):
    return(np.array([q[0],q[0]**2,np.sum(q)/500.0,np.sum(q**2)/500.0]))


lp = td.zeroMeanAR1

WN = nuts.NUTSampler(debug=False) # vanilla WALNUTS
WN.run(lpFun=lp,q0=q0,step=hmc.adaptHMCstepE(),tp0=hmc.HMCtuningPars(hMacro=0.06,delta=0.05),niter=3000,nwarmup=1000) # vanilla WALNUTS


WNIPD = wp.WALNUTSP0(debug=False,discardPassDoubling=True)  # WALNUTS-IP with discarding doubling that involves a passage
WNIPD.run(lpFun=lp,q0=q0,step=iso.adaptIKstepE(),tp0=iso.IKtuningPars(hMacro=1.0,delta=0.05),niter=3000,nwarmup=1000)


# WALNUTS ess per grads 
print("WALNUTS")
print(az.ess(WN.samples[0,1000:3000])/np.sum(WN.diagnostics.gradEvals[1000:3000]))
print("WALNUTS-IPD")
print(az.ess(WNIPD.samples[0,1000:3000])/np.sum(WNIPD.diagnostics.gradEvals[1000:3000]))
print("rel:")
print((az.ess(WNIPD.samples[0,:])/np.sum(WNIPD.diagnostics.gradEvals))/(az.ess(WN.samples[0,:])/np.sum(WN.diagnostics.gradEvals)))