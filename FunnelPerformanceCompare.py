

# script showing difference in performance for AR(1) model


import targets as td
import numpy as np

import NUTSsampler as nuts
import WALNUTSP as wp

import isokinetic as iso
import hamiltonian as hmc

import arviz as az

import matplotlib.pyplot as plt


np.random.seed(3)
q0 = np.random.normal(size=11)
q0[0] = 0.01

lp = td.funnel10 # td.stdGauss # 

def gen(q):
    return(np.array([q[0],q[0]**2,np.sum(q[1:11]**2)/10.0]))
    


WNIP = wp.WALNUTSP0(debug=False)  # WALNUTS-IP with discarding doubling that involves a passage
WNIP.run(lpFun=lp,q0=q0,step=iso.adaptIKstepE(),tp0=iso.IKtuningPars(hMacro=1.5,delta=0.7),L=8,niter=5000,nwarmup=0)



WNIPD = wp.WALNUTSP0(debug=False,orbit2k=True)  # WALNUTS-IP with discarding doubling that involves a passage
WNIPD.run(lpFun=lp,q0=q0,step=iso.adaptIKstepE(),tp0=iso.IKtuningPars(hMacro=1.5,delta=0.7),L=8,niter=5000,nwarmup=0)


# WALNUTS ess per grads 
print("WALNUTS-IP")
print(az.ess(WNIP.samples[0,:])/np.sum(WNIP.diagnostics.gradEvals))
print("WALNUTS-IPD")
print(az.ess(WNIPD.samples[0,:])/np.sum(WNIPD.diagnostics.gradEvals))
print("rel:")
print((az.ess(WNIP.samples[0,:])/np.sum(WNIP.diagnostics.gradEvals))/(az.ess(WNIPD.samples[0,:])/np.sum(WNIPD.diagnostics.gradEvals)))
