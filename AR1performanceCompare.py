# script showing difference in performance for AR(1) model


import targets as td
import numpy as np

import NUTSsampler as nuts
import WALNUTSP as wp

import isokinetic as iso

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
WN.run(lpFun=lp,q0=q0,niter=11000) # vanilla WALNUTS

WNi = nuts.NUTSampler(debug=False) # isokinetic WALNUTS
WNi.run(lpFun=lp,q0=q0,step=iso.adaptIKstepE(),tp0=iso.IKtuningPars(),niter=11000) # isokinetic WALNUTS


WNIP = wp.WALNUTSP(debug=False)  # WALNUTS-IP
WNIP.run(lpFun=lp,q0=q0,step=iso.adaptIKstepE(),tp0=iso.IKtuningPars(),niter=11000)


WNIPD = wp.WALNUTSP(debug=False,discardPassDoubling=True)  # WALNUTS-IP with discarding doubling that involves a passage
WNIPD.run(lpFun=lp,q0=q0,step=iso.adaptIKstepE(),tp0=iso.IKtuningPars(),niter=11000)


# WALNUTS ess per grads 
print("WALNUTS")
print(az.ess(WN.samples[0,1000:11000])/np.sum(WN.diagnostics.gradEvals[1000:11000]))
print("WALNUTS-I")
print(az.ess(WNi.samples[0,1000:11000])/np.sum(WNi.diagnostics.gradEvals[1000:11000]))
print("WALNUTS-IP")
print(az.ess(WNIP.samples[0,1000:11000])/np.sum(WNIP.diagnostics.gradEvals[1000:11000]))
print("WALNUTS-IPD")
print(az.ess(WNIPD.samples[0,1000:11000])/np.sum(WNIPD.diagnostics.gradEvals[1000:11000]))


# Typical output
#WALNUTS
#0.0007586456442470352
#WALNUTS-I
#0.0008088716934726283
#WALNUTS-IP
#0.0011457292903842477
#WALNUTS-IPD
#0.0008262088101250175