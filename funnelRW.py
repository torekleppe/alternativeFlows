import numpy as np
import math
import arviz as az
import copy
import matplotlib.pyplot as plt
import hamiltonian as hmc
#import relativistic as rel
import isokinetic as iso
#import correlatedMomentum as cor
import targets as td
import pandas as pd
#import MCMCutils as ut
import dualAverage as da
import P2quantile as P2
import NUTSsampler as nuts
import dill

np.random.seed(0)


lp = td.funnel10 # td.stdGauss # 

def gen(q):
    return(np.array([q[0],q[0]**2,np.sum(q[1:11]**2)/10.0]))
    


niter = 2000
q0 = np.random.normal(size=11)

wHMC = nuts.NUTSampler(debug=False)
wHMC.run(lp,q0,step=hmc.adaptHMCstepE(), tp0=hmc.HMCtuningPars(hMacro=0.2,delta=0.3),generated=gen,niter=niter)

wHMC2 = nuts.NUTSampler(debug=False)
wHMC2.run(lp,q0,step=hmc.adaptHMCstepE(), tp0=hmc.HMCtuningPars(hMacro=0.2,delta=0.3),generated=gen,niter=niter,partialMomentumRefresh=0.99)

plt.figure(1)
plt.plot(wHMC.diagnostics.Hamiltonian)
plt.plot(wHMC2.diagnostics.Hamiltonian)


plt.figure(1)
plt.plot(wHMC.samples[0,:])
plt.plot(wHMC2.samples[0,:])

