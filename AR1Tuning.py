
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import hamiltonian as hmc
import relativistic as rel
import isokinetic as iso
import vecSundman as vs
import correlatedMomentum as cor
import noseHoover as nh
import targets as td
import pandas as pd
import MCMCutils as ut
import NUTSsampler as nuts
import WASPSampler as wasps

rho = 0.99
lp = td.zeroMeanAR1
q0 = np.random.normal(size=100)
for i in range(len(q0)-1):
    q0[i+1] = rho*q0[i] + np.sqrt(1.0-rho**2)*q0[i+1]
    
niter = 10000

nrep = 10

res = np.zeros((nrep,6))

facs = np.exp(np.linspace(np.log(4.0),np.log(20.0),num=nrep))

for k in range(nrep):

    wnh = wasps.WASPSampler()
    wnh.run(lp,q0,step=nh.adaptNHstepE(),tp0=nh.NHtuningPars(Cmin=2,thermostatFac=facs[k]),niter=niter)
    res[k,0] = 2
    res[k,1] = np.mean(wnh.diagnostics.gradEvals[1000:niter])
    res[k,2] = az.ess(wnh.samples[0,1000:niter])
    res[k,3] = az.ess(wnh.samples[0,1000:niter]**2)


print(res[:,2]/res[:,1])
