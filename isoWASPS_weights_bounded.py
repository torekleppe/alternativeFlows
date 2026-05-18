

import numpy as np
import isokinetic as iso 
import WASPSampler as wasps
import NUTSsampler as nuts
import targets as td
import matplotlib.pyplot as plt
import pandas as pd
import arviz as az

import MCMCutils as ut

lp = td.corrGauss

q0 = np.array([0.5,0.55])

npo = 10


nrep = 2

scales = np.linspace(0.0, 20.0,num=npo)
ranges = np.array([0.25,0.5,0.75,1.0]) #np.linspace(0.25, 1.0,num=npr)
npr = len(ranges)
esses = np.zeros((nrep,npo,npr))
Mess = np.zeros((npo,npr))
alphas = np.zeros((nrep,npo,npr))
Malphas = np.zeros((npo,npr))

def linWts(i,weightScale,weightRange,medianOrbitSteps):
    return 1.0 + weightScale*(np.abs(i/(medianOrbitSteps)))


for i in range(npo):
    for i2 in range(npr):
        for j in range(nrep):
            Wr = wasps.WASPSampler(debug=False)
            Wr.run(lp,q0=q0,step=iso.adaptIKstepE(),tp0=iso.IKtuningPars(),niter=6000,
                   weightScale=scales[i],weightRange=ranges[i2])
            esses[j,i,i2] = az.ess(Wr.samples[0,1000:6000])
            alphas[j,i,i2] = np.mean(Wr.diagnostics.alpha[1000:6000])
        Malphas[i,i2] = np.mean(alphas[:,i,i2])
        Mess[i,i2] = np.mean(esses[:,i,i2])

print(Mess)
print(Malphas)