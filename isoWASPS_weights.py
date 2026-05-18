

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

nrep = 10

scales = np.linspace(0.0, 20.0,num=npo)
esses = np.zeros((nrep,npo))
Mess = 0.0*scales
alphas = np.zeros((nrep,npo))
Malphas = 0.0*scales

def linWts(i,weightScale,weightRange,medianOrbitSteps):
    return 1.0 + weightScale*(np.abs(i/(medianOrbitSteps)))


for i in range(npo):
    for j in range(nrep):
        Wr = wasps.WASPSampler(debug=False)
        Wr.run(lp,q0=q0,step=iso.adaptIKstepE(),tp0=iso.IKtuningPars(),niter=6000,
           weightFun=linWts,
           weightScale=scales[i])
        esses[j,i] = az.ess(Wr.samples[0,1000:6000])
        alphas[j,i] = np.mean(Wr.diagnostics.alpha[1000:6000])
    Malphas[i] = np.mean(alphas[:,i])
    Mess[i] = np.mean(esses[:,i])


plt.plot(scales,Mess)