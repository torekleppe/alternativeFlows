import numpy as np
import math
import arviz as az
import copy
import matplotlib.pyplot as plt
import hamiltonian as hmc
import relativistic as rel
import isokinetic as iso
import vecSundman as vs
import correlatedMomentum as cor
import targets as td
import pandas as pd
import MCMCutils as ut
import dualAverage as da
import P2quantile as P2
import NUTSsampler as nuts


np.random.seed(0)

q0 = np.random.normal(size=2)
lp = td.modFunnel #td.funnel1 # td.corrGauss #    td.stdGauss # 
 

niter = 2000

wVS = nuts.NUTSampler(debug=False)
wVS.run(lp,q0,step=vs.adaptVSstepE(), 
        tp0=vs.VStuningPars(hMacro=0.2,delta=0.3,procDef=vs.expPnorm(alpha=0.1)),
        niter=niter,nwarmup=1000,deltaOff=True)


wHMC = nuts.NUTSampler(debug=False)
wHMC.run(lp,q0,step=hmc.adaptHMCstepE(), tp0=hmc.HMCtuningPars(hMacro=0.2,delta=0.3),niter=niter,nwarmup=1000)
