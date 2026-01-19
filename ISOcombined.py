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

q0 = np.random.normal(size=11)
lp = td.funnel10 # td.funnel1 # td.modFunnel  # td.corrGauss #    td.stdGauss # 
 

nw = 1000

niter = 10000

wISO = nuts.NUTSampler(debug=False)
wISO.run(lp,q0,step=iso.adaptIKstepE(), 
        tp0=iso.IKtuningPars(hMacro=1.5,delta=0.5),
        niter=nw,nwarmup=nw)

wHMC = nuts.NUTSampler(debug=False)
wHMC.run(lp,q0,step=hmc.adaptHMCstepE(), 
         tp0=hmc.HMCtuningPars(hMacro=0.2,delta=0.3),niter=nw,nwarmup=nw)


samples = np.zeros((len(q0),niter))


q = wHMC.samples[:,-1]

wISOr = nuts.NUTSampler(debug=False)
wHMCr = nuts.NUTSampler(debug=False)

wISOr.run(lp,q,step=iso.adaptIKstepE(), 
          tp0=wISO.tp,
          niter=niter,nwarmup=0)

wHMCr.run(lp,q,step=hmc.adaptHMCstepE(), 
         tp0=wHMC.tp,
         niter=niter,nwarmup=0)

wISOs = nuts.NUTSampler(debug=False)
wHMCs = nuts.NUTSampler(debug=False)

for i in range(niter):
    if(np.random.uniform()<0.5):
        wISOs.run(lp,q,step=iso.adaptIKstepE(), 
                  tp0=wISO.tp,
                  niter=1,nwarmup=0)
        q = wISOs.samples[:,-1]
    else:
        wHMCs.run(lp,q,step=hmc.adaptHMCstepE(), 
                 tp0=wHMC.tp,
                 niter=1,nwarmup=0)
        q = wHMCs.samples[:,-1]
    samples[:,i] = q
    
    




