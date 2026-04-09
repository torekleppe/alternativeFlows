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
import relativisticSundman as relis


np.random.seed(0)

q0 = np.random.normal(size=2)
lp = td.stdGauss # td.corrGauss # td.funnel1 #  td.modFunnel #        
 

niter = 2000

wR = nuts.NUTSampler(debug=False)
wR.run(lp,q0,step=relis.adaptRELISstepE(), 
        tp0=relis.RELIStuningPars(hMacro=0.2,delta=0.3,c=2.0,alpha=1.0/6.0,eta=1.0),
        niter=niter,nwarmup=1000)

wR0 = nuts.NUTSampler(debug=False)
wR0.run(lp,q0,step=relis.adaptRELISstepE(), 
        tp0=relis.RELIStuningPars(hMacro=0.2,delta=0.3,c=2.0,alpha=0.0,eta=1.0),
        niter=niter,nwarmup=1000)


wHMC = nuts.NUTSampler(debug=False)
wHMC.run(lp,q0,step=hmc.adaptHMCstepE(), tp0=hmc.HMCtuningPars(hMacro=0.2,delta=0.3),niter=niter,nwarmup=1000)
