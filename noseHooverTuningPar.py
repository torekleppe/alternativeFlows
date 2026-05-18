import numpy as np
import math
import arviz as az
import copy
import matplotlib.pyplot as plt
import hamiltonian as hmc

import isokinetic as iso
import noseHooverVec as nhv
import targets as td
import pandas as pd
#import MCMCutils as ut
import dualAverage as da
import P2quantile as P2
import NUTSsampler as nuts
import WASPSampler as wasps



import dill

np.random.seed(0)


lp = td.funnel10 # td.stdGauss # 

def gen(q):
    return(np.array([q[0],q[0]**2,np.sum(q[1:11]**2)/10.0]))
    


niter = 51000
nrep = 10

Osq = np.exp(np.linspace(np.log(0.1),np.log(15.0),num=nrep))

ESSperGrad = 0.0*Osq
ESSperGradA = 0.0*Osq


q0 = np.random.normal(size=11)
q0[0] = 0.0

for i in range(nrep):
    
    
    wNH = nuts.NUTSampler(debug=False)
    wNH.run(lp,q0,step=nhv.adaptNHstepE(), tp0=nhv.NHtuningPars(hMacro=0.2,delta=0.3,oSigmaSq=Osq[i]),generated=gen,niter=niter)
    
    ESSperGrad[i] = az.ess(wNH.samples[0,1000:niter])/np.sum(wNH.diagnostics.gradEvals[1000:niter])
    
    plt.semilogx(Osq,ESSperGrad)
    
    
    #wNHa = wasps.WASPSampler()
    #wNHa.run(lp,q0,step=nhv.adaptNHstepE(), tp0=nhv.NHtuningPars(hMacro=0.2,delta=0.3,oSigmaSq=Osq[i]),generated=gen,niter=niter)
    
    #ESSperGradA[i] = az.ess(wNHa.samples[0,1000:niter])/np.sum(wNHa.diagnostics.gradEvals[1000:niter])


