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
    


niter = 11000
nrep = 10

aISO = []
aHMC = []
aNH = []

aaISO = []
aaHMC = []
aaNH = []


for i in range(nrep):
    q0 = np.random.normal(size=11)
    
    wNH = nuts.NUTSampler(debug=False)
    wNH.run(lp,q0,step=nhv.adaptNHstepE(), tp0=nhv.NHtuningPars(hMacro=0.2,delta=0.3,oSigmaSq=4.0),generated=gen,niter=niter)
    aNH.append(wNH)
    
    waNH = wasps.WASPSampler()
    waNH.run(lp,q0,step=nhv.adaptNHstepE(), tp0=nhv.NHtuningPars(hMacro=0.2,delta=0.3,oSigmaSq=4.0),generated=gen,niter=niter)
    aaNH.append(waNH)
    
    wISO = nuts.NUTSampler(debug=False)
    wISO.run(lp,q0,step=iso.adaptIKstepE(), tp0=iso.IKtuningPars(hMacro=0.2,delta=0.3),generated=gen, niter=niter)
    aISO.append(wISO)
    
    waISO = wasps.WASPSampler()
    waISO.run(lp,q0,step=iso.adaptIKstepE(), tp0=iso.IKtuningPars(hMacro=0.2,delta=0.3),generated=gen, niter=niter)
    aaISO.append(waISO)

    wHMC = nuts.NUTSampler(debug=False)
    wHMC.run(lp,q0,step=hmc.adaptHMCstepE(), tp0=hmc.HMCtuningPars(hMacro=0.2,delta=0.3),generated=gen,niter=niter)
    aHMC.append(wHMC)
    
    waHMC = wasps.WASPSampler()
    waHMC.run(lp,q0,step=hmc.adaptHMCstepE(), tp0=hmc.HMCtuningPars(hMacro=0.2,delta=0.3),generated=gen,niter=niter)
    aaHMC.append(waHMC)
    
    
    
    


# need to unwrap in order to be able store arrays
samplesISO = np.zeros((wISO.samples.shape[0],wISO.samples.shape[1],nrep))
samplesNH = np.zeros((wISO.samples.shape[0],wISO.samples.shape[1],nrep))
samplesHMC = np.zeros((wISO.samples.shape[0],wISO.samples.shape[1],nrep))

samplesISOa = np.zeros((waISO.samples.shape[0],waISO.samples.shape[1],nrep))
samplesNHa = np.zeros((waISO.samples.shape[0],waISO.samples.shape[1],nrep))
samplesHMCa = np.zeros((waISO.samples.shape[0],waISO.samples.shape[1],nrep))

diagISO = np.zeros((wISO.diagnostics.shape[0],wISO.diagnostics.shape[1],nrep))
diagHMC = np.zeros((wISO.diagnostics.shape[0],wISO.diagnostics.shape[1],nrep))
diagNH = np.zeros((wISO.diagnostics.shape[0],wISO.diagnostics.shape[1],nrep))

diagISOa = np.zeros((waISO.diagnostics.shape[0],waISO.diagnostics.shape[1],nrep))
diagHMCa = np.zeros((waISO.diagnostics.shape[0],waISO.diagnostics.shape[1],nrep))
diagNHa = np.zeros((waISO.diagnostics.shape[0],waISO.diagnostics.shape[1],nrep))


for i in range(nrep):
    samplesISO[:,:,i] = aISO[i].samples
    samplesHMC[:,:,i] = aHMC[i].samples
    samplesNH[:,:,i] = aNH[i].samples
    diagISO[:,:,i] = aISO[i].diagnostics
    diagHMC[:,:,i] = aHMC[i].diagnostics
    diagNH[:,:,i] = aNH[i].diagnostics
    
    samplesISOa[:,:,i] = aaISO[i].samples
    samplesHMCa[:,:,i] = aaHMC[i].samples
    samplesNHa[:,:,i] = aaNH[i].samples
    diagISOa[:,:,i] = aaISO[i].diagnostics
    diagHMCa[:,:,i] = aaHMC[i].diagnostics
    diagNHa[:,:,i] = aaNH[i].diagnostics


np.save("funnelSamISO.npy",samplesISO)
np.save("funnelSamHMC.npy",samplesHMC)
np.save("funnelSamNH.npy",samplesNH)
np.save("funnelDiagISO.npy",diagISO)
np.save("funnelDiagHMC.npy",diagHMC)
np.save("funnelDiagNH.npy",diagNH)


np.save("funnelSamISOa.npy",samplesISOa)
np.save("funnelSamHMCa.npy",samplesHMCa)
np.save("funnelSamNHa.npy",samplesNHa)
np.save("funnelDiagISOa.npy",diagISOa)
np.save("funnelDiagHMCa.npy",diagHMCa)
np.save("funnelDiagNHa.npy",diagNHa)

