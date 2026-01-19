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
    


niter = 101000
nrep = 10

aISO = []
aHMC = []


for i in range(nrep):
    q0 = np.random.normal(size=11)
    
    wISO = nuts.NUTSampler(debug=False)
    wISO.run(lp,q0,step=iso.adaptIKstepE(), tp0=iso.IKtuningPars(hMacro=0.2,delta=0.3),generated=gen, niter=niter)
    aISO.append(wISO)

    wHMC = nuts.NUTSampler(debug=False)
    wHMC.run(lp,q0,step=hmc.adaptHMCstepE(), tp0=hmc.HMCtuningPars(hMacro=0.2,delta=0.3),generated=gen,niter=niter)
    aHMC.append(wHMC)
    
# need to unwrap in order to be able store arrays
samplesISO = np.zeros((wISO.samples.shape[0],wISO.samples.shape[1],nrep))
samplesHMC = np.zeros((wISO.samples.shape[0],wISO.samples.shape[1],nrep))
diagISO = np.zeros((wISO.diagnostics.shape[0],wISO.diagnostics.shape[1],nrep))
diagHMC = np.zeros((wISO.diagnostics.shape[0],wISO.diagnostics.shape[1],nrep))
for i in range(nrep):
    samplesISO[:,:,i] = aISO[i].samples
    samplesHMC[:,:,i] = aHMC[i].samples
    diagISO[:,:,i] = aISO[i].diagnostics
    diagHMC[:,:,i] = aHMC[i].diagnostics


np.save("funnelSamISO.npy",samplesISO)
np.save("funnelSamHMC.npy",samplesHMC)
np.save("funnelDiagISO.npy",diagISO)
np.save("funnelDiagHMC.npy",diagHMC)




print(np.mean(wISO.diagnostics.propBasic[1000:niter]))
print(np.mean(wHMC.diagnostics.propBasic[1000:niter]))
#print(np.mean(wREL.diagnostics.propBasic[1000:niter]))
#print(np.mean(wCM.diagnostics.propBasic[1000:niter]))
print('--')
print(np.mean(wISO.diagnostics.lwtsRange[1000:niter]))
print(np.mean(wHMC.diagnostics.lwtsRange[1000:niter]))
#print(np.mean(wREL.diagnostics.lwtsRange[1000:niter]))
#print(np.mean(wCM.diagnostics.lwtsRange[1000:niter]))
print('--')
print(np.quantile(wISO.diagnostics.lwtsRange[1000:niter],0.9))
print(np.quantile(wHMC.diagnostics.lwtsRange[1000:niter],0.9))

print('--')
print(1000*az.ess(wISO.samples[0,1000:niter])/np.sum(wISO.diagnostics.gradEvals[1000:niter]))
print(1000*az.ess(wHMC.samples[0,1000:niter])/np.sum(wHMC.diagnostics.gradEvals[1000:niter]))


