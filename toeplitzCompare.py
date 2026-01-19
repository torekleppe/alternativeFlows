





import numpy as np
import math
import arviz as az
import copy
import matplotlib.pyplot as plt
import hamiltonian as hmc
import isokinetic as iso

import targets as td
import pandas as pd

import dualAverage as da
import P2quantile as P2
import NUTSsampler as nuts
import dill

np.random.seed(1)
rho = 0.99



def zeroMeanAR1(q):
    d = len(q)
    lp = -0.5*q[0]**2 - 0.5/(1.0-rho**2)*np.sum((q[1:d]-rho*q[0:(d-1)])**2)
    g = np.zeros_like(q)
    g[0] = q[0]-rho*q[1]
    g[1:(d-1)] = (1.0+rho**2)*q[1:(d-1)] - rho*(q[0:(d-2)] + q[2:d])
    g[-1] = q[-1] - rho*q[-2]
    g*=(1.0/(rho**2-1.0))
    
    return(lp,g)


lp = zeroMeanAR1

def gen(q):
    return(np.array([q[0],q[0]**2,np.sum(q)/len(q)]))
    


d = 500


niter = 6000
nrep = 10

aISO = []
anISO = []
aHMC = []
anHMC = []
#aREL = []

for i in range(nrep):
    q0 = np.random.normal(size=d)
    for i in range(1,d):
        q0[i] = rho*q0[i-1] + np.sqrt(1.0-rho**2)*q0[i]

    wISO = nuts.NUTSampler(debug=False)
    wISO.run(lp,q0,step=iso.adaptIKstepE(), tp0=iso.IKtuningPars(hMacro=1.5,delta=0.5,Cmin=4),generated=gen, niter=niter,basicTarget=0.95)
    aISO.append(wISO)

    wHMC = nuts.NUTSampler(debug=False)
    wHMC.run(lp,q0,step=hmc.adaptHMCstepE(), tp0=hmc.HMCtuningPars(hMacro=0.2,delta=0.3,Cmin=4),generated=gen,niter=niter,basicTarget=0.95)
    aHMC.append(wHMC)
    
    nISO = nuts.NUTSampler(debug=False)
    nISO.run(lp,q0,step=iso.adaptIKstepE(), tp0=iso.IKtuningPars(hMacro=0.5,delta=0.1,Cmax=0),generated=gen, niter=niter,basicTarget=0.99)
    anISO.append(nISO)
    
    nHMC = nuts.NUTSampler(debug=False)
    nHMC.run(lp,q0,step=hmc.adaptHMCstepE(), tp0=hmc.HMCtuningPars(hMacro=0.01,delta=0.1,Cmax=0),generated=gen,niter=niter,basicTarget=0.99 )
    anHMC.append(nHMC)
    

# need to unwrap in order to be able store arrays
samplesISO = np.zeros((wISO.samples.shape[0],wISO.samples.shape[1],nrep))
samplesHMC = np.zeros((wISO.samples.shape[0],wISO.samples.shape[1],nrep))
samplesnISO = np.zeros((wISO.samples.shape[0],wISO.samples.shape[1],nrep))
samplesnHMC = np.zeros((wISO.samples.shape[0],wISO.samples.shape[1],nrep))
diagISO = np.zeros((wISO.diagnostics.shape[0],wISO.diagnostics.shape[1],nrep))
diagnISO = np.zeros((wISO.diagnostics.shape[0],wISO.diagnostics.shape[1],nrep))
diagHMC = np.zeros((wISO.diagnostics.shape[0],wISO.diagnostics.shape[1],nrep))
diagnHMC = np.zeros((wISO.diagnostics.shape[0],wISO.diagnostics.shape[1],nrep))
for i in range(nrep):
    samplesISO[:,:,i] = aISO[i].samples
    samplesnISO[:,:,i] = anISO[i].samples
    samplesHMC[:,:,i] = aHMC[i].samples
    samplesnHMC[:,:,i] = anHMC[i].samples
    diagISO[:,:,i] = aISO[i].diagnostics
    diagnISO[:,:,i] = anISO[i].diagnostics
    diagHMC[:,:,i] = aHMC[i].diagnostics
    diagnHMC[:,:,i] = anHMC[i].diagnostics


np.save("toepSamISO.npy",samplesISO)
np.save("toepSamnISO.npy",samplesnISO)
np.save("toepSamHMC.npy",samplesHMC)
np.save("toepSamnHMC.npy",samplesnHMC)
np.save("toepDiagISO.npy",diagISO)
np.save("toepDiagnISO.npy",diagnISO)
np.save("toepDiagHMC.npy",diagHMC)
np.save("toepDiagnHMC.npy",diagnHMC)

   

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
#print(np.quantile(wREL.diagnostics.lwtsRange[1000:niter],0.9))
#print(np.quantile(wCM.diagnostics.lwtsRange[1000:niter],0.9))
print('--')
print(1000*az.ess(wISO.samples[0,1000:niter])/np.sum(wISO.diagnostics.gradEvals[1000:niter]))
print(1000*az.ess(wHMC.samples[0,1000:niter])/np.sum(wHMC.diagnostics.gradEvals[1000:niter]))
#print(1000*az.ess(wREL.samples[0,1000:niter])/np.sum(wREL.diagnostics.gradEvals[1000:niter]))
#print(1000*az.ess(wCM.samples[0,1000:niter])/np.sum(wCM.diagnostics.gradEvals[1000:niter]))

