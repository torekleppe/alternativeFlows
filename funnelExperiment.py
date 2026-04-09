
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

np.random.seed(0)

lp = td.funnel10rescaled
q0 = np.random.normal(size=11)
niter = 6000

nrep = 10

res = np.zeros((8*nrep,6))

k = 0

# Hamiltonian

for i in range(nrep):

    wh = wasps.WASPSampler()
    wh.run(lp,q0,step=hmc.adaptHMCstepE(),tp0=hmc.HMCtuningPars(Cmin=2),niter=niter)
    res[k,0] = 0
    res[k,1] = np.mean(wh.diagnostics.gradEvals[1000:niter])
    res[k,2] = az.ess(wh.samples[0,1000:niter])
    res[k,3] = az.ess(wh.samples[0,1000:niter]**2)
    k += 1
    # isokinetic
    
    wi = wasps.WASPSampler()
    wi.run(lp,q0,step=iso.adaptIKstepE(),tp0=iso.IKtuningPars(Cmin=2),niter=niter)
    res[k,0] = 1
    res[k,1] = np.mean(wi.diagnostics.gradEvals[1000:niter])
    res[k,2] = az.ess(wi.samples[0,1000:niter])
    res[k,3] = az.ess(wi.samples[0,1000:niter]**2)
    k += 1
    
    # nose-hoover
    
    wnh = wasps.WASPSampler()
    wnh.run(lp,q0,step=nh.adaptNHstepE(),tp0=nh.NHtuningPars(Cmin=2,oSigmaSq=4.0),niter=niter)
    res[k,0] = 2
    res[k,1] = np.mean(wnh.diagnostics.gradEvals[1000:niter])
    res[k,2] = az.ess(wnh.samples[0,1000:niter])
    res[k,3] = az.ess(wnh.samples[0,1000:niter]**2)
    k += 1
    
    # relativistic
    
    wr = wasps.WASPSampler()
    wr.run(lp,q0,step=rel.adaptRELIstepE(),tp0=rel.RELtuningPars(Cmin=2,c=1.0),niter=niter)
    res[k,0] = 3
    res[k,1] = np.mean(wr.diagnostics.gradEvals[1000:niter])
    res[k,2] = az.ess(wr.samples[0,1000:niter])
    res[k,3] = az.ess(wr.samples[0,1000:niter]**2)
    k += 1
    
    np.save("funnelExperiment", res)    
    
    
    wwh = nuts.NUTSampler(debug=False)
    wwh.run(lp,q0,step=hmc.adaptHMCstepE(),tp0=hmc.HMCtuningPars(Cmin=2),niter=niter)
    res[k,0] = 4
    res[k,1] = np.mean(wwh.diagnostics.gradEvals[1000:niter])
    res[k,2] = az.ess(wwh.samples[0,1000:niter])
    res[k,3] = az.ess(wwh.samples[0,1000:niter]**2)
    k += 1
    # isokinetic
    
    wwi =nuts.NUTSampler(debug=False)
    wwi.run(lp,q0,step=iso.adaptIKstepE(),tp0=iso.IKtuningPars(Cmin=2),niter=niter)
    res[k,0] = 5
    res[k,1] = np.mean(wwi.diagnostics.gradEvals[1000:niter])
    res[k,2] = az.ess(wwi.samples[0,1000:niter])
    res[k,3] = az.ess(wwi.samples[0,1000:niter]**2)
    k += 1
    
    # nose-hoover
    
    wwnh = nuts.NUTSampler(debug=False)
    wwnh.run(lp,q0,step=nh.adaptNHstepE(),tp0=nh.NHtuningPars(Cmin=2,oSigmaSq=4.0),niter=niter)
    res[k,0] = 6
    res[k,1] = np.mean(wwnh.diagnostics.gradEvals[1000:niter])
    res[k,2] = az.ess(wwnh.samples[0,1000:niter])
    res[k,3] = az.ess(wwnh.samples[0,1000:niter]**2)
    k += 1
    
    # relativistic
    
    wwr = nuts.NUTSampler(debug=False)
    wwr.run(lp,q0,step=rel.adaptRELIstepE(),tp0=rel.RELtuningPars(Cmin=2,c=1.0),niter=niter)
    res[k,0] = 7
    res[k,1] = np.mean(wwr.diagnostics.gradEvals[1000:niter])
    res[k,2] = az.ess(wwr.samples[0,1000:niter])
    res[k,3] = az.ess(wwr.samples[0,1000:niter]**2)
    k += 1
    
    np.save("funnelExperiment", res)    
    