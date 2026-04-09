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
import noseHoover as nh
import targets as td
import pandas as pd
import MCMCutils as ut
import dualAverage as da
import P2quantile as P2
import NUTSsampler as nuts


np.random.seed(1)

q0 = np.random.normal(size=2)
lp = td.funnel1
 

niter = 1000

wHMC = nuts.NUTSampler(debug=False)
wHMC.run(lp,q0,step=hmc.adaptHMCstepE(), tp0=hmc.HMCtuningPars(hMacro=0.2,delta=0.3),niter=niter,nwarmup=1000)

q01 = 6.0
q0 = np.array([q01,np.exp(q01/2)*2.0])

s0 = hmc.HMCstate()
s0.firstEval(lp, q=q0, tp=wHMC.tp)
s0.momentumRefresh(lp, tp=wHMC.tp)


import NUTSsampler as nuts

wS = nuts.NUTSampler(debug=True)
wS.singleIter(lp, s0, step=hmc.adaptHMCstepE(), tp0=wHMC.tp)


plt.plot(wS.fo.qs[0,:],wS.fo.qs[1,:])
plt.scatter(wS.fo.qs[0,:],wS.fo.qs[1,:],c=wS.fo.it)
plt.plot(wS.s0.q[0],wS.s0.q[1],'gs')
plt.plot(wS.sc.q[0],wS.sc.q[1],'rs')
plt.plot(np.array([wS.stopState1.q[0],wS.stopState2.q[0]]),np.array([wS.stopState1.q[1],wS.stopState2.q[1]]),'r')

