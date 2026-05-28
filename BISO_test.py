
import numpy as np
import targets as td
import WALNUTSP as wp
import isokinetic as iso
import blockedIsoKinetic as biso
import arviz as az
import MCMCutils as ut


q0 = np.random.normal(size=11)
q0[0] = 0.0
lp = td.funnel10

tpb = biso.BIKtuningPars()
tpb.blockTwoC(len(q0))


WI = wp.WALNUTSP()
WIB = wp.WALNUTSP()


WI.run(lp,q0,step=iso.adaptIKstepE(),tp0=iso.IKtuningPars(),niter=11000)
WIB.run(lp,q0,step=biso.adaptBIKstepE(),tp0=tpb,niter=11000)



