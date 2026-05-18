import NUTSsampler as nuts
import WASPSampler as wasps
import targets as td
import isokinetic as iso
import noseHooverScalar as snh
import noseHooverVec as vsh
import matplotlib.pyplot as plt
import numpy as np
        
lp = td.funnel10 #td.modFunnel   # td.stdGauss #  
wISO = wasps.WASPSampler(debug=False)
q0 = np.random.normal(size=11)
wISO.run(lp,q0,step=iso.adaptIKstepE() , tp0=iso.IKtuningPars(),niter=2000)

