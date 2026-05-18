import performanceMeasure as pm
import targets as td
import numpy as np

import NUTSsampler as nuts
import WASPSampler as wasps
import WALNUTSP as wp

np.random.seed(1)

if(1==0):
    out = pm.performanceMeasureCompute(td.stdGauss, 
                                       q0=np.random.normal(size=2),
                                       samplers=[nuts.NUTSampler,
                                                 wasps.WASPSampler,
                                                 wasps.fixedCenterWASPSampler],
                                       niter=2000,
                                       cores=10,
                                       save="tmp.csv")



if(1==1):
    def gen(q):
        return(np.array([q[0],q[0]**2,np.sum(q[1:11]**2)/10.0]))
        
    
    np.random.seed(1)
    
    q0 = np.random.normal(size=11)
    q0[0] = 0.0
    
    out = pm.performanceMeasureCompute(td.funnel10, 
                                       q0=q0,
                                       generated=gen,
                                       samplers=[nuts.NUTSampler,
                                                 wp.WALNUTSP,
                                                 wp.WALNUTSP0],
                                       niter=51000,
                                       cores=10,
                                       save="funnel10.csv")

    


if(1==1):
    
    np.random.seed(1)
    q0 = np.random.normal(size=500)
    for i in range(499):
        q0[i+1] = 0.95*q0[i] + np.sqrt(1.0-0.95**2)*q0[i+1]
    
    
    def gen(q):
        return(np.array([q[0],q[0]**2,np.sum(q)/500.0,np.sum(q**2)/500.0]))
    
    out = pm.performanceMeasureCompute(td.zeroMeanAR1,
                                       q0=q0,
                                       generated=gen,
                                       samplers=[nuts.NUTSampler,
                                                 wp.WALNUTSP,
                                                 wp.WALNUTSP0],
                                       niter=11000,
                                       cores=10,
                                       save="AR1model.csv")
    



if(1==1):
    np.random.seed(1)
    
    import stockwastsonLp as swlp
    out = pm.performanceMeasureCompute(swlp.lpFunSW,
                                       q0=swlp.q0,
                                       samplers=[nuts.NUTSampler,
                                                 wp.WALNUTSP
                                                 ],
                                       generated=swlp.generatedSW,
                                       niter=11000,
                                       cores=10,
                                       save="SWmodel.csv")
    