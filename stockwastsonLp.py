
import bridgestan as brs
import numpy as np

mdlInnov = brs.StanModel("sw_innov.stan","swdata.json")

d = mdlInnov.param_unc_num()

def lpFunSW(q,hessian=False):
    try:
        f,g = mdlInnov.log_density_gradient(q)
        return(f,g)
    except:
        #print("numeric exception")
        return(np.nan,0.0*q)

nms = mdlInnov.param_names(include_tp=True)


genInds = [nms.index("sigma"),nms.index("z.1"),nms.index("x.1"),nms.index("tau.1")]

def generatedSW(q):
    return(mdlInnov.param_constrain(q,include_tp=True)[genInds])

nms = mdlInnov.param_names(include_tp=True)

q0 = np.load("initq.npy")
dg = len(generatedSW(q0))
