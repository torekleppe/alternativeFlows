import numpy as np


# computes (1-exp(h*o))/o

def linOdefacScalar(h,o):
    if(np.abs(o)>1.0e-14):
        return(-(np.expm1(h*o))/o)
    else:
        return(-h-0.5*o*h*h-0.16666666666666666*o*o*h*h*h)
    
def linOdefacVector(h,o):
    reg = np.abs(o)<1.0e-14
    ret = -(np.expm1(h*o))/o
    if(any(reg)):
        ret[reg] = -h-0.5*o[reg]*h*h-0.16666666666666666*o[reg]*o[reg]*h*h*h
    return(ret)



#o = np.array([2.0e-14,0.9e-14,0.1])
#h = 0.1
#print(linOdefacVector(h, o))
#print((1-np.exp(h*o))/o)
#print(-(np.expm1(h*o))/o)
