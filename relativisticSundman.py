import numpy as np
import scipy as sp

class RELISstate:
    
        
    
    
    def __init__(self):
        self.Ham = np.nan
    
    def ell(self,g,tp):
        return(-tp.alpha*np.log(tp.eta + np.dot(g,g)))

    
    
    def firstEval(self,lpFun,q,tp):
        [f,g] = lpFun(q)
        self.q = q
        self.f = f
        self.g = g
        self.p = np.repeat(np.nan,len(q))
        self.v = np.repeat(np.nan,len(q))
        self.o = np.nan
        self.Ham = np.nan
        
        
    def momentumRefresh(self,lpFun,tp):
        [f,g] = lpFun(q)
        self.p = RELunivarMomentum().rng(size=len(self.q),c=tp.c)
        self.v = self.p/np.sqrt(1.0 + (self.p/tp.c)**2)
        self.o = self.ell(g,tp) + 
        self.Ham = -self.f + (tp.cSq)*np.sum(np.sqrt(1.0 + (self.p/tp.c)**2))
        
        
    def velocity(self):
        
        return(self.v)
        
    
    def momentumFlip(self):
        self.p = -self.p
        self.v = -self.v
    
    def __str__(self):
        return ("RELIState class:\nq: " + str(self.q) + "\np: " + str(self.p) + "\nHamiltonian: " + str(self.Ham))

    def __repr__(self):
        return ("RELIState class:\nq: " + str(self.q) + "\np: " + str(self.p) + "\nHamiltonian: " + str(self.Ham))


class RELIStuningPars:
    
    def __init__(self, c=1.0, alpha=0.5, eta=0.1, hMacro=0.1, delta=0.1, Cmin=0,Cmax=16):
        self.c = c
        self.alpha = alpha
        self.eta = eta
        
        self.cSq = c**2
        self.hMacro = hMacro
        self.delta = delta
        self.Cmin = Cmin
        self.Cmax = Cmax

