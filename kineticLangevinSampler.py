import numpy as np
import copy
import targets as td
import hamiltonian as hmc
import NUTSsampler as nuts
import pandas as pd
import MCMCutils as ut


class kineticLangevinSampler:
    
    
    
    def run(self,lpFun,q0,
            step=hmc.adaptHMCstepE(),
            tp0=hmc.HMCtuningPars(),
            generated=lambda q : q,
            gamma=2.0,
            niter=1000,
            nwarmup=1000,
            nthin=30,
            basicTarget=0.8
            ):
        
        self.tp = copy.deepcopy(tp0)
        self.step = copy.deepcopy(step)
        d = len(q0)
        g0 = generated(q0)
        
        self.samples = np.zeros((len(g0),niter+1))
        self.samples[:,0] = g0
        
        diagnostics = []
        
        hQA = nuts.quantileArray()
        
        s = step.getState()
        s.firstEval(lpFun,q0,self.tp)
        s.momentumRefresh(lpFun, self.tp)
        
        
        
        for i in range(niter):
            if((i+1) % 1000 == 0): print("iteration # " + str(i+1))
            self.step.reset()
            nacc = 0
            nstep = 0
            for ti in range(nthin):
                
                s.kineticLangevinRefresh(gamma,self.tp.hMacro)
                
                (s1,ljac) = self.step(s,lpFun,self.tp)
                nstep += 1
                lwt = -s1.Ham + ljac + s.Ham
                
                if(np.random.uniform()<np.exp(min(0.0,lwt))):
                   # accept
                   s = copy.deepcopy(s1)
                   nacc += 1
                else:
                    s.momentumFlip()
                    
            tuningDiag = pd.Series([self.tp.delta,self.tp.hMacro,nacc/nstep],index=['delta','h','alpha'])
            
            diagnostics.append(pd.concat([tuningDiag,self.step.diagnostics()]))
                    
            if(i < nwarmup):
                hQA.pushVec(np.log(self.step.Cobs))
                if(i > 10): self.tp.hMacro = (self.tp.delta*(2.0**self.tp.Cmin)**2/np.exp(hQA.quantile(basicTarget)))**(1.0/3.0)
                print("h = " + str(self.tp.hMacro))    
            
            
            self.samples[:,i+1] = generated(s.q)          
            
            
        self.diagnostics = pd.DataFrame(diagnostics)
                
                
        

        


# kls = kineticLangevinSampler()

# tp = hmc.HMCtuningPars()

# step = hmc.adaptHMCstepE()
# lp = td.modFunnel

# q0 = np.random.normal(size=2)

# kls.run(lp,q0,step,tp) 


