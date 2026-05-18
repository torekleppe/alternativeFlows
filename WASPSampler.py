import numpy as np
import copy
import targets as td
import hamiltonian as hmc
import isokinetic as iso
import noseHooverVec as nh
import NUTSsampler as nuts
import pandas as pd
import MCMCutils as ut
import matplotlib.pyplot as plt
import arviz as az
import P2quantile as p2q



class WASPSampler:
    
    def __init__(self,debug=False,orbitStats=False): # make consistent with NUTS
        self.orbitStats = orbitStats
    
    def defaultWts(self,i,weightScale,weightRange,medianOrbitSteps):
        return 1.0 + weightScale*(1.0 - np.exp(-0.5*(i/(weightRange*medianOrbitSteps))**2))

    def name(self):
        return("WASPS")
    
    def run(self,lpFun,q0,
            step=hmc.adaptHMCstepE(),
            tp0=hmc.HMCtuningPars(),
            generated=lambda q : q,
            weightFun=defaultWts,
            center=0.0,
            niter=10000,
            nwarmup=1000,
            L=2**10,
            orbitLWtRangeTarget=0.3,
            orbitLWtRangeQuantile=0.9,
            basicTarget=0.8,
            weightRange=0.5,# 0.25,
            weightScale=10.0, # 5.0,
            deltaOff=False,
            adaptCenterOff=False):
        
        self.tp = copy.deepcopy(tp0)
        self.step = copy.deepcopy(step)
        d = len(q0)
        g0 = generated(q0)
        
        self.samples = np.zeros((len(g0),niter+1))
        self.samples[:,0] = g0
        
        if(self.orbitStats):
            self.orbitMax = np.zeros((len(g0),niter))
            self.orbitMin = np.zeros((len(g0),niter))
            self.sorbitMax = np.zeros((len(g0),niter))
            self.sorbitMin = np.zeros((len(g0),niter))
        
        diagnostics = []
        
        deltaQA = nuts.quantileArray()
        hQA = nuts.quantileArray()
        
        
        s = step.getState()
        s.firstEval(lpFun,q0,self.tp)
        s.momentumRefresh(lpFun, self.tp)
        
        
        if (np.ndim(center) == 0):
            cen = np.repeat(center, d)
        else:
            cen = center
        
        if(not adaptCenterOff):
            cenP2s = p2q.P2vector(d,prob=0.5)
        
        
        lwts = nuts.lwtVector(2*L)
        
        diagnostics = []
        
        medianOrbitSteps = L
        numOrbtiSteps = np.zeros(nwarmup+1)
        
        for it in range(niter):
            if((it+1) % 1000 == 0): print("iteration # " + str(it+1))
            self.step.reset()
            lwts.reset()
            lwts[0] = -s.Ham
            sf = copy.deepcopy(s)
            sb = copy.deepcopy(s)

            sampledForward = copy.deepcopy(s)
            sampledBackward = copy.deepcopy(s)

            nf = np.random.randint(0,L) #np.random.random_integers(low=0,high=L-1,size=1)[0]
            nb = L - nf - 1
            
            
            ljacf = 0.0
            ljacb = 0.0
            
            wtSumF = 0.0
            wtSumB = 0.0
            
            a = 0
            b = 0
            
            iSampledForward = 0
            iSampledBackward = 0
            
            
            deadf = 0
            deadb = 0
            
            z1 = np.random.normal(size=d)
            z2 = np.random.normal(size=d)
            eta = (1.0/np.sqrt(np.sum(z1**2)))*z1
            z2 = z2 - (np.sum(z2*eta))*eta
            gam = (1.0/np.sqrt(np.sum(z2**2)))*z2
            
            if(self.orbitStats):
                tmpg = generated(s.q)
                self.orbitMax[:,it] = tmpg
                self.orbitMin[:,it] = tmpg
                self.sorbitMax[:,it] = tmpg
                self.sorbitMin[:,it] = tmpg
            
            # forward integration
            if(nf>0):
                for i in range(nf):
                    qOld = copy.deepcopy(sf.q) 
                    (sf,lj) = self.step(sf,lpFun,self.tp) 
                    
                    if(self.orbitStats):
                        tmpg = generated(sf.q)
                        self.orbitMax[:,it] = np.maximum(self.orbitMax[:,it],tmpg)
                        self.orbitMin[:,it] = np.minimum(self.orbitMin[:,it],tmpg)
                    
                    cqs = sf.q-cen
                    cq = qOld-cen
                   
                    cqseta = np.dot(cqs,eta)
                    cqeta = np.dot(cq,eta)
                    
                    ts = cqeta/(cqeta-cqseta)
                    
                    if(ts>0.0 and ts < 1.0):
                        if((1.0-ts)*np.dot(cq,gam)+ts*np.dot(cqs,gam)>0.0):
                            break
                         
                   
                    #stop1 = np.dot(cqs,eta)*np.dot(cq,eta) < 0.0 
                    #stop2 = max(np.dot(gam,cqs),np.dot(gam,cq)) > 0.0
                   
                    #if(stop1 and stop2):
                    #    break
                    
                    
                    ljacf += lj
                    
                    if(ljacf < -690.0):
                        deadf = 1
                        break
                    
                    
                    
                    b += 1
                    lwts[b] = -sf.Ham + ljacf
                    
                    if(self.orbitStats):
                        self.sorbitMax[:,it] = self.orbitMax[:,it]
                        self.sorbitMin[:,it] = self.orbitMin[:,it]
                    
                    wtFac = self.defaultWts(b,weightScale,weightRange,medianOrbitSteps)
                    
                    wt = lwts.normalizedWt(b)*wtFac
                    wtSumF += wt
                    
                    if(wtSumF > 1.0e-14 and np.random.uniform() < wt/wtSumF):
                        iSampledForward = b
                        sampledForward = copy.deepcopy(sf)
                    
            if(nb>0):
                for i in range(nb):
                    qOld = copy.deepcopy(sb.q)
                    sb.momentumFlip()
                    (sb,lj) = self.step(sb,lpFun,self.tp) 
                    sb.momentumFlip()
                    
                    if(self.orbitStats):
                        tmpg = generated(sb.q)
                        self.orbitMax[:,it] = np.maximum(self.orbitMax[:,it],tmpg)
                        self.orbitMin[:,it] = np.minimum(self.orbitMin[:,it],tmpg)
                    
                    cqs = sb.q-cen
                    cq = qOld-cen
                    
                    cqseta = np.dot(cqs,eta)
                    cqeta = np.dot(cq,eta)
                    
                    ts = cqeta/(cqeta-cqseta)
                    
                    if(ts>0.0 and ts < 1.0):
                        if((1.0-ts)*np.dot(cq,gam)+ts*np.dot(cqs,gam)>0.0):
                            break
                        
                    
                    
                    ljacb += lj
                    
                    if(ljacb < -690.0):
                        deadb = 1
                        break
                    
                    a -= 1
                    lwts[a] = -sb.Ham + ljacb
                    
                    if(self.orbitStats):
                        self.sorbitMax[:,it] = self.orbitMax[:,it]
                        self.sorbitMin[:,it] = self.orbitMin[:,it]
                    
                    wtFac = self.defaultWts(a,weightScale,weightRange,medianOrbitSteps)
                    
                    wt = lwts.normalizedWt(a)*wtFac
                    wtSumB += wt
                    
                    if(wtSumB > 1.0e-14 and np.random.uniform() < wt/wtSumB):
                        iSampledBackward = a
                        sampledBackward = copy.deepcopy(sb)
                    
            
            
            ww = np.array([self.defaultWts(0,weightScale,weightRange,medianOrbitSteps),wtSumB,wtSumF])
        
            wtsSumF = np.sum(ww)
            ww = ww/wtsSumF
        
            branch = np.random.choice(np.array([0,1,2]),p=ww)
            ii = 0
            sprop = copy.deepcopy(s)
            if(branch == 1):
                sprop = copy.deepcopy(sampledBackward)
                ii = iSampledBackward
            elif(branch==2):
                sprop = copy.deepcopy(sampledForward)
                ii = iSampledForward
        
        
            ## accept-reject step related to weights
        
            indsb = np.linspace(start=ii-b, stop=ii-a,num=(b-a+1))
            wtFacsb = self.defaultWts(indsb,weightScale,weightRange,medianOrbitSteps) #weightFun(indsb,weightRange,medianOrbitSteps)
            
            wtsSumB = np.sum(lwts.normalizedWts(a,b)*wtFacsb)
        
            #print([a,b,b-a+1])
            #print(ii)
            #print(len(lwts.normalizedWts(a, b)))
            #print(lwts.normalizedWts(a, b))
            #print(indsb)
            #print("wtsSumB :" + str(wtsSumB))
            #print("wtsSumF :" + str(wtsSumF))
            #print(wtsSumF/wtsSumB)
            
            alpha = min(1.0,wtsSumF/wtsSumB)
        
            if(np.random.uniform()< alpha):
                # accept proposal
                s = copy.deepcopy(sprop)
            
            
            self.samples[:,it+1] = generated(s.q)
            
            ESSfrac = lwts.wtsESS()
            
            lwtsRange = lwts.lwtsRange()
            
            
            tuningDiag = pd.Series([self.tp.delta,self.tp.hMacro,ESSfrac,lwtsRange,np.min(cen),np.max(cen)],index=['delta','h','ESSfrac','lwtsRange','cenMin','cenMax'])
            
            indexProp = 0.5
            if(b-a>0): indexProp = (ii-a)/(b-a)
            
            orbitDiag = pd.Series([deadf,deadb,ii,indexProp,a,b,alpha],index=['revRejectF','revRejectB','index','indexProp','a','b','alpha'])
            
            diagnostics.append(pd.concat([tuningDiag,orbitDiag,self.step.diagnostics()]))
            
            
            if(it<nwarmup):
                
                numOrbtiSteps[it] = b-a+1
                
                if(hasattr(self.step,'conservedRange')):
                    conservedRange = self.step.conservedRange()
                else:
                    conservedRange = lwtsRange
                    
                deltaQA.push(np.log(conservedRange/self.tp.delta))
                hQA.pushVec(np.log(self.step.Cobs))
                
                if(not adaptCenterOff): cenP2s.push(s.q)
                
                if(it>10 and (not deltaOff)): self.tp.delta = orbitLWtRangeTarget/np.exp(deltaQA.quantile(orbitLWtRangeQuantile))
                if(it>10): self.tp.hMacro = (self.tp.delta*(2.0**self.tp.Cmin)**2/np.exp(hQA.quantile(basicTarget)))**(1.0/3.0)
                if(it>10): medianOrbitSteps = max(1.0,np.median(numOrbtiSteps[0:it]))
                #if(it>10 and hasattr(self.tp,'oSigmaSq')):
                #    if(self.tp.thermostatFac>0.0):
                #        self.tp.oSigmaSq = 1.0/(self.tp.thermostatFac*self.tp.hMacro*2**(-self.tp.Cmin))
                
                
                if(it>10 and not adaptCenterOff): cen=cenP2s.quantile()
                if((it+1) % 100 == 0): print((np.min(cen),np.max(cen)))
            
            s.momentumRefresh(lpFun, self.tp)
            
            
        self.diagnostics = pd.DataFrame(diagnostics)



class fixedCenterWASPSampler(WASPSampler):
    def name(self):
        return("WASPS0")
    
    def run(self,lpFun,q0,
            step=hmc.adaptHMCstepE(),
            tp0=hmc.HMCtuningPars(),
            generated=lambda q : q,
            weightFun=WASPSampler.defaultWts,
            center=0.0,
            niter=10000,
            nwarmup=1000,
            L=2**10,
            orbitLWtRangeTarget=0.3,
            orbitLWtRangeQuantile=0.9,
            basicTarget=0.8,
            weightRange=0.25,
            weightScale=5.0,
            deltaOff=False):
        super().run(lpFun,q0,step,tp0,generated,weightFun,center,niter,nwarmup,L,orbitLWtRangeTarget,
                    orbitLWtRangeQuantile,basicTarget,weightRange,weightScale,deltaOff,adaptCenterOff=True)
        

    

#w = fixedCenterdWASPSampler()

# # tp = hmc.HMCtuningPars()
# # #tp = iso.IKtuningPars()
#tp = hmc.HMCtuningPars()
#tp.Cmin = 2

#step = hmc.adaptHMCstepE()
# # #step = iso.adaptIKstepE()
#step = nh.adaptNHstepE()

#lp = td.smileDistr #  td.stdGauss # td.modFunnel #td.funnel1 #  td.stdGauss #          td.corrGauss #  td.funnel10 #     

#q0 = np.random.normal(size=2)

#w.run(lp,q0,step,tp,niter=2000)

# wrs = np.linspace(0.2,4.0,num=10)
# ESSs = np.zeros_like(wrs)
# alphas = np.zeros_like(wrs)

# q0 = np.random.normal(size=2)

# for i in range(len(wrs)):
#     w.run(lp,q0,step=nh.adaptNHstepE(),tp0=nh.NHtuningPars(oSigmaSq=wrs[i]),
      
#       niter=11000)
#     ESSs[i] = az.ess(w.samples[0,1000:11000])
#     alphas[i] = np.mean(w.diagnostics.alpha[1000:11000])
       

