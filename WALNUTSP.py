import numpy as np
import math
import copy
import NUTSsampler as nuts
import hamiltonian as hmc
import pandas as pd
import dualAverage as da
import P2quantile as p2q
import matplotlib.pyplot as plt


class fullOrbitW:
    def __init__(self,s):
        self.ind = 0
        self.qs = s.q.reshape(-1,1)
        self.it = np.array([0])
        self.qd = []
        self.eta = None
        self.gam = None
        self.C = None
        
    def pushPlane(self,eta,gam,C):
        self.eta=eta
        self.gam=gam
        self.C = C
        

    def pushF(self,s,it):
        self.qs = np.hstack([self.qs, s.q.reshape(-1,1)])
        self.it = np.concatenate((self.it,np.array([it])))
        
    
    def pushB(self,s,it):
        self.qs = np.hstack([s.q.reshape(-1,1),self.qs])
        self.it = np.concatenate((np.array([it]),self.it))
        self.ind += 1
        
    def pushD(self,s):
        self.qd.append(s.q)

    def plot(self):
        
        plt.plot(self.qs[0,:],self.qs[1,:],".-")
        plt.plot(self.qs[0,self.ind],self.qs[1,self.ind],"gs")
        if(len(self.qd)>0):
            for i in range(len(self.qd)):
                plt.plot(self.qd[i][0],self.qd[i][1],'x')
        ax = plt.axis()
        if(not( self.eta is None )):
            plt.plot([self.C[0],self.C[0]+10*self.gam[0]],
                     [self.C[1],self.C[1]+10*self.gam[1]])
        plt.axis(ax)
class WALNUTSP:
    
    def __init__(self,debug=False,orbitStats=False,discardPassDoubling=False): # make consistent with NUTS
        self.debug = debug
        self.orbitStats = orbitStats
        self.discardPassDoubling = discardPassDoubling
    
    def name(self):
        return("WALNUTS-P")
    
    
    def buildOrbit(self,sc,it):
        
        d = len(sc.q)
        sf = copy.deepcopy(sc)
        sb = copy.deepcopy(sc)
        
        
        self.step.reset()
        self.lwts.reset()
        self.lwts[0] = -sc.Ham
        
        
        deadf = False
        deadb = False
        donef = False
        doneb = False
        
        z1 = np.random.normal(size=d)
        z2 = np.random.normal(size=d)
        eta = (1.0/np.sqrt(np.sum(z1**2)))*z1
        z2 = z2 - (np.sum(z2*eta))*eta
        gam = (1.0/np.sqrt(np.sum(z2**2)))*z2
        
        
        
        if(self.orbitStats):
            tmpg = self.generated(sc.q)
            self.orbitMax[:,it] = tmpg
            self.orbitMin[:,it] = tmpg
            self.sorbitMax[:,it] = tmpg
            self.sorbitMin[:,it] = tmpg
        
        a = 0
        b = 0
        
        ljacf = 0.0
        ljacb = 0.0
        
        accWtsum = 1.0
        iSampled = 0
        Sampled = copy.deepcopy(sc)
        
        if(self.debug):
            fo = fullOrbitW(sc)
            fo.pushPlane(eta, gam, self.cen)
            
        sSampled = copy.deepcopy(sc)
        
        for doubCount in range(self.L):
            
            nstep = 2**doubCount
            
            forward = np.random.uniform() < 0.5
            
            wtSum = 0.0
            
            if(forward and (not deadf) and (not donef)):
                # forward integration
                
                for i in range(nstep):
                    qOld = copy.deepcopy(sf.q) 
                    (sf,lj) = self.step(sf,self.lp,self.tp) 
                    
                    
                    
                    if(self.orbitStats):
                        tmpg = self.generated(sf.q)
                        self.orbitMax[:,it] = np.maximum(self.orbitMax[:,it],tmpg)
                        self.orbitMin[:,it] = np.minimum(self.orbitMin[:,it],tmpg)
                    
                    cqs = sf.q-self.cen
                    cq = qOld-self.cen
                   
                    cqseta = np.dot(cqs,eta)
                    cqeta = np.dot(cq,eta)
                    
                    ts = cqeta/(cqeta-cqseta)
                    
                    if(ts>0.0 and ts < 1.0):
                        if((1.0-ts)*np.dot(cq,gam)+ts*np.dot(cqs,gam)>0.0):
                            donef = True
                            if(self.discardPassDoubling): wtSum = 0.0
                            if(self.debug):
                                fo.pushD(sf)
                            break
                         
                    ljacf += lj
                    
                    if(ljacf < -690.0):
                        deadf = True
                        if(self.debug):
                            fo.pushD(sf)
                        break
                    
                    if(self.debug):
                        fo.pushF(sf,b+1)
                    
                    b += 1
                    self.lwts[b] = -sf.Ham + ljacf
                    
                    if(self.orbitStats):
                        self.sorbitMax[:,it] = self.orbitMax[:,it]
                        self.sorbitMin[:,it] = self.orbitMin[:,it]
                    
                     
                    wt = self.lwts.normalizedWt(b)
                    wtSum += wt
                    
                    if(wtSum > 1.0e-14 and np.random.uniform() < wt/wtSum):
                        iSampledLoc = b
                        sampledLoc = copy.deepcopy(sf)
                
                    
                
                
            elif((not forward) and (not deadb) and (not doneb)):
                # backward integration
                
                for i in range(nstep):
                    qOld = copy.deepcopy(sb.q)
                    sb.momentumFlip()
                    (sb,lj) = self.step(sb,self.lp,self.tp) 
                    sb.momentumFlip()
                    
                    
                    
                    if(self.orbitStats):
                        tmpg = self.generated(sb.q)
                        self.orbitMax[:,it] = np.maximum(self.orbitMax[:,it],tmpg)
                        self.orbitMin[:,it] = np.minimum(self.orbitMin[:,it],tmpg)
                    
                    cqs = sb.q-self.cen
                    cq = qOld-self.cen
                   
                    cqseta = np.dot(cqs,eta)
                    cqeta = np.dot(cq,eta)
                    
                    ts = cqeta/(cqeta-cqseta)
                    
                    if(ts>0.0 and ts < 1.0):
                        if((1.0-ts)*np.dot(cq,gam)+ts*np.dot(cqs,gam)>0.0):
                            doneb = True
                            if(self.discardPassDoubling): wtSum = 0.0
                            if(self.debug):
                                fo.pushD(sb)
                            break
                         
                    ljacb += lj
                    
                    if(ljacb < -690.0):
                        deadb = True
                        if(self.debug):
                            fo.pushD(sb)
                        break
                    
                    a -= 1
                    self.lwts[a] = -sb.Ham + ljacb
                    
                    if(self.orbitStats):
                        self.sorbitMax[:,it] = self.orbitMax[:,it]
                        self.sorbitMin[:,it] = self.orbitMin[:,it]
                    
                    if(self.debug):
                        fo.pushB(sb,a-1)
                     
                    wt = self.lwts.normalizedWt(a)
                    wtSum += wt
                    
                    if(wtSum > 1.0e-14 and np.random.uniform() < wt/wtSum):
                        iSampledLoc = a
                        sampledLoc = copy.deepcopy(sb)
                
                
            ## done integration part
            if(np.random.uniform()<wtSum/accWtsum):
                #print("accepted state from proposed subOrbit: a = " + str(subOrbitWtSum/accWtsum))
                iSampled = iSampledLoc
                sSampled = copy.deepcopy(sampledLoc)
            
            if((donef or deadf) and (doneb or deadb)):
                #print("done")
                #print((donef,deadf,doneb,deadb))
                break
            
            
            accWtsum += wtSum
            #print((a,b,iSampled))
        if(self.debug):
            fo.plot()
        
        dd = pd.Series([1*deadf,1*deadb,1*donef,1*doneb,
                        iSampled,a,b],
                       index=['revRejectF','revRejectB','planeHitF','planeHitB','j','a','b'])
        #print(sc)
        return(sSampled,dd)
    
    
    
    def run(self,lpFun,q0,
            step=hmc.adaptHMCstepE(),
            tp0=hmc.HMCtuningPars(),
            generated=lambda q : q,
            center=0.0,
            niter=10000,
            nwarmup=1000,
            L=10,
            orbitLWtRangeTarget=0.3,
            orbitLWtRangeQuantile=0.9,
            basicTarget=0.8,
            deltaOff=False,
            adaptCenterOff=False):
        
        self.lp = lpFun
        self.tp = copy.deepcopy(tp0)
        self.step = copy.deepcopy(step)
        self.generated = generated
        self.L = L
        self.lwts = nuts.lwtVector(4*(2**L))
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
            self.cen = np.repeat(center, d)
        else:
            self.cen = center
        
        if(not adaptCenterOff):
            cenP2s = p2q.P2vector(d,prob=0.5)
        
        
        lwts = nuts.lwtVector(2*L)
        
        diagnostics = []
        
        medianOrbitSteps = L
        numOrbtiSteps = np.zeros(nwarmup+1)
    
        
        for it in range(niter):
            if((it+1) % 1000 == 0): print("iteration # " + str(it+1))
            
            s.momentumRefresh(lpFun, self.tp)
            (s,dd) = self.buildOrbit(s,it)
    
    
            self.samples[:,it+1] = generated(s.q)
            
            ESSfrac = self.lwts.wtsESS()
            
            lwtsRange = self.lwts.lwtsRange()
            
            
            tuningDiag = pd.Series([self.tp.delta,self.tp.hMacro,
                                    ESSfrac,lwtsRange,np.min(self.cen),np.max(self.cen),medianOrbitSteps],
                                   index=['delta','h','ESSfrac','lwtsRange','cenMin','cenMax','medOrbitSteps'])
            
            diagnostics.append(pd.concat([tuningDiag,dd,self.step.diagnostics()]))
    
    
            if(it<nwarmup):
                
                numOrbtiSteps[it] = dd.b-dd.a+1
                
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
                
                
                if(it>10 and not adaptCenterOff): self.cen=cenP2s.quantile()
                if((it+1) % 100 == 0): print((np.min(self.cen),np.max(self.cen)))
            
    
        self.diagnostics = pd.DataFrame(diagnostics)


class WALNUTSP0(WALNUTSP):
    def name(self):
        return("WALNUTS-P0")
    
    def run(self,lpFun,q0,
            step=hmc.adaptHMCstepE(),
            tp0=hmc.HMCtuningPars(),
            generated=lambda q : q,
            center=0.0,
            niter=10000,
            nwarmup=1000,
            L=10,
            orbitLWtRangeTarget=0.3,
            orbitLWtRangeQuantile=0.9,
            basicTarget=0.8,
            deltaOff=False,
            adaptCenterOff=False):
        super().run(lpFun,q0,step,tp0,generated,center,niter,nwarmup,L,orbitLWtRangeTarget,
                    orbitLWtRangeQuantile,basicTarget,deltaOff,adaptCenterOff=True)
import targets as td


#samp = WALNUTSP(debug=False)
#samp.run(td.smileDistr ,q0=np.random.normal(size=2),niter=11000)

