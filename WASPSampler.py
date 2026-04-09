import numpy as np
import copy
import targets as td
import hamiltonian as hmc
import isokinetic as iso
import noseHoover as nh
import NUTSsampler as nuts
import pandas as pd
import MCMCutils as ut
import matplotlib.pyplot as plt
import arviz as az

# class WASPSampler:
    
    
    
#     def run(self,lpFun,q0,
#             step=hmc.adaptHMCstepE(),
#             tp0=hmc.HMCtuningPars(),
#             generated=lambda q : q,
#             center=0.0,
#             niter=100000,
#             nwarmup=1000,
#             N=10,
#             basicTarget=0.8
#             ):
        
#         self.tp = copy.deepcopy(tp0)
#         self.step = copy.deepcopy(step)
#         d = len(q0)
#         g0 = generated(q0)
        
#         self.samples = np.zeros((len(g0),niter+1))
#         self.samples[:,0] = g0
        
#         diagnostics = []
        
#         hQA = nuts.quantileArray()
        
#         lwts = nuts.lwtVector(2**(N+1))
        
#         s = step.getState()
#         s.firstEval(lpFun,q0,self.tp)
#         s.momentumRefresh(lpFun, self.tp)
        
        
#         if (np.ndim(center) == 0):
#             cen = np.repeat(center, d)
#         else:
#             cen = center
        
#         #plt.plot(s.q[0],s.q[1],'.')
        
#         for it in range(niter):
#             if((it+1) % 1000 == 0): print("iteration # " + str(it+1))
#             self.step.reset()
#             sf = copy.deepcopy(s)
#             sb = copy.deepcopy(s)
#             subSampled = copy.deepcopy(s)
#             lwts.reset()
#             lwts[0] = -s.Ham
#             ljf = 0.0
#             ljb = 0.0
#             a = 0
#             b = 0
            
            
#             accWtsum = 1.0
#             subOrbitWtSum = 0.0
            
            
#             z1 = np.random.normal(size=d)
#             z2 = np.random.normal(size=d)
#             eta = (1.0/np.sum(z1**2))*z1
#             z2 = z2 - (np.sum(z2*eta))*eta
#             gam = (1.0/np.sum(z2**2))*z2
            
#             forwardDone = False
#             backwardDone = False
            
#             for i in range(N-1):
#                 nstep = 2**i
                
                
#                 accWtsum += subOrbitWtSum
#                 subOrbitWtSum = 0.0
                
#                 if(np.random.uniform()<0.5):
#                     #print("f")
#                     # forward integration
                    
                        
#                     for j in range(nstep):
#                         qOld = copy.deepcopy(sf.q)
#                         (sf,lj) = self.step(sf,lpFun,self.tp)
                        
#                         cqs = sf.q-cen
#                         cq = qOld-cen
                        
#                         stop1 = np.dot(cqs,eta)*np.dot(cq,eta) < 0.0 
#                         stop2 = max(np.dot(gam,cqs),np.dot(gam,cq)) > 0.0
                        
#                         if(stop1 and stop2):
#                             forwardDone = True
#                             break
                        
                        
#                         b += 1
#                         ljf += lj
#                         lwts[b] = -sf.Ham + ljf
                        
#                         wt = lwts.normalizedWt(b)
                        
#                         subOrbitWtSum += wt
#                         if(subOrbitWtSum> 0.0 and np.random.uniform() < wt/subOrbitWtSum):
#                             subSampled = copy.deepcopy(sf)
                        
#                         #plt.plot(sf.q[0],sf.q[1],'r.')
                    
                
#                 else:
#                     #print("b")
#                     # backward integration
                   
#                     for j in range(nstep):
#                         qOld = copy.deepcopy(sb.q)
#                         sb.momentumFlip()
#                         (sb,lj) = self.step(sb,lpFun,self.tp)
                        
#                         cqs = sf.q-cen
#                         cq = qOld-cen
                        
#                         stop1 = np.dot(cqs,eta)*np.dot(cq,eta) < 0.0 
#                         stop2 = max(np.dot(gam,cqs),np.dot(gam,cq)) > 0.0
                        
#                         if(stop1 and stop2):
#                             backwardDone = True
#                             break
                        
                        
                        
                        
#                         a -= 1
#                         sb.momentumFlip()
#                         ljb += lj
                        
#                         lwts[a] = -sb.Ham + ljb
                        
#                         wt = lwts.normalizedWt(a)
#                         subOrbitWtSum += wt
#                         if(subOrbitWtSum> 0.0 and np.random.uniform() < wt/subOrbitWtSum):
#                             subSampled = copy.deepcopy(sb)
                            
#                             #plt.plot(sb.q[0],sb.q[1],'g.')
                        
                
#                 if(forwardDone or backwardDone):
#                     break
                
#                 if(np.random.uniform()<subOrbitWtSum/accWtsum):
#                     sSampled = copy.deepcopy(subSampled)
                
                
#             s = copy.deepcopy(sSampled)
            
#             self.samples[:,it+1] = generated(s.q) 
#             #plt.plot(sSampled.q[0],sSampled.q[1],'rs')
#             s.momentumRefresh(lpFun, self.tp)



class WASPSampler:
    
    def defaultWts(self,i,weightScale,weightRange,medianOrbitSteps):
        return 1.0 + weightScale*(1.0 - np.exp(-0.5*(i/(weightRange*medianOrbitSteps))**2))

    
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
            weightRange=0.25,
            weightScale=5.0,
            deltaOff=False):
        
        self.tp = copy.deepcopy(tp0)
        self.step = copy.deepcopy(step)
        d = len(q0)
        g0 = generated(q0)
        
        self.samples = np.zeros((len(g0),niter+1))
        self.samples[:,0] = g0
        
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
            eta = (1.0/np.sum(z1**2))*z1
            z2 = z2 - (np.sum(z2*eta))*eta
            gam = (1.0/np.sum(z2**2))*z2
            
            # forward integration
            if(nf>0):
                for i in range(nf):
                    qOld = copy.deepcopy(sf.q) 
                    (sf,lj) = self.step(sf,lpFun,self.tp) 
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
                    #print("f")
                    #print(-sf.Ham + ljacf)
                    
                    
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
                    
                    
                    
                    cqs = sb.q-cen
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
                    
                    
                    ljacb += lj
                    
                    if(ljacb < -690.0):
                        deadb = 1
                        break
                    
                    a -= 1
                    lwts[a] = -sb.Ham + ljacb
                    #print("b")
                    
                    #print(-sb.Ham + ljacb)
                    
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
            
            
            tuningDiag = pd.Series([self.tp.delta,self.tp.hMacro,ESSfrac,lwtsRange],index=['delta','h','ESSfrac','lwtsRange'])
            
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
                
                
                if(it>10 and (not deltaOff)): self.tp.delta = orbitLWtRangeTarget/np.exp(deltaQA.quantile(orbitLWtRangeQuantile))
                if(it>10): self.tp.hMacro = (self.tp.delta*(2.0**self.tp.Cmin)**2/np.exp(hQA.quantile(basicTarget)))**(1.0/3.0)
                if(it>10): medianOrbitSteps = max(1.0,np.median(numOrbtiSteps[0:it]))
                if(it>10 and hasattr(self.tp,'oSigmaSq')):
                    if(self.tp.thermostatFac>0.0):
                        self.tp.oSigmaSq = 1.0/(self.tp.thermostatFac*self.tp.hMacro*2**(-self.tp.Cmin))
                
                
                print("medianOrbitSteps: " + str(medianOrbitSteps))
            
            s.momentumRefresh(lpFun, self.tp)
            
            
        self.diagnostics = pd.DataFrame(diagnostics)
        
#w = WASPSampler()

# # tp = hmc.HMCtuningPars()
# # #tp = iso.IKtuningPars()
#tp = nh.NHtuningPars(oSigmaSq=0.5**2)
#tp.Cmin = 2

# # step = hmc.adaptHMCstepE()
# # #step = iso.adaptIKstepE()
#step = nh.adaptNHstepE()

#lp =  td.corrGauss # td.modFunnel #td.funnel1 #  td.stdGauss #   td.smileDistr #        td.corrGauss #  td.funnel10 #     

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
       

