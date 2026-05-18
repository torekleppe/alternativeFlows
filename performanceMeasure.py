from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor

import numpy as np
import pandas as pd
from scipy.stats import t
import NUTSsampler as nuts
import WASPSampler as wasps
import hamiltonian as hmc
import isokinetic as iso 
import noseHooverVec as nhv
import noseHooverScalar as nhs
import targets as td
import arviz as az
import sys
import matplotlib.pyplot as plt



def performanceMeasureCompute(lpFun,q0,
                              generated=lambda q : q,
                              samplers=[nuts.NUTSampler,
                                        wasps.WASPSampler],
                              steps=[hmc.adaptHMCstepE,
                                     iso.adaptIKstepE,
                                     nhv.adaptNHstepE,
                                     nhs.adaptSNHstepE],
                              tps=[hmc.HMCtuningPars(),
                                   iso.IKtuningPars(),
                                   nhv.NHtuningPars(oSigmaSq=1.0),
                                   nhs.SNHtuningPars(oSigmaSq=1.0)],
                              labels=["HMC","ISO","VNH1","SNH1"],
                              diagVars=['delta','h','ESSfrac','lwtsRange'],
                              nreplica=10,
                              cores=5,
                              nwarmup=1000,
                              niter=11000,
                              save=None):
    
    if(len(steps) != len(tps) or len(steps) != len(labels)) : sys.exit("bad steps, tps, labels")
    genLen = len(generated(q0))
    
    #raw = np.zeros((nreplica*len(samplers)*len(steps),2*genLen+3*len(diagVars) + 3))
    
    
    raw = []
    essnms = []
    wessnms = []
    diagnms = []
    
    for i in range(genLen): essnms.append("ESS_gen_" + str(i))
    
    for i in range(genLen): wessnms.append("ESSperGrad_gen_" + str(i))
    
    for i in range(len(diagVars)): 
        diagnms.append(diagVars[i] + "_mean") 
        diagnms.append(diagVars[i] + "_min")
        diagnms.append(diagVars[i] + "_max")
    
    for samp in range(len(samplers)):
        
        for stp in range(len(steps)):
            print("-------------------------------------")
            print(samplers[samp](debug=False).name()  + " + " + steps[stp]().name())
            print("-------------------------------------")
            
            #for rep in range(nreplica):
            def replica(rep):
                
                np.random.seed(rep)
                sampler = samplers[samp](debug=False)
                
                sampler.run(lpFun=lpFun,q0=q0,
                            step=steps[stp](),tp0=tps[stp],
                            generated=generated,
                            nwarmup=nwarmup,
                            niter=niter)
                
                essVec = np.zeros(genLen)
                
                
                
                for i in range(genLen):
                    essVec[i] = az.ess(sampler.samples[i,nwarmup:niter])
                    #[k,genLen + i] = raw[k,i]/sum(sampler.diagnostics.gradEvals[nwarmup:niter])
                
                essrow = pd.Series(essVec,essnms)
                wessrow = pd.Series(essVec/sum(sampler.diagnostics.gradEvals[nwarmup:niter]),wessnms)
                
                diags = []
                
                for i in range(len(diagVars)):
                    diags.append(np.mean(sampler.diagnostics[diagVars[i]][nwarmup:niter]))
                    diags.append(np.min(sampler.diagnostics[diagVars[i]][nwarmup:niter]))
                    diags.append(np.max(sampler.diagnostics[diagVars[i]][nwarmup:niter]))
                 
                diagsrow = pd.Series(diags,diagnms)
                
                metarow = pd.Series([rep,sampler.name(),steps[stp]().name(),labels[stp]],["replica","sampler","step","label"])
                
                return(pd.concat([essrow,wessrow,diagsrow,metarow]))
                #return(rep*2)
            
            if(cores>1):
                rr = Parallel(n_jobs=min(cores,nreplica),verbose=10)(delayed(replica)(ii) for ii in range(nreplica))
                for i in range(nreplica):
                    raw.append(rr[i])
                
                get_reusable_executor().shutdown(wait=True)
                
            else:
                for i in range(nreplica):
                    raw.append(replica(i))
            
            ret = pd.DataFrame(raw)
            if save is not None:
                ret.to_csv(save,index=False)
    
            
    return(ret)

def performanceMeasurePlot(data,alpha=0.95,plotfn="fig.pdf"):
    
    scols = ['k','b','r','c','m']
    
    plt.figure(figsize=(14,4))
    
    samplers = data['sampler'].unique()
    
    labels = data['label'].unique()
    
    cols = data.columns.tolist()
    
    
    wESSlabs = [item for item in cols if "ESSperGrad_gen_" in item]
    
    
    
    for i in range(len(wESSlabs)):
        ax=plt.subplot(1,len(wESSlabs),i+1)
        ax.set_title("generated # " + str(i+1))
        offset = 0.0
        ticks = []
        tickLabs = []
        for samp in range(len(samplers)):
            subData = data[data.sampler==samplers[samp]]
            for lab in range(len(labels)):
                subsubData = subData[subData.label==labels[lab]]
                plt.plot(np.array([offset,offset]),
                         np.array([np.min(subsubData[wESSlabs[i]]),
                                   np.max(subsubData[wESSlabs[i]])]),scols[samp]+'--')
                
                mn = np.mean(subsubData[wESSlabs[i]])
                sd = np.sqrt(np.var(subsubData[wESSlabs[i]]))
                dd = len(subsubData[wESSlabs[i]])
                lb = mn + t.ppf(0.5*(1.0-alpha),df=dd-1)*sd/np.sqrt(dd)
                ub = mn + t.ppf(1.0-0.5*(1.0-alpha),df=dd-1)*sd/np.sqrt(dd)
                
                plt.plot(offset,mn,'.'+scols[samp])
                plt.plot(np.array([offset-0.2,offset+0.2]),np.repeat(lb, 2),scols[samp])
                plt.plot(np.array([offset-0.2,offset+0.2]),np.repeat(ub, 2),scols[samp])
                plt.plot(np.repeat(offset,2),np.array([lb,ub]),scols[samp])
                
                ticks.append(offset)
                tickLabs.append(samplers[samp] + " + " + labels[lab])
                offset+=1.0
            offset+=1.0
        ax.set_xticks(ticks,tickLabs,rotation=-90,ha='center')
        
    plt.savefig(plotfn,bbox_inches='tight')
    plt.show()

def performanceMeasureSummary(data):
    samplers = data['sampler'].unique()
    
    labels = data['label'].unique()
    
    cols = data.columns.tolist()
    
    ESSlabs = [item for item in cols if "ESS_gen_" in item]
    wESSlabs = [item for item in cols if "ESSperGrad_gen_" in item]
    print(ESSlabs)
    
    for samp in range(len(samplers)):
        print("--------------------------------")
        print(samplers[samp])
        print("--------------------------------")
        for i in range(len(ESSlabs)):
            print("gen_" + str(i) + "\t\t",end="")
        print("")
        
        subData = data[data.sampler==samplers[samp]]
        
        
        
        for lab in range(len(labels)):
            print("-------------- " + labels[lab] + " --------------")
            subsubData = subData[subData.label==labels[lab]]
            for i in range(len(ESSlabs)):
                print(str(round(np.mean(subsubData[ESSlabs[i]]),1)) + "\t\t",end="")
            print("")
            for i in range(len(ESSlabs)):
                print(str(round(np.mean(1000*subsubData[wESSlabs[i]]),2)) + "\t\t",end="")
            print("")
    

#data = performanceMeasureCompute(td.stdGauss, q0=np.random.normal(size=3),niter=2000,nreplica=10,save="tmp.csv")


    
#



