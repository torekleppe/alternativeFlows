import numpy as np
import targets as td
import NUTSsampler as nuts
import matplotlib.pyplot as plt

def weightedWASPS(q0,lpFun,h=0.1,L=64,niter=1000000):
    
    
    
    
    
    qc = q0
    d = len(q0)
    
    lwts = nuts.lwtVector(2*L+2)
    
    samples = np.zeros((d,niter+1))
    samples[:,0] = q0
    
    for it in range(niter):
        pc = np.random.normal(size=d)
        
        nf = np.random.randint(0,L)
        nb = L - nf - 1
        Hlwts = np.zeros(2*L+1)
        [f,g] = lpFun(qc)
        
        Hlwts[L] = -f + 0.5*np.dot(pc,pc)
        
        qs = np.zeros((d,2*L+1))
        qs[:,L] = qc
        
        
        
        
        z1 = np.random.normal(size=d)
        z2 = np.random.normal(size=d)
        eta = (1.0/np.sum(z1**2))*z1
        z2 = z2 - (np.sum(z2*eta))*eta
        gam = (1.0/np.sum(z2**2))*z2
    
        a = 0
        b = 0
    
        if(nf>0):
            q = qc
            p = pc
            [f,g] = lpFun(q)
            for i in range(nf):
                qOld = q
                
                ph = p + 0.5*h*g
                q = q+h*ph
                [f,g] = lpFun(q)
                p = ph + 0.5*h*g
                
                cqs = q
                cq = qOld
                
                cqseta = np.dot(cqs,eta)
                cqeta = np.dot(cq,eta)
                
                ts = cqeta/(cqeta-cqseta)
                
                if(ts>0.0 and ts < 1.0):
                    if((1.0-ts)*np.dot(cq,gam)+ts*np.dot(cqs,gam)>0.0):
                        break
                
                
                b += 1
                #print("b = " + str(b))
                qs[:,L+b] = q
                Hlwts[L+b] = -f + 0.5*np.dot(p,p)
                
                
        if(nb>0):
            q = qc
            p = -pc
            [f,g] = lpFun(q)
            for i in range(nb):
                qOld = q
                
                ph = p + 0.5*h*g
                q = q+h*ph
                [f,g] = lpFun(q)
                p = ph + 0.5*h*g
                
                cqs = q
                cq = qOld
                
                cqseta = np.dot(cqs,eta)
                cqeta = np.dot(cq,eta)
                
                ts = cqeta/(cqeta-cqseta)
                
                if(ts>0.0 and ts < 1.0):
                    if((1.0-ts)*np.dot(cq,gam)+ts*np.dot(cqs,gam)>0.0):
                        break
                
                
                a -= 1
                #print("a = " + str(a))
                qs[:,L+a] = q
                Hlwts[L+a] = -f + 0.5*np.dot(p,p)
                        
        
                
        inds = np.linspace(start=a, stop=b,num=(b-a+1))
        
        mlw = np.max(Hlwts[(L+a):(L+b+1)])
        
        lwts = Hlwts[(L+a):(L+b+1)] - mlw + 0.5*np.log(np.abs(inds)+1.0)
        
        wts = np.exp(lwts)
        wtsSum = np.sum(wts)
        wts = wts/wtsSum
        
        i = np.random.choice(inds,p=wts)
        
        indsb = np.linspace(start=i-b, stop=i-a,num=(b-a+1))
        
        print(i)
        print(indsb)

        lwtsb = Hlwts[(L+a):(L+b+1)] - mlw + 0.5*np.log(np.abs(indsb)+1.0)
        
        wtsSumb = np.sum(np.exp(lwtsb))
        
        print(wtsSum/wtsSumb)
        
        if(np.random.uniform()<wtsSum/wtsSumb):
            qc = qs[:,L+int(i)]
            
            
        samples[:,it+1] = qc
        
        
        if(False):
            plt.subplot(1,2,1)
            plt.plot(qs[0,(L+a):(L+b+1)],qs[1,(L+a):(L+b+1)],'.-')
        
            plt.plot(q0[0],q0[1],'gs')
            xx = np.array([-1.0,1.0])
            plt.plot(xx,-xx*eta[0]/eta[1])
        
            plt.subplot(1,2,2)
            plt.plot(inds,lwts)
        
        
    return(samples)
lp = td.stdGauss # td.smileDistr
q0 = np.array([0.5,1.0])

s=weightedWASPS(q0,lp)



