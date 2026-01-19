import numpy as np
import matplotlib.pyplot as plt
import arviz as az

sISO=np.load("funnelSamISO.npy")
sHMC=np.load("funnelSamHMC.npy")
dISO=np.load("funnelDiagISO.npy")
dHMC=np.load("funnelDiagHMC.npy")

niter = 101000

tabISO = np.zeros((10,8))
tabHMC = np.zeros((10,8))
mns1ISO = np.zeros((10,100000))
mns1HMC = np.zeros((10,100000))

mns2ISO = np.zeros((10,100000))
mns2HMC = np.zeros((10,100000))

mns3ISO = np.zeros((10,100000))
mns3HMC = np.zeros((10,100000))

for i in range(10):
    tabISO[i,0] = az.ess(sISO[0,1000:niter,i])
    tabHMC[i,0] = az.ess(sHMC[0,1000:niter,i])
    
    tabISO[i,1] = 1000*tabISO[i,0]/np.sum(dISO[1000:niter,11,i])
    tabHMC[i,1] = 1000*tabHMC[i,0]/np.sum(dHMC[1000:niter,11,i])
    
    tabISO[i,2] = az.ess(sISO[1,1000:niter,i])
    tabHMC[i,2] = az.ess(sHMC[1,1000:niter,i])
    
    tabISO[i,3] = 1000*tabISO[i,2]/np.sum(dISO[1000:niter,11,i])
    tabHMC[i,3] = 1000*tabHMC[i,2]/np.sum(dHMC[1000:niter,11,i])
    
    
    tabISO[i,4] = az.ess(sISO[2,1000:niter,i])
    tabHMC[i,4] = az.ess(sHMC[2,1000:niter,i])
    
    tabISO[i,5] = 1000*tabISO[i,4]/np.sum(dISO[1000:niter,11,i])
    tabHMC[i,5] = 1000*tabHMC[i,4]/np.sum(dHMC[1000:niter,11,i])
    
    tabISO[i,6] = np.mean(dISO[:,3,i])
    tabHMC[i,6] = np.mean(dHMC[:,3,i])
    
    tabISO[i,7] = np.mean(dISO[:,-1,i])
    tabHMC[i,7] = np.mean(dHMC[:,-1,i])
    
    mns1ISO[i,:] = np.cumsum(sISO[0,1000:niter,i])/np.linspace(start=1,stop=100000, num=100000)
    mns1HMC[i,:] = np.cumsum(sHMC[0,1000:niter,i])/np.linspace(start=1,stop=100000, num=100000)

    mns2ISO[i,:] = np.cumsum(sISO[1,1000:niter,i])/np.linspace(start=1,stop=100000, num=100000)
    mns2HMC[i,:] = np.cumsum(sHMC[1,1000:niter,i])/np.linspace(start=1,stop=100000, num=100000)
    
    mns3ISO[i,:] = np.cumsum(sISO[2,1000:niter,i])/np.linspace(start=1,stop=100000, num=100000)
    mns3HMC[i,:] = np.cumsum(sHMC[2,1000:niter,i])/np.linspace(start=1,stop=100000, num=100000)



print(np.round(np.mean(tabISO,axis=0),2))
print(np.round(np.mean(tabISO,axis=0)+1.96*np.sqrt(np.var(tabISO,axis=0)/10),2))
print(np.round(np.mean(tabISO,axis=0)-1.96*np.sqrt(np.var(tabISO,axis=0)/10),2))


print(np.round(np.mean(tabHMC,axis=0),2))
print(np.round(np.mean(tabHMC,axis=0)+1.96*np.sqrt(np.var(tabHMC,axis=0)/10),2))
print(np.round(np.mean(tabHMC,axis=0)-1.96*np.sqrt(np.var(tabHMC,axis=0)/10),2))


gradAxISO = np.mean(np.cumsum(dISO[1000:niter,11,],axis=0),axis=1)
gradAxHMC = np.mean(np.cumsum(dISO[1000:niter,11,],axis=0),axis=1)

plt.subplot(1,2,1)

plt.semilogx(gradAxISO,
             0*gradAxISO,
             ':')

plt.semilogx(gradAxHMC,np.mean(mns1HMC,axis=0),'b--',linewidth=2)
#plt.semilogx(gradAxHMC,np.mean(mns1HMC,axis=0)+2.0*np.sqrt(np.var(mns1HMC,axis=0)),'b--')
#plt.semilogx(gradAxHMC,np.mean(mns1HMC,axis=0)-2.0*np.sqrt(np.var(mns1HMC,axis=0)),'b--')
plt.fill_between(gradAxHMC,
                 np.mean(mns1HMC,axis=0)-2.0*np.sqrt(np.var(mns1HMC,axis=0)),
                 np.mean(mns1HMC,axis=0)+2.0*np.sqrt(np.var(mns1HMC,axis=0)),
                 color="blue",alpha=0.3)
                 


plt.semilogx(gradAxISO,np.mean(mns1ISO,axis=0),'red',linewidth=2)
#plt.semilogx(gradAxISO,np.mean(mns1ISO,axis=0)+2.0*np.sqrt(np.var(mns1ISO,axis=0)),'k')
#plt.semilogx(gradAxISO,np.mean(mns1ISO,axis=0)-2.0*np.sqrt(np.var(mns1ISO,axis=0)),'k')
plt.fill_between(gradAxISO,
                 np.mean(mns1ISO,axis=0)-2.0*np.sqrt(np.var(mns1ISO,axis=0)),
                 np.mean(mns1ISO,axis=0)+2.0*np.sqrt(np.var(mns1ISO,axis=0)),
                 color="red",alpha=0.3)



plt.title("$E(\\omega)$")
plt.xlabel("# gradient evaluations")
plt.axis([1e3,2e7,-4,4])

plt.subplot(1,2,2)


plt.semilogx(gradAxISO,
             np.repeat(9.0,len(gradAxISO)),
             ':')

plt.semilogx(gradAxHMC,np.mean(mns2HMC,axis=0),'b--',linewidth=2,label="WALNUTS run. mean")
#plt.semilogx(gradAxHMC,np.mean(mns2HMC,axis=0)+2.0*np.sqrt(np.var(mns2HMC,axis=0)),'b--')
#plt.semilogx(gradAxHMC,np.mean(mns2HMC,axis=0)-2.0*np.sqrt(np.var(mns2HMC,axis=0)),'b--')
plt.fill_between(gradAxHMC,np.mean(mns2HMC,axis=0)+2.0*np.sqrt(np.var(mns2HMC,axis=0)),
                 np.mean(mns2HMC,axis=0)-2.0*np.sqrt(np.var(mns2HMC,axis=0)),
                 color='blue',alpha=0.3,label="WALNUTS 95% int.")


plt.semilogx(gradAxISO,np.mean(mns2ISO,axis=0),'red',linewidth=2,label="iWALNUTS run. mean")
#plt.semilogx(gradAxISO,np.mean(mns2ISO,axis=0)+2.0*np.sqrt(np.var(mns2ISO,axis=0)),'r')
#plt.semilogx(gradAxISO,np.mean(mns2ISO,axis=0)-2.0*np.sqrt(np.var(mns2ISO,axis=0)),'r')

plt.fill_between(gradAxISO,
                np.mean(mns2ISO,axis=0)-2.0*np.sqrt(np.var(mns2ISO,axis=0)),
                np.mean(mns2ISO,axis=0)+2.0*np.sqrt(np.var(mns2ISO,axis=0)),
                color='red', alpha=0.3,label="iWALNUTS 95% int.")



plt.title("$E(\\omega^2)$")
plt.xlabel("# gradient evaluations")
plt.axis([1e3,2e7,-10,20])
plt.legend()

plt.savefig("runningMeansFunnel.pdf")


# plt.figure()
# plt.subplot(1,2,1)
# plt.plot(sISO[0,1000:2000,1])

# plt.subplot(1,2,2)
# plt.plot(sHMC[0,1000:2000,1])


