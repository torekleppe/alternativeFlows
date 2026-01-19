import numpy as np
import matplotlib.pyplot as plt
import arviz as az

sISO=np.load("toepSamISO.npy")
snISO=np.load("toepSamnISO.npy")
sHMC=np.load("toepSamHMC.npy")
snHMC=np.load("toepSamnHMC.npy")
dISO=np.load("toepDiagISO.npy")
dnISO=np.load("toepDiagnISO.npy")
dHMC=np.load("toepDiagHMC.npy")
dnHMC=np.load("toepDiagnHMC.npy")

niter = 6000
nsample = niter-1000

s2Var = 0.3193203822569591


tabISO = np.zeros((10,6))
tabnISO = np.zeros((10,6))
tabHMC = np.zeros((10,6))
tabnHMC = np.zeros((10,6))
mns1ISO = np.zeros((10,nsample))
mns1HMC = np.zeros((10,nsample))

mns2ISO = np.zeros((10,nsample))
mns2HMC = np.zeros((10,nsample))

mns3ISO = np.zeros((10,nsample))
mns3HMC = np.zeros((10,nsample))

for i in range(10):
    tabISO[i,0] = az.ess(sISO[2,1000:niter,i])
    tabnISO[i,0] = az.ess(snISO[2,1000:niter,i])
    tabHMC[i,0] = az.ess(sHMC[2,1000:niter,i])
    tabnHMC[i,0] = az.ess(snHMC[2,1000:niter,i])
    
    tabISO[i,1] = 1000*tabISO[i,0]/np.sum(dISO[1000:niter,11,i])
    tabnISO[i,1] = 1000*tabnISO[i,0]/np.sum(dnISO[1000:niter,11,i])
    tabHMC[i,1] = 1000*tabHMC[i,0]/np.sum(dHMC[1000:niter,11,i])
    tabnHMC[i,1] = 1000*tabnHMC[i,0]/np.sum(dnHMC[1000:niter,11,i])
    
    tabISO[i,2] = az.ess(sISO[2,1000:niter,i]**2)
    tabnISO[i,2] = az.ess(snISO[2,1000:niter,i]**2)
    tabHMC[i,2] = az.ess(sHMC[1,1000:niter,i]**2)
    tabnHMC[i,2] = az.ess(snHMC[1,1000:niter,i]**2)
    
    tabISO[i,3] = 1000*tabISO[i,2]/np.sum(dISO[1000:niter,11,i])
    tabnISO[i,3] = 1000*tabnISO[i,2]/np.sum(dnISO[1000:niter,11,i])
    tabHMC[i,3] = 1000*tabHMC[i,2]/np.sum(dHMC[1000:niter,11,i])
    tabnHMC[i,3] = 1000*tabnHMC[i,2]/np.sum(dnHMC[1000:niter,11,i])
    
    tabISO[i,4] = np.mean(dISO[:,3,i])
    tabHMC[i,4] = np.mean(dHMC[:,3,i])
    tabnISO[i,4] = np.mean(dnISO[:,3,i])
    tabnHMC[i,4] = np.mean(dnHMC[:,3,i])
    
    tabISO[i,5] = np.mean(dISO[:,-1,i])
    tabHMC[i,5] = np.mean(dHMC[:,-1,i])
    tabnISO[i,5] = np.mean(dnISO[:,-1,i])
    tabnHMC[i,5] = np.mean(dnHMC[:,-1,i])
    
    mns1ISO[i,:] = np.cumsum(sISO[0,1000:niter,i])/np.linspace(start=1,stop=nsample, num=nsample)
    mns1HMC[i,:] = np.cumsum(sHMC[0,1000:niter,i])/np.linspace(start=1,stop=nsample, num=nsample)

    mns2ISO[i,:] = np.cumsum(sISO[2,1000:niter,i])/np.linspace(start=1,stop=nsample, num=nsample)
    mns2HMC[i,:] = np.cumsum(sHMC[2,1000:niter,i])/np.linspace(start=1,stop=nsample, num=nsample)
    
    


print(np.round(np.mean(tabISO,axis=0),2))
print(np.round(np.mean(tabISO,axis=0)+1.96*np.sqrt(np.var(tabISO,axis=0)/10),2))
print(np.round(np.mean(tabISO,axis=0)-1.96*np.sqrt(np.var(tabISO,axis=0)/10),2))

print('--')
print(np.round(np.mean(tabHMC,axis=0),2))
print(np.round(np.mean(tabHMC,axis=0)+1.96*np.sqrt(np.var(tabHMC,axis=0)/10),2))
print(np.round(np.mean(tabHMC,axis=0)-1.96*np.sqrt(np.var(tabHMC,axis=0)/10),2))

print('--')

print(np.round(np.mean(tabnISO,axis=0),2))
print(np.round(np.mean(tabnISO,axis=0)+1.96*np.sqrt(np.var(tabnISO,axis=0)/10),2))
print(np.round(np.mean(tabnISO,axis=0)-1.96*np.sqrt(np.var(tabnISO,axis=0)/10),2))

print('--')
print(np.round(np.mean(tabnHMC,axis=0),2))
print(np.round(np.mean(tabnHMC,axis=0)+1.96*np.sqrt(np.var(tabnHMC,axis=0)/10),2))
print(np.round(np.mean(tabnHMC,axis=0)-1.96*np.sqrt(np.var(tabnHMC,axis=0)/10),2))

gradAxISO = np.mean(np.cumsum(dISO[1000:niter,11,],axis=0),axis=1)
gradAxHMC = np.mean(np.cumsum(dISO[1000:niter,11,],axis=0),axis=1)

plt.subplot(1,2,1)

plt.semilogx(gradAxISO,
             0*gradAxISO,
             ':')
plt.semilogx(gradAxISO,np.mean(mns1ISO,axis=0),'k',linewidth=2)
plt.semilogx(gradAxISO,np.mean(mns1ISO,axis=0)+2.0*np.sqrt(np.var(mns1ISO,axis=0)),'k')
plt.semilogx(gradAxISO,np.mean(mns1ISO,axis=0)-2.0*np.sqrt(np.var(mns1ISO,axis=0)),'k')

plt.semilogx(gradAxHMC,np.mean(mns1HMC,axis=0),'b--',linewidth=2)
plt.semilogx(gradAxHMC,np.mean(mns1HMC,axis=0)+2.0*np.sqrt(np.var(mns1HMC,axis=0)),'b--')
plt.semilogx(gradAxHMC,np.mean(mns1HMC,axis=0)-2.0*np.sqrt(np.var(mns1HMC,axis=0)),'b--')
plt.title("$E(q_1)$")
plt.xlabel("# gradient evaluations")
#plt.axis([1e3,2e7,-2,2])

plt.subplot(1,2,2)

plt.semilogx(gradAxISO,
             np.repeat(0.0,len(gradAxISO)),
             ':')
plt.semilogx(gradAxISO,np.mean(mns2ISO,axis=0),'k',linewidth=2,label="iWALNUTS")
plt.semilogx(gradAxISO,np.mean(mns2ISO,axis=0)+2.0*np.sqrt(np.var(mns2ISO,axis=0)),'k')
plt.semilogx(gradAxISO,np.mean(mns2ISO,axis=0)-2.0*np.sqrt(np.var(mns2ISO,axis=0)),'k')

plt.semilogx(gradAxHMC,np.mean(mns2HMC,axis=0),'b--',linewidth=2,label="WALNUTS")
plt.semilogx(gradAxHMC,np.mean(mns2HMC,axis=0)+2.0*np.sqrt(np.var(mns2HMC,axis=0)),'b--')
plt.semilogx(gradAxHMC,np.mean(mns2HMC,axis=0)-2.0*np.sqrt(np.var(mns2HMC,axis=0)),'b--')
plt.title("$E(\\sum_i q_i/d)$")
plt.xlabel("# gradient evaluations")
#plt.axis([1e3,2e7,-2,2])
plt.legend()

plt.savefig("runningMeansToep.pdf")





# plt.subplot(1,3,3)

# plt.semilogx(gradAxISO,
#              np.repeat(9.0,len(gradAxISO)),
#              ':')
# plt.semilogx(gradAxISO,np.mean(mns3ISO,axis=0),'k')
# plt.semilogx(gradAxISO,np.mean(mns3ISO,axis=0)+2.0*np.sqrt(np.var(mns3ISO,axis=0)),'k')
# plt.semilogx(gradAxISO,np.mean(mns3ISO,axis=0)-2.0*np.sqrt(np.var(mns3ISO,axis=0)),'k')

# plt.semilogx(gradAxHMC,np.mean(mns3HMC,axis=0),'b--')
# plt.semilogx(gradAxHMC,np.mean(mns3HMC,axis=0)+2.0*np.sqrt(np.var(mns3HMC,axis=0)),'b--')
# plt.semilogx(gradAxHMC,np.mean(mns3HMC,axis=0)-2.0*np.sqrt(np.var(mns3HMC,axis=0)),'b--')







#plt.axis([1e4,2e7,-10,20])
