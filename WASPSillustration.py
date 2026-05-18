import numpy as np
import isokinetic as iso
import targets as td
import matplotlib.pyplot as plt
import copy



np.random.seed(1)
d = 2

z1 = np.random.normal(size=d)
z2 = np.random.normal(size=d)
eta = (1.0/np.sqrt(np.sum(z1**2)))*z1
z2 = z2 - (np.sum(z2*eta))*eta
gam = (1.0/np.sqrt(np.sum(z2**2)))*z2




plt.figure(figsize=(5,5))
fig,ax = plt.subplots()
plt.axis([-2,2,-2,2])
ax.set_box_aspect(1)


#plt.arrow(0,0,eta[0],eta[1])

plt.plot(0,0,'sb',zorder=11)

plt.quiver(0,0,eta[0], eta[1],angles='xy', scale_units='xy',scale=1,width=0.004)
plt.text(eta[0], eta[1]-0.01,'$\\eta$')

plt.plot([0,10*gam[0]],[0,10*gam[1]],'r',lw=3)
plt.quiver(0,0,gam[0], gam[1],angles='xy', scale_units='xy',scale=1,zorder=10,width=0.004)
plt.text(gam[0], gam[1]-0.05,'$\\gamma$')


lp = td.stdGauss
s = iso.IKState()
q0 = np.array([1.5,0.0])

plt.plot(q0[0],q0[1],'og',zorder=11)

tp = iso.IKtuningPars(hMacro=0.3)

step = iso.adaptIKstepE()

s.firstEval(lp, q0, tp)
s.momentumRefresh(lp, tp)

sf = copy.deepcopy(s)
nf = 100
qsf = np.zeros((2,100))
qsf[:,0] = sf.q

for i in range(100):
    qOld = sf.q
    (sf,ljac) = step(sf,lp,tp)
    qsf[:,i+1] = sf.q
    
    cqs = sf.q
    cq = qOld
    
    cqseta = np.dot(cqs,eta)
    cqeta = np.dot(cq,eta)
    
    ts = cqeta/(cqeta-cqseta)
    
    if(ts>0.0 and ts < 1.0):
        if((1.0-ts)*np.dot(cq,gam)+ts*np.dot(cqs,gam)>0.0):
            nf=i+1
            break
    
    
    

plt.plot(qsf[0,0:nf],qsf[1,0:nf],'.k')
plt.plot(qsf[0,0:(nf+1)],qsf[1,0:(nf+1)],'-k',lw=0.5)
plt.plot(sf.q[0],sf.q[1],'x',color='k')


sb = copy.deepcopy(s)
sb.momentumFlip()
nb = 100
qsb = np.zeros((2,100))
qsb[:,0] = sb.q

for i in range(100):
    qOld = sb.q
    (sb,ljac) = step(sb,lp,tp)
    qsb[:,i+1] = sb.q
    
    cqs = sb.q
    cq = qOld
    
    cqseta = np.dot(cqs,eta)
    cqeta = np.dot(cq,eta)
    
    ts = cqeta/(cqeta-cqseta)
    
    if(ts>0.0 and ts < 1.0):
        if((1.0-ts)*np.dot(cq,gam)+ts*np.dot(cqs,gam)>0.0):
            nb=i+1
            break
    
    
    

plt.plot(qsb[0,0:nb],qsb[1,0:nb],'.',color='grey')
plt.plot(qsb[0,0:(nb+1)],qsb[1,0:(nb+1)],'-',color='grey',lw=0.5)
plt.plot(sb.q[0],sb.q[1],'x',color="grey")


plt.savefig("WASPSill.pdf")

