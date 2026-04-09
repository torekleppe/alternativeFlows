import numpy as np
import scipy as sp
import targets as td
import matplotlib.pyplot as plt
import copy


def BSsolver(p0,o0,g,oSigmaSq,h,eps=1.0e-14):
    z0 = np.array([p0,o0])
    
    maxN = 14
    Tp = np.zeros((maxN,maxN))
    To = np.zeros((maxN,maxN))
    for j in range(maxN):
        n = 2*(j+1)
        hh = h/n

        print("j="+str(j))
        print("n="+str(n))
     
        z = copy.deepcopy(z0)
        for m in range(n):
            z[1] = z[1] + 0.5*hh*oSigmaSq*(z[0]**2-1.0)
            z[0] = z[0]*np.exp(-hh*z[1]) + g*((1.0-np.exp(-hh*z[1]))/z[1])
            z[1] = z[1] + 0.5*hh*oSigmaSq*(z[0]**2-1.0)
        
        zf = z

        Tp[j,0] = zf[0]
        To[j,0] = zf[1]
        print(zf)
        
        for l in range(1,j+1):
            fac = 1.0/((n/(2*(j-l+1)))**2-1.0)
            Tp[j,l] = Tp[j,l-1] + fac*(Tp[j,l-1]-Tp[j-1,l-1])
            To[j,l] = To[j,l-1] + fac*(To[j,l-1]-To[j-1,l-1])
        
        errp = np.abs(Tp[j,j]-Tp[j,j-1])/(eps*(1.0+np.abs(Tp[j,j])))
        erro = np.abs(To[j,j]-To[j,j-1])/(eps*(1.0+np.abs(To[j,j])))
        
        pBest = Tp[j,j]
        oBest = To[j,j]
        
        print(errp)
        print(erro)
        if(errp<1.0 and erro<1.0):
            print("success")
            break
        
    return(pBest,oBest,max(errp,erro))
g = 30.0       
oSigmaSq = 4.1
h = 0.3
p0 = 1.0
o0 = 1.2
(p1,o1,err) = BSsolver(p0, o0, g, oSigmaSq, h=h)

(p0b,o0b,errb) = BSsolver(-p1, -o1, g, oSigmaSq, h=h)


def odeS(t,y):
    return(np.array([g-y[0]*y[1],oSigmaSq*(y[0]**2-1.0)]))

outS = sp.integrate.solve_ivp(odeS,t_span=(0,h),y0=np.array([p0,o0]),rtol=1.0e-13,atol=1.0e-13)

outSB = sp.integrate.solve_ivp(odeS,t_span=(0,h),y0=np.array([-outS.y[0,-1],-outS.y[1,-1]]),rtol=1.0e-13,atol=1.0e-13)




def Bsolver(p0,o0,g,oSigmaSq,h,eps=1.0e-14):
    oldp = p0
    oldo = o0
    for c in range(20):
        nstep = 2**c
        hh = h/nstep
        p = p0
        o = o0
        
        for i in range(nstep):
            o = o + 0.5*hh*oSigmaSq*(p**2-1.0)
            p = p*np.exp(-hh*o) + g*((1.0-np.exp(-hh*o))/o)
            o = o + 0.5*hh*oSigmaSq*(p**2-1.0)
        
        dev = max(np.max(np.abs(p-oldp)),np.max(np.abs(o-oldo)))
        
        
        print(dev)
        if(dev<eps):
            break
        
        oldp = p
        oldo = o
        
        
    
    


lp = td.funnel1 # td.corrGauss # td.modFunnel #    td.stdGauss # 
sigmaSq = np.sqrt(1.0)**2

def ode(t,y):
    d = len(y)//4
    q = y[0:d]
    p = y[d:(2*d)]
    o = y[(2*d):(3*d)]
    eta = y[(3*d):(4*d)]
    
    [f,g] = lp(q)
    
    qdot = p
    pdot = g - o*p
    odot = sigmaSq*(p**2-1.0)
    etadot = o*eta 
    
    return(np.concat((qdot,pdot,odot,etadot)))


def Ham(y):
    d = len(y)//4
    q = y[0:d]
    p = y[d:(2*d)]
    o = y[(2*d):(3*d)]
    eta = y[(3*d):(4*d)]
    
    [f,_] = lp(q)
    
    Ham = -f + 0.5*np.dot(p,p) + 0.5/sigmaSq*np.dot(o,o) + np.sum(np.log(eta)) 
    
    return(Ham)


q0 =  np.array([9.55,-100.5]) #np.random.normal(size=2)  # 
p0 = np.random.normal(size=len(q0))
o0 = np.sqrt(sigmaSq)*np.random.normal(size=len(q0))
eta0 = np.repeat(1.0, len(q0))

y0 = np.concat((q0,p0,o0,eta0))

Tmax = 200.0    
out = sp.integrate.solve_ivp(ode,t_span=(0,Tmax),y0=y0,t_eval=np.linspace(0, Tmax,num=2000),atol=1e-10,rtol=1e-10)

y0b = copy.deepcopy(out.y[:,-1])
d = (len(y0b))//4
y0b[d:(2*d)] = -y0b[d:(2*d)]
y0b[(2*d):(3*d)] = -y0b[(2*d):(3*d)]

Hams = np.zeros_like(out.t)
for i in range(len(Hams)):
     Hams[i] = Ham(out.y[:,i])


outb = sp.integrate.solve_ivp(ode,t_span=(0,Tmax),y0=y0b,t_eval=np.linspace(0, Tmax,num=2000),atol=1e-10,rtol=1e-10)


plt.subplot(1,3,1)
plt.plot(out.y[0,:],out.y[1,:])
plt.plot(outb.y[0,:],outb.y[1,:])

plt.subplot(1,3,2)
plt.plot(out.y[4,:])
plt.plot(out.y[5,:])
plt.plot(-outb.y[4,:])
plt.plot(-outb.y[5,:])
# plt.plot(-outb.y[-1,:])
# #plt.plot(-np.flip(outb.y[-2,:]))


plt.subplot(1,3,3)
plt.plot(Hams)



