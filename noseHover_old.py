import numpy as np
import scipy as sp
import targets as td
import matplotlib.pyplot as plt
import copy


lp = td.funnel1 # td.corrGauss # td.stdGauss # td.modFunnel # 
sigmaSq = 2.0**2

def ode(t,y):
    d = (len(y)-1)//2
    q = y[0:d]
    p = y[d:(2*d)]
    o = y[-1]
    
    
    [f,g] = lp(q)
    
    qdot = p
    pdot = g - o*p
    odot = sigmaSq*(np.dot(p,p)-d)
    
    
    return(np.concat((qdot,pdot,np.array([odot]))))


def Ham(y):
    d = (len(y)-1)//2
    q = y[0:d]
    p = y[d:(2*d)]
    o = y[-1]
    
    [f,_] = lp(q)
    
    Ham = -f + 0.5*np.dot(p,p) + 0.5/sigmaSq*(o**2)
    
    return(Ham)


q0 = np.random.normal(size=2) # np.array([-0.55,-0.5]) # 
p0 = np.random.normal(size=2)

o0 = np.sqrt(sigmaSq)*np.random.normal()

# y0 = np.concat((q0,p0,np.array([o0])))

# Tmax = 10.0    
# out = sp.integrate.solve_ivp(ode,t_span=(0,Tmax),y0=y0,t_eval=np.linspace(0, Tmax,num=2000),atol=1e-10,rtol=1e-10)

# y0b = copy.deepcopy(out.y[:,-1])
# d = (len(y0b)-1)//2
# y0b[d:(2*d)] = -y0b[d:(2*d)]
# y0b[-1] = -y0b[-1]

# Hams = np.zeros_like(out.t)
# for i in range(len(Hams)):
#     Hams[i] = Ham(out.y[:,i])


# outb = sp.integrate.solve_ivp(ode,t_span=(0,Tmax),y0=y0b,t_eval=np.linspace(0, Tmax,num=2000),atol=1e-10,rtol=1e-10)


# plt.subplot(1,3,1)
# plt.plot(out.y[0,:],out.y[1,:])
# plt.plot(outb.y[0,:],outb.y[1,:])

# plt.subplot(1,3,2)
# plt.plot(out.y[-1,:])
# #plt.plot(out.y[-2,:])
# plt.plot(-outb.y[-1,:])
# #plt.plot(-np.flip(outb.y[-2,:]))


# plt.subplot(1,3,3)
# plt.plot(Hams)




class nhIntegrator:
    
    def integrate(self,lpFun,q0,p0,o0,h=0.05,nstep=8*64):
        
        q = q0
        p = p0
        o = o0
        
        d = len(q)
        
        eta = 1.0
        
        [f,g] = lpFun(q)
        
        qs = np.zeros((len(q),nstep+1))
        qs[:,0] = q
        
        HE = np.zeros(nstep+1)
        H = np.zeros(nstep+1)
        lwt = np.zeros(nstep+1)
        HE[0] = -f + 0.5*np.dot(p,p) + (0.5/sigmaSq)*o**2 + np.log(eta)
        H[0] = -f + 0.5*np.dot(p,p) + (0.5/sigmaSq)*o**2
        
        
        os = np.zeros(nstep+1)
        os[0] = o
        ljac = 0.0
        
        for i in range(nstep):
            ph = p*np.exp(-0.5*h*o) + g*((1.0-np.exp(-0.5*h*o))/o)
            etah = eta*np.exp(0.5*h*d*o)
            
            q = q + h*ph
            [f,g] = lpFun(q)
            oOld = o
            o = o + h*sigmaSq*(np.dot(ph,ph)-d)
            
            ljac -= 0.5*h*d*(o+oOld)
            
            p = ph*np.exp(-0.5*h*o) + g*((1.0-np.exp(-0.5*h*o))/o)
            eta = etah*np.exp(0.5*h*d*o)
            
            qs[:,i+1] = q
            HE[i+1] = -f + 0.5*np.dot(p,p) + (0.5/sigmaSq)*o**2 + np.log(eta)
            H[i+1] = -f + 0.5*np.dot(p,p) + (0.5/sigmaSq)*o**2
            
            lwt[i+1] = H[0]-H[i+1] + ljac
            
            os[i+1] = o
            
        
        print(ljac)
        plt.subplot(1,3,1)
        plt.plot(qs[0,:],qs[1,:])
        plt.subplot(1,3,2)
        plt.plot(lwt)
        plt.subplot(1,3,3)
        plt.plot(os)
        
        
        

#q0 = np.array([2.0])
#p0 = np.array([1.0])
#o0 = -1.0
        
        

ii = nhIntegrator()
ii.integrate(lp, q0, p0, o0)






