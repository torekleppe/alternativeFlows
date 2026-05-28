import numpy as np
import copy
import pandas as pd


LOG_ZERO_THRESH = -700.0

def log_mu_prod(theta):
    d = theta.shape[0]
    N = d // 11
    s = 0.0
    for k in range(N):
        v = theta[11*k]
        x2 = 0.0
        for i in range(1, 11):
            x2 += theta[11*k+i] * theta[11*k+i]
        s += -v*v/18.0 - 5.0*v - 0.5*np.exp(-v)*x2
    return s



def grad_log_mu_prod(theta):
    d = theta.shape[0]
    N = d // 11
    out = np.empty(d)
    for k in range(N):
        v = theta[11*k]
        x2 = 0.0
        for i in range(1, 11):
            x2 += theta[11*k+i] * theta[11*k+i]
        e_neg = np.exp(-v)
        out[11*k] = -v/9.0 - 5.0 + 0.5*e_neg*x2
        for i in range(1, 11):
            out[11*k+i] = -e_neg * theta[11*k+i]
    return out


def funnelProdLp(q,hessian=False):
    return(log_mu_prod(q),grad_log_mu_prod(q))


def G_diag_prod(theta):
    d = theta.shape[0]
    N = d // 11
    G = np.empty(d)
    for k in range(N):
        v = theta[11*k]
        e_neg = np.exp(-v)
        x2 = 0.0
        for i in range(1, 11):
            x2 += theta[11*k+i] * theta[11*k+i]
        G[11*k] = 1.0/9.0 + 0.5*e_neg*x2
        for i in range(1, 11):
            G[11*k+i] = e_neg + 0.05   # wide-end floor: caps 1/G and binding freq at v>>0
    return G







def hat_H(theta, theta_t, rho, rho_t, omega):
    d = theta.shape[0]
    G_main = G_diag_prod(theta); G_aux = G_diag_prod(theta_t)
    kin_main = 0.0; kin_aux = 0.0; logdet_main = 0.0; logdet_aux = 0.0
    for i in range(d):
        kin_main    += rho[i]*rho[i] / G_aux[i]
        kin_aux     += rho_t[i]*rho_t[i] / G_main[i]
        logdet_main += np.log(G_main[i])
        logdet_aux  += np.log(G_aux[i])
    bind = 0.0
    for i in range(d):
        diff = theta[i] - theta_t[i]
        bind += diff*diff
    return (-log_mu_prod(theta)
            + 0.5*kin_main + 0.5*logdet_aux
            + 0.5*kin_aux  + 0.5*logdet_main
            + 0.5*omega*bind)

class funneProdState:
    
    def firstEval(self,lpFun,q,tp):
        [f,g] = funnelProdLp(q)
        self.q = q.copy()
        self.qt = q.copy()
        self.f = f
        self.g = g
        self.p = np.repeat(np.nan,len(q))
        self.pt = np.repeat(np.nan,len(q))
        self.Ham = np.nan
        
        
    def momentumRefresh(self,lpFun,tp):
        self.qt = self.q.copy() + np.sqrt(1.0/tp.omega)*np.random.normal(size=len(self.q))
        G_main = G_diag_prod(self.q); G_aux = G_diag_prod(self.qt)
        self.p = np.sqrt(G_aux)*np.random.normal(size=len(self.q))
        self.pt = np.sqrt(G_main)*np.random.normal(size=len(self.q))
        self.Ham = hat_H(self.q,self.qt,self.p,self.pt,tp.omega)
        
    def momentumFlip(self):
        self.p = -self.p
        self.pt = -self.pt
    
    
    
    
    def velocity(self):
        return(self.p)
    
    def __str__(self):
        return ("funneProdState class:\nq: " + str(self.q) + "\np: " + str(self.p) + "\nHamiltonian: " + str(self.Ham))

    def __repr__(self):
        return ("funneProdState class:\nq: " + str(self.q) + "\np: " + str(self.p) + "\nHamiltonian: " + str(self.Ham))

class funneProdTuningPars:
    
    def __init__(self, hMacro=0.1, delta=0.1, Cmin=0,Cmax=10,omega=5.0):
        
        self.hMacro = hMacro
        self.delta = delta
        
        self.Cmin = Cmin
        self.Cmax = Cmax
        self.omega = omega





class FPintegrator:
    
    def phi_AC_half(self,theta, theta_t, rho, rho_t, h, omega):
        """Flow of H_A + H_C/2 for time h. theta held fixed.
           Exact harmonic for (tilde_theta, tilde_rho); closed-form kick on rho."""
        # Fully-fused, allocation-light version: one block loop, scalar locals,
        # G and grad-log-mu inlined.  Only 3 output arrays are allocated.
        d = theta.shape[0]; Nf = d // 11
        tt_out = np.empty(d); rho_out = np.empty(d); rt_out = np.empty(d)
        for k in range(Nf):
            b = 11*k; vv = theta[b]; e = np.exp(-vv)
            x2 = 0.0
            for ii in range(1, 11):
                x2 += theta[b+ii]*theta[b+ii]
            Gv = 1.0/9.0 + 0.5*e*x2
            s_kin = 0.0; s_logdet = 0.0; ir0 = 0.0
            for c in range(11):
                i = b + c
                if c == 0:
                    G_i = Gv; dg = -0.5*e*x2
                    glm = -vv/9.0 - 5.0 + 0.5*e*x2
                else:
                    G_i = e + 0.05; dg = -e
                    glm = -e*theta[i]
                u_i = theta_t[i] - theta[i]
                om2 = omega/(2.0*G_i); om = np.sqrt(om2)
                coh = np.cos(om*h); sih = np.sin(om*h); si2h = np.sin(2.0*om*h)
                cosq = sih*sih
                iu = u_i/om*sih + 2.0*rho_t[i]/omega*(1.0 - coh)
                A = 0.5*h + si2h/(4.0*om); B = 0.5*h - si2h/(4.0*om)
                ir = rho_t[i]*rho_t[i]*A + G_i*G_i*om2*u_i*u_i*B - G_i*rho_t[i]*u_i*cosq
                tt_out[i] = theta[i] + (u_i*coh + rho_t[i]/(G_i*om)*sih)
                rt_out[i] = rho_t[i]*coh - G_i*om*u_i*sih
                rho_out[i] = rho[i] + 0.5*omega*iu + h*glm
                s_kin += dg/(G_i*G_i)*ir
                s_logdet += dg/G_i
                if c == 0:
                    ir0 = ir
            rho_out[b] += 0.5*s_kin - 0.5*h*s_logdet
            for jj in range(1, 11):
                sk = (e*theta[b+jj])/(Gv*Gv)*ir0
                sl = (e*theta[b+jj])/Gv
                rho_out[b+jj] += 0.5*sk - 0.5*h*sl
        return theta, tt_out, rho_out, rt_out
    
    
    
    def phi_BC_half(self,theta, theta_t, rho, rho_t, h, omega):
        """Flow of H_B + H_C/2 for time h. tilde_theta held fixed.
           Exact harmonic for (theta, rho); closed-form kick on tilde_rho."""
        # Fully-fused, allocation-light version (metric at theta_t; no potential).
        d = theta.shape[0]; Nf = d // 11
        th_out = np.empty(d); r_out = np.empty(d); rt_out = np.empty(d)
        for k in range(Nf):
            b = 11*k; vk = theta_t[b]; e = np.exp(-vk)
            x2 = 0.0
            for ii in range(1, 11):
                x2 += theta_t[b+ii]*theta_t[b+ii]
            Gv = 1.0/9.0 + 0.5*e*x2
            s_kin = 0.0; s_logdet = 0.0; ir0 = 0.0
            for c in range(11):
                i = b + c
                if c == 0:
                    G_i = Gv; dg = -0.5*e*x2
                else:
                    G_i = e + 0.05; dg = -e
                diff = theta[i] - theta_t[i]
                om2 = omega/(2.0*G_i); om = np.sqrt(om2)
                coh = np.cos(om*h); sih = np.sin(om*h); si2h = np.sin(2.0*om*h)
                cosq = sih*sih
                iv = diff/om*sih + 2.0*rho[i]/omega*(1.0 - coh)
                A = 0.5*h + si2h/(4.0*om); B = 0.5*h - si2h/(4.0*om)
                ir = rho[i]*rho[i]*A + G_i*G_i*om2*diff*diff*B - G_i*rho[i]*diff*cosq
                th_out[i] = theta_t[i] + (diff*coh + rho[i]/(G_i*om)*sih)
                r_out[i] = rho[i]*coh - G_i*om*diff*sih
                rt_out[i] = rho_t[i] + 0.5*omega*iv
                s_kin += dg/(G_i*G_i)*ir
                s_logdet += dg/G_i
                if c == 0:
                    ir0 = ir
            rt_out[b] += 0.5*s_kin - 0.5*h*s_logdet
            for jj in range(1, 11):
                sk = (e*theta_t[b+jj])/(Gv*Gv)*ir0
                sl = (e*theta_t[b+jj])/Gv
                rt_out[b+jj] += 0.5*sk - 0.5*h*sl
        return th_out, theta_t, r_out, rt_out
    
    
    
    def strang_step(self,theta, theta_t, rho, rho_t, h, omega):
        theta, theta_t, rho, rho_t = self.phi_AC_half(theta, theta_t, rho, rho_t, 0.5*h, omega)
        theta, theta_t, rho, rho_t = self.phi_BC_half(theta, theta_t, rho, rho_t, h, omega)
        theta, theta_t, rho, rho_t = self.phi_AC_half(theta, theta_t, rho, rho_t, 0.5*h, omega)
        return theta, theta_t, rho, rho_t
    

    def strangIntegrator(self,s,lpFun,omega,h=0.1,nstep=1):
        q = s.q.copy()
        qt = s.qt.copy()
        p = s.p.copy()
        pt = s.pt.copy()
        
        for i in range(nstep):
            (q,qt,p,pt) = self.strang_step(q,qt,p,pt,h,omega)
        
        sOut = funneProdState()
        [f,g] = funnelProdLp(q)
        sOut.f = f
        sOut.g = g
        sOut.q = q.copy()
        sOut.qt = qt.copy()
        sOut.p = p.copy()
        sOut.pt = pt.copy()
        sOut.Ham = hat_H(q,qt,p,pt,omega)
        
        return(sOut)
        


class adaptfunnelProdstepE(FPintegrator):
    
    def __init__(self):
        self.gradEval = 0
        self.Hams = []
        self.Ifs = []
        self.Ibs = []
        self.basic = []
        self.Cobs = []
    
    def name(self):
        return("adaptfunnelProdstepE")
    
    def reset(self):
        self.gradEval = 0
        self.Hams = []
        self.Ifs = []
        self.Ibs = []
        self.basic = []
        self.Cobs = []

    def diagnostics(self):
        
        return (pd.Series([self.gradEval, np.max(self.Hams)-np.min(self.Hams),
                          np.min(self.Ifs),
                          np.max(self.Ifs),
                          np.mean(self.basic)],
                         index=['gradEvals', 'energyErr',
                                'minIf', 'maxIf', 'propBasic']))


    
    def getState(self):
        return(funneProdState())
    
    def propBasic(self):
        return(np.mean(self.basic))
    
    def Cobs(self):
        return(np.median(self.Cobs))
    
    def __call__(self,s,lpFun,tp):
        if not self.Hams:
            self.Hams.append(s.Ham)
            
        If = tp.Cmax
        
        for c in range(tp.Cmin,tp.Cmax+1):
            nstep = 2**c
            
            sOut = self.strangIntegrator(s,lpFun,tp.omega,h=tp.hMacro/nstep,nstep=nstep)
            self.gradEval += nstep
            
            Eerr = np.abs(sOut.Ham-s.Ham)
            
            Cobs = np.abs(Eerr)*nstep**2/(tp.hMacro**3)
            
            if(Eerr < tp.delta):
                If = c
                break
            
        Ib = If
        self.Cobs.append(Cobs)
        
        if(If>tp.Cmin):
            self.basic.append(0)
            sB = copy.deepcopy(sOut)
            sB.momentumFlip()
            
            for c in range(tp.Cmin,If):
                nstep = 2**c
                
                sTmp = self.strangIntegrator(sB,lpFun,tp.omega,h=tp.hMacro/nstep,nstep=nstep)
                self.gradEval += nstep
                
                Eerr = np.abs(sTmp.Ham-sB.Ham)
                
                if(Eerr < tp.delta):
                    Ib = c
                    break
        
        else:
            self.basic.append(1)

        self.Ifs.append(If)
        self.Ibs.append(Ib)
        self.Hams.append(sOut.Ham)
        
        lwt = 0.0 if Ib==If else LOG_ZERO_THRESH
        
        
        return(sOut,lwt)





#s = funneProdState()
#q = np.random.normal(size=3*11)

#tp = funneProdTuningPars()
#s.firstEval(0, q, tp)
#s.momentumRefresh(0, tp)


#ii = FPintegrator()
#s1 = ii.strangIntegrator(s, 0, tp.omega,h=0.1,nstep=10)

#s1.momentumFlip()
#s0b = ii.strangIntegrator(s1, 0, tp.omega,h=0.1,nstep=10)



import WALNUTSP as wp
import isokinetic as iso
import hamiltonian as hmc
import matplotlib.pyplot as plt
import arviz as az

np.random.seed(1)

N = 4
q0 = np.random.normal(size=N*11)
for i in range(N):
    q0[11*i] = 0.0

# wfp = wp.WALNUTSP0() 
# ww = wp.WALNUTSP0()



# # self-tuning
# wfp.run(0,q0=q0,step=adaptfunnelProdstepE(),tp0=funneProdTuningPars(),niter=6000)
# ww.run(lpFun=funnelProdLp,q0=q0,step=iso.adaptIKstepE(),tp0=iso.IKtuningPars(),niter=6000)

# print(az.ess(wfp.samples[0,1000:6000])/np.sum(wfp.diagnostics.gradEvals[1000:6000]))
# print(az.ess(ww.samples[0,1000:6000])/np.sum(ww.diagnostics.gradEvals[1000:6000]))

# typical output
#5.064632552142031e-06
#3.796776855558673e-05

wfpf = wp.WALNUTSP0() # offpiste
wwf = wp.WALNUTSP0() # isokinetic
wwhf = wp.WALNUTSP0() # hamiltonian

# static tuning
wfpf.run(0,q0=q0,step=adaptfunnelProdstepE(),tp0=funneProdTuningPars(hMacro=0.1,delta=0.7),niter=5000,nwarmup=0)
wwf.run(lpFun=funnelProdLp,q0=q0,step=iso.adaptIKstepE(),tp0=iso.IKtuningPars(hMacro=0.1,delta=0.7),niter=5000,nwarmup=0)
wwhf.run(lpFun=funnelProdLp,q0=q0,step=hmc.adaptHMCstepE(),tp0=hmc.HMCtuningPars(hMacro=0.1,delta=0.7),niter=5000,nwarmup=0)

print(1000*az.ess(wfpf.samples[0,:])/np.sum(wfpf.diagnostics.gradEvals))
print(1000*az.ess(wwf.samples[0,:])/np.sum(wwf.diagnostics.gradEvals))
print(1000*az.ess(wwhf.samples[0,:])/np.sum(wwhf.diagnostics.gradEvals))

# typical output
#0.1292671928197213
#0.024160139429897906
#0.028620482400480693