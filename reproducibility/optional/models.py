"""HH (mV, ms, µA/cm²) and Izh (mV-scale, ms, model input units).

Uniform samples are right continuous at Izh resets. Internal RK4 steps are
split at a threshold root, then the remaining fraction is integrated.
"""
import numpy as np
from numba import njit
from scipy.integrate import solve_ivp

@njit(cache=True)
def vtrap(z):
    if abs(z)<1e-5:
        return 10.+z/2.+z*z/120.-z**4/720000.
    return z/(-np.expm1(-z/10.))

@njit(cache=True)
def rates(v):
    return np.array([.1*vtrap(v+40),4*np.exp(-(v+65)/18),
        .07*np.exp(-(v+65)/20),1/(1+np.exp(-(v+35)/10)),
        .01*vtrap(v+55),.125*np.exp(-(v+65)/80)])

@njit(cache=True)
def initial(model):
    if model==1:
        return np.array([-65.,-13.])
    a=rates(-65.)
    return np.array([-65.,a[0]/(a[0]+a[1]),a[2]/(a[2]+a[3]),a[4]/(a[4]+a[5])])

@njit(cache=True)
def rhs(model,t,x,j0,amp,period,kappa,beta):
    cur=kappa*(j0+amp*np.sin(2*np.pi*t/period))+beta
    if model==1:
        v,u=x
        return np.array([.04*v*v+5*v+140-u+cur,.02*(.2*v-u)])
    v,m,h,n=x
    rr=rates(v)
    return np.array([cur-120*m**3*h*(v-50)-36*n**4*(v+77)-.3*(v+54.387),
        rr[0]*(1-m)-rr[1]*m,rr[2]*(1-h)-rr[3]*h,rr[4]*(1-n)-rr[5]*n])

@njit(cache=True)
def rk4(model,t,x,h,j0,amp,period,kappa,beta):
    a=rhs(model,t,x,j0,amp,period,kappa,beta)
    b=rhs(model,t+h/2,x+h*a/2,j0,amp,period,kappa,beta)
    c=rhs(model,t+h/2,x+h*b/2,j0,amp,period,kappa,beta)
    d=rhs(model,t+h,x+h*c,j0,amp,period,kappa,beta)
    return x+h*(a+2*b+2*c+d)/6

@njit(cache=True)
def simulate(model,j0,amp,period,kappa,beta,transient,duration,dt=.2,h=.01):
    stride=int(round(dt/h)); ntr=int(round(transient/h)); nobs=int(round(duration/h))
    assert abs(stride*h-dt)<1e-9 and nobs%stride==0
    x=initial(model)
    states=np.empty((nobs//stride+1,len(x)))
    spikes=[-1.]
    threshold=30. if model==1 else 0.
    for i in range(ntr+nobs):
        t=i*h
        if i==ntr: states[0]=x
        z=rk4(model,t,x,h,j0,amp,period,kappa,beta)
        if x[0]<threshold<=z[0]:
            lo=0.;hi=h
            for _ in range(24):
                mid=(lo+hi)/2
                trial=rk4(model,t,x,mid,j0,amp,period,kappa,beta)
                if trial[0]<threshold: lo=mid
                else: hi=mid
            te=(lo+hi)/2
            if i>=ntr: spikes.append(t+te-transient)
            if model==1:
                event=rk4(model,t,x,te,j0,amp,period,kappa,beta)
                event[0]=-65.;event[1]+=8.
                z=rk4(model,t+te,event,h-te,j0,amp,period,kappa,beta)
        x=z
        if i>=ntr and (i-ntr+1)%stride==0:
            states[(i-ntr+1)//stride]=x
    return states,np.asarray(spikes[1:])

def jacobian(model,t,x,pars):
    if model==1:
        return np.array([[.08*x[0]+5,-1],[.004,-.02]])
    v,m,h,n=x
    q=rates(v);dq=(rates(v+1e-4)-rates(v-1e-4))/2e-4
    J=np.zeros((4,4))
    J[0]=[-120*m**3*h-36*n**4-.3,-360*m*m*h*(v-50),-120*m**3*(v-50),-144*n**3*(v+77)]
    J[1,0]=dq[0]*(1-m)-dq[1]*m; J[1,1]=-q[0]-q[1]
    J[2,0]=dq[2]*(1-h)-dq[3]*h; J[2,2]=-q[2]-q[3]
    J[3,0]=dq[4]*(1-n)-dq[5]*n; J[3,3]=-q[4]-q[5]
    return J

def saltation(t,pre,pars):
    post=np.array([-65.,pre[1]+8.])
    fminus=rhs(1,t,pre,*pars);fplus=rhs(1,t,post,*pars)
    DR=np.array([[0.,0.],[0.,1.]])
    return DR+np.outer(fplus-DR@fminus,np.array([1.,0.]))/fminus[0]

def flow_variational(model,x,period,pars,rtol=2e-10,atol=1e-11,reset_derivative='saltation'):
    """Stroboscopic derivative; prescribed forcing phase fixed at t=0.

    For autonomous guard g=v-30, R=(-65,u+8), use Xi=DR+(f+−DRf−)nᵀ/(nᵀf−).
    DR alone is deliberately exposed only for the diagnostic comparison.
    """
    d=len(x); z=np.r_[x,np.eye(d).ravel()];t=0.; events=[]
    def fun(t,z):
        xx=z[:d]; Q=z[d:].reshape(d,d)
        return np.r_[rhs(model,t,xx,*pars),(jacobian(model,t,xx,pars)@Q).ravel()]
    def event(t,z): return z[0]-30.
    event.direction=1.;event.terminal=True
    while t<period-1e-12:
        sol=solve_ivp(fun,(t,period),z,method='DOP853',rtol=rtol,atol=atol,max_step=.1,
                      events=event if model==1 else None)
        if not sol.success: raise RuntimeError(sol.message)
        t=sol.t[-1];z=sol.y[:,-1]
        if model==1 and len(sol.t_events[0]):
            pre=z[:d].copy();events.append(t)
            S=saltation(t,pre,pars) if reset_derivative=='saltation' else np.array([[0.,0.],[0.,1.]])
            z[d:]=(S@z[d:].reshape(d,d)).ravel()
            z[0]=-65.;z[1]+=8.
        else: break
    return z[:d],z[d:].reshape(d,d),np.asarray(events)

def firing_rate(spikes):
    return 1000./np.mean(np.diff(spikes)) if len(spikes)>2 else 0.
