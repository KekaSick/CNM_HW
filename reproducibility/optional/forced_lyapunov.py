"""Model tangent largest Lyapunov estimates with prescribed forcing phase.

The tangent has only physical state coordinates: the phase of the sinusoidal
input is never perturbed. Thus a stable forced orbit has a negative exponent;
an autonomous periodic orbit at A=0 retains its neutral time-shift direction.
At every Izh reset the saltation matrix includes the change in event time.
All state components, event times and per-cycle log growth are persisted.
"""
from pathlib import Path
import os
os.environ.setdefault('NUMBA_CACHE_DIR','/tmp/cnm_optional_numba')
import json,hashlib,time
import numpy as np
import pandas as pd
from numba import njit
from models import initial, rhs, rk4, rates
ROOT=Path(__file__).resolve().parent

@njit(cache=True)
def _vtrap_derivative(z):
    if abs(z)<1e-3:
        return .5+z/60.-z**3/180000.
    den=-np.expm1(-z/10.)
    return (den-z*np.exp(-z/10.)/10.)/(den*den)

@njit(cache=True)
def analytic_jacobian(model,x):
    if model==1:
        return np.array([[.08*x[0]+5.,-1.],[.004,-.02]])
    v,m,h,n=x
    q=rates(v)
    dq=np.array([.1*_vtrap_derivative(v+40.),-q[1]/18.,-q[2]/20.,q[3]*(1.-q[3])/10.,.01*_vtrap_derivative(v+55.),-q[5]/80.])
    J=np.zeros((4,4))
    J[0]=np.array([-120.*m**3*h-36.*n**4-.3,-360.*m*m*h*(v-50.),-120.*m**3*(v-50.),-144.*n**3*(v+77.)])
    J[1,0]=dq[0]*(1.-m)-dq[1]*m;J[1,1]=-q[0]-q[1]
    J[2,0]=dq[2]*(1.-h)-dq[3]*h;J[2,2]=-q[2]-q[3]
    J[3,0]=dq[4]*(1.-n)-dq[5]*n;J[3,3]=-q[4]-q[5]
    return J

@njit(cache=True)
def _tangent_rk4(model,t,x,Q,h,pars):
    a=rhs(model,t,x,*pars);A=analytic_jacobian(model,x)@Q
    xb=x+h*a/2.;QB=Q+h*A/2.
    b=rhs(model,t+h/2.,xb,*pars);B=analytic_jacobian(model,xb)@QB
    xc=x+h*b/2.;QC=Q+h*B/2.
    c=rhs(model,t+h/2.,xc,*pars);C=analytic_jacobian(model,xc)@QC
    xd=x+h*c;QD=Q+h*C
    d=rhs(model,t+h,xd,*pars);D=analytic_jacobian(model,xd)@QD
    return x+h*(a+2.*b+2.*c+d)/6., Q+h*(A+2.*B+2.*C+D)/6.

@njit(cache=True)
def _advance(model,t,x,Q,h,pars):
    y,M=_tangent_rk4(model,t,x,Q,h,pars)
    threshold=30. if model==1 else 0.
    event_time=-1.
    if x[0]<threshold<=y[0]:
        lo=0.;hi=h
        for _ in range(30):
            mid=(lo+hi)/2.
            if rk4(model,t,x,mid,*pars)[0]<threshold:lo=mid
            else:hi=mid
        te=(lo+hi)/2.;event_time=t+te
        if model==1:
            pre,QM=_tangent_rk4(model,t,x,Q,te,pars)
            pre[0]=30.
            post=np.array([-65.,pre[1]+8.])
            fm=rhs(1,event_time,pre,*pars);fp=rhs(1,event_time,post,*pars)
            # (DR + (f+ - DR f-) n^T/(n^T f-)) Q, n=(1,0).
            SQ=QM.copy()
            SQ[0,:]=fp[0]/fm[0]*QM[0,:]
            SQ[1,:]=QM[1,:]+(fp[1]-fm[1])/fm[0]*QM[0,:]
            y,M=_tangent_rk4(model,event_time,post,SQ,h-te,pars)
    return y,M,event_time

@njit(cache=True)
def period_map(model,x,period,pars,h=.01):
    steps=int(round(period/h))
    assert abs(steps*h-period)<1e-8
    Q=np.eye(len(x));xx=x.copy();events=[-1.]
    for i in range(steps):
        xx,Q,te=_advance(model,i*h,xx,Q,h,pars)
        if te>=0.:events.append(te)
    return xx,Q,np.asarray(events[1:])

@njit(cache=True)
def _record(model,pars,transient,duration,h,dt,tangent):
    T=pars[2];cycle=int(round(T/h));stride=int(round(dt/h))
    ntr=int(round(transient/h));nobs=int(round(duration/h))
    assert abs(cycle*h-T)<1e-8 and abs(stride*h-dt)<1e-8
    assert ntr%cycle==0 and nobs%cycle==0 and nobs%stride==0
    x=initial(model);Q=tangent.reshape(len(x),1).copy();Q/=np.linalg.norm(Q)
    states=np.empty((nobs//stride+1,len(x)));strobe=np.empty((nobs//cycle+1,len(x)))
    loggrowth=np.empty(nobs//cycle);events=[-1.]
    for i in range(ntr+nobs):
        if i==ntr:states[0]=x;strobe[0]=x
        x,Q,te=_advance(model,i*h,x,Q,h,pars)
        if i>=ntr and te>=0.:events.append(te-transient)
        if (i+1)%cycle==0:
            norm=np.linalg.norm(Q)
            assert np.isfinite(norm) and norm>0.
            if i>=ntr:
                loggrowth[(i-ntr+1)//cycle-1]=np.log(norm)
                strobe[(i-ntr+1)//cycle]=x
            Q/=norm
        if i>=ntr and (i-ntr+1)%stride==0:states[(i-ntr+1)//stride]=x
    return states,strobe,np.asarray(events[1:]),loggrowth,Q[:,0]

def lyapunov_recording(model,pars,transient=4000.,duration=32000.,h=.01,dt=.2,initial_tangent=None):
    """Return growth per forcing cycle after state and tangent transient removal."""
    tangent=np.ones(2 if model else 4) if initial_tangent is None else np.asarray(initial_tangent,dtype=float)
    states,strobe,spikes,logs,final_tangent=_record(model,tuple(map(float,pars)),float(transient),float(duration),float(h),float(dt),tangent)
    return {'states':states,'stroboscopic_states':strobe,'spikes_ms':spikes,
            'log_growth_per_cycle':logs,'final_unit_tangent':final_tangent}

def check(duration=32000.,transient=4000.):
    start=time.time();p=json.loads((ROOT/'calibration.json').read_text());rows=[];checks=[]
    from models import flow_variational
    for label,model,x,pars in [('HH',0,np.array([-65.,.05,.6,.32]),(10.,1.,20.,1.,0.)),
                               ('Izh',1,np.array([-40.,-8.]),(10.,10.,20.,p['kappa'],p['beta']))]:
        yy,MM,ss=flow_variational(model,x,20.,pars)
        for h in [.02,.01,.005]:
            y,M,s=period_map(model,x,20.,pars,h=h)
            checks.append({'model':label,'h_ms':h,'state_error_max':np.max(np.abs(y-yy)),
                           'matrix_relative_error':np.linalg.norm(M-MM)/np.linalg.norm(MM),
                           'event_time_error_max_ms':np.max(np.abs(s-ss)) if model==1 and len(s) else np.nan})
    for label,model,kind,A in [('HH',0,'weak',1.),('Izh',1,'strong',10.),
                              ('HH',0,'DC',0.),('Izh',1,'DC',0.),
                              ('HH',0,'strong',10.),('Izh',1,'weak',1.),
                              ('Izh',1,'orbit_control',13.8)]:
        hs=[.02,.01,.005] if kind in ['weak','strong'] and (label,kind) in [('HH','weak'),('Izh','strong')] else [.01]
        for h in hs:
            du=duration if kind=='DC' or (label,kind) in [('HH','weak'),('Izh','strong')] else min(8000.,duration)
            pars=(10.,A,20.,1. if model==0 else p['kappa'],0. if model==0 else p['beta'])
            result=lyapunov_recording(model,pars,transient,du,h)
            file=f'data/forced_lyapunov_{label}_{kind}_h{h:g}.npz'
            np.savez_compressed(ROOT/file,**result,initial_state=initial(model),initial_unit_tangent=np.ones(2 if model else 4)/np.sqrt(2 if model else 4),
                                dt_ms=.2,h_ms=h,transient_ms=transient,duration_ms=du,T_ms=20.,
                                J0_uA_cm2=10.,A_uA_cm2=A,kappa=pars[3],beta=pars[4])
            logs=result['log_growth_per_cycle'];T=20.
            for endpoint in [x for x in [4000.,8000.,16000.,32000.] if x<=du]:
                sample=logs[:int(round(endpoint/T))]
                blocks=np.array_split(sample,4);rates=np.array([q.mean()/T for q in blocks])
                rows.append({'signal':f'{label}_{kind}','model':label,'A_uA_cm2':A,
                             'h_ms':h,'transient_ms':transient,'observation_ms':endpoint,
                             'lambda_max_per_ms':sample.mean()/T,'lambda_max_per_s':1000.*sample.mean()/T,
                             'quarter_min_per_ms':rates.min(),'quarter_max_per_ms':rates.max(),
                             **{f'quarter{i+1}_lambda_per_ms':v for i,v in enumerate(rates)},
                             'data_file':file})
            print(label,kind,h,'LLE',logs.mean()/T,'ms^-1',flush=True)
    pd.DataFrame(rows).to_csv(ROOT/'tables/forced_lyapunov.csv',index=False)
    pd.DataFrame(checks).to_csv(ROOT/'tables/forced_lyapunov_map_checks.csv',index=False)
    settings={'definition':'largest physical-state Lyapunov exponent at fixed prescribed forcing phase; deterministic initial tangent, normalization each forcing period',
              'units':'ms^-1; per-second values multiply by 1000','integrator':'augmented RK4 with analytic Jacobian, 30 bisections for Izh reset and exact saltation at event',
              'phase_coordinate_in_tangent':False,'T_ms':20.,'state_and_tangent_transient_ms':transient,
              'observation_ms':duration,'calibration':p,'runtime_seconds':time.time()-start,
              'source_SHA256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['forced_lyapunov.py','models.py']}}
    (ROOT/'data/forced_lyapunov_settings.json').write_text(json.dumps(settings,indent=2))
    print('Fixed-phase Lyapunov calculation complete',settings['runtime_seconds'],flush=True)
    return pd.DataFrame(rows)

if __name__=='__main__':check()
