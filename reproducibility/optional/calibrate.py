"""Training-only rate fit; the four validation currents never enter optimization."""
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from models import simulate, firing_rate

ROOT=Path(__file__).resolve().parent

def calibration():
    target=[]
    train=np.array([0.,5.,8.,10.,15.,20.]);valid=np.array([7.,9.,12.,18.])
    for j in train:
        x,s=simulate(0,j,0.,20.,1.,0.,1000.,2000.,.2,.01)
        np.savez_compressed(ROOT/'data'/f'constant_HH_J{j:g}.npz',states=x,spikes_ms=s,dt_ms=.2,transient_ms=1000.)
        target.append(firing_rate(s))
    target=np.array(target)
    # An inexpensive deterministic lookup provides the global initial search.
    grid=np.arange(-10.,120.001,.25);f=[]
    for cur in grid:
        x,s=simulate(1,cur,0.,20.,1.,0.,1000.,1500.,1.,.02)
        f.append(firing_rate(s))
    f=np.array(f)
    pd.DataFrame({'I_Izh_model_units':grid,'rate_Hz':f}).to_csv(ROOT/'tables/izh_lookup.csv',index=False)
    def approx(p):return np.sum((np.interp(np.exp(p[0])*train+p[1],grid,f)-target)**2)
    opt=differential_evolution(approx,[(-3,2),(-20,40)],seed=0,tol=1e-9,popsize=10)
    def exact(p):
        return np.sum([(firing_rate(simulate(1,float(np.exp(p[0])*j+p[1]),0.,20.,1.,0.,1000.,2000.,1.,.01)[1])-y)**2 for j,y in zip(train,target)])
    fitted=minimize(exact,opt.x,method='Nelder-Mead',options={'xatol':2e-5,'fatol':2e-5,'maxiter':120})
    k=float(np.exp(fitted.x[0]));b=float(fitted.x[1]);rows=[]
    for split,js in [('train',train),('validation',valid)]:
        for j in js:
            rate=[]
            for model,label in [(0,'HH'),(1,'Izh')]:
                x,s=simulate(model,j,0.,20.,1. if model==0 else k,0. if model==0 else b,1000.,2000.,.2,.01)
                np.savez_compressed(ROOT/'data'/f'constant_{label}_J{j:g}.npz',states=x,spikes_ms=s,dt_ms=.2,transient_ms=1000.)
                rate.append(firing_rate(s))
            rows.append({'split':split,'J_uA_cm2':j,'I_Izh_model_units':k*j+b,'HH_Hz':rate[0],'Izh_Hz':rate[1],'error_Hz':rate[1]-rate[0]})
    df=pd.DataFrame(rows);df.to_csv(ROOT/'tables/calibration.csv',index=False)
    params={'kappa':k,'beta':b,'criterion':'unweighted SSE (Hz²); rate=1000/mean(ISI) for at least 3 observed spikes, else 0; stationarity audited separately',
            'rate_estimator':{'minimum_observed_spikes':3,'otherwise_Hz':0,'transient_ms':1000,'duration_ms':2000,'h_ms':.01},
            'postfit_stationarity_audit':'tables/calibration_tail_stability.csv; not an optimization constraint',
            'training_J':train.tolist(),'validation_J':valid.tolist(),'optimizer_success':bool(fitted.success),'SSE_Hz2':float(fitted.fun),'global_seed':0}
    state=np.random.RandomState(0).get_state()
    params['optimizer_rng_initial_state']={'generator':state[0],'key':state[1].tolist(),'pos':int(state[2]),'has_gauss':int(state[3]),'cached_gaussian':float(state[4])}
    (ROOT/'calibration.json').write_text(json.dumps(params,indent=2))
    print(params,flush=True);print(df.to_string(index=False),flush=True)
    return k,b
if __name__=='__main__':
    for d in ['tables','data','figures']: (ROOT/d).mkdir(exist_ok=True)
    calibration()
