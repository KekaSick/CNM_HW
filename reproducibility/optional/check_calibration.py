"""Independent multi-start and active-set checks for the training-only fit."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution,minimize
from models import simulate,firing_rate
ROOT=Path(__file__).resolve().parent

def check():
    data=pd.read_csv(ROOT/'tables/calibration.csv');data=data[data.split=='train']
    js=data.J_uA_cm2.to_numpy();ys=data.HH_Hz.to_numpy()
    lookup=pd.read_csv(ROOT/'tables/izh_lookup.csv');grid=lookup.iloc[:,0].to_numpy();rates=lookup.iloc[:,1].to_numpy()
    rows=[];states=[]
    def approx(p):
        currents=np.exp(p[0])*js+p[1]
        if currents.max()>grid.max(): return 1e8+currents.max()
        return np.sum((np.interp(currents,grid,rates)-ys)**2)
    for seed in [1,2,3,4,5]:
        rng=np.random.RandomState(seed);st=rng.get_state();states.append({'seed':seed,'generator':st[0],'key':st[1].tolist(),'pos':int(st[2]),'has_gauss':int(st[3]),'cached_gaussian':float(st[4])})
        fit=differential_evolution(approx,[(np.log(.05),np.log(4.)),(-25.,35.)],seed=seed,popsize=12,tol=1e-9)
        rows.append({'method':f'independent_DE_seed_{seed}','kappa':np.exp(fit.x[0]),'beta':fit.x[1],'lookup_SSE_Hz2':fit.fun,'exact_SSE_Hz2':np.nan})
    # Probe the competing discontinuous active sets explicitly.
    for label,bounds in [('J0_active',[(np.log(.05),np.log(4.)),(3.81,35.)]),('J5_quiet',[(np.log(.05),np.log(4.)),(-25.,3.8)])]:
        def obj(p):
            if label=='J5_quiet' and np.exp(p[0])*5+p[1]>3.75: return 1e6+(np.exp(p[0])*5+p[1]-3.75)*1e5
            return approx(p)
        fit=differential_evolution(obj,bounds,seed=0,tol=1e-7,popsize=12)
        def exact(p):
            if label=='J0_active' and p[1]<3.81:return 1e6+(3.81-p[1])*1e5
            if label=='J5_quiet' and np.exp(p[0])*5+p[1]>3.75:return 1e6+(np.exp(p[0])*5+p[1]-3.75)*1e5
            return np.sum([(firing_rate(simulate(1,j,0.,20.,np.exp(p[0]),p[1],1000.,1500.,1.,.02)[1])-y)**2 for j,y in zip(js,ys)])
        exactfit=minimize(exact,fit.x,method='Nelder-Mead',options={'maxiter':90,'xatol':1e-4,'fatol':1e-3})
        rows.append({'method':label+'_refined','kappa':np.exp(exactfit.x[0]),'beta':exactfit.x[1],'lookup_SSE_Hz2':approx(exactfit.x),'exact_SSE_Hz2':exactfit.fun})
    pp=json.loads((ROOT/'calibration.json').read_text());rows.append({'method':'selected_exact_train_fit','kappa':pp['kappa'],'beta':pp['beta'],'lookup_SSE_Hz2':approx([np.log(pp['kappa']),pp['beta']]),'exact_SSE_Hz2':pp['SSE_Hz2']})
    pd.DataFrame(rows).to_csv(ROOT/'tables/calibration_optimization_checks.csv',index=False)
    (ROOT/'data/calibration_check_rng_states.json').write_text(json.dumps(states,indent=2))
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
if __name__=='__main__':check()
