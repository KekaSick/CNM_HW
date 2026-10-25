"""Detect and resolve stroboscopic sampling alias in dimension estimates.

At Δt=.2 and T=20 exactly 100 phases are sampled on a locked orbit.
Using Δt=.07 gives 2000 phases (T/Δt=2000/7), with the same h=.01.
The secondary output is for sensitivity, not an alteration of the common main comparison.
"""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from models import simulate
from scalar import standardize,box_count,correlation_sum,fit_scaling
ROOT=Path(__file__).resolve().parent
sys.path.append(str(ROOT.parent/'src'))
from publication import install_publication_exports
install_publication_exports()

def check():
    p=json.loads((ROOT/'calibration.json').read_text());rows=[]
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for col,(label,model,kind,A) in enumerate([('HH',0,'strong',10.),('Izh',1,'weak',1.)]):
        name=f'{label}_{kind}'
        for dt in [.2,.07,.013]:
            if dt==.2:
                data=np.load(ROOT/'data'/f'selected_{name}.npz');x=data['states'];s=data['spikes_ms']
            else:
                du=7000. if dt==.07 else 2600.;h=.01 if dt==.07 else .001
                x,s=simulate(model,10.,A,20.,1. if model==0 else p['kappa'],0. if model==0 else p['beta'],2000.,du,dt,h)
                np.savez_compressed(ROOT/'data'/f'sampling_{name}_dt{str(dt).replace(chr(46),chr(48))}.npz',states=x,spikes_ms=s,dt_ms=dt,transient_ms=2000.,A_uA_cm2=A,T_ms=20.)
            z=standardize(x[:,0]);lag=int(round(1/dt));eps=np.geomspace(.025,.7,22)
            counts=box_count(z,3,lag,eps);slopes=[]
            for shift,c in zip([0.,.25,.5,.75],counts):
                mask=(eps>=.08)&(eps<=.4)&(c<len(z)/2)
                fit=np.polyfit(np.log(1/eps[mask]),np.log(c[mask]),1);slopes.append(fit[0])
            result=correlation_sum(z,4,lag,int(round(40/dt)),np.geomspace(.008,.9,55),maxpoints=3500)
            fit=fit_scaling(result,.06,.35,min_count=200)
            rows.append({'signal':name,'dt_ms':dt,'N':len(z),'actual_lag_ms':dt*lag,'D0_m3_offset_mean':np.mean(slopes),'D0_min':min(slopes),'D0_max':max(slopes),'D2_m4':fit['estimate'],'max_boxes_smallest_eps':int(counts[:,0].max()),'distinct_forcing_phases_theoretical':100 if dt==.2 else (2000 if dt==.07 else 20000)})
            axs[col].loglog(1/eps,np.mean(counts,axis=0),'o-',label=f'dt={dt:g} ms')
            np.savez_compressed(ROOT/'data'/f'sampling_GP_{name}_{str(dt).replace(".","")}.npz',**result)
        axs[col].set(title=name,xlabel='1/ε [standardized units⁻¹]',ylabel='occupied boxes m=3');axs[col].legend()
    pd.DataFrame(rows).to_csv(ROOT/'tables/sampling_dimension_check.csv',index=False)
    fig.tight_layout();fig.savefig(ROOT/'figures/sampling_dimension_alias.png');plt.close(fig)
    print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__':check()
