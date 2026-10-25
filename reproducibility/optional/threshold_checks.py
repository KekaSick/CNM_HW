"""First evoked spike and finite-time stationary sustained firing thresholds."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from models import simulate, initial
from firing import firing_diagnostics, CRITERION
ROOT=Path(__file__).resolve().parent


def threshold(model,label,k,b,tr,du,h,kind,limits,tag):
    """Bisect an operational criterion; persist full final endpoint recordings."""
    def evaluate(j):
        x,s=simulate(model,j,0.,20.,k,b,tr,du,1.,h)
        diagnostics=firing_diagnostics(s,du)
        active=len(s)>=1 if kind=='first_spike' else diagnostics['sustained']
        return x,s,diagnostics,bool(active)
    lo,hi=limits
    assert not evaluate(lo)[3] and evaluate(hi)[3], 'Invalid operational threshold bracket'
    while hi-lo>2e-5:
        mid=(lo+hi)/2
        if evaluate(mid)[3]:hi=mid
        else:lo=mid
    result={'model':label,'definition':kind,'h_ms':h,'transient_ms':tr,
            'duration_ms':du,'observation_ms':du,'J_lower_uA_cm2':lo,
            'J_upper_uA_cm2':hi,'threshold_mid':(lo+hi)/2,
            'bracket_width_uA_cm2':hi-lo,
            'criterion':'one event in observation' if kind=='first_spike' else 'stationary_four_windows_and_tail'}
    for side,j in [('lower',lo),('upper',hi)]:
        x,s,d,active=evaluate(j)
        name=f'threshold_{label}_{tag}_h{h:g}_{side}.npz'
        np.savez_compressed(ROOT/'data'/name,states=x,spikes_ms=s,dt_ms=1.,h_ms=h,
                            transient_ms=tr,duration_ms=du,J_uA_cm2=j,J0_uA_cm2=j,
                            A_uA_cm2=0.,T_ms=20.,kappa=k,beta=b,
                            initial_state=initial(model),criterion_json=json.dumps(CRITERION),
                            criterion_active=active,**d)
        result[side+'_data_file']='data/'+name
        result[side+'_events']=len(s)
        result[side+'_last_event_ms']=d['last_event_ms']
        result[side+'_window_count_min']=d['window_count_min']
        result[side+'_window_rate_relative_range']=d['window_rate_relative_range']
        result[side+'_final_gap_ISI']=d['last_gap_in_tail_ISI']
        assert active==(side=='upper')
    return result


def calibration_tail_check():
    """Audit saved calibration observations; never modify the fit or its objective."""
    rows=[]
    for row in pd.read_csv(ROOT/'tables/calibration.csv').itertuples():
        for label in ['HH','Izh']:
            file=f'constant_{label}_J{row.J_uA_cm2:g}.npz'
            data=np.load(ROOT/'data'/file);x=data['states'];s=data['spikes_ms']
            dt=float(data['dt_ms']);du=(len(x)-1)*dt;d=firing_diagnostics(s,du)
            tail=x[-int(round(500/dt)):]
            quiet=not len(s) and float(np.ptp(tail[:,0]))<1e-6
            rows.append({'model':label,'split':row.split,'J_uA_cm2':row.J_uA_cm2,
                         'recording_file':'data/'+file,'duration_ms':du,
                         'quiet_equilibrium_tail':bool(quiet),'tail_V_range_mV':float(np.ptp(tail[:,0])),
                         **{key:value for key,value in d.items() if key not in ['window_counts','window_rates_Hz']},
                         **{f'window{i+1}_events':int(n) for i,n in enumerate(d['window_counts'])},
                         **{f'window{i+1}_rate_Hz':float(v) for i,v in enumerate(d['window_rates_Hz'])}})
            assert quiet or d['sustained'],file
    df=pd.DataFrame(rows);df.to_csv(ROOT/'tables/calibration_tail_stability.csv',index=False)
    return df


def check():
    p=json.loads((ROOT/'calibration.json').read_text());rows=[]
    for model,label,k,b in [(0,'HH',1.,0.),(1,'Izh',p['kappa'],p['beta'])]:
        for kind,tr,du in [('first_spike',0.,200.),('sustained_long',3000.,6000.)]:
            rows.append(threshold(model,label,k,b,tr,du,.01,kind,
                                  (-5.,10.) if model else (0.,10.),kind))
    pd.DataFrame(rows).to_csv(ROOT/'tables/threshold_definition_checks.csv',index=False)
    (ROOT/'data/firing_criterion.json').write_text(json.dumps(CRITERION,indent=2))
    calibration_tail_check()
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
    return pd.DataFrame(rows)


if __name__=='__main__':check()
