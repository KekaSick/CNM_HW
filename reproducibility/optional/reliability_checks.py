"""Targeted step/transient/grid checks; raw variables retained for every new run.

These are finite-time sensitivity checks, not proofs of global attractor
uniqueness, chaos, or exact bifurcation locations. Calibration stays frozen.
"""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from models import simulate,initial
from scalar import standardize,ordinal,false_nearest,correlation_sum,fit_scaling
from firing import firing_diagnostics,CRITERION
ROOT=Path(__file__).resolve().parent


def recurrence(model,states,T,dt):
    scale=np.array([100.,1.,1.,1.]) if model==0 else np.array([100.,30.])
    tail=states[::int(round(T/dt))][-101:]
    errors=np.asarray([np.sqrt(np.mean(((tail[p:]-tail[:-p])/scale)**2)) for p in range(1,9)])
    period=next((i+1 for i,e in enumerate(errors) if e<1e-5),0)
    return period,errors


def recording(model,label,A,h,tr,du,tag,k,b):
    T=20.;dt=.2;k0,b0=(1.,0.) if model==0 else (k,b)
    x,s=simulate(model,10.,A,T,k0,b0,tr,du,dt,h)
    path=ROOT/'data'/f'reliability_{tag}_{label}_A{A:.2f}_h{h:g}.npz'
    np.savez_compressed(path,states=x,spikes_ms=s,dt_ms=dt,h_ms=h,
                        transient_ms=tr,duration_ms=du,J0_uA_cm2=10.,A_uA_cm2=A,
                        T_ms=T,kappa=k0,beta=b0,initial_state=initial(model))
    return x,s,str(path.relative_to(ROOT))


def check():
    p=json.loads((ROOT/'calibration.json').read_text());k,b=p['kappa'],p['beta']
    sweep=pd.read_csv(ROOT/'tables/periodic_sweep.csv');rows=[]
    targets=[(0,'HH',[1.1,1.2,1.3,1.6,1.7,2.7,2.8]),
             (1,'Izh',[.3,.4,2.8,2.9,3.,3.1,4.4,4.5,5.7,8.3,8.4,10.,11.9,12.4,12.5])]
    for model,label,amps in targets:
        for A in amps:
            base=sweep[(sweep.model==label)&(sweep.T_ms==20)&(abs(sweep.A_uA_cm2-A)<1e-8)].iloc[0]
            for h in [.01,.005]:
                x,s,file=recording(model,label,A,h,16000.,4000.,'transition',k,b)
                period,errors=recurrence(model,x,20.,.2)
                isi=np.diff(s)
                rows.append({'model':label,'T_ms':20.,'A_uA_cm2':A,'h_ms':h,
                             'transient_ms':16000.,'duration_ms':4000.,'data_file':file,
                             'baseline_transient_ms':2000.,'baseline_h_ms':.01,
                             'baseline_period':int(base.period_forcing_cycles),'period':period,
                             'period_matches':int(base.period_forcing_cycles)==period,
                             'baseline_spikes_per_cycle':base.spikes_per_forcing_period,
                             'spikes_per_cycle':len(s)/200.,
                             'baseline_recurrence_RMS':base.recurrence_best_scaled_RMS,
                             'recurrence_RMS':float(errors.min()),
                             'p1_recurrence_RMS':errors[0],'p2_recurrence_RMS':errors[1],
                             'rate_Hz':1000./isi.mean(),'ISI_CV':isi.std()/isi.mean()})
        print('reliability transitions',label,'complete',flush=True)
    pd.DataFrame(rows).to_csv(ROOT/'tables/transition_convergence.csv',index=False)
    grid=[]
    for model,label,amps in [(0,'HH',np.round(np.arange(2.70,2.801,.01),2)),
                             (1,'Izh',np.round(np.arange(.30,.401,.01),2))]:
        for A in amps:
            x,s,file=recording(model,label,A,.005,16000.,4000.,'grid',k,b)
            period,errors=recurrence(model,x,20.,.2)
            grid.append({'model':label,'T_ms':20.,'A_uA_cm2':A,'h_ms':.005,
                         'transient_ms':16000.,'duration_ms':4000.,'data_file':file,
                         'period':period,'spikes_per_cycle':len(s)/200.,
                         'recurrence_RMS':float(errors.min()),'p1_recurrence_RMS':errors[0],
                         'p2_recurrence_RMS':errors[1]})
    pd.DataFrame(grid).to_csv(ROOT/'tables/transition_grid_refinement.csv',index=False)
    selected=pd.read_csv(ROOT/'tables/selected_summary.csv');fnn=[];scalar=[]
    for row in selected.itertuples():
        data=np.load(ROOT/'data'/f'selected_{row.signal}.npz');z=standardize(data['states'][:,0])
        fn=false_nearest(z,5,maxdim=7,window=200,maxpoints=5000)
        np.savez_compressed(ROOT/'data'/f'common_lag_FNN_{row.signal}.npz',FNN=fn,lag_samples=5,W_samples=200,dt_ms=.2)
        fnn.append({'signal':row.signal,'lag_ms':1.,'m_GP':4,
                    'FNN_m3':fn[2,1],'FNN_m4':fn[3,1],'FNN_m5':fn[4,1],'FNN_m7':fn[6,1]})
        if row.signal not in ['HH_weak','Izh_strong']:continue
        model=int(row.model=='Izh')
        x,s,file=recording(model,row.model,row.A_uA_cm2,.005,2000.,8000.,'scalar',k,b)
        fine=standardize(x[:,0]);oo=ordinal(fine,5,5);isi=np.diff(s)
        gp=correlation_sum(fine,4,5,200,np.geomspace(.008,.9,55),maxpoints=3500)
        fit=fit_scaling(gp,.06,.35,min_count=200)
        np.savez_compressed(ROOT/'data'/f'reliability_GP_{row.signal}.npz',**gp)
        scalar.append({'signal':row.signal,'coarse_h_ms':.01,'fine_h_ms':.005,
                       'transient_ms':2000.,'observation_ms':8000.,'data_file':file,
                       'coarse_events':len(data['spikes_ms']),'fine_events':len(s),
                       'coarse_rate_Hz':row.rate_Hz,'fine_rate_Hz':1000./isi.mean(),
                       'coarse_ISI_CV':row.ISI_CV,'fine_ISI_CV':isi.std()/isi.mean(),
                       'coarse_H':row.H_ell5_q5,'fine_H':oo['H'],
                       'coarse_C':row.C_ell5_q5,'fine_C':oo['C'],
                       'coarse_D2':row.D2_fit,'fine_D2':fit['estimate']})
    pd.DataFrame(fnn).to_csv(ROOT/'tables/common_lag_FNN.csv',index=False)
    pd.DataFrame(scalar).to_csv(ROOT/'tables/scalar_integration_convergence.csv',index=False)
    false_positive=[]
    for tag,J,tr,du in [('short',6.260234832763672,1500.,3000.),('long',6.2602996826171875,3000.,6000.)]:
        x,s=simulate(0,J,0.,20.,1.,0.,tr,du,1.,.01);d=firing_diagnostics(s,du)
        name=f'data/old_threshold_false_positive_HH_{tag}.npz'
        np.savez_compressed(ROOT/name,states=x,spikes_ms=s,dt_ms=1.,h_ms=.01,transient_ms=tr,
                            duration_ms=du,J_uA_cm2=J,A_uA_cm2=0.,T_ms=20.,
                            kappa=1.,beta=0.,initial_state=initial(0),criterion_json=json.dumps(CRITERION),**d)
        false_positive.append({'model':'HH','old_definition':tag,'J_uA_cm2':J,
                               'transient_ms':tr,'duration_ms':du,'data_file':name,
                               'old_total_count_passed':len(s)>=3,
                               **{key:value for key,value in d.items() if key not in ['window_counts','window_rates_Hz']}})
    pd.DataFrame(false_positive).to_csv(ROOT/'tables/threshold_false_positive_regression.csv',index=False)
    settings={'description':__doc__,'forcing_J0_uA_cm2':10.,'forcing_T_ms':20.,
              'targeted_amplitudes':{label:amps for _,label,amps in targets},
              'recurrence_scaled_RMS_tolerance':1e-5,'maximum_stroboscopic_period':8,
              'transition_h_ms':[.01,.005],'transition_transient_ms':16000.,
              'transition_observation_ms':4000.,'transition_amplitude_grid_step':.01,
              'full_scalar_step_check_signals':['HH_weak','Izh_strong'],
              'scalar_transient_ms':2000.,'scalar_observation_ms':8000.,
              'common_lag_FNN_ms':1.,'firing_criterion':CRITERION,
              'calibration_kappa':k,'calibration_beta':b,
              'source_SHA256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in ROOT.glob('*.py')}}
    (ROOT/'data/reliability_settings.json').write_text(json.dumps(settings,indent=2))
    print('Reliability checks complete:',len(rows),'transition checks;',len(grid),'grid points;',len(scalar),'full scalar repeats',flush=True)


if __name__=='__main__':check()
