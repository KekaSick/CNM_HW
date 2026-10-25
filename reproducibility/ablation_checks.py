"""Matched sampling, frozen-input calibration ablation, and evidence inventory.

All added outputs have an ablation prefix or live in data/ablation. Existing
recordings/tables are read-only. HR time is in model units; HH/Izh time is ms.
No parameters are fitted and no new random values are generated.
"""
from pathlib import Path
import os,sys,json,hashlib,time,math
os.environ.setdefault('NUMBA_CACHE_DIR','/tmp/cnm_optional_numba')
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from cnm import ordinal,entropy_complexity
sys.path.insert(0,str(ROOT/'optional'))
from models import simulate,initial,firing_rate


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lag_samples(physical_lag,dt):
    """Require an exactly representable positive physical element separation."""
    if dt<=0 or physical_lag<=0:raise ValueError('Positive lag and sampling step required')
    q=int(round(physical_lag/dt))
    if q<1 or not np.isclose(q*dt,physical_lag,atol=1e-10,rtol=0):
        raise ValueError('Physical lag must be a positive integral number of samples')
    return q


def matched_samples(raw,source_dt,target_dt,duration,start=0.):
    """Select the same half-open physical observation interval without filtering.

    The nominal record duration is N*dt; the sample span is (N-1)*dt. Keeping
    these quantities separate avoids claiming that decimated endpoint times
    coincide. The source can contain an additional endpoint observation.
    """
    factor=lag_samples(target_dt,source_dt)
    n=lag_samples(duration,target_dt)
    offset=int(round(start/source_dt))
    if start<0 or not np.isclose(offset*source_dt,start,atol=1e-9,rtol=0):
        raise ValueError('Start must be represented on the source grid')
    ids=offset+np.arange(n)*factor
    if ids[-1]>=len(raw):raise ValueError('Source does not cover the observation interval')
    x=np.asarray(raw)[ids]
    return x,{'N':len(x),'dt':float(target_dt),'start_time':float(start),
              'end_time_exclusive':float(start+duration),'record_duration':float(duration),
              'sample_span':float((len(x)-1)*target_dt),'decimation_factor':factor}


def load_hr_signals():
    signals={};files={};dt=.2;duration=30000.
    for name in ['spiking','bursting','chaotic']:
        file=ROOT/'data'/f'{name}.npz'
        with np.load(file) as z:
            assert np.isclose(float(z['dt']),dt) and np.isclose(float(z['duration']),duration)
            signals[name]=z['state'][:,0].copy();files[name]=str(file.relative_to(ROOT))
    file=ROOT/'data/controls.npz'
    with np.load(file) as z:
        assert np.isclose(float(z['dt']),dt)
        for name in ['white_noise','shuffled','quasiperiodic']:
            signals[name]=z[name].copy();files[name]=str(file.relative_to(ROOT))
    return signals,files,dt,duration


def sampling_ablation():
    signals,files,source_dt,duration=load_hr_signals();rows=[];counts={}
    for name,raw in signals.items():
        for ell in [4,5]:
            for dt in [.2,.4,1.,2.]:
                for arm in ['fixed_physical_lag','fixed_q10']:
                    q=lag_samples(2.,dt) if arm=='fixed_physical_lag' else 10
                    for segment in [-1,0,1,2,3,4,5]:
                        span=duration if segment<0 else duration/6.
                        start=0. if segment<0 else segment*span
                        x,meta=matched_samples(raw,source_dt,dt,span,start)
                        result=ordinal(x,ell,q);key=f'case_{len(rows):04d}'
                        counts[key]=result['counts']
                        rows.append({'signal':name,'source_file':files[name],'arm':arm,
                                     'ell':ell,'q':q,'physical_element_lag':q*dt,
                                     'physical_ordinal_window':(ell-1)*q*dt,
                                     'time_unit':'HR_model_time','segment':segment,
                                     'scope':'full' if segment<0 else 'segment',**meta,
                                     **{k:result[k] for k in ['H','C','patterns','occupied','ties','patterns_per_possible']},
                                     'counts_key':key})
    frame=pd.DataFrame(rows);frame.to_csv(ROOT/'tables/ablation_sampling_HC.csv',index=False)
    np.savez_compressed(ROOT/'data/ablation/sampling_ordinal_counts.npz',**counts)
    summary=[]
    for (signal,ell,dt),g in frame.groupby(['signal','ell','dt'],sort=False):
        row={'signal':signal,'ell':ell,'dt':dt,'record_duration':duration,'N':int(round(duration/dt))}
        for arm in ['fixed_physical_lag','fixed_q10']:
            a=g[g.arm==arm];full=a[a.scope=='full'].iloc[0];seg=a[a.scope=='segment']
            row.update({f'{arm}_{k}':full[k] for k in ['q','physical_element_lag','physical_ordinal_window','H','C','patterns','patterns_per_possible']})
            row.update({f'{arm}_{k}_segment_mean':seg[k].mean() for k in ['H','C']})
            row.update({f'{arm}_{k}_segment_sd':seg[k].std(ddof=1) for k in ['H','C']})
        row['confounded_minus_matched_H']=row['fixed_q10_H']-row['fixed_physical_lag_H']
        row['confounded_minus_matched_C']=row['fixed_q10_C']-row['fixed_physical_lag_C']
        summary.append(row)
    comparison=pd.DataFrame(summary)
    for metric in ['H','C']:
        for arm in ['fixed_physical_lag','fixed_q10']:
            ref=comparison[comparison.dt==.2].set_index(['signal','ell'])[f'{arm}_{metric}']
            comparison[f'{arm}_{metric}_minus_dt0p2']=[r[f'{arm}_{metric}']-ref.loc[(r['signal'],r['ell'])] for r in comparison.to_dict('records')]
    comparison.to_csv(ROOT/'tables/ablation_sampling_comparison.csv',index=False)
    return frame,comparison


def constant_izh_record(J,kappa,beta,transient=1000.,duration=2000.,h=.02,dt=.2):
    states,spikes=simulate(1,float(J),0.,20.,float(kappa),float(beta),float(transient),float(duration),float(dt),float(h))
    return {'states':states,'spikes_ms':spikes,'J_uA_cm2':float(J),
            'I_Izh_model_units':float(kappa*J+beta),'kappa':float(kappa),'beta':float(beta),
            'initial_state':initial(1),'transient_ms':float(transient),'duration_ms':float(duration),
            'h_ms':float(h),'dt_ms':float(dt),'rate_Hz':float(firing_rate(spikes))}


def calibration_ablation():
    fitted=pd.read_csv(ROOT/'optional/tables/calibration.csv')
    params=json.loads((ROOT/'optional/calibration.json').read_text());k,b=params['kappa'],params['beta']
    rows=[]
    for old in fitted.itertuples():
        # Split labels are carried through for evaluation only; nothing is fitted.
        row={'split':old.split,'J_uA_cm2':old.J_uA_cm2,'HH_Hz':old.HH_Hz,
             'existing_fitted_Hz_h0p01':old.Izh_Hz,'existing_fitted_error_Hz_h0p01':old.error_Hz,
             'transient_ms':1000.,'duration_ms':2000.,'paired_h_ms':.02}
        for arm,k0,b0 in [('identity',1.,0.),('frozen_fitted',k,b)]:
            record=constant_izh_record(old.J_uA_cm2,k0,b0)
            file=f'data/ablation/Izh_{arm}_J{old.J_uA_cm2:g}_h0.02.npz'
            np.savez_compressed(ROOT/file,**record,split=old.split,arm=arm)
            row.update({f'{arm}_Hz':record['rate_Hz'],f'{arm}_error_Hz':record['rate_Hz']-old.HH_Hz,
                        f'{arm}_input_model_units':record['I_Izh_model_units'],f'{arm}_data_file':file})
        row['frozen_h0p02_minus_existing_h0p01_Hz']=row['frozen_fitted_Hz']-old.Izh_Hz
        rows.append(row)
    frame=pd.DataFrame(rows);frame.to_csv(ROOT/'tables/ablation_calibration_rates.csv',index=False)
    scores=[]
    for split,g in frame.groupby('split',sort=False):
        for arm,column in [('identity','identity_Hz'),('frozen_fitted','frozen_fitted_Hz'),('existing_fitted_h0p01','existing_fitted_Hz_h0p01')]:
            error=g[column]-g.HH_Hz
            scores.append({'split':split,'arm':arm,'currents':len(g),'h_ms':.01 if arm=='existing_fitted_h0p01' else .02,
                           'transient_ms':1000.,'duration_ms':2000.,'RMSE_Hz':np.sqrt(np.mean(error**2)),
                           'MAE_Hz':np.mean(abs(error)),'max_abs_Hz':np.max(abs(error)),
                           'SSE_Hz2':np.sum(error**2),'parameters_refitted':False})
    score=pd.DataFrame(scores);score.to_csv(ROOT/'tables/ablation_calibration_scores.csv',index=False)
    return frame,score


def evidence_inventory(sampling,calibration):
    rows=[]
    def add(name,kind,changed,fixed,sources,selection,observed,limit):
        rows.append({'evidence':name,'evidence_type':kind,'changed':changed,'held_fixed':fixed,
                     'source_tables':sources,'source_selection':selection,
                     'observed_metrics_json':json.dumps(observed,ensure_ascii=False,sort_keys=True),
                     'interpretation_limit':limit})
    def read(name):return pd.read_csv(ROOT/name)
    def span(x):return [float(x.min()),float(x.max())]
    hc=read('tables/entropy_complexity.csv').set_index('signal')
    signals,_,_,_=load_hr_signals()
    assert np.array_equal(np.sort(signals['chaotic'][:150000]),np.sort(signals['shuffled']))
    add('Temporal order removed','surrogate control','shuffle HR chaotic sample order','exact empirical marginal, N=150000, dt=.2, ell5,q10',
        'tables/entropy_complexity.csv','signal=chaotic vs shuffled,ell5,q10',
        {'H_original':float(hc.loc['chaotic','H']),'H_shuffled':float(hc.loc['shuffled','H']),
         'C_original':float(hc.loc['chaotic','C']),'C_shuffled':float(hc.loc['shuffled','C']),
         'empirical_marginal_exactly_preserved':True},'Tests temporal order dependence; a shuffled surrogate is not proof of deterministic chaos.')
    for control in ['white_noise','quasiperiodic']:
        add(control,'reference control','signal origin','N=150000,dt=.2,ell5,q10',
            'tables/entropy_complexity.csv',f'signal={control},ell5,q10',
            {'H':float(hc.loc[control,'H']),'C':float(hc.loc[control,'C'])},'A benchmark signal, not removal of one model mechanism.')
    noise=read('tables/measurement_noise.csv')
    for signal in ['spiking','chaotic']:
        g=noise[(noise.signal==signal)&(noise.ell==5)&(noise.q==10)];base=g[g.eta==0].iloc[0];high=g[g.eta==.2]
        add(f'Measurement noise {signal}','robustness stress test','eta0 to .2, five independent nonzero realizations','same solved signal,N,dt,ell5,q10',
            'tables/measurement_noise.csv',f'signal={signal},ell5,q10,eta0 vs .2',
            {'nonzero_replicates':len(high),'H_base':float(base.H),'H_noise_mean':float(high.H.mean()),'H_noise_sd':float(high.H.std()),
             'C_base':float(base.C),'C_noise_mean':float(high.C.mean()),'C_noise_sd':float(high.C.std())},'Changes observation noise, not model dynamics; no claim of a general classifier error rate.')
    sensitivity=read('tables/correlation_sensitivity.csv')
    for varied in ['W','lag','N']:
        for regime,g0 in sensitivity.groupby('regime',sort=False):
            g=g0[g0.varied==varied]
            add(f'GP {varied} {regime}','estimator sensitivity',varied,'source signal and other tabulated settings; see row filters',
                'tables/correlation_sensitivity.csv',f'regime={regime},varied={varied}',
                {'varied_values':g.value.tolist(),'D2_fit_range':span(g.estimate),'m':sorted(g.m.unique().tolist()),
                 'lag_samples':sorted(g.lag.unique().tolist()),'W_samples':sorted(g.W.unique().tolist()),'N':sorted(g.N.unique().tolist())},
                'Finite-scale estimator robustness; W=0 also diagnoses omission of temporal exclusion. Not a biological-mechanism ablation.')
    dimensions=read('tables/correlation_dimensions.csv')
    for regime,g in dimensions.groupby('regime',sort=False):
        add(f'GP embedding {regime}','estimator sensitivity','embedding dimension m','source,N,lag,W; normalized fit-scale convention',
            'tables/correlation_dimensions.csv',f'regime={regime}',{'m':g.m.tolist(),'D2_fit_range':span(g.estimate)},'Fit radii follow dimensional scale convention; this is not a pure fixed-geometry mechanism removal.')
    boxes=read('tables/capacity_dimensions.csv')
    for regime in ['spiking','chaotic']:
        g=boxes[(boxes.regime==regime)&(boxes.m==3)&(boxes.record_N==boxes.record_N.max())]
        add(f'Box grid {regime}','estimator sensitivity','grid offset','same scalar record,m3,N and fit-scale interval',
            'tables/capacity_dimensions.csv',f'regime={regime},m3,max record_N',
            {'grid_shift':g.grid_shift.tolist(),'D0_fit_range':span(g.D0)},'Measures grid dependence of finite-scale fit; does not establish an asymptotic capacity dimension.')
    orbit=read('optional/tables/periodic_orbits.csv');g=orbit[orbit.model=='Izh'].iloc[0]
    add('Izh reset time derivative','mechanistic ablation of numerical linearization','replace saltation by DR only','identical refined orbit,forcing,model and smooth variational integration',
        'optional/tables/periodic_orbits.csv','model=Izh',
        {'correct_spectral_radius':float(g.spectral_radius),'DR_only_spectral_radius':float(g.wrong_DR_only_spectral_radius),
         'difference_DR_minus_correct':float(g.wrong_DR_only_spectral_radius-g.spectral_radius),'correct_FD_relative_error':float(g.FD_relative_error)},
        'DR-only derivative is deliberately incorrect; this diagnoses event-time correction. Both radii happen to remain below1 here.')
    steps=read('tables/integration_observables.csv')
    for regime,g in steps.groupby('regime',sort=False):
        add(f'RK step {regime}','numerical convergence','h=.04,.02,.01','same initial condition,transient,observation duration and output step',
            'tables/integration_observables.csv',f'regime={regime}',
            {'h':g.h.tolist(),'H_range':span(g.H),'C_range':span(g.C),'mean_ISI_range':span(g.mean_isi)},'Numerical validation, not a mechanism ablation; chaotic trajectories need statistical rather than pointwise agreement.')
    transients=read('tables/integration_convergence.csv')
    for regime,g0 in transients.groupby('regime',sort=False):
        g=g0[(g0.h==.02)&(g0.duration==15000)]
        add(f'Transient {regime}','numerical sensitivity','discard5000 vs20000 model-time units','h=.02,observation15000,same initial condition and scalar signal',
            'tables/integration_convergence.csv',f'regime={regime},h=.02,duration15000',
            {'transient':g.transient.tolist(),'mean_ISI_range':span(g.mean_isi),'mean_x_range':span(g.mean_x),'spikes':g.spikes.tolist()},'Finite transient sensitivity cannot rule out arbitrarily slow transients or other basins.')
    tr=read('optional/tables/transition_convergence.csv');g=tr[tr.h_ms==.01]
    add('Forced transition transient','numerical sensitivity','100T to800T discarded','h=.01,forcing and200T observation,22 amplitude/model points',
        'optional/tables/transition_convergence.csv','h_ms=.01',
        {'cases':len(g),'period_matches':int(g.period_matches.sum()),'max_abs_spikes_per_cycle_change':float(abs(g.spikes_per_cycle-g.baseline_spikes_per_cycle).max())},'Only selected amplitudes and recurrence p<=8; agreement does not identify bifurcation type.')
    scores=calibration.set_index(['split','arm'])
    for split in ['train','validation']:
        a=scores.loc[(split,'identity')];b=scores.loc[(split,'frozen_fitted')]
        add(f'Input calibration {split}','input-mapping ablation','identity kappa1,beta0 vs fixed fitted transform','same currents,IC,h=.02,1000ms transient,2000ms observation,rate estimator',
            'tables/ablation_calibration_scores.csv',f'split={split},identity vs frozen_fitted',
            {'identity_RMSE_Hz':float(a.RMSE_Hz),'frozen_RMSE_Hz':float(b.RMSE_Hz),'identity_MAE_Hz':float(a.MAE_Hz),'frozen_MAE_Hz':float(b.MAE_Hz)},'No refitting or use of validation for tuning. Tests input matching, not changing Izh mechanisms.')
    for signal,g in sampling[sampling.ell==5].groupby('signal',sort=False):
        add(f'Observation sampling {signal}','matched sensitivity with confounding diagnostic','dt=.2,.4,1,2; matched physical lag2 versus fixed q10',
            'same half-open30000 model-time interval and ell5; physical lag fixed only in matched arm',
            'tables/ablation_sampling_comparison.csv',f'signal={signal},ell5',
            {'matched_H_range':span(g.fixed_physical_lag_H),'matched_C_range':span(g.fixed_physical_lag_C),
             'max_abs_fixed_q_minus_matched_H':float(abs(g.confounded_minus_matched_H).max()),
             'max_abs_fixed_q_minus_matched_C':float(abs(g.confounded_minus_matched_C).max())},'N and estimator variance vary with dt; physical duration is held fixed, not sample count. Decimation has no antialias filter.')
    frame=pd.DataFrame(rows);frame.to_csv(ROOT/'tables/ablation_evidence_map.csv',index=False)
    (ROOT/'data/ablation/evidence_map.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    return frame


def verify():
    hc=pd.read_csv(ROOT/'tables/ablation_sampling_HC.csv')
    assert len(hc)==672 and set(hc.signal)=={'spiking','bursting','chaotic','white_noise','shuffled','quasiperiodic'}
    with np.load(ROOT/'data/ablation/sampling_ordinal_counts.npz') as counts:
        for row in hc.itertuples():
            assert row.patterns==row.N-(row.ell-1)*row.q
            assert row.physical_element_lag==row.dt*row.q
            assert row.physical_ordinal_window==(row.ell-1)*row.physical_element_lag
            np.testing.assert_allclose([row.N*row.dt,row.sample_span],[row.record_duration,(row.N-1)*row.dt],rtol=0,atol=1e-9)
            if row.arm=='fixed_physical_lag':assert np.isclose(row.physical_element_lag,2.)
            c=counts[row.counts_key];assert c.sum()==row.patterns and len(c)==math.factorial(row.ell)
            np.testing.assert_allclose(entropy_complexity(c/c.sum()),[row.H,row.C],rtol=1e-12,atol=1e-14)
    base=hc[hc.dt==.2]
    a=base[base.arm=='fixed_physical_lag'].sort_values(['signal','ell','segment'])
    b=base[base.arm=='fixed_q10'].sort_values(['signal','ell','segment'])
    np.testing.assert_allclose(a[['H','C']].to_numpy(),b[['H','C']].to_numpy(),rtol=0,atol=0)
    rates=pd.read_csv(ROOT/'tables/ablation_calibration_rates.csv')
    assert set(rates[rates.split=='train'].J_uA_cm2)=={0,5,8,10,15,20}
    assert set(rates[rates.split=='validation'].J_uA_cm2)=={7,9,12,18}
    for row in rates.itertuples():
        for arm in ['identity','frozen_fitted']:
            with np.load(ROOT/getattr(row,arm+'_data_file')) as z:
                assert z['states'].shape==(10001,2) and z['states'][:,0].max()<30.00001
                assert np.isfinite(z['states']).all() and np.all(np.diff(z['spikes_ms'])>0)
                np.testing.assert_allclose(firing_rate(z['spikes_ms']),getattr(row,arm+'_Hz'),atol=1e-12)
                assert float(z['h_ms'])==.02 and float(z['transient_ms'])==1000 and float(z['duration_ms'])==2000
                if arm=='identity':assert float(z['kappa'])==1 and float(z['beta'])==0 and float(z['I_Izh_model_units'])==row.J_uA_cm2
    scores=pd.read_csv(ROOT/'tables/ablation_calibration_scores.csv')
    for row in scores.itertuples():
        g=rates[rates.split==row.split]
        col='existing_fitted_Hz_h0p01' if row.arm=='existing_fitted_h0p01' else row.arm+'_Hz'
        err=g[col]-g.HH_Hz
        np.testing.assert_allclose([np.sqrt(np.mean(err**2)),abs(err).mean()],[row.RMSE_Hz,row.MAE_Hz],atol=1e-12)
    return {'sampling_rows':len(hc),'sampling_comparison_rows':48,'calibration_currents':len(rates),'new_model_recordings':20,'all_checks_passed':True}


def check():
    start=time.time();(ROOT/'data/ablation').mkdir(exist_ok=True)
    sources=[ROOT/'data'/f'{s}.npz' for s in ['spiking','bursting','chaotic']]+[ROOT/'data/controls.npz']
    hashes={str(p.relative_to(ROOT)):digest(p) for p in sources}
    _,sampling=sampling_ablation();_,calibration=calibration_ablation();evidence=evidence_inventory(sampling,calibration)
    outcome=verify()
    assert hashes=={str(p.relative_to(ROOT)):digest(p) for p in sources},'A source recording changed during the computation'
    params=json.loads((ROOT/'optional/calibration.json').read_text())
    settings={'experiments':{'sampling':{'source_dt':.2,'observation_dt':[.2,.4,1.,2.],
                'fixed_physical_element_lag':2.,'matched_q':[10,5,2,1],'confounded_arm_q':10,
                'ell':[4,5],'record_duration':30000.,'segment_count':6,'segment_duration':5000.,
                'interval_convention':'half-open; N*dt is record duration, (N-1)*dt is sample span',
                'time_unit':'HR model time','decimation':'deterministic observation subselection, no filtering'},
                'calibration':{'arms':['identity','frozen_fitted'],'kappa_frozen':params['kappa'],'beta_frozen':params['beta'],
                'h_ms':.02,'dt_ms':.2,'transient_ms':1000.,'duration_ms':2000.,
                'training_J':[0,5,8,10,15,20],'validation_J':[7,9,12,18],
                'rate_estimator':'1000/mean(ISI) if at least3 observed spikes, else0; unchanged original calibration estimator',
                'new_fitting':False,'existing_fit_reference':'optional/tables/calibration.csv (h=.01)'}},
              'source_recording_SHA256':hashes,'source_code_SHA256':{str(p.relative_to(ROOT)):digest(p) for p in [ROOT/'ablation_checks.py',ROOT/'src/cnm.py',ROOT/'optional/models.py']},
              'runtime_seconds':time.time()-start}
    (ROOT/'data/ablation/settings.json').write_text(json.dumps(settings,ensure_ascii=False,indent=2))
    (ROOT/'data/ablation/verification.json').write_text(json.dumps(outcome,indent=2))
    print(calibration.to_string(index=False),flush=True)
    print('Ablation checks complete',settings['runtime_seconds'],'seconds;',outcome,flush=True)
    return settings

if __name__=='__main__':check()
