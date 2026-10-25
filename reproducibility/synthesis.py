"""Scientific synthesis tables for parts D and 5, without document or audit tooling."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from cnm import equilibrium,jacobian


def summarize_hr():
    """Combine model classification, scalar dimensions and ordinal statistics (part D)."""
    read=lambda name:pd.read_csv(ROOT/'tables'/name)
    params=json.loads((ROOT/'data/reconstruction_parameters.json').read_text())
    lya=read('lyapunov.csv');d2=read('correlation_dimensions.csv');d0=read('capacity_dimensions.csv')
    ec=read('entropy_complexity.csv');noise=read('measurement_noise.csv')
    gp_ranges=read('reconstruction_refinement_summary_gp.csv')
    cap=read('reconstruction_refinement_summary_capacity.csv')
    cap=cap[(cap.m==6)&(cap.tau==2)]
    summary=[]
    for name,current,label in [('spiking',4.,'regular spiking'),('bursting',2.5,'periodic bursting'),('chaotic',3.25,'deterministic chaos: numerical evidence')]:
        p=params[name];le=lya[(lya.regime==name)&(lya.h==.01)].iloc[0];hc=ec[ec.signal==name].iloc[0]
        ds=d2[(d2.regime==name)&(d2.m>=3)];bs=d0[(d0.regime==name)&(d0.m==4)&(d0.record_N==150000)]
        if name=='chaotic':
            dp=f'{gp_ranges.estimate_min.min():.3f}-{gp_ranges.estimate_max.max():.3f}: finite-scale; IC2 1.910'
            bc=cap[(cap['case']=='main')&(cap.method=='curve')].iloc[0]
            bp=f'{bc.estimate_min:.3f}-{bc.estimate_max:.3f}: polyline, finite-scale; points 1.689-1.747 unreliable'
            reconstruction='10; 6; 656'
        else:
            dp=f'{ds.estimate.min():.3f}-{ds.estimate.max():.3f}: consistent with 1'
            bp=f'{bs.D0.min():.3f}-{bs.D0.max():.3f}: finite-scale' if len(bs) else 'not estimated'
            reconstruction=f'{p["lag"]}; {p["m"]}; {p["W"]}'
        summary.append({'origin':f'HR {name}','I':current,'classification':label,'lambda_max':le.lambda1,'k; m; W':reconstruction,'D2':dp,'D0':bp,'H':hc.H,'C':hc.C})
    eq=equilibrium(1.);quiet=np.load(ROOT/'data/steady.npz')
    summary.append({'origin':'HR steady','I':1.,'classification':'stable equilibrium','lambda_max':max(np.linalg.eigvals(jacobian(eq)).real),'k; m; W':'not applicable','D2':'0: theory','D0':'0: theory','H':'not applicable','C':'not applicable'})
    for name,label in [('white_noise','IID Gaussian'),('shuffled','random permutation of HR chaos'),('quasiperiodic','two incommensurate frequencies')]:
        hc=ec[ec.signal==name].iloc[0]
        summary.append({'origin':name,'I':'not applicable','classification':label,'lambda_max':'not applicable','k; m; W':'not estimated','D2':'not estimated','D0':'not estimated','H':hc.H,'C':hc.C})
    hn=noise[(noise.signal=='spiking')&(noise.eta==.05)&(noise.ell==5)&(noise.q==10)]
    summary.append({'origin':'HR spiking + measurement noise eta=.05','I':4.,'classification':'periodic source + measurement noise','lambda_max':'source model approximately 0','k; m; W':f'{params["spiking"]["lag"]}; 2-6; {params["spiking"]["W"]}','D2':'no common plateau; see C5','D0':'not estimated','H':hn.H.mean(),'C':hn.C.mean()})
    sdf=pd.DataFrame(summary);sdf.to_csv(ROOT/'tables/summary.csv',index=False)


def summarize_optional():
    """Aggregate calibrated HH/Izh responses with per-estimate reliability labels."""
    read=lambda name:pd.read_csv(ROOT/'optional/tables'/name)
    sel=read('selected_summary.csv');sweep=read('periodic_sweep.csv')
    dim=read('dimension_sensitivity.csv');box=read('box_sensitivity.csv')
    sampling=read('sampling_dimension_check.csv');lle=read('forced_lyapunov.csv')
    corrected=sampling[sampling.dt_ms==sampling.dt_ms.min()]
    selected_lle=[]
    for name in sel.signal:
        data=lle[lle.signal==name]
        data=data[data.observation_ms==data.observation_ms.max()]
        selected_lle.append(data.sort_values('h_ms').iloc[0])
    selected_lle=pd.DataFrame(selected_lle)
    match=[]
    for T in [10.,20.,40.]:
        g=sweep[sweep.T_ms==T].pivot(index='A_uA_cm2',columns='model',values='spikes_per_forcing_period')
        match.append({'T_ms':T,'A_points':len(g),'mean_abs_rate_ratio_difference':float(abs(g.HH-g.Izh).mean()),'matching_ratio_abs_lt_0.02':int((abs(g.HH-g.Izh)<.02).sum())})
    dsum=[]
    for name,g in dim.groupby('signal'):
        base=g[(g.lag_samples==5)&(g.W_samples==200)&(g.N==g.N.max())&(g.m>=3)]
        gg=box[box.signal==name]
        vals=g['estimate'].dropna()
        dsum.append({'signal':name,'D2_m3to6_min':base['estimate'].min(),'D2_m3to6_max':base['estimate'].max(),'D2_all_settings_min':vals.min(),'D2_all_settings_max':vals.max(),'D0_all_min':gg.D0_fit.min(),'D0_all_max':gg.D0_fit.max()})
    classification=[]
    for row in sel.itertuples():
        if row.A_uA_cm2==0:
            desc='autonomous regular firing; neutral time-shift direction'
        else:
            z=sweep[(sweep.model==row.model)&(sweep.T_ms==row.T_ms)&(abs(sweep.A_uA_cm2-row.A_uA_cm2)<1e-7)].iloc[0]
            if row.signal=='Izh_strong':
                desc='numerically chaotic: positive model exponent at three steps and in every 8-s block'
            elif row.signal=='HH_weak':
                desc='period <=8 unresolved; near-neutral exponent; quasiperiodicity plausible, unproved'
            else:
                desc=f'stable stroboscopic period {int(z.period_forcing_cycles)}T; λ<0'
        classification.append({'signal':row.signal,'classification':desc})
    reliability=[]
    for row in sel.itertuples():
        if row.signal.endswith('_DC'):
            d2status='consistent with 1';d0status='consistent with 1'
        elif row.signal in ['HH_strong','Izh_weak']:
            d2status='finite-scale; 100 observed phases';d0status='unreliable: 100 observed phases'
        elif row.signal=='Izh_strong':
            d2status='unreliable: D2>D0';d0status='unreliable: D0<D2'
        else:
            d2status='unreliable: no broad plateau';d0status='unreliable: parameter sensitivity'
        extra=corrected[corrected.signal==row.signal]
        reliability.append({'signal':row.signal,'D2_fit':row.D2_fit,'D2_status':d2status,
                            'D0_raw_m3':row.D0_mean_m3,'D0_raw_dt_ms':.2,'D0_status':d0status,
                            'D0_phase_rich_m3':float(extra.D0_m3_offset_mean.iloc[0]) if len(extra) else np.nan,
                            'D0_phase_rich_dt_ms':float(extra.dt_ms.iloc[0]) if len(extra) else np.nan,
                            'D0_phase_rich_min':float(extra.D0_min.iloc[0]) if len(extra) else np.nan,
                            'D0_phase_rich_max':float(extra.D0_max.iloc[0]) if len(extra) else np.nan})
    reliability=pd.DataFrame(reliability)
    final_summary=sel.merge(reliability,on=['signal','D2_fit']).merge(pd.DataFrame(classification),on='signal').merge(
        selected_lle[['signal','h_ms','observation_ms','lambda_max_per_ms']],on='signal')
    tables=ROOT/'optional/tables'
    reliability.to_csv(tables/'dimension_reliability.csv',index=False)
    final_summary.to_csv(tables/'selected_summary_reliability.csv',index=False)
    pd.DataFrame(classification).to_csv(tables/'selected_classification.csv',index=False)
    pd.DataFrame(dsum).to_csv(tables/'dimension_ranges.csv',index=False)
    pd.DataFrame(match).to_csv(tables/'synchronization_comparison.csv',index=False)


if __name__=='__main__':
    summarize_hr()
    summarize_optional()
    print('Part D and HH/Izh synthesis tables regenerated.')
