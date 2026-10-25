"""Sensitivity to embedding, record density, duration and ordinal settings.

These checks extend, rather than replace, the original unsuccessful estimates.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from run import (ROOT,DATA,TABLES,FIG,DT,H,TRANSIENT,DURATION,IC1,run_record,
                 read_record,csv,savefig,plt,log)
from cnm import standardize,correlation_sum,fit_scaling,box_count,ordinal,false_nearest


def selected_dimension_sensitivity():
    p=json.loads((DATA/'reconstruction_parameters.json').read_text())['chaotic']
    m=p['m'];x=standardize(read_record('chaotic')['state'][:-1,0]);rows=[]
    lo,hi=.012,.07
    experiments=[]
    for factor in [.5,1.,1.5]:
        lag=max(1,round(p['lag']*factor));experiments.append(('lag',lag,x,lag,p['W']))
    experiments.append(('acf_lag',p['acf_lag'],x,p['acf_lag'],p['W']))
    for w in [0,p['W']//2,p['W'],2*p['W'],4*p['W']]:experiments.append(('W',w,x,p['lag'],w))
    for fraction in [.25,.5,1.]:
        xx=x[:int(len(x)*fraction)];experiments.append(('N',len(xx),xx,p['lag'],p['W']))
    for j,(var,value,xx,lag,w) in enumerate(experiments):
        res=correlation_sum(xx,m,lag,w,maxpoints=12000)
        np.savez_compressed(DATA/f'gp_chaotic_selected_m_sensitivity_{j:02d}.npz',**res)
        rows.append({'regime':'chaotic','varied':var,'value':value,'m':m,'lag':lag,'W':w,'N':len(xx),
                     **fit_scaling(res,lo*np.sqrt(m),hi*np.sqrt(m))})
    baseline=dict(np.load(DATA/f'gp_chaotic_m{m}.npz'))
    for a,b in [(lo*1.3,hi),(lo,hi/1.3)]:
        rows.append({'regime':'chaotic','varied':'range','value':a,'m':m,'lag':p['lag'],'W':p['W'],'N':len(x),
                     **fit_scaling(baseline,a*np.sqrt(m),b*np.sqrt(m))})
    csv('correlation_selected_m_sensitivity.csv',rows)
    log(f'Sensitivity: sensitivity at the selected chaotic dimension m={m} completed')


def dense_fnn_check():
    p=json.loads((DATA/'reconstruction_parameters.json').read_text())['chaotic']
    x=standardize(read_record('chaotic')['state'][:-1,0])
    values=false_nearest(x,p['lag'],12,p['W'],60000)
    np.savez_compressed(DATA/'fnn_chaotic_dense_check.npz',fnn=values,lag=p['lag'],W=p['W'],maxpoints=60000)
    csv('fnn_chaotic_dense_check.csv',[{'m':int(row[0]),'FNN':row[1],
                                     'usable':int(row[2]),'omitted':int(row[3]),'maxpoints':60000} for row in values])
    log('Sensitivity: chaotic FNN repeated with 60000 reference points')


def capacity_sampling_duration():
    base=read_record('chaotic');mu=base['state'][:-1,0].mean();sigma=base['state'][:-1,0].std()
    lag_base=json.loads((DATA/'reconstruction_parameters.json').read_text())['chaotic']['lag']
    cases=[('baseline',base,DT),('longer_record',None,DT),('finer_sampling',None,.04)]
    rows=[];eps=np.geomspace(.02,.5,24);fig,axs=plt.subplots(2,3,figsize=(15,8))
    for col,(case,a,dt) in enumerate(cases):
        if a is None:
            duration=60000. if case=='longer_record' else DURATION
            path=DATA/'convergence'/f'chaotic_{case}.npz'
            run_record(path,3.25,IC1,TRANSIENT,duration,dt,H);a=np.load(path)
        else:duration=DURATION
        x=(a['state'][:-1,0]-mu)/sigma
        lag=round(lag_base*DT/dt)
        for m,style in [(3,'-'),(4,'--')]:
            counts=box_count(x,m,lag,eps)
            np.savez_compressed(DATA/f'boxes_chaotic_{case}_m{m}.npz',eps=eps,counts=counts,
                                dt=dt,lag=lag,record_N=len(x),duration=duration)
            for j,c in enumerate(counts):
                mask=(eps>=.09)&(eps<=.4)&(c<len(x)*.25)
                local=-np.gradient(np.log(c),np.log(eps))
                value=np.polyfit(np.log(1/eps[mask]),np.log(c[mask]),1)[0] if mask.sum()>=4 else np.nan
                rows.append({'case':case,'dt':dt,'duration':duration,'N':len(x),'m':m,'lag':lag,
                             'grid_shift':[0,.25,.5,.75][j],'D0_finite_scale':value,
                             'eps_min':eps[mask].min(),'eps_max':eps[mask].max(),
                             'local_min':local[mask].min(),'local_max':local[mask].max(),
                             'nscales':mask.sum(),'reliable':False})
            average=counts.mean(axis=0)
            axs[0,col].loglog(1/eps,average,style,label=f'm={m}')
            axs[0,col].fill_between(1/eps,counts.min(axis=0),counts.max(axis=0),alpha=.15)
            axs[1,col].semilogx(eps,-np.gradient(np.log(average),np.log(eps)),style,label=f'm={m}')
        axs[0,col].set(title=f'{case}\nT={duration:g}, dt={dt:g}',xlabel='1/epsilon [1/std x]',ylabel='occupied boxes')
        axs[1,col].axvspan(.09,.4,color='gray',alpha=.14)
        axs[1,col].set(xlabel='epsilon [std x]',ylabel='local box slope',ylim=(0,3))
        for ax in axs[:,col]:ax.legend()
    fig.tight_layout();savefig('B_capacity_additional_checks.png',fig);csv('capacity_additional_checks.csv',rows)
    log('Sensitivity: capacity estimates with independent duration/sampling changes completed')


def ordinal_robustness():
    import itertools
    names=['spiking','bursting','chaotic'];rows=[];allsegments=[]
    for ell,q in [(4,1),(4,10),(5,10),(5,50)]:
        for name in names:
            x=read_record(name)['state'][:-1,0]
            full=ordinal(x,ell,q)
            for j,chunk in enumerate(np.array_split(x,6)):
                result=ordinal(chunk,ell,q)
                allsegments.append({'signal':name,'ell':ell,'q':q,'dt':DT,'segment':j,
                                    'H':result['H'],'C':result['C'],'full_H':full['H'],'full_C':full['C'],
                                    'patterns_per_possible':result['patterns_per_possible']})
    seg=csv('ordinal_robustness_segments.csv',allsegments)
    for (ell,q),g in seg.groupby(['ell','q']):
        for first,second in itertools.combinations(names,2):
            a=g[g.signal==first];b=g[g.signal==second]
            mean_a=a[['H','C']].mean().to_numpy();mean_b=b[['H','C']].mean().to_numpy()
            spread_a=np.linalg.norm(a[['H','C']].to_numpy()-mean_a,axis=1).max()
            spread_b=np.linalg.norm(b[['H','C']].to_numpy()-mean_b,axis=1).max()
            rows.append({'ell':ell,'q':q,'dt':DT,'first':first,'second':second,
                         'distance_between_segment_means':np.linalg.norm(mean_a-mean_b),
                         'sum_max_segment_radii':spread_a+spread_b,
                         'segment_discs_disjoint':np.linalg.norm(mean_a-mean_b)>spread_a+spread_b})
    csv('ordinal_robustness_pairs.csv',rows)
    sens=pd.read_csv(TABLES/'entropy_sensitivity.csv');length=[]
    for (name,ell,q,dt),g in sens.groupby(['signal','ell','q','dt']):
        short=g[g.fraction==.1].iloc[0];full=g[g.fraction==1.].iloc[0]
        length.append({'signal':name,'ell':ell,'q':q,'dt':dt,
                       'HC_drift_short_to_full':np.hypot(short.H-full.H,short.C-full.C),
                       'short_N':short.N,'short_patterns_per_possible':short.patterns_per_possible})
    csv('ordinal_record_length_effect.csv',length)
    log('Sensitivity: explicit segment and record-length ordinal robustness completed')


def run_all():
    selected_dimension_sensitivity();dense_fnn_check();capacity_sampling_duration();ordinal_robustness()


if __name__=='__main__':run_all()
