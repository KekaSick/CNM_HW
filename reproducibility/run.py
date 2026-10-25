#!/usr/bin/env python3
"""Reproduce A–D: python run.py [simulate|analyze|report|all]."""
from __future__ import annotations
import argparse
import math
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', '/tmp/cnm_mpl')
os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/cnm_numba')
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cnm import *
from hc_bounds import complexity_bounds, draw_complexity_bounds

DATA, TABLES, FIG = ROOT/'data', ROOT/'tables', ROOT/'figures'
REGIMES = {'spiking':4., 'bursting':2.5, 'chaotic':3.25, 'steady':1.}
COLORS = {'spiking':'#167a7a', 'bursting':'#df8f28', 'chaotic':'#9b3d68',
          'steady':'#5d6d7e','white_noise':'#7b68b5','shuffled':'#3974b9','quasiperiodic':'#658a34'}
DT, H, TRANSIENT, DURATION = .2, .02, 10000., 30000.
plt.rcParams.update({'font.size':10, 'axes.grid':True, 'grid.alpha':.2,
                     'figure.dpi':120, 'savefig.dpi':160, 'axes.spines.top':False,
                     'axes.spines.right':False})
from publication import install_publication_exports
install_publication_exports()


def log(message):
    print(message, flush=True)


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    def convert(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        if isinstance(x, Path): return str(x)
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=convert)+'\n')


def csv(name, rows):
    df = pd.DataFrame(rows)
    df.to_csv(TABLES/name, index=False, float_format='%.12g')
    return df


def savefig(name, fig):
    fig.savefig(FIG/name, bbox_inches='tight')
    plt.close(fig)


def run_record(path, current, initial, transient, duration, dt=DT, h=H):
    path.parent.mkdir(parents=True, exist_ok=True)
    y, spikes, section, peaks, peak_times = integrate(current, initial, transient, duration, dt, h)
    np.savez_compressed(path, t=np.arange(len(y))*dt, state=y, spikes=spikes,
                        section=section, peaks=peaks, peak_times=peak_times,
                        initial=initial, transient_state=y[0], final=y[-1],
                        I=current, transient=transient, duration=duration, dt=dt, h=h)
    return {'state':y, 'spikes':spikes, 'section':section, 'peaks':peaks, 'peak_times':peak_times}


def read_record(name):
    return np.load(DATA/f'{name}.npz')


def describe_record(a):
    x = a['state'][:, 0]
    isi = np.diff(a['spikes'])
    return {'mean_x':np.mean(x), 'std_x':np.std(x), 'range_x':np.ptp(x),
            'spikes':len(a['spikes']), 'mean_isi':np.mean(isi) if len(isi) else np.nan,
            'std_isi':np.std(isi) if len(isi) else np.nan,
            'min_isi':np.min(isi) if len(isi) else np.nan,
            'max_isi':np.max(isi) if len(isi) else np.nan,
            'mean_peak':np.mean(a['peaks']) if len(a['peaks']) else np.nan}


def equilibria():
    f = lambda current: np.max(np.linalg.eigvals(jacobian(equilibrium(current))).real)
    hopf = brentq(f, 1.35, 1.37, xtol=5e-15)
    lo, hi = hopf-2e-13, hopf+2e-13
    v = equilibrium(hopf)
    A = jacobian(v)
    vals, vecs = np.linalg.eig(A.astype(complex))
    idx = np.argmax(vals.imag)
    omega, q = vals[idx].imag, vecs[:, idx]
    vp, vectors = np.linalg.eig(A.T.astype(complex))
    p = vectors[:, np.argmin(abs(vp+1j*omega))]
    p /= np.conjugate(np.vdot(p, q))
    def B(u, w): return np.array([(-6*v[0]+6)*u[0]*w[0], -10*u[0]*w[0], 0j])
    def C(u, w, z): return np.array([-6*u[0]*w[0]*z[0], 0j, 0j])
    g = (C(q,q,q.conj())-2*B(q,np.linalg.solve(A,B(q,q.conj())))
         +B(q.conj(),np.linalg.solve(2j*omega*np.eye(3)-A,B(q,q))))
    l1 = np.vdot(p, g).real/(2*omega)
    result = {'I':hopf, 'bracket':[lo,hi], 'bracket_values':[f(lo),f(hi)],
              'equilibrium':v, 'omega':omega, 'first_lyapunov_coefficient':l1,
              'crossing_derivative':(f(hopf+1e-5)-f(hopf-1e-5))/2e-5,
              'residual':np.max(np.abs(rhs(v,hopf))), 'normalization_inner_product':[np.vdot(p,q).real,np.vdot(p,q).imag]}
    dump(DATA/'critical_point.json', result)
    rows = []
    for current in np.linspace(0,4,801):
        eq = equilibrium(current)
        vals = sorted(np.linalg.eigvals(jacobian(eq)), key=lambda z:z.real)
        row = {'I':current, 'x':eq[0], 'y':eq[1], 'z':eq[2], 'stable':f(current)<0,
               'residual':np.max(np.abs(rhs(eq,current)))}
        for j,e in enumerate(vals): row.update({f'lambda{j+1}_real':e.real, f'lambda{j+1}_imag':e.imag})
        rows.append(row)
    df = csv('equilibria.csv',rows)
    spectrum = []
    for delta in [-1e-3,-1e-4,-1e-6,0.,1e-6,1e-4,1e-3]:
        eq = equilibrium(hopf+delta)
        for e in np.linalg.eigvals(jacobian(eq)):
            spectrum.append({'I':hopf+delta,'delta_I':delta,'real':e.real,'imag':e.imag})
    csv('critical_spectrum.csv',spectrum)
    fig, ax = plt.subplots(1,2,figsize=(11,4))
    for stable,style,label in [(True,'-','stable'),(False,'--','unstable')]:
        z=df[df.stable==stable];ax[0].plot(z.I,z.x,style,label=label)
    ax[0].scatter(hopf,v[0],c='black',s=25,label=f'Hopf, I={hopf:.9f}')
    ax[0].set(xlabel='I [model units]',ylabel='equilibrium x [model units]');ax[0].legend()
    near=np.linspace(hopf-.01,hopf+.01,200)
    eig=np.array([np.linalg.eigvals(jacobian(equilibrium(i))) for i in near])
    ax[1].plot(near,np.max(eig.real,axis=1),label='max Re(lambda)')
    ax[1].axhline(0,color='black',lw=.7);ax[1].axvline(hopf,color='black',ls=':')
    ax[1].set(xlabel='I [model units]',ylabel='Re(lambda) [1/model time]')
    savefig('A_equilibria.png',fig)
    log(f'Equilibrium Hopf I={hopf:.12f}, l1={l1:.8g}')


def run_sweeps():
    """Recompute the four original protocols with resolved quiet transients.

    Continuation always passes the actual final state into the next current.
    Saved maxima and section events are retained even for numerically flat tails.
    """
    overrides = {'1.300':12000., '1.350':80000.}
    tail_points, tolerance, flat_range = 1000, 1e-7, 1e-9
    settings_path = DATA/'settings.json'
    settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}
    settings.update({'sweep_grid':np.linspace(0,4,81), 'sweep_transient':6000.,
                     'sweep_transient_overrides':overrides, 'sweep_duration':4000.,
                     'sweep_dt':1., 'sweep_internal_step':H,
                     'sweep_stationarity':{'quiet_rule':'no positive x=0 crossings',
                         'tail_points':tail_points, 'max_state_distance':tolerance,
                         'max_rhs_infinity_norm':tolerance,
                         'numerically_flat_x_range':flat_range,
                         'raw_extrema_retained':True}})
    dump(settings_path,settings)
    rows, diagnostics, failures = [], [], []
    for mode in ['forward','backward','independent_ic1','independent_ic2']:
        grid=np.linspace(0,4,81)
        if mode=='backward': grid=grid[::-1]
        initial=IC2.copy() if mode in ['backward','independent_ic2'] else IC1.copy()
        for current in grid:
            if mode=='independent_ic1': initial=IC1.copy()
            if mode=='independent_ic2': initial=IC2.copy()
            transient=overrides.get(f'{current:.3f}',6000.)
            a=run_record(DATA/'sweeps'/mode/f'I_{current:.3f}.npz',
                         current,initial,transient,4000.,1.)
            description=describe_record(a)
            rows.append({'mode':mode,'I':current,**description})
            eq=equilibrium(current)
            stable=np.max(np.linalg.eigvals(jacobian(eq)).real)<0
            quiet=len(a['spikes'])==0
            candidate=quiet and stable
            distance, residual = np.nan, np.nan
            if candidate:
                tail=a['state'][-tail_points:]
                distance=np.max(np.linalg.norm(tail-eq,axis=1))
                residual=max(np.max(np.abs(rhs(v,current))) for v in tail)
            passed=candidate and distance<tolerance and residual<tolerance
            numerically_flat=passed and description['range_x']<=flat_range
            diagnostics.append({'mode':mode,'I':current,'transient':transient,
                'duration':4000.,'dt':1.,'h':H,'spikes':len(a['spikes']),
                'stable_equilibrium':stable,'quiet_candidate':candidate,
                'tail_points':tail_points if candidate else 0,
                'tail_max_state_distance':distance,'tail_max_rhs_infinity_norm':residual,
                'quiet_stationarity_passed':passed,'range_x':description['range_x'],
                'numerically_flat':numerically_flat,'raw_peak_count':len(a['peaks']),
                'numerically_flat_peak_count':len(a['peaks']) if numerically_flat else 0})
            if candidate and not passed:
                failures.append(f'{mode}, I={current:.3f}: distance={distance:.3g}, rhs={residual:.3g}')
            if mode in ['forward','backward']: initial=a['state'][-1].copy()
        log(f'Sweep {mode}: 81 currents completed')
    csv('sweep_statistics.csv',rows)
    csv('sweep_stationarity.csv',diagnostics)
    if failures:
        raise RuntimeError('Unresolved quiet sweep records: '+'; '.join(failures))
    log(f'Sweep stationarity: {sum(row["quiet_candidate"] for row in diagnostics)} quiet records passed; raw extrema retained')


def simulate():
    settings={'model':'Hindmarsh–Rose','parameters':{'a':1,'b':3,'c':1,'d':5,'r':.006,'s':4,'x_R':-1.6},
      'initial_conditions':[IC1,IC2], 'integrator':'classical fixed-step RK4, float64',
      'internal_step':H,'selected_dt':DT,'selected_transient':TRANSIENT,'selected_duration':DURATION,
      'sweep_grid':np.linspace(0,4,81),'sweep_transient':6000,'sweep_duration':4000,'sweep_dt':1.,
      'event_rule':'Hermite interpolation at internal steps; x=0, positive direction',
      'versions':{p:importlib.metadata.version(p) for p in ['numpy','scipy','matplotlib','numba','pandas']},
      'python':sys.version,'platform':platform.platform()}
    dump(DATA/'settings.json',settings)
    equilibria()
    run_sweeps()
    # Dense region serves both the bifurcation zoom and the entropy path (21 > 15).
    for current in np.linspace(2.8,3.4,31):
        run_record(DATA/'transition'/f'I_{current:.3f}.npz',current,IC1,10000.,12000.,DT)
    # Near the Hopf point: critical slowing and coexistence need longer transients.
    nearrows=[]
    for current in [1.33,1.34,1.35,1.355,1.358,1.359,1.36,1.37,1.38,1.39,1.4]:
        for label, initial in [('ic1',IC1),('ic2',IC2),('near_eq',equilibrium(current)+np.array([1e-5,0,0]))]:
            a=run_record(DATA/'onset'/f'{label}_I_{current:.3f}.npz',current,initial,40000.,6000.,1.)
            nearrows.append({'initial':label,'I':current,**describe_record(a),
                            'distance_to_eq':np.linalg.norm(a['state'][-1]-equilibrium(current))})
    csv('onset_statistics.csv',nearrows)
    conv=[]
    for name,current in REGIMES.items():
        base=run_record(DATA/f'{name}.npz',current,IC1,TRANSIENT,DURATION)
        log(f'Selected {name} I={current}: {len(base["spikes"])} spikes')
        if name=='steady':continue
        for h in [.04,.01]:
            a=run_record(DATA/'convergence'/f'{name}_h{h}.npz',current,IC1,TRANSIENT,DURATION,DT,h)
            conv.append({'regime':name,'h':h,'transient':TRANSIENT,'duration':DURATION,**describe_record(a)})
        conv.append({'regime':name,'h':H,'transient':TRANSIENT,'duration':DURATION,**describe_record(base)})
        for tr in [5000.,20000.]:
            a=run_record(DATA/'convergence'/f'{name}_transient{tr:.0f}.npz',current,IC1,tr,15000.,DT,H)
            conv.append({'regime':name,'h':H,'transient':tr,'duration':15000.,**describe_record(a)})
        for h in [.04,.02,.01]:
            a=lyapunov(current,IC1,TRANSIENT,80000.,h,1.,1000.)
            np.save(DATA/f'lyapunov_{name}_h{h}.npy',a)
            log(f'Lyapunov {name}, h={h}: {a[-1,1:4]}')
    csv('integration_convergence.csv',conv)
    for qr in [.5,2.]:
        np.save(DATA/f'lyapunov_chaotic_qr{qr}.npy',lyapunov(3.25,IC1,10000.,30000.,H,qr,1000.))
    short=[]
    for current in [1.,2.5,3.25,4.]:
        times=np.arange(101)*DT
        ref=solve_ivp(lambda t,v:rhs(v,current),(0,20),IC1,method='DOP853',
                      t_eval=times,rtol=2e-13,atol=2e-14).y.T
        np.savez_compressed(DATA/'convergence'/f'dop853_I{current}.npz',t=times,state=ref,
                            rtol=2e-13,atol=2e-14,initial=IC1)
        for h in [.04,.02,.01]:
            a=run_record(DATA/'convergence'/f'short_I{current}_h{h}.npz',current,IC1,0.,20.,DT,h)
            short.append({'I':current,'h':h,'max_abs_error':np.max(np.abs(a['state']-ref)),
                          'final_l2_error':np.linalg.norm(a['state'][-1]-ref[-1])})
    csv('short_time_accuracy.csv',short)
    onset_extra()


def onset_extra():
    """Resolve the very slow growth/decay of a tiny perturbation near Hopf."""
    rows=[]
    for current in [1.358,1.359]:
        initial=equilibrium(current)+np.array([1e-5,0.,0.])
        a=run_record(DATA/'onset'/f'extended_I_{current:.3f}.npz',current,initial,1000000.,6000.,1.)
        rows.append({'I':current,'transient':1000000.,**describe_record(a),
                     'distance_to_eq':np.linalg.norm(a['state'][-1]-equilibrium(current))})
    csv('onset_extended.csv',rows)


def figure_dynamics():
    fig,axs=plt.subplots(4,2,figsize=(12,11),sharex='col')
    stationarity=pd.read_csv(TABLES/'sweep_stationarity.csv')
    for row,mode in enumerate(['forward','backward','independent_ic1','independent_ic2']):
        for path in sorted((DATA/'sweeps'/mode).glob('*.npz')):
            a=np.load(path);I=float(a['I']);isi=np.diff(a['spikes'])
            gate=stationarity[(stationarity['mode']==mode)&np.isclose(stationarity.I,I)].iloc[0]
            if not gate.numerically_flat:
                axs[row,0].plot(np.full(len(a['peaks']),I),a['peaks'],'.',ms=.7,color='#315a7a')
            axs[row,1].plot(np.full(len(isi),I),isi,'.',ms=.7,color='#9b3d68')
        axs[row,0].set(ylabel=f'{mode}\nlocal max x [m.u.]')
        axs[row,1].set(ylabel='ISI [model time]')
    for row in axs:
        for a in row:a.set_xlim(0,4)
    for a in axs[-1]:a.set_xlabel('I [model units]')
    savefig('A_sweeps.png',fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    for path in sorted((DATA/'transition').glob('*.npz')):
        a=np.load(path);I=float(a['I']);isi=np.diff(a['spikes'])
        axs[0].plot(np.full(len(a['peaks']),I),a['peaks'],'.',ms=1,color='#315a7a')
        axs[1].plot(np.full(len(isi),I),isi,'.',ms=1,color='#9b3d68')
    axs[0].set(xlabel='I [model units]',ylabel='local max x [model units]')
    axs[1].set(xlabel='I [model units]',ylabel='ISI [model time]')
    savefig('A_transition_zoom.png',fig)
    fig=plt.figure(figsize=(14,10));rows=[];periodrows=[];blockrows=[]
    for j,name in enumerate(['spiking','bursting','chaotic']):
        a=read_record(name);t=a['t'];y=a['state'];s=a['spikes'];isi=np.diff(s)
        ax=fig.add_subplot(3,3,j*3+1);sel=t<650
        ax.plot(t[sel],y[sel,0],lw=.8,color=COLORS[name]);ax.set(title=f'{name}, I={REGIMES[name]}',xlabel='t [model time]',ylabel='x [model units]')
        ax=fig.add_subplot(3,3,j*3+2,projection='3d');v=y[t<6000] if name=='chaotic' else y[t<1000]
        ax.plot(*v.T,lw=.3,color=COLORS[name],alpha=.8);ax.set(xlabel='x [m.u.]',ylabel='y [m.u.]',zlabel='z [m.u.]')
        ax=fig.add_subplot(3,3,j*3+3);ax.scatter(isi[:-1],isi[1:],s=7,c=COLORS[name]);ax.set(xlabel='ISI(n) [model time]',ylabel='ISI(n+1) [model time]')
        if name=='spiking':
            center=isi.mean();ax.set(xlim=(center-.05,center+.05),ylim=(center-.05,center+.05))
            ax.ticklabel_format(useOffset=False)
            ax.text(.04,.9,f'ISI spread < {np.ptp(isi):.1e}',transform=ax.transAxes,fontsize=8)
        rows.append({'regime':name,'I':REGIMES[name],**describe_record(a)})
        for block,ids in enumerate(np.array_split(np.arange(len(y)-1),6)):
            xx=y[ids,0]
            blockrows.append({'regime':name,'block':block,'start_time':t[ids[0]],
                              'end_time':t[ids[-1]],'mean_x':xx.mean(),'std_x':xx.std(),
                              'min_x':xx.min(),'max_x':xx.max()})
        sec=a['section']
        for p in range(1,13):
            d=np.linalg.norm(sec[p:]-sec[:-p],axis=1)
            Ts=s[p:]-s[:-p]
            periodrows.append({'regime':name,'p_spikes':p,'section_median_error':np.median(d),
                              'section_max_error':np.max(d),'mean_period':Ts.mean(),'std_period':Ts.std()})
    fig.subplots_adjust(wspace=.35,hspace=.38)
    savefig('A_selected_regimes.png',fig);csv('selected_regimes.csv',rows);csv('periodicity.csv',periodrows);csv('block_statistics.csv',blockrows)
    lrows=[];fig,ax=plt.subplots(1,3,figsize=(14,4))
    for j,name in enumerate(['spiking','bursting','chaotic']):
        for h,ls in [(.04,':'),(.02,'-'),(.01,'--')]:
            a=np.load(DATA/f'lyapunov_{name}_h{h}.npy')
            ax[j].plot(a[:,0],a[:,1],ls,label=f'h={h}')
            row={'regime':name,'h':h,'lambda1':a[-1,1],'lambda2':a[-1,2],'lambda3':a[-1,3],
                 'block_mean':a[:,4].mean(),'block_std':a[:,4].std(ddof=1),
                 'first_half_mean':a[:40,4].mean(),'second_half_mean':a[40:,4].mean(),
                 'sum_minus_divergence':np.sum(a[-1,1:4])-a[-1,10]}
            lrows.append(row)
        ax[j].axhline(0,color='black',lw=.7);ax[j].set(title=name,xlabel='averaging time [model time]',ylabel='largest LE [1/model time]');ax[j].legend()
    savefig('A_lyapunov_convergence.png',fig);csv('lyapunov.csv',lrows)


def reconstruction():
    parameters={};rows=[];sensitivity=[];base_results={}
    fig,ax=plt.subplots(3,3,figsize=(14,11))
    figrec=plt.figure(figsize=(12,4))
    for row,name in enumerate(['spiking','bursting','chaotic']):
        a=read_record(name);x=standardize(a['state'][:-1,0]);n=len(x)
        ac=acf(x,1500);firstzero=int(np.flatnonzero(ac<=0)[0])
        mi={bins:mutual_information(x,400,bins) for bins in [16,32,64]}
        # First pronounced minimum: Gaussian smoothing sigma=1 sample,
        # prominence .005 nats. Raw curves are retained as diagnostics.
        minima={b:find_peaks(-gaussian_filter1d(v,1),prominence=.005)[0] for b,v in mi.items()}
        first_mi=int(minima[32][0]) if len(minima[32]) else firstzero
        lag=first_mi
        # A short lag in a long silent part of a periodic burst is ill-conditioned.
        # The third MI32 minimum is validated by FNN across dimensions, not by H/C.
        if name=='bursting' and len(minima[32])>=3:
            lag=int(minima[32][2])
        window=max(firstzero,lag*4)
        fnn=false_nearest(x,lag,12,window,12000)
        fnn_first=false_nearest(x,first_mi,12,max(firstzero,first_mi*4),12000)
        fnn_acf=false_nearest(x,firstzero,12,firstzero*4,12000)
        eligible=np.flatnonzero((fnn[:,1]<.01) & (np.maximum.accumulate(fnn[::-1,1])[::-1]<.01))
        candidates=fnn[fnn[:,0]>=3]
        m=(int(fnn[eligible[0],0]) if len(eligible)
           else int(candidates[np.argmin(candidates[:,1]),0]))
        m=max(3,m)
        parameters[name]={'acf_lag':firstzero,'mi_lag':first_mi,'lag':lag,'m':m,'W':window,'dt':DT,
                          'fnn_embedding_confirmed':bool(len(eligible)),
                          'selection_reason':('third MI minimum validated with FNN' if name=='bursting'
                              else 'first pronounced MI minimum; if FNN is unresolved, trial m minimizes observed FNN over m=3..12'),
                          'mi_minima':{str(b):list(v[:6]) for b,v in minima.items()},
                          'fnn_threshold':.01,'samples':n,'standardization_mean':a['state'][:-1,0].mean(),
                          'standardization_std':a['state'][:-1,0].std()}
        np.savez_compressed(DATA/f'reconstruction_{name}.npz',acf=ac,mi16=mi[16],mi32=mi[32],mi64=mi[64],fnn=fnn,fnn_first_mi=fnn_first,fnn_acf=fnn_acf)
        ax[row,0].plot(np.arange(len(ac))*DT,ac);ax[row,0].axhline(0,color='black',lw=.7);ax[row,0].axvline(firstzero*DT,ls=':');ax[row,0].set(title=name,xlabel='delay [model time]',ylabel='ACF')
        for b,v in mi.items():ax[row,1].plot(np.arange(len(v))*DT,v,label=f'{b} bins')
        ax[row,1].axvline(lag*DT,ls=':',color='black');ax[row,1].axvline(first_mi*DT,ls='--',color='gray',lw=.8);ax[row,1].set(xlabel='delay [model time]',ylabel='mutual information [nat]');ax[row,1].legend(fontsize=8)
        for f,label,style in [(fnn,'selected','o-'),(fnn_first,'first MI','--'),(fnn_acf,'ACF',':')]:
            ax[row,2].semilogy(f[:,0],np.maximum(f[:,1],1e-5),style,label=label)
        ax[row,2].axhline(.01,ls=':',color='black');ax[row,2].set(xlabel='embedding dimension m',ylabel='FNN fraction (floor 1e-5)');ax[row,2].legend(fontsize=8)
        ar=figrec.add_subplot(1,3,row+1,projection='3d');Z=embed(x,3,lag)
        if name=='chaotic':
            ar.scatter(*Z[::3].T,s=.12,color=COLORS[name],alpha=.25,rasterized=True)
        else:
            ar.plot(*Z[:5000].T,lw=.4,color=COLORS[name])
        ar.set(title=f'{name}, tau={lag*DT:.1f}',xlabel='X1 [std x]',ylabel='X2 [std x]',zlabel='X3 [std x]')
        # Initial dimension curves are computed for a common normalized radius r/sqrt(m).
        curves={}
        for mm in [2,3,4,5,6,8]:
            result=correlation_sum(x,mm,lag,window,maxpoints=12000)
            curves[mm]=result
            np.savez_compressed(DATA/f'gp_{name}_m{mm}.npz',**result)
        base_results[name]=curves
        log(f'Reconstruction {name}: ACF={firstzero}, MI={lag}, m={m}, W={window}')
    fig.tight_layout();savefig('B_reconstruction_diagnostics.png',fig);savefig('B_reconstructed_attractors.png',figrec)
    dump(DATA/'reconstruction_parameters.json',parameters)
    # Scaling intervals are fixed in r/sqrt(m), then inspected across m.
    # Conservative intervals avoid C~1 and small-scale finite sample saturation.
    ranges={'spiking':(.018,.09),'bursting':(.014,.07),'chaotic':(.012,.07)}
    fig,ax=plt.subplots(3,2,figsize=(12,12))
    for j,name in enumerate(['spiking','bursting','chaotic']):
        p=parameters[name];x=standardize(read_record(name)['state'][:-1,0])
        lo,hi=ranges[name]
        for mm,result in base_results[name].items():
            fit=fit_scaling(result,lo*np.sqrt(mm),hi*np.sqrt(mm))
            rows.append({'regime':name,'m':mm,'lag':p['lag'],'W':p['W'],'record_N':len(x),'maxpoints':12000,**fit})
            ax[j,0].loglog(result['r']/np.sqrt(mm),result['C'],label=f'm={mm}, D={fit["estimate"]:.2f}')
            ax[j,1].semilogx(result['r']/np.sqrt(mm),result['slope'],label=f'm={mm}')
        for a in ax[j]:a.axvspan(lo,hi,color='gray',alpha=.14)
        ax[j,0].set(title=name,xlabel='r/sqrt(m) [std x]',ylabel='C(r)');ax[j,0].legend(fontsize=8)
        ax[j,1].set(xlabel='r/sqrt(m) [std x]',ylabel='local d log(C)/d log(r)',ylim=(0,4));ax[j,1].legend(fontsize=8)
        for factor in [.5,1.5]:
            lag=max(1,round(p['lag']*factor));res=correlation_sum(x,5,lag,p['W'],maxpoints=10000)
            sensitivity.append({'regime':name,'varied':'lag','value':lag,'m':5,'lag':lag,'W':p['W'],'N':len(x),**fit_scaling(res,lo*np.sqrt(5),hi*np.sqrt(5))})
        for w in [0,p['W']//2,2*p['W'],4*p['W']]:
            res=correlation_sum(x,5,p['lag'],w,maxpoints=10000)
            sensitivity.append({'regime':name,'varied':'W','value':w,'m':5,'lag':p['lag'],'W':w,'N':len(x),**fit_scaling(res,lo*np.sqrt(5),hi*np.sqrt(5))})
        for fraction in [.25,.5,1.]:
            xx=x[:int(len(x)*fraction)];res=correlation_sum(xx,5,p['lag'],p['W'],maxpoints=10000)
            sensitivity.append({'regime':name,'varied':'N','value':len(xx),'m':5,'lag':p['lag'],'W':p['W'],'N':len(xx),**fit_scaling(res,lo*np.sqrt(5),hi*np.sqrt(5))})
        # ACF candidate and alternative scaling boundaries are separate checks.
        res=correlation_sum(x,5,p['acf_lag'],p['W'],maxpoints=10000)
        sensitivity.append({'regime':name,'varied':'acf_lag','value':p['acf_lag'],'m':5,'lag':p['acf_lag'],'W':p['W'],'N':len(x),**fit_scaling(res,lo*np.sqrt(5),hi*np.sqrt(5))})
        for a,b in [(lo*1.3,hi),(lo,hi/1.3)]:
            sensitivity.append({'regime':name,'varied':'range','value':a,'m':5,'lag':p['lag'],'W':p['W'],'N':len(x),**fit_scaling(base_results[name][5],a*np.sqrt(5),b*np.sqrt(5))})
    fig.tight_layout();savefig('B_correlation_dimension.png',fig)
    csv('correlation_dimensions.csv',rows);csv('correlation_sensitivity.csv',sensitivity)
    log('Correlation dimension sensitivity completed')
    boxrows=[];fig,ax=plt.subplots(2,2,figsize=(12,9))
    eps=np.geomspace(.008,.8,30)
    for j,name in enumerate(['spiking','chaotic']):
        x=standardize(read_record(name)['state'][:-1,0]);p=parameters[name]
        for mm in [3,4,5]:
            for fraction in [.25,.5,1.]:
                xx=x[:int(len(x)*fraction)]
                counts=box_count(xx,mm,p['lag'],eps)
                np.savez_compressed(DATA/f'boxes_{name}_m{mm}_fraction{fraction}.npz',eps=eps,counts=counts)
                for k,c in enumerate(counts):
                    lower,upper=(.025,.16) if name=='spiking' else (.09,.4)
                    mask=(eps>=lower)&(eps<=upper)&(c<len(xx)*.25)
                    slope,intercept=np.polyfit(np.log(1/eps[mask]),np.log(c[mask]),1)
                    local=-np.gradient(np.log(c),np.log(eps))
                    boxrows.append({'regime':name,'m':mm,'record_N':len(xx),'grid_shift':[0,.25,.5,.75][k],
                                    'D0':slope,'eps_min':eps[mask].min(),'eps_max':eps[mask].max(),
                                    'occupied_min':c[mask].min(),'occupied_max':c[mask].max(),
                                    'nscales':mask.sum(),'local_min':local[mask].min(),'local_max':local[mask].max()})
                    if mm==4 and fraction==1.:
                        ax[j,0].loglog(1/eps,c,'o-',ms=3,label=f'shift={[0,.25,.5,.75][k]}')
                        ax[j,1].semilogx(eps,local,label=f'shift={[0,.25,.5,.75][k]}')
        ax[j,0].set(title=f'{name}, m=4',xlabel='1/epsilon [1/std x]',ylabel='occupied boxes');ax[j,0].legend()
        ax[j,1].axvspan(lower,upper,color='gray',alpha=.15);ax[j,1].set(xlabel='epsilon [std x]',ylabel='local box slope',ylim=(0,4));ax[j,1].legend()
    fig.tight_layout();savefig('B_capacity_dimension.png',fig);csv('capacity_dimensions.csv',boxrows)
    fig,ax=plt.subplots(1,2,figsize=(12,4))
    sens=pd.DataFrame(sensitivity)
    for name in ['spiking','bursting','chaotic']:
        for j,var in enumerate(['W','N']):
            z=sens[(sens.regime==name)&(sens.varied==var)]
            ax[j].plot(z.value,z.estimate,'o-',label=name,color=COLORS[name])
    ax[0].set(xlabel='Theiler W [samples]',ylabel='D2 estimate');ax[1].set(xlabel='record length N [samples]',ylabel='D2 estimate')
    for a in ax:a.legend()
    savefig('B_dimension_sensitivity.png',fig)


def entropy_study():
    rng=np.random.Generator(np.random.PCG64(20261024))
    dump(DATA/'rng_initial_state.json',rng.bit_generator.state)
    signals={name:read_record(name)['state'][:-1,0].copy() for name in ['spiking','bursting','chaotic']}
    N=len(signals['spiking']);t=np.arange(N)*DT
    T0=np.diff(read_record('spiking')['spikes']).mean()
    signals.update({'white_noise':rng.normal(size=N), 'shuffled':rng.permutation(signals['chaotic']),
                    'quasiperiodic':np.sin(2*np.pi*t/T0)+.5*np.sin(2*np.pi*np.sqrt(2)*t/T0)})
    np.savez_compressed(DATA/'controls.npz',t=t,white_noise=signals['white_noise'],
                        shuffled=signals['shuffled'],quasiperiodic=signals['quasiperiodic'],T0=T0,dt=DT)
    # Equal main parameters for every neuronal and control signal.
    ell,q=5,10
    main=[];segments=[];sens=[]
    for name,x in signals.items():
        out=ordinal(x,ell,q)
        main.append({'signal':name,'ell':ell,'q':q,'dt':DT,'N':N,**{k:v for k,v in out.items() if k!='counts'}})
        np.save(DATA/f'ordinal_counts_{name}.npy',out['counts'])
        for j,chunk in enumerate(np.array_split(x,6)):
            out=ordinal(chunk,ell,q)
            segments.append({'signal':name,'segment':j,'ell':ell,'q':q,'dt':DT,'N':len(chunk),**{k:v for k,v in out.items() if k!='counts'}})
        for e in [4,5]:
            for step in [1,2,5]:
                for delay in [1,5,10,25,50]:
                    for fraction in [.1,.5,1.]:
                        xx=x[:int(N*fraction):step];out=ordinal(xx,e,delay)
                        sens.append({'signal':name,'ell':e,'q':delay,'dt':DT*step,'physical_delay':DT*step*delay,'fraction':fraction,'N':len(xx),**{k:v for k,v in out.items() if k!='counts'}})
    df=csv('entropy_complexity.csv',main);seg=csv('entropy_segments.csv',segments);csv('entropy_sensitivity.csv',sens)
    transition=[]
    for path in sorted((DATA/'transition').glob('*.npz')):
        a=np.load(path);x=a['state'][:-1,0];out=ordinal(x,ell,q)
        transition.append({'I':float(a['I']),'ell':ell,'q':q,'dt':DT,**describe_record(a),**{k:v for k,v in out.items() if k!='counts'}})
    z=csv('entropy_transition.csv',transition)
    noise=[];gp=[];parameters=json.loads((DATA/'reconstruction_parameters.json').read_text())
    noise_state=[]
    for name in ['spiking','chaotic']:
        x=signals[name];sigma=x.std()
        for eta in [0.,.01,.05,.1,.2]:
            for realization in range(1 if eta==0 else 5):
                state=rng.bit_generator.state
                observed=x.copy() if eta==0 else x+eta*sigma*rng.normal(size=N)
                noise_state.append({'regime':name,'eta':eta,'realization':realization,'initial_rng_state':state})
                np.savez_compressed(DATA/'noise'/f'{name}_eta{eta}_r{realization}.npz',observed=observed,sigma=sigma,eta=eta,dt=DT)
                for e,delay in [(4,10),(5,10),(5,50)]:
                    out=ordinal(observed,e,delay)
                    noise.append({'signal':name,'eta':eta,'realization':realization,'ell':e,'q':delay,'dt':DT,**{k:v for k,v in out.items() if k!='counts'}})
                if name=='spiking' and eta==.05 and realization==0:
                    p=parameters[name];xx=standardize(observed)
                    for m in [2,3,4,5,6]:
                        res=correlation_sum(xx,m,p['lag'],p['W'],maxpoints=12000)
                        np.savez_compressed(DATA/f'gp_noisy_spiking_m{m}.npz',**res)
                        for scale,lo,hi in [('clean_range',.018,.09),('alternative_range',.04,.1)]:
                            gp.append({'signal':name,'eta':eta,'m':m,'scale':scale,**fit_scaling(res,lo*np.sqrt(m),hi*np.sqrt(m))})
    dump(DATA/'noise_rng_states.json',noise_state);dump(DATA/'rng_final_state.json',rng.bit_generator.state)
    nd=csv('measurement_noise.csv',noise);csv('noisy_correlation_dimension.csv',gp)
    plot_entropy_results()
    fig,ax=plt.subplots(1,2,figsize=(12,4))
    for m in [2,3,4,5,6]:
        z=np.load(DATA/f'gp_noisy_spiking_m{m}.npz')
        ax[0].loglog(z['r']/np.sqrt(m),z['C'],label=f'm={m}')
        ax[1].semilogx(z['r']/np.sqrt(m),z['slope'],label=f'm={m}')
    ax[0].set(xlabel='r/sqrt(m) [std x]',ylabel='C(r)',title='Spiking + measurement noise eta=0.05');ax[1].set(xlabel='r/sqrt(m) [std x]',ylabel='local slope',ylim=(0,7))
    for a in ax:a.legend(fontsize=8)
    savefig('C_noisy_dimension.png',fig)
    log('Entropy controls, 31-current path, sensitivity and noise completed')


def plot_entropy_results():
    """Render H-C results and their theoretical region from saved tables only."""
    df=pd.read_csv(TABLES/'entropy_complexity.csv');seg=pd.read_csv(TABLES/'entropy_segments.csv')
    sens=pd.read_csv(TABLES/'entropy_sensitivity.csv');transition=pd.read_csv(TABLES/'entropy_transition.csv')
    noise=pd.read_csv(TABLES/'measurement_noise.csv')
    for ell in [4,5]:
        lower,upper=complexity_bounds(math.factorial(ell))
        np.savez_compressed(DATA/f'complexity_bounds_K{math.factorial(ell)}.npz',
                            lower=lower,upper=upper,K=math.factorial(ell))
    fig,ax=plt.subplots(figsize=(9,6))
    for name in df.signal:
        row=df[df.signal==name].iloc[0];g=seg[seg.signal==name]
        ax.scatter(g.H,g.C,s=20,alpha=.35,color=COLORS[name],zorder=3)
        ax.errorbar(row.H,row.C,xerr=g.H.std(ddof=1),yerr=g.C.std(ddof=1),fmt='o',
                    capsize=3,color=COLORS[name],label=name,zorder=4)
    ax.set(xlabel='normalized permutation entropy H',ylabel='Jensen-Shannon complexity C',
           title='ell=5, K=120, q=10, dt=0.2; 6 consecutive segments')
    draw_complexity_bounds(ax,5);savefig('C_entropy_complexity.png',fig)
    fig,axs=plt.subplots(2,2,figsize=(11,8.5));z=transition
    axs[0,0].plot(z.I,z.H,'o-');axs[0,1].plot(z.I,z.C,'o-')
    for a,label in zip(axs[0],['H','C']):a.set(xlabel='I [model units]',ylabel=label)
    for ax in axs[1]:
        ax.plot(z.H,z.C,color='gray',lw=.8,zorder=3)
        sc=ax.scatter(z.H,z.C,c=z.I,cmap='viridis',zorder=4)
        ax.set(xlabel='H',ylabel='C')
    draw_complexity_bounds(axs[1,0],5)
    axs[1,0].set_title('ell=5, K=120; theoretical region')
    axs[1,1].set(xlim=(z.H.min()-.001,z.H.max()+.001),
                 ylim=(z.C.min()-.001,z.C.max()+.001),title='transition path: detailed view')
    fig.colorbar(sc,ax=axs[1,1],label='I [model units]')
    fig.tight_layout();savefig('C_transition_path.png',fig)
    fig,axs=plt.subplots(2,2,figsize=(12,9))
    for rr,ell in enumerate([4,5]):
        for name in df.signal:
            z=sens[(sens.signal==name)&(sens.ell==ell)&(sens.dt==DT)&(sens.fraction==1.)]
            axs[rr,0].plot(z.physical_delay,z.H,'o-',label=name,color=COLORS[name])
            axs[rr,1].plot(z.H,z.C,'o-',label=name,color=COLORS[name],zorder=3)
        axs[rr,0].set(title=f'ell={ell}',xlabel='q*dt [model time]',ylabel='H')
        axs[rr,1].set(title=f'ell={ell}, K={math.factorial(ell)}',xlabel='H',ylabel='C')
        draw_complexity_bounds(axs[rr,1],ell,legend=False)
        axs[rr,1].legend(handles=axs[rr,1].lines[-2:],fontsize=8,loc='upper left')
    axs[0,0].legend(fontsize=8);savefig('C_parameter_sensitivity.png',fig)
    fig,axs=plt.subplots(3,1,figsize=(7.5,12.5))
    markers={0.:'o',.01:'s',.05:'^',.1:'D',.2:'X'}
    for ax,(ell,q) in zip(axs,[(4,10),(5,10),(5,50)]):
        for name in ['spiking','chaotic']:
            z=noise[(noise.signal==name)&(noise.ell==ell)&(noise.q==q)]
            means=z.groupby('eta')[['H','C']].mean();std=z.groupby('eta')[['H','C']].std().fillna(0)
            ax.errorbar(means.H,means.C,xerr=std.H,yerr=std.C,fmt='-',color=COLORS[name],
                        capsize=3,label=name,zorder=3)
            for eta,row in means.iterrows():
                ax.scatter(row.H,row.C,marker=markers[eta],s=40,color=COLORS[name],zorder=4)
        ax.set(title=f'ell={ell}, K={math.factorial(ell)}, q={q}, dt={DT}',xlabel='H',ylabel='C')
        ax.tick_params(labelsize=10)
        ax.xaxis.label.set_size(11);ax.yaxis.label.set_size(11)
        draw_complexity_bounds(ax,ell,legend=False);ax.legend(fontsize=9,loc='upper left')
    from matplotlib.lines import Line2D
    handles=[Line2D([],[],color='#333333',marker=marker,linestyle='none',label=f'{eta:g}')
             for eta,marker in markers.items()]
    fig.legend(handles=handles,loc='lower center',ncol=5,title='noise level eta',fontsize=10)
    fig.tight_layout(rect=(0,.055,1,1));savefig('C_measurement_noise.png',fig)


def analyze():
    figure_dynamics();reconstruction();entropy_study()
    integration_observables()
    from sensitivity_analysis import run_all
    run_all()


def integration_observables():
    rows=[]
    for name in ['spiking','bursting','chaotic']:
        for h in [.04,.02,.01]:
            path=DATA/f'{name}.npz' if h==.02 else DATA/'convergence'/f'{name}_h{h}.npz'
            a=np.load(path);out=ordinal(a['state'][:-1,0],5,10)
            rows.append({'regime':name,'h':h,**describe_record(a),'H':out['H'],'C':out['C']})
    csv('integration_observables.csv',rows)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',choices=['simulate','sweeps','analyze','plot-entropy','all'],nargs='?',default='all');args=ap.parse_args()
    for p in [DATA,TABLES,FIG,DATA/'noise']:p.mkdir(parents=True,exist_ok=True)
    start=time.time()
    if args.stage in ['simulate','all']:simulate()
    if args.stage=='sweeps':run_sweeps()
    if args.stage in ['analyze','all']:analyze()
    if args.stage=='plot-entropy':plot_entropy_results()
    log(f'{args.stage} completed in {time.time()-start:.1f} s')


if __name__=='__main__':main()
