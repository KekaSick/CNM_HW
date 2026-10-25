"""Reproduce optional sections 5.1–5.3. Run from any working directory."""
from pathlib import Path
import os,time,json,platform,sys,math
os.environ.setdefault('MPLCONFIGDIR','/tmp/cnm_optional_mpl')
os.environ.setdefault('NUMBA_CACHE_DIR','/tmp/cnm_optional_numba')
import numpy as np
import pandas as pd
import scipy,numba,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import root
from models import simulate,initial,rates,firing_rate,flow_variational
from calibrate import calibration
from scalar import standardize,acf,mutual_information,false_nearest,embed,correlation_sum,fit_scaling,box_count,ordinal
ROOT=Path(__file__).resolve().parent
sys.path.append(str(ROOT.parent/'src'))
from hc_bounds import draw_complexity_bounds
from publication import install_publication_exports
install_publication_exports()
for d in ['tables','data','figures']: (ROOT/d).mkdir(exist_ok=True)
plt.rcParams.update({'figure.dpi':130,'font.size':9,'axes.grid':True,'grid.alpha':.2})
COL={'HH':'#006c9e','Izh':'#c34a24'}

def savefig(name):
    plt.tight_layout();plt.savefig(ROOT/'figures'/f'{name}.png');plt.close()

def plot_isi_return(ax,isi,name,label):
    """Keep microscopic timing error from looking like resolved ISI variability."""
    ax.scatter(isi[:-1],isi[1:],s=3,c=COL[label])
    ax.set(title=name,xlabel='ISIₙ [ms]',ylabel='ISIₙ₊₁ [ms]')
    mean=float(np.mean(isi));sigma=float(np.std(isi))
    if sigma/mean<1e-4:
        ax.set(xlim=(mean-.05,mean+.05),ylim=(mean-.05,mean+.05))
        ax.ticklabel_format(axis='both',style='plain',useOffset=False)
        ax.text(.03,.97,f'σ(ISI)={sigma:.2g} ms\nmax−min={np.ptp(isi):.2g} ms',
                transform=ax.transAxes,va='top',fontsize=8,
                bbox={'facecolor':'white','edgecolor':'none','alpha':.85})

def replot_isi_returns():
    """Only rebuild the return plot using the persisted selected spike times."""
    fig,axs=plt.subplots(3,2,figsize=(11,9))
    for rr,kind in enumerate(['DC','weak','strong']):
        for cc,label in enumerate(['HH','Izh']):
            name=f'{label}_{kind}'
            with np.load(ROOT/'data'/f'selected_{name}.npz') as data:
                plot_isi_return(axs[rr,cc],np.diff(data['spikes_ms']),name,label)
    savefig('ISI_returns')

def thresholds(k,b):
    from threshold_checks import threshold
    rows=[]
    for model,label,k0,b0,limits in [(0,'HH',1.,0.,(5.,8.)),(1,'Izh',k,b,(-10.,10.))]:
        for h in [.02,.01]:
            rows.append(threshold(model,label,k0,b0,1500.,3000.,h,
                                  'sustained_short',limits,'sustained_short'))
    pd.DataFrame(rows).to_csv(ROOT/'tables/thresholds.csv',index=False)
    return rows


def sweeps(k,b):
    rows=[];sh=[]
    for T in [10.,20.,40.]:
        amps=np.round(np.arange(0.,15.0001,.1 if T==20 else .5),4)
        for label,model in [('HH',0),('Izh',1)]:
            for A in amps:
                # 100 forcing periods of transient; 200 periods observed, all phase aligned.
                tr=100*T;du=200*T
                x,s=simulate(model,10.,A,T,1. if model==0 else k,0. if model==0 else b,tr,du,.2,.01)
                st=x[::int(round(T/.2))]
                isi=np.diff(s);tail=st[-101:]
                # Full-state stroboscopic recurrence on final 100 forcing cycles.
                scales=np.array([100.,1.,1.,1.]) if model==0 else np.array([100.,30.])
                errs=[]
                for p in range(1,9):
                    errs.append(float(np.sqrt(np.mean(((tail[p:]-tail[:-p])/scales)**2))))
                p=next((ii+1 for ii,e in enumerate(errs) if e<1e-5),0)
                nper=np.bincount(np.minimum((s/T).astype(int),199),minlength=200)
                ratio=len(s)/200.
                np.savez_compressed(ROOT/'data'/f'sweep_{label}_T{T:g}_A{A:.1f}.npz',states=x,spikes_ms=s,stroboscopic_states=st,dt_ms=.2,transient_ms=tr,T_ms=T,A_uA_cm2=A)
                row={'model':label,'T_ms':T,'A_uA_cm2':A,'rate_Hz':firing_rate(s),'spikes_per_forcing_period':ratio,'period_forcing_cycles':p,
                     'ISI_mean_ms':float(np.mean(isi)) if len(isi) else np.nan,'ISI_CV':float(np.std(isi)/np.mean(isi)) if len(isi) else np.nan,
                     'spikes_min_per_cycle':int(nper.min()),'spikes_max_per_cycle':int(nper.max()),'recurrence_best_scaled_RMS':min(errs)}
                rows.append(row)
                if T==20:
                    for j,v in enumerate(isi[-100:]):sh.append({'model':label,'A_uA_cm2':A,'kind':'ISI_ms','value':v})
                    for v in st[-100:,0]:sh.append({'model':label,'A_uA_cm2':A,'kind':'stroboscopic_mV','value':v})
            print('sweep',label,T,'complete',flush=True)
    df=pd.DataFrame(rows);df.to_csv(ROOT/'tables/periodic_sweep.csv',index=False)
    cloud=pd.DataFrame(sh);cloud.to_csv(ROOT/'tables/detailed_cloud.csv',index=False)
    fig,axs=plt.subplots(2,3,figsize=(13,6))
    for col,T in enumerate([10.,20.,40.]):
        for label in ['HH','Izh']:
            sel=df[(df.model==label)&(df.T_ms==T)]
            axs[0,col].plot(sel.A_uA_cm2,sel.spikes_per_forcing_period,label=label,c=COL[label])
            axs[1,col].scatter(sel.A_uA_cm2,sel.period_forcing_cycles,s=8,c=COL[label],label=label)
        axs[0,col].set(title=f'T={T:g} ms, J₀=10 µA/cm²',ylabel='spikes / forcing cycle')
        axs[1,col].set(xlabel='A [µA/cm²]',ylabel='detected period [cycles]; 0 unresolved')
        axs[0,col].legend()
    savefig('forcing_overview')
    fig,axs=plt.subplots(2,2,figsize=(12,7))
    for col,label in enumerate(['HH','Izh']):
        for rr,kind in enumerate(['ISI_ms','stroboscopic_mV']):
            z=cloud[(cloud.model==label)&(cloud.kind==kind)]
            axs[rr,col].scatter(z.A_uA_cm2,z.value,s=.35,c=COL[label])
            axs[rr,col].set(xlabel='A [µA/cm²]',ylabel='ISI [ms]' if rr==0 else 'V(nT) [mV scale]',title=f'{label}; T=20 ms')
    savefig('detailed_T20')
    return df

def orbit_stability(k,b):
    rows=[];results={}
    for label,model in [('HH',0),('Izh',1)]:
        # Choose a well resolved one-cycle response at high amplitude, without fitting parameters.
        sweep=pd.read_csv(ROOT/'tables/periodic_sweep.csv')
        poss=sweep[(sweep.model==label)&(sweep.T_ms==20)&(sweep.period_forcing_cycles==1)&(sweep.A_uA_cm2>5)]
        if len(poss)==0: raise RuntimeError('No detected period-one response to refine')
        A=float(poss.iloc[len(poss)//2].A_uA_cm2);T=20.;pars=(10.,A,T,1. if model==0 else k,0. if model==0 else b)
        archive=np.load(ROOT/'data'/f'sweep_{label}_T20_A{A:.1f}.npz');x0=archive['states'][-1]
        def fun(x):
            y,M,_=flow_variational(model,x,T,pars)
            return y-x,M-np.eye(len(x))
        sol=root(fun,x0,jac=True,tol=1e-10)
        y,M,ev=flow_variational(model,sol.x,T,pars)
        vals=np.linalg.eigvals(M)
        cols=[]
        for j in range(len(x0)):
            eps=1e-4 if j==0 or model==1 else 1e-6
            e=np.eye(len(x0))[j]*eps
            cols.append((flow_variational(model,sol.x+e,T,pars)[0]-flow_variational(model,sol.x-e,T,pars)[0])/(2*eps))
        FD=np.column_stack(cols)
        rel=float(np.linalg.norm(M-FD)/np.linalg.norm(FD))
        _,Mf,_=flow_variational(model,sol.x,T,pars,rtol=2e-11,atol=1e-12)
        naive=np.nan
        if model==1: naive=float(np.max(np.abs(np.linalg.eigvals(flow_variational(model,sol.x,T,pars,reset_derivative='DR')[1]))))
        np.savez_compressed(ROOT/'data'/f'orbit_{label}.npz',phase0_state=sol.x,monodromy=M,multipliers=vals,finite_difference=FD,spikes_ms=ev,A_uA_cm2=A,T_ms=T)
        rows.append({'model':label,'T_ms':T,'A_uA_cm2':A,'root_success':bool(sol.success),'residual_max':float(np.max(np.abs(y-sol.x))),
                     'spectral_radius':float(np.max(np.abs(vals))),'multipliers':str(vals),'FD_relative_error':rel,'tight_tolerance_matrix_difference':float(np.linalg.norm(M-Mf)),
                     'wrong_DR_only_spectral_radius':naive,'events_per_period':len(ev) if model==1 else np.nan})
        results[label]=(A,sol.x,vals)
    pd.DataFrame(rows).to_csv(ROOT/'tables/periodic_orbits.csv',index=False)
    return results

def selected_analysis(k,b):
    # Identical forcing, observation interval and voltage-only processing for both models.
    sets=[('DC',0.,20.),('weak',1.,20.),('strong',10.,20.)]
    allhc=[];isirows=[];summary=[];sensitivity=[];boxrows=[];diag=[];conv=[]
    figtrace,axtrace=plt.subplots(3,2,figsize=(13,8))
    figrec,axrec=plt.subplots(3,2,figsize=(11,9))
    figdelay,axdelay=plt.subplots(3,2,figsize=(12,8))
    figcorr,axcorr=plt.subplots(3,2,figsize=(12,10))
    figbox,axbox=plt.subplots(3,2,figsize=(12,9))
    figr,axr=plt.subplots(3,2,figsize=(11,9))
    for rr,(kind,A,T) in enumerate(sets):
        for cc,(label,model) in enumerate([('HH',0),('Izh',1)]):
            name=f'{label}_{kind}';tr=2000.;duration=8000.;dt=.2
            x,s=simulate(model,10.,A,T,1. if model==0 else k,0. if model==0 else b,tr,duration,dt,.01)
            np.savez_compressed(ROOT/'data'/f'selected_{name}.npz',states=x,spikes_ms=s,dt_ms=dt,transient_ms=tr,T_ms=T,A_uA_cm2=A)
            # Fine integration on first 400 ms, independent run from specified initial state.
            xf,sf=simulate(model,10.,A,T,1. if model==0 else k,0. if model==0 else b,tr,400.,dt,.005)
            coarse=s[s<=400.]
            conv.append({'signal':name,'coarse_h_ms':.01,'fine_h_ms':.005,'max_spike_difference_ms':float(np.max(np.abs(coarse-sf))) if len(coarse)==len(sf) else np.nan,
                         'coarse_spikes':len(coarse),'fine_spikes':len(sf),'voltage_RMS_mV':float(np.sqrt(np.mean((x[:len(xf),0]-xf[:,0])**2)))})
            z=standardize(x[:,0]);aa=acf(z,200);mi=mutual_information(z,200)
            ac=int(np.where(aa<=0)[0][0]) if np.any(aa<=0) else 0
            mins=np.where((mi[1:-1]<mi[:-2])&(mi[1:-1]<mi[2:]))[0]+1
            lag=int(mins[0]) if len(mins) else 5
            fn=false_nearest(z,lag,maxdim=7,window=200,maxpoints=5000)
            np.savez_compressed(ROOT/'data'/f'diagnostics_{name}.npz',acf=aa,MI_nats=mi,FNN=fn,lag_samples=lag)
            pd.DataFrame(fn,columns=['m','false_fraction','admissible_references','omitted_references']).to_csv(ROOT/'tables'/f'fnn_{name}.csv',index=False)
            diag.append({'signal':name,'ACF_first_zero_samples':ac,'MI_first_min_samples':lag,'ACF_zero_ms':ac*dt,'MI_min_ms':lag*dt,'FNN_m3':fn[2,1],'FNN_m5':fn[4,1]})
            axtrace[rr,cc].plot(np.arange(1501)*dt,x[:1501,0],lw=.8,c=COL[label]);axtrace[rr,cc].set(title=f'{name}, J₀=10, A={A:g} µA/cm²; T=20 ms',xlabel='t after transient [ms]',ylabel='V [mV scale]')
            E=embed(z,3,5);axrec[rr,cc].plot(E[:10000,0],E[:10000,1],lw=.3,c=COL[label]);axrec[rr,cc].set(title=f'{name}: m=3, lag=1 ms',xlabel='z(t) [standard deviations]',ylabel='z(t+1 ms)')
            axdelay[rr,cc].plot(np.arange(201)*dt,aa,label='ACF')
            axdelay[rr,cc].plot(np.arange(201)*dt,mi/mi[0],label='MI / MI(0)');axdelay[rr,cc].axvline(lag*dt,c='k',ls=':',lw=.7)
            axdelay[rr,cc].set(title=name,xlabel='lag [ms]',ylabel='normalized diagnostic');axdelay[rr,cc].legend()
            # GP sensitivity: several m, a physical common lag, MI lag, N and Theiler width.
            radii=np.geomspace(.008,.9,55);configs=[(m,5,200,len(z)) for m in [2,3,4,5,6]]+[(4,l,200,len(z)) for l in [max(1,lag),10]]+[(4,5,w,len(z)) for w in [0,100,400]]+[(4,5,200,len(z)//2)]
            basefit=None
            for m,l,W,n in configs:
                result=correlation_sum(z[:n],m,l,W,radii,maxpoints=3500)
                fit=fit_scaling(result,.06,.35,min_count=200)
                sensitivity.append({'signal':name,'m':m,'lag_samples':l,'lag_ms':l*dt,'W_samples':W,'W_ms':W*dt,'N':n,**fit})
                if l==5 and W==200 and n==len(z):
                    axcorr[rr,cc].plot(result['r'],result['C'],label=f'm={m}')
                    np.savez_compressed(ROOT/'data'/f'GP_{name}_m{m}.npz',**result)
                    if m==4:basefit=fit
            axcorr[rr,cc].axvspan(.06,.35,color='gray',alpha=.15)
            axcorr[rr,cc].set(xscale='log',yscale='log',title=name,xlabel='r [standardized Euclidean units]',ylabel='C(r), W=40 ms');axcorr[rr,cc].legend()
            # Box count sensitivity in m, offset and record length, fixed physical delay 1 ms.
            eps=np.geomspace(.025,.7,22);basebox=[]
            for m in [2,3,4]:
                for n in [len(z)//2,len(z)]:
                    counts=box_count(z[:n],m,5,eps)
                    for shift,c in zip([0.,.25,.5,.75],counts):
                        mask=(eps>=.08)&(eps<=.4)&(c<.5*n)
                        fit=np.polyfit(np.log(1/eps[mask]),np.log(c[mask]),1)
                        boxrows.append({'signal':name,'m':m,'N':n,'offset':shift,'D0_fit':fit[0],'eps_min':float(eps[mask].min()),'eps_max':float(eps[mask].max())})
                        if m==3 and n==len(z):
                            axbox[rr,cc].plot(1/eps,c,label=f'offset={shift:g}');basebox.append(fit[0])
            axbox[rr,cc].set(xscale='log',yscale='log',title=name,xlabel='1/ε [standardized units⁻¹]',ylabel='occupied boxes, m=3');axbox[rr,cc].legend(fontsize=7)
            isi=np.diff(s)
            plot_isi_return(axr[rr,cc],isi,name,label)
            for ell in [4,5]:
                for q in [1,5,25]:
                    for dec in [1,2]:
                        zz=z[::dec]
                        pp=ordinal(zz,ell,q)
                        for segment in range(4):
                            seg=zz[segment*(len(zz)//4):(segment+1)*(len(zz)//4)]
                            h=ordinal(seg,ell,q)
                            allhc.append({'signal':name,'model':label,'segment':segment,'ell':ell,'q':q,'dt_ms':dt*dec,'element_lag_ms':q*dt*dec,**{key:h[key] for key in ['H','C','patterns','occupied','ties','patterns_per_possible']}})
            # Raw timing vs resolution-aware ISI ordinal; near-constant patterns are numerical jitter.
            for resolution in [0.,.001,.01]:
                seq=isi if resolution==0 else np.round(isi/resolution)*resolution
                for ell in [3,4,5]:
                    oo=ordinal(seq,ell,1)
                    isirows.append({'signal':name,'timing_rounding_ms':resolution,'ell':ell,'ISI_count':len(isi),**{key:oo[key] for key in ['H','C','patterns','occupied','ties','patterns_per_possible']}})
            hc=ordinal(z,5,5)
            summary.append({'signal':name,'model':label,'J0_uA_cm2':10.,'A_uA_cm2':A,'T_ms':T,'rate_Hz':firing_rate(s),'ISI_CV':float(np.std(isi)/np.mean(isi)),
                            'reconstruction_lag_ms':1.,'m_GP':4,'W_ms':40.,'D2_fit':basefit.get('estimate',np.nan),'D2_local_min':basefit.get('local_min',np.nan),'D2_local_max':basefit.get('local_max',np.nan),
                            'D0_mean_m3':float(np.mean(basebox)),'H_ell5_q5':hc['H'],'C_ell5_q5':hc['C'],'ISI_count':len(isi)})
            print('scalar',name,'complete',flush=True)
    for fig,name in [(figtrace,'selected_traces'),(figrec,'reconstruction'),(figdelay,'delay_diagnostics'),(figcorr,'correlation_sums'),(figbox,'box_counts'),(figr,'ISI_returns')]:
        plt.figure(fig.number);savefig(name)
    for rows,name in [(allhc,'HC_segments'),(isirows,'ISI_ordinal'),(summary,'selected_summary'),(sensitivity,'dimension_sensitivity'),(boxrows,'box_sensitivity'),(diag,'reconstruction_choices'),(conv,'integration_convergence')]:
        pd.DataFrame(rows).to_csv(ROOT/'tables'/f'{name}.csv',index=False)
    # Local slopes are shown separately: high regression R² alone is insufficient.
    fig,axs=plt.subplots(3,2,figsize=(12,9))
    for rr,(kind,A,T) in enumerate(sets):
        for cc,label in enumerate(['HH','Izh']):
            name=f'{label}_{kind}'
            for m in [2,3,4,5,6]:
                r=np.load(ROOT/'data'/f'GP_{name}_m{m}.npz');axs[rr,cc].semilogx(r['r'],r['slope'],label=f'm={m}')
            axs[rr,cc].axvspan(.06,.35,color='gray',alpha=.15);axs[rr,cc].set(ylim=(0,4),title=name,xlabel='r [standardized Euclidean units]',ylabel='d ln C / d ln r');axs[rr,cc].legend(fontsize=7)
    savefig('correlation_slopes')
    hc=pd.DataFrame(allhc);fig,axs=plt.subplots(1,2,figsize=(11,5))
    for col,q in enumerate([5,25]):
        for name,z in hc[(hc.ell==5)&(hc.q==q)&(hc.dt_ms==.2)].groupby('signal'):
            axs[col].errorbar(z.H.mean(),z.C.mean(),xerr=z.H.std(),yerr=z.C.std(),fmt='o',capsize=3,label=name)
        axs[col].set(xlabel='normalized entropy H',ylabel='statistical complexity C',title=f'ℓ=5, q={q}, dt=.2 ms')
        draw_complexity_bounds(axs[col],5)
    savefig('HC_plane')
    return pd.DataFrame(summary)

def calibration_plot():
    df=pd.read_csv(ROOT/'tables/calibration.csv');fig,ax=plt.subplots(figsize=(7,4))
    for label,col in [('HH','HH_Hz'),('Izh','Izh_Hz')]:
        for split,marker in [('train','o'),('validation','s')]:
            z=df[df.split==split];ax.scatter(z.J_uA_cm2,z[col],label=f'{label} {split}',edgecolors=COL[label],marker=marker,facecolors='none' if split=='validation' else COL[label])
    ax.set(xlabel='J [µA/cm², common input]',ylabel='steady firing rate [Hz]');ax.legend(ncol=2);savefig('calibration')

def main():
    start=time.time()
    rng=np.random.default_rng(0)
    settings={'models':{'HH':{'Cm_uF_cm2':1,'gNa_mS_cm2':120,'gK_mS_cm2':36,'gL_mS_cm2':.3,'ENa_mV':50,'EK_mV':-77,'EL_mV':-54.387,'initial_state':initial(0).tolist()},
                        'Izh':{'a':.02,'b':.2,'c_mV':-65,'d_model_units':8,'initial_state':initial(1).tolist()}},
              'integrator':'RK4 h=.01 ms, event root bisection 24 iterations, split Izh reset step; selected checks h=.005 ms',
              'calibration_transient_ms':1000,'calibration_duration_ms':2000,'sweep_transient_cycles':100,'sweep_observed_cycles':200,'sweep_dt_ms':.2,'sweep_J0_uA_cm2':10.,'sweep_T_ms':[10.,20.,40.],'sweep_A_grid':{'T20':'.0:.1:15','T10_T40':'0:.5:15'},
              'selected_transient_ms':2000,'selected_duration_ms':8000,'selected_dt_ms':.2,'rng_initial_state':rng.bit_generator.state,
              'versions':{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'numba':numba.__version__,'matplotlib':matplotlib.__version__,'pandas':pd.__version__}}
    (ROOT/'settings.json').write_text(json.dumps(settings,indent=2))
    if '--reuse-calibration' in sys.argv and (ROOT/'calibration.json').exists():
        p=json.loads((ROOT/'calibration.json').read_text());k,b=p['kappa'],p['beta']
    else:k,b=calibration()
    from check_calibration import check as check_calibration
    from threshold_checks import check as check_thresholds
    check_calibration();check_thresholds()
    calibration_plot();thresholds(k,b);print('thresholds complete',flush=True)
    sweeps(k,b);orbit_stability(k,b);selected_analysis(k,b)
    from reliability_checks import check as reliability_checks
    reliability_checks()
    from sampling_check import check as sampling_check
    sampling_check()
    settings['runtime_seconds']=time.time()-start
    settings['runtime_scope']='core optional stages through calibration checks and sampling refinement; forced tangent dynamics and extra ablations have separate settings'
    (ROOT/'settings.json').write_text(json.dumps(settings,indent=2));print('ALL COMPUTATIONS COMPLETE',settings['runtime_seconds'],flush=True)
    from supplement import supplement
    supplement()
    if '--core-only' not in sys.argv:
        from forced_lyapunov import check as forced_lyapunov
        forced_lyapunov()
        sys.path.append(str(ROOT.parent))
        from ablation_checks import check as ablation_checks
        ablation_checks()
        from plot_reliability import plot
        plot()
if __name__=='__main__':
    if '--plot-isi-only' in sys.argv:replot_isi_returns()
    elif '--thresholds-only' in sys.argv:
        p=json.loads((ROOT/'calibration.json').read_text())
        thresholds(p['kappa'],p['beta'])
        from threshold_checks import check
        check()
    else:main()
