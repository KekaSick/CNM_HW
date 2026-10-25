"""H–C controls, pattern-count diagnostics, and measured synchronization phase."""
from pathlib import Path
import sys
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/cnm_optional_mpl')
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scalar import ordinal
ROOT=Path(__file__).resolve().parent
sys.path.append(str(ROOT.parent/'src'))
from hc_bounds import draw_complexity_bounds
from publication import install_publication_exports
install_publication_exports()


def replot_hc():
    """Refresh the two H-C diagrams from saved statistics, without new simulation."""
    response=pd.read_csv(ROOT/'tables/HC_segments.csv')
    controls=pd.read_csv(ROOT/'tables/HC_controls.csv')
    for include_controls,name in [(True,'HC_with_controls'),(False,'HC_plane')]:
        combined=pd.concat([controls,response[response.dt_ms==.2]],ignore_index=True) if include_controls else response[response.dt_ms==.2]
        fig,axs=plt.subplots(2,1,figsize=(7.5,10.5))
        for col,q in enumerate([5,25]):
            for signal,g in combined[(combined.ell==5)&(combined.q==q)].groupby('signal'):
                axs[col].errorbar(g.H.mean(),g.C.mean(),xerr=g.H.std(),yerr=g.C.std(),fmt='o',ms=5,capsize=2,label=signal,zorder=3)
            axs[col].set(xlabel='normalized permutation entropy H',ylabel='statistical complexity C',
                         title=f'ell=5, K=120, element lag={q*.2:g} ms; 4 segments')
            draw_complexity_bounds(axs[col],5,legend=False)
            axs[col].legend(fontsize=8.5,loc='upper left')
        fig.tight_layout();fig.savefig(ROOT/'figures'/f'{name}.png');plt.close(fig)

def supplement():
    sweep=pd.read_csv(ROOT/'tables/periodic_sweep.csv');rows=[]
    for row in sweep.itertuples():
        path=ROOT/'data'/f'sweep_{row.model}_T{row.T_ms:g}_A{row.A_uA_cm2:.1f}.npz'
        z=np.load(path);s=z['spikes_ms'];ph=2*np.pi*s/row.T_ms
        rows.append({'model':row.model,'T_ms':row.T_ms,'A_uA_cm2':row.A_uA_cm2,
                     'phase_resultant_R1':float(abs(np.mean(np.exp(1j*ph)))) if len(ph) else np.nan,
                     'phase_resultant_R2':float(abs(np.mean(np.exp(2j*ph)))) if len(ph) else np.nan,
                     'mean_phase_rad':float(np.angle(np.mean(np.exp(1j*ph)))) if len(ph) else np.nan})
    pd.DataFrame(rows).to_csv(ROOT/'tables/synchronization_phase.csv',index=False)
    data=np.load(ROOT/'data/selected_HH_DC.npz');x=data['states'][:,0];sp=data['spikes_ms'];N=len(x);dt=.2
    T0=float(np.mean(np.diff(sp)));t=np.arange(N)*dt
    rng=np.random.default_rng(20261024)
    (ROOT/'data/control_rng_initial_state.json').write_text(json.dumps(rng.bit_generator.state,indent=2))
    controls={'Gaussian':rng.standard_normal(N),'shuffled_HH_DC':rng.permutation(x),'quasiperiodic':np.sin(2*np.pi*t/T0)+.5*np.sin(2*np.pi*np.sqrt(2)*t/T0)}
    np.savez_compressed(ROOT/'data/HC_controls.npz',**controls,dt_ms=dt,T0_ms=T0)
    rows=[]
    for name,z in controls.items():
        for ell in [4,5]:
            for q in [1,5,25]:
                for seg in range(4):
                    signal=z[seg*(N//4):(seg+1)*(N//4)];oo=ordinal(signal,ell,q)
                    rows.append({'signal':name,'segment':seg,'ell':ell,'q':q,'dt_ms':dt,**{key:oo[key] for key in ['H','C','patterns','occupied','ties','patterns_per_possible']}})
    df=pd.DataFrame(rows);df.to_csv(ROOT/'tables/HC_controls.csv',index=False)
    replot_hc()
    fig,axs=plt.subplots(3,2,figsize=(10,8))
    for rr,kind in enumerate(['DC','weak','strong']):
        for cc,label in enumerate(['HH','Izh']):
            name=f'{label}_{kind}';data=np.load(ROOT/'data'/f'diagnostics_{name}.npz');fn=data['FNN']
            axs[rr,cc].plot(fn[:,0],fn[:,1],'o-')
            axs[rr,cc].set(title=name,xlabel='embedding dimension m',ylabel='false-neighbor fraction',ylim=(0,max(.05,fn[:,1].max()*1.05)))
            axs[rr,cc].grid(alpha=.2)
    fig.tight_layout();fig.savefig(ROOT/'figures/FNN_diagnostics.png');plt.close(fig)
    isi=pd.read_csv(ROOT/'tables/ISI_ordinal.csv');fig,axs=plt.subplots(1,2,figsize=(12,4))
    for col,ell in enumerate([3,5]):
        for name,g in isi[isi.ell==ell].groupby('signal'):
            axs[col].plot(g.timing_rounding_ms,g.H,'o-',label=name)
        axs[col].set(xlabel='timing quantization [ms]; 0 = raw',ylabel='ISI permutation entropy H',title=f'ℓ={ell}, q=1');axs[col].legend(fontsize=7)
    fig.tight_layout();fig.savefig(ROOT/'figures/ISI_ordinal_resolution.png');plt.close(fig)
if __name__=='__main__':
    if '--plot-hc-only' in sys.argv:replot_hc()
    else:supplement()
