"""Publication exports of saved Lyapunov and frozen-calibration comparisons.

This plotting stage never integrates a model or changes analysis parameters.
"""
from pathlib import Path
import os,sys
os.environ.setdefault('MPLCONFIGDIR','/tmp/cnm_optional_mpl')
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'src'))
from publication import install_publication_exports
install_publication_exports()


def plot():
    lle=pd.read_csv(ROOT/'tables/forced_lyapunov.csv')
    styles={.02:('#6d737a','s','--'),.01:('#006c9e','o','-'),.005:('#c34a24','^','-')}
    fig,axes=plt.subplots(1,2,figsize=(7.5,4.25))
    for ax,name,title in zip(axes,['HH_weak','Izh_strong'],['HH weak; A=1','Izh strong; A=10']):
        for h,(color,marker,line) in styles.items():
            g=lle[(lle.signal==name)&(lle.h_ms==h)].sort_values('observation_ms')
            ax.plot(g.observation_ms/1000,g.lambda_max_per_s,color=color,marker=marker,
                    ls=line,lw=1.3,ms=4,label=f'h={h:g} ms')
        last=lle[(lle.signal==name)&(lle.h_ms==.005)].sort_values('observation_ms').iloc[-1]
        ax.errorbar(last.observation_ms/1000,last.lambda_max_per_s,
                    yerr=[[1000*(last.lambda_max_per_ms-last.quarter_min_per_ms)],
                          [1000*(last.quarter_max_per_ms-last.lambda_max_per_ms)]],
                    fmt='none',color=styles[.005][0],capsize=4,lw=1.2)
        ax.axhline(0,color='#353b40',lw=.8,ls=':')
        ax.set(title=title,xlabel='Observation [s]',ylabel='Largest exponent [s⁻¹]',
               xticks=[4,8,16,32],xlim=(2,35))
        ax.grid(alpha=.18);ax.legend(frameon=False,fontsize=9)
    fig.suptitle('Model tangent dynamics at fixed forcing phase; T=20 ms',fontsize=11)
    fig.savefig(ROOT/'figures/forced_lyapunov_convergence.png');plt.close(fig)
    scores=pd.read_csv(ROOT.parent/'tables/ablation_calibration_scores.csv')
    fig,axes=plt.subplots(1,2,figsize=(7.5,3.9))
    x=np.arange(2);splits=['train','validation'];width=.32
    for ax,metric,title in zip(axes,['RMSE_Hz','MAE_Hz'],['RMSE','MAE']):
        for arm,shift,color,label in [('identity',-width/2,'#7b838c','Identity input'),
                                      ('frozen_fitted',width/2,'#c34a24','Frozen calibration')]:
            g=scores[scores.arm==arm].set_index('split').loc[splits]
            bars=ax.bar(x+shift,g[metric],width,color=color,label=label)
            ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=9)
        ax.set(title=title,ylabel='Rate error [Hz]',xticks=x,xticklabels=splits,ylim=(0,scores[metric].max()*1.18))
        ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True);ax.legend(frameon=False,fontsize=9)
    fig.suptitle('Input-mapping ablation; paired h=0.02 ms; no refitting',fontsize=11)
    fig.savefig(ROOT/'figures/calibration_ablation.png');plt.close(fig)
    print('Saved Lyapunov and calibration-ablation publication figures',flush=True)

if __name__=='__main__':plot()
