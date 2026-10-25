#!/usr/bin/env python3
"""Publication figures computed strictly from saved ablation/statistics tables."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/cnm_mpl')
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from publication import install_publication_exports
install_publication_exports()
COLORS={'spiking':'#167a7a','bursting':'#df8f28','chaotic':'#9b3d68'}

def main():
    d=pd.read_csv(ROOT/'tables/ablation_sampling_comparison.csv')
    fig,axes=plt.subplots(2,2,figsize=(10,8),sharex=True)
    for j,(arm,title) in enumerate([('fixed_physical_lag','Matched physical lag = 2'),('fixed_q10','Fixed q = 10: physical lag changes')]):
        for name,color in COLORS.items():
            g=d[(d.signal==name)&(d.ell==5)].sort_values('dt')
            for row,metric in enumerate(['H','C']):
                axes[row,j].plot(g.dt,g[f'{arm}_{metric}'],'o-',color=color,label=name)
                axes[row,j].set(ylabel=metric+' [dimensionless]')
        axes[0,j].set_title(title)
        axes[1,j].set_xlabel('Observation interval [HR model time]')
        axes[0,j].legend(frameon=False)
    fig.savefig(ROOT/'figures/C_sampling_ablation.png');plt.close(fig)
    d=pd.read_csv(ROOT/'tables/noise_statistics_distances.csv')
    c=pd.read_csv(ROOT/'tables/noise_statistics_convergence.csv')
    fig,axes=plt.subplots(2,1,figsize=(8,8))
    for (ell,q),g in d.groupby(['ell','q']):
        line,=axes[0].plot(g.eta,g.centroid_distance,'o-',label=f'ell={ell}, q={q}')
        axes[0].fill_between(g.eta,g.centroid_ci_low,g.centroid_ci_high,color=line.get_color(),alpha=.25)
    axes[0].set(xlabel='Relative measurement noise eta',ylabel='Distance between mean (H,C)',title='30 independent replicate tuples; pointwise 95% bootstrap CI')
    axes[0].legend(frameon=False)
    c=c[(c.estimand=='source_mean')&(c.eta>0)]
    for metric in ['H','C']:
        g=c[c.metric==metric].groupby('n').difference_from_n30.agg(lambda v:abs(v).max())
        axes[1].plot(g.index,g.values,'o-',label=metric)
    axes[1].set(xlabel='Number of independent noise realizations',ylabel='Maximum absolute mean shift vs n=30',title='Conditional Monte Carlo convergence across sources and settings')
    axes[1].legend(frameon=False)
    fig.savefig(ROOT/'figures/C_noise_statistics.png');plt.close(fig)

if __name__=='__main__':main()
