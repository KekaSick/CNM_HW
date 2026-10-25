"""Attainable Shannon/Jensen-Shannon complexity bounds for K outcomes.

Extremal families: Martin, Plastino & Rosso (2006), Physica A 369, 439-462,
https://doi.org/10.1016/j.physa.2005.11.053. The normalization is that of lab.md.
"""
from functools import lru_cache
import math
import numpy as np
from scipy.special import xlogy


def _hc(p):
    K=p.shape[-1];s=-xlogy(p,p).sum(axis=-1)
    middle=(p+1/K)/2
    j=-xlogy(middle,middle).sum(axis=-1)-s/2-np.log(K)/2
    jmax=np.log(2)+.5*np.log(K)-(K+1)*np.log(K+1)/(2*K)
    H=s/np.log(K)
    return np.column_stack((H,H*np.maximum(j,0)/jmax))


def _ordered(points):
    points=points[np.argsort(points[:,0],kind='stable')]
    _,unique=np.unique(points[:,0],return_index=True)
    points=points[unique]
    points[0]=[0.,0.];points[-1]=[1.,0.]
    points.flags.writeable=False
    return points


@lru_cache(maxsize=8)
def complexity_bounds(K):
    """Return (Cmin, Cmax) arrays with columns (H,C), sorted by entropy.

    Lower: one dominant probability p in [1/K,1], all other K-1 equal.
    Upper: support n=2..K, one small p in [0,1/n], n-1 equal large
    probabilities (1-p)/(n-1), and K-n zeros. Zero entries remain in the
    normalization/mixing with the full K-outcome uniform distribution.
    """
    if not isinstance(K,(int,np.integer)) or K<2:raise ValueError('K must be an integer >=2')
    t=(1-np.cos(np.linspace(0,np.pi,4097)))/2
    p=1/K+(1-1/K)*t
    lower=np.full((len(p),K),0.);lower[:,0]=p;lower[:,1:]=(1-p[:,None])/(K-1)
    upper=[]
    for n in range(2,K+1):
        samples=4097 if n==2 else 257
        q=(1-np.cos(np.linspace(0,np.pi,samples)))/(2*n)
        dist=np.zeros((samples,K));dist[:,0]=q;dist[:,1:n]=(1-q[:,None])/(n-1)
        upper.append(_hc(dist))
    return _ordered(_hc(lower)),_ordered(np.vstack(upper))


def draw_complexity_bounds(ax,ell,full_domain=True,legend=True):
    """Draw both bounds for K=ell!, without altering empirical observations."""
    lower,upper=complexity_bounds(math.factorial(int(ell)))
    ax.plot(lower[:,0],lower[:,1],color='#424a54',ls='--',lw=1.25,
            label=r'$C_{\min}(H)$',zorder=1)
    ax.plot(upper[:,0],upper[:,1],color='#20252b',lw=1.35,
            label=r'$C_{\max}(H)$',zorder=1)
    grid=np.linspace(0,1,1501)
    ax.fill_between(grid,np.interp(grid,lower[:,0],lower[:,1]),
                    np.interp(grid,upper[:,0],upper[:,1]),color='#a8b1bb',alpha=.10,zorder=0)
    if full_domain:ax.set(xlim=(-.015,1.025),ylim=(-.012,float(upper[:,1].max())+.045))
    if legend:ax.legend(fontsize=7,loc='upper left',framealpha=.92)
    return lower,upper
