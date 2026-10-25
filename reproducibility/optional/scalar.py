"""Standalone scalar estimators shared in algorithm with ../src/cnm.py.
No hidden model variables enter the analysis.
"""
import math
import numpy as np
from numba import njit
from scipy.spatial import cKDTree
from scipy.signal import fftconvolve

def standardize(x):
    x = np.asarray(x, dtype=float)
    if x.std() == 0:
        raise ValueError('Constant recording cannot be normalized for reconstruction')
    return (x-x.mean())/x.std()


def embed(x, m, lag):
    n = len(x)-(m-1)*lag
    if n < 1 or lag < 1:
        raise ValueError('Insufficient recording for embedding')
    return np.column_stack([x[j*lag:j*lag+n] for j in range(m)])


def acf(x, maxlag):
    z = x-np.mean(x)
    values = fftconvolve(z, z[::-1], mode='full')[len(z)-1:len(z)+maxlag]
    return values/values[0]


def mutual_information(x, maxlag, bins=32):
    codes = np.minimum(np.floor((x-x.min())/(np.ptp(x)+1e-15)*bins).astype(int), bins-1)
    out = np.zeros(maxlag+1)
    for k in range(maxlag+1):
        a, b = (codes, codes) if k == 0 else (codes[:-k], codes[k:])
        p = np.bincount(a*bins+b, minlength=bins*bins).reshape(bins, bins)/len(a)
        product = p.sum(axis=1)[:, None]*p.sum(axis=0)[None, :]
        mask = p > 0
        out[k] = np.sum(p[mask]*np.log(p[mask]/product[mask]))
    return out


def false_nearest(x, lag, maxdim=8, window=100, maxpoints=12000):
    """Nearest admissible neighbor: common m+1 index set, original-index Theiler.

    Deterministic evenly spread reference cloud; adaptively expand tree queries
    until an admissible neighbor has been found for every reference point.
    """
    rows = []
    sigma = np.std(x)
    for m in range(1, maxdim+1):
        full = embed(x, m+1, lag)
        ids = np.unique(np.linspace(0, len(full)-1, min(maxpoints, len(full))).astype(int))
        X, extra = full[ids, :m], full[ids, m]
        tree = cKDTree(X)
        kval = min(32, len(X))
        while True:
            dist, nn = tree.query(X, k=kval, workers=1)
            eligible = np.abs(ids[nn]-ids[:, None]) > window
            found = eligible.any(axis=1)
            if found.all() or kval == len(X):
                break
            kval = min(2*kval, len(X))
        pos = eligible.argmax(axis=1)
        jj = nn[np.arange(len(X)), pos]
        r = dist[np.arange(len(X)), pos]
        delta = np.abs(extra-extra[jj])
        good = found & ((r > 0) | (delta > 0))
        false = ((delta > 10*r) | (np.sqrt(r*r+delta*delta) > 2*sigma)) & good
        rows.append([m, false.sum()/good.sum(), good.sum(), len(X)-good.sum()])
    return np.array(rows)


@njit(cache=True)
def excluded_histogram(X, ids, window, radii):
    hist = np.zeros(len(radii)+1, dtype=np.int64)
    excluded = 0
    for i in range(len(X)):
        for j in range(i+1, len(X)):
            if ids[j]-ids[i] > window:
                break
            excluded += 1
            d2 = 0.
            for k in range(X.shape[1]):
                d2 += (X[i, k]-X[j, k])**2
            d = np.sqrt(d2)
            # count_neighbors uses <= r; use the same boundary convention.
            pos = np.searchsorted(radii, d)
            hist[pos] += 1
    return np.cumsum(hist[:-1]), excluded


def correlation_sum(x, m, lag, window, radii=None, maxpoints=10000):
    full = embed(x, m, lag)
    ids = np.unique(np.linspace(0, len(full)-1, min(maxpoints, len(full))).astype(int))
    X = full[ids]
    if radii is None:
        radii = np.geomspace(.003*np.sqrt(m), 2*np.sqrt(m), 60)
    tree = cKDTree(X)
    ordered = tree.count_neighbors(tree, radii)
    removed, excluded = excluded_histogram(X, ids, int(window), radii)
    counts = (ordered-len(X))//2-removed
    pairs = len(X)*(len(X)-1)//2-excluded
    C = counts/pairs
    slopes = np.full(len(C), np.nan)
    valid = C > 0
    slopes[valid] = np.gradient(np.log(C[valid]), np.log(radii[valid]))
    return {'r':radii, 'C':C, 'slope':slopes, 'counts':counts,
            'pairs':pairs, 'points':len(X), 'ids':ids}


def fit_scaling(result, lower, upper, min_count=200):
    r, C = result['r'], result['C']
    mask = (r >= lower) & (r <= upper) & (result['counts'] >= min_count) & (C > 0) & (C < .15)
    if mask.sum() < 4:
        return {'estimate':np.nan, 'r2':np.nan, 'nscales':int(mask.sum())}
    a, b = np.polyfit(np.log(r[mask]), np.log(C[mask]), 1)
    err = np.log(C[mask])-(a*np.log(r[mask])+b)
    total = np.sum((np.log(C[mask])-np.log(C[mask]).mean())**2)
    return {'estimate':float(a), 'intercept':float(b), 'r2':float(1-np.sum(err*err)/total),
            'nscales':int(mask.sum()), 'rmin':float(r[mask][0]), 'rmax':float(r[mask][-1]),
            'min_pairs':int(result['counts'][mask].min()),
            'local_min':float(np.min(result['slope'][mask])),
            'local_max':float(np.max(result['slope'][mask]))}


def box_count(x, m, lag, eps, offsets=(0., .25, .5, .75)):
    X = embed(x, m, lag)
    origin = X.min(axis=0)
    out = []
    for shift in offsets:
        counts = []
        for e in eps:
            cells = np.floor((X-origin)/e+shift).astype(np.int64)
            counts.append(len(np.unique(cells, axis=0)))
        out.append(counts)
    return np.asarray(out)


def entropy_complexity(p):
    p = np.asarray(p, dtype=float)
    if p.ndim != 1 or len(p) < 2 or np.any(p < 0) or not np.isclose(p.sum(), 1):
        raise ValueError('Expected a normalized nonnegative probability vector')
    K = len(p)
    def S(a):
        a = a[a > 0]
        return -np.sum(a*np.log(a))
    H = S(p)/np.log(K)
    J = S((p+1/K)/2)-S(p)/2-np.log(K)/2
    # The maximum JS divergence from uniform is attained by a point mass.
    jmax = np.log(2)+.5*np.log(K)-((K+1)/(2*K))*np.log(K+1)
    return float(H), float(max(J, 0)*H/jmax)


def ordinal(x, length=5, delay=10):
    """Stable sorting resolves exact ties by earlier time index, no added jitter.

    Permutations are encoded by their Lehmer ranks in [0, length!-1].
    Every one of the length! bins, including unobserved patterns, is retained.
    """
    X = embed(np.asarray(x), length, delay)
    order = np.argsort(X, axis=1, kind='stable')
    code = np.zeros(len(X), dtype=np.int64)
    for j in range(length-1):
        code += np.sum(order[:, j+1:] < order[:, j, None], axis=1)*math.factorial(length-1-j)
    counts = np.bincount(code, minlength=math.factorial(length))
    p = counts/counts.sum()
    H, C = entropy_complexity(p)
    tie_fraction = np.mean(np.any(np.diff(np.sort(X, axis=1), axis=1) == 0, axis=1))
    return {'H':H, 'C':C, 'counts':counts, 'patterns':len(X),
            'occupied':int(np.count_nonzero(counts)), 'ties':float(tie_fraction),
            'patterns_per_possible':len(X)/len(counts)}
