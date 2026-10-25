"""Compute reference results with ordpy and JiTCODE, independently of the custom solvers.

The frozen reconstruction experiment is left intact. This module checks library
agreement and records model-based exponents using automatic differentiation.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import math
import sys
import numpy as np
import pandas as pd
import ordpy

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'src'))


def ordinal_library(x, ell=5, q=10):
    """ordpy with exact lexicographic (value, time-index) tie handling.

    Unique global ranks preserve every within-window order, resolving equal
    values by time without adding random jitter or changing untied orders.
    """
    x = np.asarray(x)
    if x.ndim != 1 or not np.isfinite(x).all() or len(x) <= (ell-1)*q:
        raise ValueError('Expected a finite scalar recording longer than one pattern')
    ranks = np.empty(len(x), dtype=np.int64)
    ranks[np.argsort(x, kind='stable')] = np.arange(len(x))
    permutations, probabilities = ordpy.ordinal_distribution(
        ranks, dx=ell, taux=q, return_missing=True)
    patterns = len(x)-(ell-1)*q
    counts = np.zeros(math.factorial(ell), dtype=np.int64)
    for permutation, probability in zip(permutations, probabilities):
        code = sum(int(np.sum(permutation[j+1:] < permutation[j]))*math.factorial(ell-1-j)
                   for j in range(ell-1))
        counts[code] = int(round(probability*patterns))
    H, C = ordpy.complexity_entropy(probabilities, dx=ell, probs=True)
    assert counts.sum() == patterns and len(counts) == math.factorial(ell)
    return {'H':float(H), 'C':float(C), 'counts':counts, 'patterns':patterns}


def model_exponents(expressions, initial, transient, duration, qr_dt=1.,
                    rtol=1e-9, compile_c=True):
    """JiTCODE automatically differentiates and integrates the tangent equations.

    A deterministic identity tangent basis avoids JiTCODE's random initializer.
    All observation intervals are equal; the returned local exponents already
    divide logarithmic growth by each interval's physical duration.
    """
    from jitcode import jitcode, jitcode_lyap
    initial = np.asarray(initial, dtype=float)
    n = len(initial)
    for length in [transient, duration]:
        if not np.isclose(length/qr_dt, round(length/qr_dt)):
            raise ValueError('Transient and duration must contain complete QR intervals')
    solver = jitcode_lyap(expressions, n_lyap=n, verbose=False)
    if compile_c:
        solver.generate_f_C()
        solver.compile_C(extra_compile_args=['-std=c11', '-O3'])
    else:
        solver.generate_lambdas()
    solver.set_integrator('dop853', rtol=rtol, atol=rtol*.01, max_step=.1)
    jitcode.set_initial_value(solver, np.r_[initial, np.eye(n).ravel()], time=0.)
    pre = int(round(transient/qr_dt)); count = int(round(duration/qr_dt))
    transient_final = initial.copy()
    for i in range(pre):
        transient_final = solver.integrate((i+1)*qr_dt)[0]
    states=[]; local=[]; vectors=[]
    for i in range(count):
        state, exponents, basis = solver.integrate(transient+(i+1)*qr_dt)
        states.append(state.copy()); local.append(exponents.copy()); vectors.append(np.array(basis))
    return {'t':transient+np.arange(1,count+1)*qr_dt,
            'state':np.asarray(states), 'local_exponents':np.asarray(local),
            'tangent_basis':np.asarray(vectors), 'initial':initial,
            'initial_tangent_basis':np.eye(n), 'transient_final':transient_final,
            'transient':transient, 'duration':duration, 'qr_dt':qr_dt, 'rtol':rtol}


def ordinal_checks():
    from cnm import ordinal
    cases=[]
    for name in ['spiking','bursting','chaotic']:
        cases.append((name, np.load(ROOT/'data'/f'{name}.npz')['state'][:-1,0]))
    controls=np.load(ROOT/'data/controls.npz')
    cases += [(name, controls[name]) for name in ['white_noise','shuffled','quasiperiodic']]
    rows=[]
    for name,x in cases:
        for ell,q in [(4,10),(5,10),(5,50)]:
            reference=ordinal_library(x,ell,q); old=ordinal(x,ell,q)
            rows.append(dict(signal=name,ell=ell,q=q,N=len(x),H=reference['H'],C=reference['C'],
                             H_difference=reference['H']-old['H'],C_difference=reference['C']-old['C'],
                             identical_counts=np.array_equal(reference['counts'],old['counts'])))
    for name in ['HH_DC','Izh_DC','HH_weak','Izh_weak','HH_strong','Izh_strong']:
        data=np.load(ROOT/'optional/data'/f'selected_{name}.npz')
        x=data['states'][:,0]
        for ell in [4,5]:
            for q in [1,5,25]:
                reference=ordinal_library(x,ell,q)
                sys.path.insert(0,str(ROOT/'optional'))
                from scalar import ordinal as optional_ordinal
                old=optional_ordinal(x,ell,q)
                rows.append(dict(signal=name,ell=ell,q=q,N=len(x),H=reference['H'],C=reference['C'],
                                 H_difference=reference['H']-old['H'],C_difference=reference['C']-old['C'],
                                 identical_counts=np.array_equal(reference['counts'],old['counts'])))
    # Exact ties in rounded ISIs are included, rather than checking continuous data only.
    for name in ['HH_DC','Izh_DC','HH_strong','Izh_strong']:
        spikes=np.load(ROOT/'optional/data'/f'selected_{name}.npz')['spikes_ms']
        x=np.round(np.diff(spikes)/.001)*.001
        for ell in [3,4,5]:
            reference=ordinal_library(x,ell,1);old=optional_ordinal(x,ell,1)
            rows.append(dict(signal=name+'_rounded_ISI',ell=ell,q=1,N=len(x),H=reference['H'],C=reference['C'],
                             H_difference=reference['H']-old['H'],C_difference=reference['C']-old['C'],
                             identical_counts=np.array_equal(reference['counts'],old['counts'])))
    frame=pd.DataFrame(rows)
    assert frame.identical_counts.all()
    assert frame[['H_difference','C_difference']].abs().to_numpy().max()<2e-13
    frame.to_csv(ROOT/'tables/library_ordinal_validation.csv',index=False)
    print('ordpy:',len(frame),'actual scalar recordings/settings checked.',flush=True)


def lyapunov_checks():
    from jitcode import y
    rows=[]
    directory=ROOT/'data/library_validation';directory.mkdir(exist_ok=True)
    for name,current in [('spiking',4.),('bursting',2.5),('chaotic',3.25)]:
        expressions=[y(1)-y(0)**3+3*y(0)**2-y(2)+current,
                     1-5*y(0)**2-y(1), .006*(4*(y(0)+1.6)-y(2))]
        for tolerance in [1e-8,1e-10]:
            result=model_exponents(expressions,[-1.6,-12,0],10000.,30000.,rtol=tolerance)
            np.savez_compressed(directory/f'{name}_rtol{tolerance:g}.npz',**result,I=current)
            exponents=result['local_exponents']; blocks=exponents.reshape(6,-1,3).mean(axis=1)
            rows.append(dict(regime=name,I=current,rtol=tolerance,transient=10000.,duration=30000.,
                             lambda1=exponents[:,0].mean(),lambda2=exponents[:,1].mean(),
                             lambda3=exponents[:,2].mean(),block_min=blocks[:,0].min(),block_max=blocks[:,0].max()))
            print('JiTCODE:',rows[-1],flush=True)
    pd.DataFrame(rows).to_csv(ROOT/'tables/library_lyapunov_validation.csv',index=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ordinal-only',action='store_true')
    parser.add_argument('--lyapunov-only',action='store_true')
    args=parser.parse_args()
    if not args.lyapunov_only:ordinal_checks()
    if not args.ordinal_only:lyapunov_checks()
    sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
             for p in [Path(__file__),ROOT/'src/cnm.py',ROOT/'optional/scalar.py']}
    (ROOT/'data/library_validation_settings.json').write_text(json.dumps({
        'versions':{k:importlib.metadata.version(k) for k in ['ordpy','jitcode','jitcxde_common','symengine','scipy']},
        'source_sha256':sources,'tie_rule':'exact global lexicographic ranks (value, time index)',
        'lyapunov_integrator':'JiTCODE automatic tangent equations + scipy dop853; QR every 1 model time unit',
        'initial_tangent_basis':'identity; no randomized initialization',
        'time_blocks':'six 5000-unit blocks; descriptive ranges, not IID confidence intervals'
    },indent=2)+'\n')


if __name__=='__main__':main()
