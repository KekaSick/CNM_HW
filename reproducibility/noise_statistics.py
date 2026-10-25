#!/usr/bin/env python3
"""Conditional paired Monte Carlo study: python submission/noise_statistics.py.

Replicate tuples, rather than individual time samples, are the uncertainty units.
All uncertainty is conditional on these two fixed noise-free HR trajectories.
One Gaussian vector is intentionally shared across sources, amplitudes and
ordinal settings within each replicate; different replicates use new draws.
Only new noise_statistics_* files are written; the original five draws remain.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('NUMBA_CACHE_DIR','/tmp/cnm_numba')
sys.path.insert(0,str(ROOT/'src'))
from cnm import ordinal, entropy_complexity

DATA, TABLES = ROOT/'data', ROOT/'tables'
SIGNALS = ['spiking','chaotic']
ETAS = np.array([0.,.01,.05,.1,.2])
ORDINAL_SETTINGS = [(4,10),(5,10),(5,50)]
REPLICATES, RESAMPLES = 30, 10000
NOISE_SEED, BOOTSTRAP_SEED = 2026102501, 2026102502


def observed_signal(clean, xi, eta, sigma=None):
    clean, xi = np.asarray(clean,dtype=float), np.asarray(xi,dtype=float)
    if clean.ndim!=1 or clean.shape!=xi.shape or not np.isfinite(eta) or eta<0:
        raise ValueError('Expected equally sized scalar signals and nonnegative eta')
    if eta==0:
        return clean.copy()
    scale = np.std(clean) if sigma is None else float(sigma)
    return clean+(eta*scale)*xi


def moments(values):
    values = np.asarray(values,dtype=float)
    if values.ndim<1 or len(values)<2 or not np.isfinite(values).all():
        raise ValueError('Expected at least two finite independent replicate values')
    constant = np.ptp(values,axis=0)==0
    mean = np.where(constant,values[0],np.mean(values,axis=0))
    sd = np.where(constant,0.,np.std(values,axis=0,ddof=1))
    return mean, sd, sd/np.sqrt(len(values))


def bootstrap_means(values, indices):
    """Resample complete replicate rows with shared indices; bounded scratch RAM."""
    values, indices = np.asarray(values,dtype=float), np.asarray(indices)
    if (values.ndim<1 or len(values)<2 or indices.ndim!=2
        or indices.shape[1]!=len(values) or not np.issubdtype(indices.dtype,np.integer)
        or np.any(indices<0) or np.any(indices>=len(values))):
        raise ValueError('Bootstrap indices must resample complete replicate tuples')
    out = np.empty((len(indices),)+values.shape[1:])
    for start in range(0,len(indices),128):
        out[start:start+128] = np.mean(values[indices[start:start+128]],axis=1)
    # Exact constants have no Monte Carlo uncertainty; avoid summation artifacts.
    return np.where(np.ptp(values,axis=0)==0,values[0],out)


def summarize_mc(values, indices):
    values = np.asarray(values,dtype=float)
    if values.ndim!=1:
        raise ValueError('Summarize one statistic per independent replicate')
    mean, sd, mcse = moments(values)
    boot = bootstrap_means(values,indices)
    low, high = np.quantile(boot,[.025,.975])
    return {'n':len(values),'mean':float(mean),'sd':float(sd),'mcse':float(mcse),
            'ci_low':float(low),'ci_high':float(high)}


def paired_contrast(a, b, indices):
    """Estimate E[b-a], preserving the covariance created by common noise."""
    a, b = np.asarray(a,dtype=float), np.asarray(b,dtype=float)
    if a.ndim!=1 or a.shape!=b.shape:
        raise ValueError('A contrast needs matched one-dimensional replicate arrays')
    result = summarize_mc(b-a,indices)
    mean_a, sd_a, _ = moments(a)
    mean_b, sd_b, _ = moments(b)
    covariance = (float(np.dot(a-mean_a,b-mean_b)/(len(a)-1))
                  if sd_a>0 and sd_b>0 else 0.)
    variance_identity = float(sd_a**2+sd_b**2-2*covariance)
    unpaired = float(np.sqrt((sd_a**2+sd_b**2)/len(a)))
    result.update({'sample_covariance':covariance,
        'sample_correlation':covariance/float(sd_a*sd_b) if sd_a>0 and sd_b>0 else None,
        'unpaired_mcse':unpaired,
        'paired_to_unpaired_mcse':result['mcse']/unpaired if unpaired>0 else None,
        'variance_identity_error':float(result['sd']**2-variance_identity)})
    return result


def centroid_distance_mcse(differences):
    """Delta-method MCSE of ||E[chaotic(H,C)-spiking(H,C)]||."""
    differences = np.asarray(differences,dtype=float)
    mean, _, _ = moments(differences)
    distance = float(np.linalg.norm(mean))
    if np.ptp(differences,axis=0).max()==0:
        return distance, 0.
    if distance==0:
        return distance, np.nan
    gradient = mean/distance
    projected = differences@gradient
    return distance, float(moments(projected)[2])


def write_csv(name, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (TABLES/name).open('w',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    def convert(item):
        if isinstance(item,np.generic):return item.item()
        if isinstance(item,np.ndarray):return item.tolist()
        raise TypeError(type(item).__name__)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,default=convert)+'\n')


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate():
    clean, source_metadata = {}, {}
    for name in SIGNALS:
        path = DATA/f'{name}.npz'
        with np.load(path) as record:
            clean[name] = record['state'][:-1,0].copy()
            if len(clean[name])!=150000 or float(record['dt'])!=.2:
                raise AssertionError('Expected the fixed 150000-point, dt=.2 clean recordings')
        source_metadata[name] = {'path':str(path.relative_to(ROOT)),
            'file_sha256':file_hash(path), 'scalar_sha256':hashlib.sha256(clean[name].tobytes()).hexdigest(),
            'sigma_ddof0':float(np.std(clean[name])), 'samples':len(clean[name]),'dt':.2,
            'right_endpoint_excluded':True}
    originals = [TABLES/'measurement_noise.csv',DATA/'noise_rng_states.json']
    originals += sorted((DATA/'noise').glob('*.npz'))
    original_hashes = {str(p.relative_to(ROOT)):file_hash(p) for p in originals}
    rng = np.random.Generator(np.random.PCG64(NOISE_SEED))
    rng_initial = copy.deepcopy(rng.bit_generator.state)
    states_before = []
    xi_all = np.empty((REPLICATES,150000))
    values = np.empty((REPLICATES,2,5,3,2))
    ties = np.empty((REPLICATES,2,5,3))
    counts = {f'counts_l{ell}_q{q}':np.empty((REPLICATES,2,5,
        24 if ell==4 else 120),dtype=np.int64) for ell,q in ORDINAL_SETTINGS}
    replicate_rows = []
    for replicate in range(REPLICATES):
        states_before.append(copy.deepcopy(rng.bit_generator.state))
        xi = rng.standard_normal(150000)
        xi_all[replicate] = xi
        for si,name in enumerate(SIGNALS):
            for ei,eta in enumerate(ETAS):
                observed = observed_signal(clean[name],xi,eta,source_metadata[name]['sigma_ddof0'])
                for ki,(ell,q) in enumerate(ORDINAL_SETTINGS):
                    key = f'counts_l{ell}_q{q}'
                    if ei==0 and replicate>0:
                        current_counts = counts[key][0,si,ei]
                        h,c = values[0,si,ei,ki]
                        tie = ties[0,si,ei,ki]
                    else:
                        result = ordinal(observed,ell,q)
                        current_counts = result['counts']
                        h,c,tie = result['H'],result['C'],result['ties']
                    counts[key][replicate,si,ei] = current_counts
                    values[replicate,si,ei,ki] = [h,c]
                    ties[replicate,si,ei,ki] = tie
                    replicate_rows.append({'replicate':replicate,'signal':name,'eta':eta,
                        'ell':ell,'q':q,'dt':.2,'H':h,'C':c,'patterns':int(current_counts.sum()),
                        'occupied':int(np.count_nonzero(current_counts)),'ties':tie})
        if (replicate+1)%5==0:
            print(f'Noise Monte Carlo: {replicate+1}/{REPLICATES} complete',flush=True)
    noise_final = copy.deepcopy(rng.bit_generator.state)
    bootstrap_rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    bootstrap_initial = copy.deepcopy(bootstrap_rng.bit_generator.state)
    indices = bootstrap_rng.integers(0,REPLICATES,size=(RESAMPLES,REPLICATES),dtype=np.int64)
    bootstrap_final = copy.deepcopy(bootstrap_rng.bit_generator.state)
    np.savez_compressed(DATA/'noise_statistics_xi.npz',xi=xi_all,dt=.2,replicate=np.arange(REPLICATES))
    np.savez_compressed(DATA/'noise_statistics_ordinal.npz',values=values,ties=ties,
        signals=np.array(SIGNALS),etas=ETAS,ordinal_settings=np.array(ORDINAL_SETTINGS),
        metric_names=np.array(['H','C']),**counts)
    np.savez_compressed(DATA/'noise_statistics_bootstrap.npz',indices=indices)
    correlation = np.corrcoef(xi_all)
    np.savez_compressed(DATA/'noise_statistics_noise_diagnostics.npz',
        replicate_mean=xi_all.mean(axis=1),replicate_sd=xi_all.std(axis=1),
        replicate_correlation=correlation)
    settings = {'replicates':REPLICATES,'samples_per_replicate':150000,
        'source_signals':SIGNALS,'etas':ETAS,'ordinal_settings':ORDINAL_SETTINGS,'dt':.2,
        'noise_model':'x_obs = x + eta * sigma_ddof0(x) * xi',
        'noise_distribution':'Gaussian N(0,1); PCG64; one new vector per replicate',
        'pairing':'within replicate the same xi is reused across sources, amplitudes and ordinal settings',
        'replicate_independence':'new disjoint Gaussian draws; independent replicate tuples are resampling units',
        'noise_seed':NOISE_SEED,'noise_initial_rng_state':rng_initial,
        'noise_rng_states_before_each_replicate':states_before,'noise_final_rng_state':noise_final,
        'bootstrap_seed':BOOTSTRAP_SEED,'bootstrap_resamples':RESAMPLES,
        'bootstrap_initial_rng_state':bootstrap_initial,'bootstrap_final_rng_state':bootstrap_final,
        'bootstrap_indices_dtype':'int64','bootstrap_method':'pointwise approximate 95% percentile bootstrap of complete replicate tuples',
        'mcse_method':'sample SD across independent replicate statistics / sqrt(n)',
        'centroid_distance_mcse':'multivariate delta method; replicate covariance retained',
        'uncertainty_scope':'conditional on these two fixed clean trajectories, their phase, sampling and preprocessing; no ODE, trajectory, biological or time-series sampling uncertainty',
        'distance_estimands':{'centroid':'norm of difference of conditional mean (H,C) points',
            'paired_realization':'expected H-C distance under the explicitly coupled common-noise design; not the distance distribution of independently noisy records'},
        'eta_zero':'one deterministic clean trajectory per source; replicated bookkeeping does not create extra independent clean trajectories; MCSE and CI width exactly zero',
        'multiplicity':'pointwise intervals, no simultaneous coverage or familywise claim; no p-values or hypothesis tests across time samples',
        'bootstrap_limitations':'30 independent replicates; percentile intervals are approximate; 10000 bootstrap resamples do not increase the experimental replicate count',
        'source_metadata':source_metadata,'original_study_sha256':original_hashes,
        'numpy_version':np.__version__,'python_version':sys.version,
        'script_sha256':file_hash(Path(__file__)),
        'ordinal_implementation_sha256':file_hash(ROOT/'src/cnm.py'),
        'max_abs_cross_replicate_xi_correlation':float(np.max(np.abs(correlation-np.eye(REPLICATES))))}
    for path,expected in original_hashes.items():
        if file_hash(ROOT/path)!=expected:
            raise AssertionError(f'Original noise study unexpectedly changed: {path}')
    write_json(DATA/'noise_statistics_settings.json',settings)
    write_csv('noise_statistics_replicates.csv',replicate_rows)
    return values, indices


def analyze(values, indices):
    mean, sd, mcse = moments(values)
    boot_points = bootstrap_means(values,indices)
    point_ci = np.quantile(boot_points,[.025,.975],axis=0)
    summaries, source_contrasts, changes, distances, distance_changes, convergence = [], [], [], [], [], []
    differences = values[:,1]-values[:,0]
    boot_differences = boot_points[:,1]-boot_points[:,0]
    paired_distances = np.linalg.norm(differences,axis=-1)
    boot_centroid_distance = np.linalg.norm(boot_differences,axis=-1)
    for ki,(ell,q) in enumerate(ORDINAL_SETTINGS):
        for ei,eta in enumerate(ETAS):
            base = {'eta':eta,'ell':ell,'q':q,'dt':.2}
            for metric_index,metric in enumerate(['H','C']):
                for si,name in enumerate(SIGNALS):
                    summaries.append({**base,'signal':name,'metric':metric,'n':REPLICATES,
                        'mean':mean[si,ei,ki,metric_index],'sd':sd[si,ei,ki,metric_index],
                        'mcse':mcse[si,ei,ki,metric_index],
                        'ci_low':point_ci[0,si,ei,ki,metric_index],
                        'ci_high':point_ci[1,si,ei,ki,metric_index]})
                    full_mean = mean[si,ei,ki,metric_index]
                    for n in [5,10,20,30]:
                        m,s,se = moments(values[:n,si,ei,ki,metric_index])
                        convergence.append({**base,'estimand':'source_mean','signal':name,
                            'metric':metric,'n':n,'mean':m,'sd':s,'mcse':se,
                            'difference_from_n30':m-full_mean})
                source_contrasts.append({**base,'contrast':'chaotic_minus_spiking','metric':metric,
                    **paired_contrast(values[:,0,ei,ki,metric_index],values[:,1,ei,ki,metric_index],indices)})
            d, d_se = centroid_distance_mcse(differences[:,ei,ki])
            d_ci = np.quantile(boot_centroid_distance[:,ei,ki],[.025,.975])
            paired = summarize_mc(paired_distances[:,ei,ki],indices)
            distances.append({**base,'n':REPLICATES,'centroid_distance':d,
                'centroid_mcse_delta_method':d_se,'centroid_ci_low':d_ci[0],'centroid_ci_high':d_ci[1],
                **{f'paired_distance_{key}':value for key,value in paired.items() if key!='n'}})
            for n in [5,10,20,30]:
                estimate,se = centroid_distance_mcse(differences[:n,ei,ki])
                convergence.append({**base,'estimand':'centroid_distance','signal':'chaotic_vs_spiking',
                    'metric':'HC_distance','n':n,'mean':estimate,'mcse':se,'difference_from_n30':estimate-d})
                m,s,se = moments(paired_distances[:n,ei,ki])
                convergence.append({**base,'estimand':'paired_distance_mean','signal':'chaotic_vs_spiking',
                    'metric':'HC_distance','n':n,'mean':m,'sd':s,'mcse':se,
                    'difference_from_n30':m-paired['mean']})
            if ei==0:
                continue
            for previous,contrast_name in [(0,'eta_minus_clean'),(ei-1,'eta_minus_previous')]:
                for si,name in enumerate(SIGNALS):
                    for metric_index,metric in enumerate(['H','C']):
                        changes.append({**base,'previous_eta':ETAS[previous],'signal':name,
                            'metric':metric,'contrast':contrast_name,
                            **paired_contrast(values[:,si,previous,ki,metric_index],
                                              values[:,si,ei,ki,metric_index],indices)})
            previous = ei-1
            prior_d,_ = centroid_distance_mcse(differences[:,previous,ki])
            paired_change = summarize_mc(paired_distances[:,ei,ki]-paired_distances[:,previous,ki],indices)
            boot_change = boot_centroid_distance[:,ei,ki]-boot_centroid_distance[:,previous,ki]
            low,high = np.quantile(boot_change,[.025,.975])
            mu_current = moments(differences[:,ei,ki])[0]
            mu_previous = moments(differences[:,previous,ki])[0]
            projected_change = (differences[:,ei,ki]@(mu_current/d)
                -differences[:,previous,ki]@(mu_previous/prior_d))
            distance_changes.append({**base,'previous_eta':ETAS[previous],
                'centroid_change':d-prior_d,'centroid_change_mcse_delta_method':moments(projected_change)[2],
                'centroid_change_ci_low':low,'centroid_change_ci_high':high,
                **{f'paired_distance_change_{key}':value for key,value in paired_change.items()}})
    write_csv('noise_statistics_summary.csv',summaries)
    write_csv('noise_statistics_paired_contrasts.csv',source_contrasts)
    write_csv('noise_statistics_noise_changes.csv',changes)
    write_csv('noise_statistics_distances.csv',distances)
    write_csv('noise_statistics_distance_changes.csv',distance_changes)
    write_csv('noise_statistics_convergence.csv',convergence)
    np.savez_compressed(DATA/'noise_statistics_bootstrap_estimates.npz',
        bootstrap_mean_points=boot_points,bootstrap_centroid_distances=boot_centroid_distance,
        mean_points=mean,paired_realization_distances=paired_distances)
    return summaries, distances


def main():
    DATA.mkdir(exist_ok=True)
    TABLES.mkdir(exist_ok=True)
    values, indices = generate()
    summaries, distances = analyze(values,indices)
    if not all(row['mcse']==0 and row['ci_low']==row['ci_high']==row['mean']
               for row in summaries if row['eta']==0):
        raise AssertionError('Zero-noise deterministic limit failed')
    print(f'Saved {len(summaries)} mean/uncertainty rows and {len(distances)} distance rows; '
          'all intervals are conditional, approximate and pointwise.',flush=True)


if __name__=='__main__':
    main()
