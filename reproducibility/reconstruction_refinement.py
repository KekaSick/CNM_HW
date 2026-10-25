#!/usr/bin/env python3
"""Scalar-only reconstruction repairs and independent sampling diagnostics.

Run ``python reconstruction_refinement.py all`` from submission/. Original
records and estimates remain untouched. New outputs are prefixed B_refined or
stored under data/reconstruction_refinement/. A polyline cover is explicitly a
cover of the continuously interpolated delay trajectory, not extra independent
observations and not a claim about an asymptotic fractal dimension.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', '/tmp/cnm_mpl')
os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/cnm_numba')
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import pandas as pd
from numba import njit
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from publication import install_publication_exports
install_publication_exports()
from cnm import embed, false_nearest, correlation_sum, fit_scaling, integrate, IC1, IC2

DATA = ROOT/'data'/'reconstruction_refinement'
TABLES = ROOT/'tables'
FIG = ROOT/'figures'


@njit(cache=True)
def _visited_codes(z, multipliers):
    """Traverse grid planes in parameter order, including both endpoints.

    Cells are half-open: floor(z). At a simultaneous crossing, positive
    coordinates change at the crossing itself and negative coordinates just
    after it. This includes a cell touched only at a mixed-direction corner.
    Crossing times are compared exactly, so short visits are not discarded.
    Integer packing is injective; the public wrapper checks for overflow.
    """
    cell = np.floor(z).astype(np.int64)
    capacity = 1 + np.abs(cell[1:]-cell[:-1]).sum()
    codes = np.empty(capacity, dtype=np.int64)
    codes[0] = np.sum(cell[0]*multipliers)
    at = 1
    if len(z) == 1:
        return codes[:1]
    for i in range(len(z)-1):
        # A cell is convex: a segment whose endpoints lie in it cannot add a
        # different cell. This avoids allocating several tiny temporary arrays
        # at every step of a very densely sampled smooth trajectory.
        if np.all(cell[i] == cell[i+1]):
            continue
        current = cell[i].copy()
        target = cell[i+1]
        code = np.sum(current*multipliers)
        direction = z[i+1]-z[i]
        step = np.sign(direction).astype(np.int64)
        while np.any(current != target):
            crossing = np.full(z.shape[1], np.inf)
            for j in range(z.shape[1]):
                if current[j] != target[j]:
                    boundary = current[j] + (1 if step[j] > 0 else 0)
                    crossing[j] = (boundary-z[i, j])/direction[j]
            first = crossing.min()
            positive = False
            negative = False
            for j in range(z.shape[1]):
                if crossing[j] == first and step[j] > 0:
                    positive = True
                    current[j] += step[j]
                    code += step[j]*multipliers[j]
                if crossing[j] == first and step[j] < 0:
                    negative = True
            if positive and negative:
                # floor at the corner includes positive steps, while negative
                # coordinates still belong to the cell on the positive side.
                codes[at] = code
                at += 1
            for j in range(z.shape[1]):
                if crossing[j] == first and step[j] < 0:
                    current[j] += step[j]
                    code += step[j]*multipliers[j]
            codes[at] = code
            at += 1
    return codes[:at]


def occupied_boxes(points, eps, shift=0., origin=None, curve=True):
    """Count point cells or every cell met by the piecewise-linear trajectory.

    ``shift`` is a scalar or one offset per coordinate, in cell-width units.
    Translating the grid changes counts at finite scale and is retained as a
    sensitivity test. The function receives reconstructed scalar coordinates;
    it cannot access ODE state variables, derivatives, or hidden events.
    """
    points = np.asarray(points, dtype=float)
    if (points.ndim != 2 or len(points) == 0 or not np.isfinite(points).all()
            or not np.isfinite(eps) or eps <= 0):
        raise ValueError('Finite nonempty point array and positive scale required')
    origin = points.min(axis=0) if origin is None else np.asarray(origin)
    z = (points-origin)/eps + np.asarray(shift)
    # Subtract a common integer cell index: preserves all crossings and packs
    # arbitrary (also negative) grid indices into nonnegative mixed-radix codes.
    z -= np.floor(z.min(axis=0))
    sizes = np.floor(z.max(axis=0)).astype(np.int64)+1
    product = 1
    multipliers = []
    for size in sizes:
        multipliers.append(product)
        product *= int(size)
    if product > np.iinfo(np.int64).max:
        raise OverflowError('Grid cannot be represented by unique int64 cell codes')
    multipliers = np.asarray(multipliers, dtype=np.int64)
    if curve:
        codes = _visited_codes(z, multipliers)
    else:
        codes = np.sum(np.floor(z).astype(np.int64)*multipliers, axis=1)
    return int(len(np.unique(codes)))


def write_json(path, value):
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)
    path.write_text(json.dumps(value, indent=2, default=convert)+'\n')


def freeze_protocol(path, protocol, source_path, record_path):
    """Persist an immutable method before a new validation record can be read.

    A byte-for-byte method snapshot is archived beside the JSON. Reusing the
    validation after changing settings or source is rejected rather than silently
    redefining an already observed record as a fresh holdout.
    """
    path, source_path, record_path = Path(path), Path(source_path), Path(record_path)
    source = source_path.read_bytes()
    source_hash = hashlib.sha256(source).hexdigest()
    if path.exists():
        frozen = json.loads(path.read_text())
        if frozen['protocol'] != protocol:
            raise ValueError('The heldout protocol is frozen; changed settings cannot reuse it')
        if frozen['method_source_sha256'] != source_hash:
            raise ValueError('The heldout source is frozen; changed source cannot reuse it')
        snapshot = path.parent/frozen['source_snapshot']
        if hashlib.sha256(snapshot.read_bytes()).hexdigest() != source_hash:
            raise ValueError('The archived frozen source checksum does not match')
        return frozen
    if record_path.exists():
        raise ValueError('A preexisting record cannot be declared unseen retrospectively')
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = path.with_name(path.stem+'_source.py')
    snapshot.write_bytes(source)
    frozen = dict(protocol=protocol, frozen_utc=datetime.now(timezone.utc).isoformat(),
        method_source_sha256=source_hash, source_snapshot=snapshot.name,
        record_path=str(record_path), record_absent_when_frozen=True)
    write_json(path, frozen)
    return frozen


def write_csv(name, rows):
    result = pd.DataFrame(rows)
    result.to_csv(TABLES/f'reconstruction_refinement_{name}.csv', index=False, float_format='%.12g')
    return result


def save_figure(name, fig):
    fig.savefig(FIG/f'B_refined_{name}.png', dpi=170, bbox_inches='tight')
    plt.close(fig)


def scalar_record(path, mean, sigma):
    # All estimators below receive only this scalar array, never other states.
    with np.load(path) as record:
        return (record['state'][:-1, 0]-mean)/sigma


def reconstruction_study():
    """Compare MI/ACF candidates with shorter lags using scalar diagnostics."""
    with np.load(ROOT/'data'/'chaotic.npz') as record:
        potential = record['state'][:-1, 0]
    mean, sigma = potential.mean(), potential.std()
    x = (potential-mean)/sigma
    rows, conditioning = [], []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for lag in [8, 10, 12, 124, 164]:
        fnn = false_nearest(x, lag, 12, 656, 12000)
        np.savez_compressed(DATA/f'fnn_k{lag}.npz', fnn=fnn, lag=lag, W=656)
        for m, fraction, usable, omitted in fnn:
            rows.append(dict(case='baseline', lag=lag, tau=.2*lag, W=656,
                             m=int(m), FNN=fraction, usable=int(usable), omitted=int(omitted)))
        axes[0].semilogy(fnn[:, 0], np.maximum(fnn[:, 1], 1/12000), 'o-',
                         label=f'k={lag}, tau={lag*.2:g}')
        for m in [4, 6, 8]:
            delay_vectors = embed(x, m, lag)
            covariance = np.cov(delay_vectors.T)
            eigenvalues = np.linalg.eigvalsh(covariance)
            correlation = np.corrcoef(delay_vectors.T)
            conditioning.append(dict(lag=lag, m=m,
                covariance_condition=eigenvalues[-1]/eigenvalues[0],
                smallest_covariance_eigenvalue=eigenvalues[0],
                largest_covariance_eigenvalue=eigenvalues[-1],
                adjacent_correlation=correlation[0, 1],
                end_to_end_correlation=correlation[0, -1],
                embedding_span_time=(m-1)*lag*.2))
    for case, xx in [('first_quarter', x[:len(x)//4]), ('first_half', x[:len(x)//2]),
                     ('last_quarter', x[-len(x)//4:])]:
        fnn = false_nearest(xx, 10, 8, 656, 12000)
        for m, fraction, usable, omitted in fnn:
            rows.append(dict(case=case, lag=10, tau=2., W=656,
                             m=int(m), FNN=fraction, usable=int(usable), omitted=int(omitted)))
    write_csv('fnn', rows)
    cond = write_csv('conditioning', conditioning)
    for m in [4, 6, 8]:
        a = cond[cond.m == m]
        axes[1].plot(a.lag*.2, a.covariance_condition, 'o-', label=f'm={m}')
    axes[0].axhline(.01, color='black', ls=':', lw=1)
    axes[0].set(xlabel='embedding dimension m', ylabel='FNN fraction (floor 1/12000)',
                title='Same W=656, 12000 scalar delay vectors')
    axes[1].set(xlabel='delay tau [model time]', ylabel='covariance condition number',
                title='Check against a nearly diagonal reconstruction')
    for ax in axes:
        ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure('reconstruction', fig)
    settings = dict(source='data/chaotic.npz', mean=mean, sigma=sigma, dt=.2,
                    selected_lag=10, selected_m=6, W=656,
                    grid_rng_initial_state=np.random.PCG64(7261).state,
                    lag_candidates=[8, 10, 12, 124, 164],
                    FNN_threshold=.01, FNN_maxpoints=12000,
                    rationale='m=6 stays below 1% FNN at all short lags 8,10,12; '
                              'short lag avoids large next-coordinate stretching; '
                              'MI and ACF alternatives retained for comparison',
                    interpretation='Finite-scale geometry; FNN and covariance diagnostics '
                                   'are necessary checks, not embedding proofs')
    write_json(DATA/'settings.json', settings)
    fig = plt.figure(figsize=(11, 4.8))
    for panel, (m, lag, coordinates, title) in enumerate([
            (8, 164, [0, 1, 2], 'Original MI choice: tau=32.8, m=8'),
            (6, 10, [0, 2, 5], 'Revised choice: tau=2, m=6')]):
        ax = fig.add_subplot(1, 2, panel+1, projection='3d')
        vectors = embed(x, m, lag)[::8, coordinates]
        ax.scatter(*vectors.T, s=.3, alpha=.35, color='#9b3d68',
                   depthshade=False, rasterized=True)
        labels = [f'x(t+{coordinate*lag*.2:g}) [std x]' for coordinate in coordinates]
        ax.set(title=title, xlabel=labels[0], ylabel=labels[1], zlabel=labels[2])
    fig.tight_layout()
    save_figure('attractor', fig)
    print('Refinement: FNN, segment checks and scalar conditioning saved', flush=True)
    return settings


def gp_study(settings):
    mean, sigma = settings['mean'], settings['sigma']
    x = scalar_record(ROOT/'data'/'chaotic.npz', mean, sigma)
    rows, results = [], {}
    experiments = []
    for lag in [8, 10, 12]:
        for m in [3, 4, 5, 6, 8, 10]:
            experiments.append((f'lag{lag}_m{m}', 'embedding', lag, m, 656, x))
    for w in [0, 328, 1312, 2624]:
        experiments.append((f'W{w}', 'W', 10, 6, w, x))
    for name, xx in [('first_quarter', x[:len(x)//4]), ('first_half', x[:len(x)//2]),
                     ('last_quarter', x[-len(x)//4:]),
                     ('longer', scalar_record(ROOT/'data'/'convergence'/'chaotic_longer_record.npz', mean, sigma))]:
        experiments.append((name, 'N', 10, 6, 656, xx))
    bands = [(.0005, .005), (.001, .01), (.002, .02), (.003, .03), (.01, .07)]
    for name, varied, lag, m, window, xx in experiments:
        radii = np.geomspace(.0001*np.sqrt(m), .15*np.sqrt(m), 100)
        result = correlation_sum(xx, m, lag, window, radii, maxpoints=16000)
        np.savez_compressed(DATA/f'gp_{name}.npz', **result, lag=lag, m=m,
                            W=window, record_N=len(xx))
        results[name] = result
        for lower, upper in bands:
            fit = fit_scaling(result, lower*np.sqrt(m), upper*np.sqrt(m))
            rows.append(dict(case=name, varied=varied, lag=lag, m=m, W=window,
                             record_N=len(xx), nominal_lower=lower, nominal_upper=upper, **fit))
        print(f'Refinement: GP {name}', flush=True)
    table = write_csv('gp', rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for m in [3, 4, 5, 6, 8, 10]:
        result = results[f'lag10_m{m}']
        axes[0].loglog(result['r']/np.sqrt(m), result['C'], label=f'm={m}')
        axes[1].semilogx(result['r']/np.sqrt(m), result['slope'], label=f'm={m}')
    for ax in axes:
        ax.axvspan(.001, .01, alpha=.12, color='grey')
        ax.set_xlabel('r/sqrt(m) [std x]')
        ax.legend(fontsize=8)
    axes[0].set(ylabel='correlation sum C(r)', title='Short delay k=10, W=656')
    axes[1].set(ylabel='local correlation slope', ylim=(0, 3.2), title='All radii; common fit band shaded')
    fig.tight_layout()
    save_figure('gp', fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    band = table[(table.nominal_lower == .001) & (table.nominal_upper == .01)]
    for lag in [8, 10, 12]:
        a = band[(band.varied == 'embedding') & (band.lag == lag)]
        axes[0].plot(a.m, a.estimate, 'o-', label=f'k={lag}')
    a = band[(band.varied == 'W') | (band.case == 'lag10_m6')].sort_values('W')
    axes[1].plot(a.W, a.estimate, 'o-')
    a = band[(band.varied == 'N') | (band.case == 'lag10_m6')]
    axes[2].plot(np.arange(len(a)), a.estimate, 'o')
    axes[2].set_xticks(np.arange(len(a)), a.case, rotation=25, ha='right', fontsize=8)
    axes[0].set(xlabel='embedding dimension m', ylabel='finite-scale D2 slope', title='Delay and m')
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel='Theiler W [original samples]', title='m=6, k=10')
    axes[2].set(title='Record duration / segment, m=6')
    fig.tight_layout()
    save_figure('gp_sensitivity', fig)


def box_curves(x, m, lag, scales, shifts):
    points = embed(x, m, lag)
    point_counts, curve_counts = [], []
    for shift in shifts:
        point_counts.append([occupied_boxes(points, e, shift, curve=False) for e in scales])
        curve_counts.append([occupied_boxes(points, e, shift, curve=True) for e in scales])
    return np.asarray(point_counts), np.asarray(curve_counts)


def box_fit_rows(case, m, lag, dt, scales, point_counts, curve_counts, bands=None):
    rows = []
    # Fixed before comparisons. All bands and local slopes are published;
    # an interval is not selected by forcing D0 to match D2.
    for method, counts in [('points', point_counts), ('curve', curve_counts)]:
        for grid, values in enumerate(counts):
            local = -np.gradient(np.log(values), np.log(scales))
            for lower, upper in (bands if bands is not None else [(.05, .25), (.09, .4), (.14, .56)]):
                mask = (scales >= lower) & (scales <= upper)
                slope = np.polyfit(np.log(1/scales[mask]), np.log(values[mask]), 1)[0]
                rows.append(dict(case=case, method=method, m=m, lag=lag, dt=dt,
                    tau=lag*dt, grid=grid, nominal_lower=lower, nominal_upper=upper,
                    estimate=slope, eps_min=scales[mask].min(), eps_max=scales[mask].max(),
                    nscales=mask.sum(), occupied_min=values[mask].min(), occupied_max=values[mask].max(),
                    local_min=local[mask].min(), local_max=local[mask].max()))
    return rows


def capacity_study(settings):
    x = scalar_record(ROOT/'data'/'chaotic.npz', settings['mean'], settings['sigma'])
    scales = np.geomspace(.03, .8, 24)
    rows, curves = [], {}
    cases = [('main', m, lag, .2, x) for m, lag in
             [(3, 164), (4, 164), (4, 8), (4, 10), (4, 12), (6, 10)]]
    for case, file_name, dt, factor in [
            ('main_finer', 'chaotic_finer_sampling.npz', .04, 5),
            ('main_longer', 'chaotic_longer_record.npz', .2, 1)]:
        xx = scalar_record(ROOT/'data'/'convergence'/file_name, settings['mean'], settings['sigma'])
        for m, lag in [(4, 164), (6, 10)]:
            cases.append((case, m, lag*factor, dt, xx))
    for case, m, lag, dt, xx in cases:
        rng = np.random.Generator(np.random.PCG64(7261))
        shifts = np.vstack((np.zeros(m), np.full(m, .25), rng.random((2, m))))
        point_counts, curve_counts = box_curves(xx, m, lag, scales, shifts)
        name = f'{case}_m{m}_k{lag}'
        np.savez_compressed(DATA/f'box_{name}.npz', eps=scales, point_counts=point_counts,
                            curve_counts=curve_counts, shifts=shifts, m=m, lag=lag, dt=dt)
        if case == 'main':
            curves[f'm{m}_k{lag}'] = (point_counts, curve_counts)
        rows.extend(box_fit_rows(case, m, lag, dt, scales, point_counts, curve_counts))
        print(f'Refinement: point/curve box covers {name}', flush=True)
    write_csv('capacity', rows)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    for row, (m, lag) in enumerate([(4, 164), (6, 10)]):
        for label, counts, style in [('points', curves[f'm{m}_k{lag}'][0], '--'),
                                     ('curve', curves[f'm{m}_k{lag}'][1], '-')]:
            average = counts.mean(axis=0)
            line = axes[row, 0].loglog(1/scales, average, style, label=label)[0]
            axes[row, 0].fill_between(1/scales, counts.min(axis=0), counts.max(axis=0),
                                      color=line.get_color(), alpha=.13)
            local = -np.gradient(np.log(average), np.log(scales))
            axes[row, 1].semilogx(scales, local, style, label=label)
        axes[row, 0].set(title=f'm={m}, tau={lag*.2:g}', xlabel='1/epsilon [1/std x]', ylabel='occupied cells')
        axes[row, 1].set(title='Local slopes, including grid effects', xlabel='epsilon [std x]',
                          ylabel='local box slope', ylim=(0, 3.4))
        axes[row, 1].axvspan(.09, .4, color='grey', alpha=.1)
        for ax in axes[row]:
            ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure('capacity', fig)


def dense_sampling_study(settings):
    """All samplings are exact subarrays of the same accurately solved path."""
    path = DATA/'dense_validation.npz'
    if not path.exists():
        state, spikes, section, peaks, peak_times = integrate(3.25, IC1, 10000., 10000., .005, .0025)
        np.savez_compressed(path, t=np.arange(len(state))*.005, state=state, spikes=spikes,
            section=section, peaks=peaks, peak_times=peak_times, initial=IC1,
            transient_state=state[0], final=state[-1], I=3.25, transient=10000.,
            duration=10000., dt=.005, h=.0025)
    with np.load(path) as record:
        potential = record['state'][:, 0]
    scales = np.geomspace(.03, .8, 24)
    rows, support, curves = [], [], {}
    for m, tau in [(3, 32.8), (4, 32.8), (6, 2.)]:
        # Independent per-coordinate shifts supplement the conventional common
        # diagonal shifts. A fixed seed is part of the saved configuration.
        rng = np.random.Generator(np.random.PCG64(7261))
        shifts = np.vstack((np.zeros(m), rng.random(m)))
        for stride in [40, 8, 2, 1]:
            dt = stride*.005
            lag = int(round(tau/dt))
            x = (potential[::stride][:-1]-settings['mean'])/settings['sigma']
            point_counts, curve_counts = box_curves(x, m, lag, scales, shifts)
            name = f'm{m}_tau{tau:g}_stride{stride}'
            np.savez_compressed(DATA/f'sampling_{name}.npz', eps=scales, point_counts=point_counts,
                curve_counts=curve_counts, shifts=shifts, m=m, lag=lag, dt=dt,
                source='dense_validation.npz', stride=stride, duration=10000.)
            rows.extend(box_fit_rows('matched_dense_path', m, lag, dt, scales, point_counts, curve_counts))
            points = embed(x, m, lag)
            jumps = np.max(np.abs(np.diff(points, axis=0)), axis=1)
            for eps in scales:
                support.append(dict(m=m, tau=tau, dt=dt, stride=stride, eps=eps,
                    max_step_infinity_norm=jumps.max(), step_p99=np.quantile(jumps, .99),
                    fraction_steps_exceeding_eps=np.mean(jumps > eps), record_N=len(x)))
            curves[(m, tau, stride)] = (point_counts, curve_counts)
            print(f'Refinement: matched sampling m={m},tau={tau:g},dt={dt:g}', flush=True)
    table = write_csv('sampling', rows)
    write_csv('sampling_support', support)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for row, (m, tau) in enumerate([(4, 32.8), (6, 2.)]):
        for stride in [40, 8, 2, 1]:
            point_counts, curve_counts = curves[(m, tau, stride)]
            label = f'dt={stride*.005:g}'
            line = axes[row, 0].loglog(1/scales, point_counts.mean(axis=0), label=label)[0]
            axes[row, 0].loglog(1/scales, curve_counts.mean(axis=0), '--', color=line.get_color(), alpha=.65)
        for method, style in [('points', 'o-'), ('curve', 's--')]:
            a = table[(table.m == m) & (table.tau.round(8) == tau) & (table.method == method)
                      & (table.nominal_lower == .09)]
            g = a.groupby('dt').estimate
            axes[row, 1].errorbar(g.mean().index, g.mean(),
                yerr=[g.mean()-g.min(), g.max()-g.mean()], fmt=style, label=method, capsize=3)
        axes[row, 0].set(title=f'One trajectory: m={m}, tau={tau:g}', xlabel='1/epsilon [1/std x]',
                          ylabel='cells; solid=points, dashed=curve')
        axes[row, 1].set(xlabel='dt [model time]', ylabel='finite-scale box slope', xscale='log',
                          title='Fixed eps band .09–.4; shifts are error bars')
        for ax in axes[row]:
            ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure('sampling', fig)


def heldout_study(settings):
    """Evaluate a frozen scalar protocol on a newly generated IC2 trajectory."""
    record_path = DATA/'heldout_IC2.npz'
    training_gp_path = TABLES/'reconstruction_refinement_gp.csv'
    training_box_path = TABLES/'reconstruction_refinement_capacity.csv'
    training_gp = pd.read_csv(training_gp_path)
    training_box = pd.read_csv(training_box_path)
    gp_envelopes = {}
    for m in [4, 6, 8]:
        values = training_gp[(training_gp.m == m) & (training_gp.nominal_lower == .001)].estimate
        gp_envelopes[str(m)] = [float(values.min()), float(values.max())]
    box_envelopes = {}
    for method in ['points', 'curve']:
        values = training_box[(training_box.m == 6) & (training_box.tau == 2.)
            & (training_box.nominal_lower == .09) & (training_box.method == method)].estimate
        box_envelopes[method] = [float(values.min()), float(values.max())]
    rng = np.random.Generator(np.random.PCG64(7261))
    shifts = np.vstack((np.zeros(6), np.full(6, .25), rng.random((2, 6))))
    protocol = dict(I=3.25, initial=[1., 0., 0.], transient=10000., duration=30000.,
        dt=.2, h=.02, normalization_mean=settings['mean'], normalization_sigma=settings['sigma'],
        selected_lag=10, selected_m=6, W=656, FNN_lags=[8, 10, 12], FNN_maxdim=8,
        FNN_maxpoints=12000, FNN_threshold=.01, GP_dimensions=[4, 6, 8],
        GP_band_normalized=[.001, .01], GP_radius_normalized=[.0001, .15], GP_nscales=100,
        GP_maxpoints=16000, GP_min_pairs=200, GP_max_C=.15,
        box_scales=np.geomspace(.03, .8, 24).tolist(), box_band=[.09, .4], box_shifts=shifts.tolist(),
        GP_disclosed_training_envelopes=gp_envelopes, box_disclosed_training_envelopes=box_envelopes,
        training_sources_sha256={
            'data/chaotic.npz':hashlib.sha256((ROOT/'data'/'chaotic.npz').read_bytes()).hexdigest(),
            'tables/reconstruction_refinement_gp.csv':hashlib.sha256(training_gp_path.read_bytes()).hexdigest(),
            'tables/reconstruction_refinement_capacity.csv':hashlib.sha256(training_box_path.read_bytes()).hexdigest(),
            'src/cnm.py':hashlib.sha256((ROOT/'src'/'cnm.py').read_bytes()).hexdigest()},
        statement='Settings were explored on IC1, then frozen before generating this new IC2 record. '
                  'No tuning or replacement is permitted on this validation record; results outside '
                  'the disclosed envelopes remain reported. This tests the chosen I only.')
    frozen_path = DATA/'heldout_frozen_settings.json'
    frozen = freeze_protocol(frozen_path, protocol, Path(__file__), record_path)
    p = frozen['protocol']
    print(f'Refinement: heldout protocol frozen at {frozen["frozen_utc"]}', flush=True)
    if not record_path.exists():
        initial = np.asarray(p['initial'])
        state, spikes, section, peaks, peak_times = integrate(p['I'], initial,
            p['transient'], p['duration'], p['dt'], p['h'])
        np.savez_compressed(record_path, t=np.arange(len(state))*p['dt'], state=state,
            spikes=spikes, section=section, peaks=peaks, peak_times=peak_times,
            initial=initial, transient_state=state[0], final=state[-1], I=p['I'],
            transient=p['transient'], duration=p['duration'], dt=p['dt'], h=p['h'])
    x = scalar_record(record_path, p['normalization_mean'], p['normalization_sigma'])
    fnn_rows, gp_rows, comparisons, gp_curves = [], [], [], {}
    for lag in p['FNN_lags']:
        fnn = false_nearest(x, lag, p['FNN_maxdim'], p['W'], p['FNN_maxpoints'])
        np.savez_compressed(DATA/f'heldout_fnn_k{lag}.npz', fnn=fnn, lag=lag, W=p['W'])
        for m, fraction, usable, omitted in fnn:
            fnn_rows.append(dict(lag=lag, m=int(m), FNN=fraction,
                                 usable=int(usable), omitted=int(omitted)))
            if int(m) == p['selected_m']:
                comparisons.append(dict(metric='FNN', setting=f'k={lag},m=6',
                    validation_min=fraction, validation_max=fraction,
                    disclosed_lower=0., disclosed_upper=p['FNN_threshold'],
                    inside_disclosed_envelope=bool(fraction <= p['FNN_threshold'])))
    write_csv('heldout_fnn', fnn_rows)
    for m in p['GP_dimensions']:
        radii = np.geomspace(*np.asarray(p['GP_radius_normalized'])*np.sqrt(m), p['GP_nscales'])
        result = correlation_sum(x, m, p['selected_lag'], p['W'], radii,
                                 maxpoints=p['GP_maxpoints'])
        gp_curves[m] = result
        np.savez_compressed(DATA/f'heldout_gp_m{m}.npz', **result, m=m,
                            lag=p['selected_lag'], W=p['W'], record_N=len(x))
        lower, upper = p['GP_band_normalized']
        fit = fit_scaling(result, lower*np.sqrt(m), upper*np.sqrt(m), p['GP_min_pairs'])
        gp_rows.append(dict(m=m, lag=p['selected_lag'], W=p['W'], record_N=len(x),
                            nominal_lower=lower, nominal_upper=upper, **fit))
        lo, hi = p['GP_disclosed_training_envelopes'][str(m)]
        comparisons.append(dict(metric='D2_effective', setting=f'k=10,m={m}',
            validation_min=fit['estimate'], validation_max=fit['estimate'],
            disclosed_lower=lo, disclosed_upper=hi,
            inside_disclosed_envelope=bool(lo <= fit['estimate'] <= hi)))
    write_csv('heldout_gp', gp_rows)
    eps = np.asarray(p['box_scales'])
    point_counts, curve_counts = box_curves(x, p['selected_m'], p['selected_lag'],
                                           eps, np.asarray(p['box_shifts']))
    np.savez_compressed(DATA/'heldout_box_m6_k10.npz', eps=eps, point_counts=point_counts,
        curve_counts=curve_counts, shifts=p['box_shifts'], m=6, lag=10, dt=p['dt'])
    box_rows = box_fit_rows('heldout_IC2', 6, 10, p['dt'], eps, point_counts, curve_counts,
                            bands=[p['box_band']])
    box_table = write_csv('heldout_capacity', box_rows)
    for method in ['points', 'curve']:
        values = box_table[box_table.method == method].estimate
        lo, hi = p['box_disclosed_training_envelopes'][method]
        comparisons.append(dict(metric='D0_effective', setting=f'{method},k=10,m=6',
            validation_min=values.min(), validation_max=values.max(),
            disclosed_lower=lo, disclosed_upper=hi,
            inside_disclosed_envelope=bool(lo <= values.min() and values.max() <= hi)))
    write_csv('heldout_comparison', comparisons)
    write_json(DATA/'heldout_manifest.json', dict(
        record_sha256=hashlib.sha256(record_path.read_bytes()).hexdigest(),
        frozen_settings_sha256=hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
        frozen_method_sha256=frozen['method_source_sha256'], record_N=len(x),
        completed_utc=datetime.now(timezone.utc).isoformat()))
    # Visualization is descriptive; no displayed scale or observed result feeds
    # back into the frozen estimation procedure above.
    train_x = scalar_record(ROOT/'data'/'chaotic.npz', p['normalization_mean'], p['normalization_sigma'])
    train_gp = np.load(DATA/'gp_lag10_m6.npz')
    train_box = np.load(DATA/'box_main_m6_k10.npz')
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].semilogx(train_gp['r']/np.sqrt(6), train_gp['slope'], color='black',
                       ls='--', label='explored IC1, m=6')
    for m in p['GP_dimensions']:
        result = gp_curves[m]
        axes[0, 0].semilogx(result['r']/np.sqrt(m), result['slope'], label=f'new IC2, m={m}')
    axes[0, 0].axvspan(*p['GP_band_normalized'], color='grey', alpha=.12)
    axes[0, 0].set(xlabel='r/sqrt(m) [std x]', ylabel='local correlation slope', ylim=(0, 3.2),
                    title='Frozen k=10, W=656, GP band .001–.01')
    for label, points, curve in [('IC1', train_box['point_counts'], train_box['curve_counts']),
                                  ('IC2', point_counts, curve_counts)]:
        line = axes[0, 1].loglog(1/eps, curve.mean(axis=0), label=f'{label} curve')[0]
        axes[0, 1].loglog(1/eps, points.mean(axis=0), '--', color=line.get_color(),
                          label=f'{label} points')
    axes[0, 1].set(xlabel='1/epsilon [1/std x]', ylabel='occupied cells',
                    title='Frozen m=6, tau=2, box band .09–.4')
    for ax, signal, name in zip(axes[1], [train_x, x], ['Explored IC1', 'New heldout IC2']):
        vectors = embed(signal, 6, 10)
        ax.scatter(vectors[:, 0], vectors[:, 5], s=.1, alpha=.12, color='#9b3d68', rasterized=True)
        ax.set(xlabel='x(t) [std x]', ylabel='x(t+10) [std x]', title=f'{name}: same projection of m=6')
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].legend(fontsize=8)
    fig.tight_layout()
    save_figure('heldout', fig)
    print('Refinement: heldout IC2 results saved without parameter changes', flush=True)


def summarize():
    """Compact sensitivity envelopes; no fitted-slope SE is called uncertainty."""
    for kind in ['gp', 'capacity', 'sampling']:
        path = TABLES/f'reconstruction_refinement_{kind}.csv'
        if not path.exists():
            continue
        data = pd.read_csv(path)
        if kind == 'gp':
            data = data[(data.m == 6) & (data.nominal_lower == .001)]
            groups = ['varied']
        else:
            data = data[data.nominal_lower == .09]
            groups = ['case', 'm', 'tau', 'dt', 'method']
        result = data.groupby(groups, dropna=False).agg(
            estimate_min=('estimate', 'min'), estimate_max=('estimate', 'max'),
            local_min=('local_min', 'min'), local_max=('local_max', 'max'),
            comparisons=('estimate', 'size')).reset_index()
        result.to_csv(TABLES/f'reconstruction_refinement_summary_{kind}.csv',
                      index=False, float_format='%.12g')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['all', 'reconstruction', 'gp', 'capacity', 'sampling', 'heldout', 'summary'], nargs='?', default='all')
    args = parser.parse_args()
    for directory in [DATA, TABLES, FIG]:
        directory.mkdir(parents=True, exist_ok=True)
    settings = (reconstruction_study() if args.stage in ['all', 'reconstruction']
                else json.loads((DATA/'settings.json').read_text()))
    if args.stage in ['all', 'gp']:
        gp_study(settings)
    if args.stage in ['all', 'capacity']:
        capacity_study(settings)
    if args.stage in ['all', 'sampling']:
        dense_sampling_study(settings)
    if args.stage in ['all', 'heldout']:
        heldout_study(settings)
    summarize()


if __name__ == '__main__':
    main()
