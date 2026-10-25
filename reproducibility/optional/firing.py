"""Finite-time firing diagnostics; no claim about an asymptotic bifurcation.

The four windows follow the discarded transient. Counts, stationarity and a
last-event guard prevent a decaying initial train from being called sustained.
"""
import numpy as np

CRITERION = {'windows':4, 'minimum_events_per_window':3,
             'maximum_window_rate_relative_range':.01,
             'maximum_final_gap_in_tail_ISI':2., 'tail_ISI_count':8}


def firing_diagnostics(spikes, duration):
    s = np.asarray(spikes, dtype=float)
    if duration <= 0 or s.ndim != 1 or not np.isfinite(s).all():
        raise ValueError('Expected finite event times and a positive observation duration')
    if len(s) and (s[0] < 0 or s[-1] > duration+1e-7 or np.any(np.diff(s) <= 0)):
        raise ValueError('Events must be ordered within the observation interval')
    edges = np.linspace(0., duration, CRITERION['windows']+1)
    counts = np.histogram(s, edges)[0]
    rates = []
    for i, (a,b) in enumerate(zip(edges[:-1],edges[1:])):
        events = s[(s>=a) & ((s<=b) if i==len(counts)-1 else (s<b))]
        rates.append(1000./np.mean(np.diff(events)) if len(events)>=3 else np.nan)
    rates = np.asarray(rates)
    isi = np.diff(s)
    median = float(np.median(isi[-CRITERION['tail_ISI_count']:])) if len(isi) else np.nan
    gap = float(duration-s[-1]) if len(s) else np.nan
    relative = float(np.ptp(rates)/np.mean(rates)) if np.isfinite(rates).all() else np.nan
    sustained = (np.all(counts>=CRITERION['minimum_events_per_window'])
                 and np.isfinite(relative)
                 and relative<=CRITERION['maximum_window_rate_relative_range']
                 and gap<=CRITERION['maximum_final_gap_in_tail_ISI']*median)
    return {'sustained':bool(sustained), 'total_events':len(s),
            'window_counts':counts, 'window_rates_Hz':rates,
            'window_count_min':int(counts.min()),
            'window_rate_relative_range':relative,
            'tail_median_ISI_ms':median, 'last_gap_ms':gap,
            'last_event_ms':float(s[-1]) if len(s) else np.nan,
            'last_gap_in_tail_ISI':gap/median if len(isi) else np.nan}
