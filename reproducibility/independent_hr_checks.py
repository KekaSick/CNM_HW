#!/usr/bin/env python3
"""Independent HR checks using only NumPy, SciPy and the Python standard library.

Run from any directory: python submission/independent_hr_checks.py
Writes only new independent_hr_* data and independent_hr_checks.csv. It does not
import cnm.py or reuse its RK4, Jacobian, variational solver or orthogonalization.
The high-precision Hopf calculation uses decimal, not an unpinned dependency.
"""
from __future__ import annotations

import csv
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, root


ROOT = Path(__file__).resolve().parent
DATA, TABLES = ROOT/'data', ROOT/'tables'
IC = np.array([-1.6, -12., 0.])
CYCLE_SETTINGS = {'rtol':2e-12, 'atol':2e-14, 'max_step':.25}
VERIFY_SETTINGS = {'rtol':2e-13, 'atol':2e-15, 'max_step':.25}
CHAOTIC_SETTINGS = {'I':3.25, 'state_transient':10000.,
    'tangent_transient':1000., 'duration':5000., 'qr_interval':1.,
    'block':1000., 'rtol':2e-10, 'atol':2e-12, 'max_step':.5}


def model_rhs(t, state, current):
    """Separately spelled HR equations; complex input is retained for checking."""
    x, y, z = state
    return np.array([y-x**3+3*x**2-z+current,
                     1-5*x**2-y, .006*(4*(x+1.6)-z)])


def model_jacobian(state):
    x = state[0]
    return np.array([[-3*x**2+6*x, 1., -1.],
                     [-10*x, -1., 0.], [.024, 0., -.006]])


def augmented_rhs(t, packed, current):
    """State, full 3x3 fundamental matrix, and integral of divergence."""
    state = packed[:3]
    matrix = packed[3:12].reshape(3, 3)
    jac = model_jacobian(state)
    return np.r_[model_rhs(t, state, current), (jac@matrix).ravel(),
                 np.trace(jac)]


def augmented_flow(initial, period, current, settings, *, dense_output=False,
                   events=None):
    result = solve_ivp(augmented_rhs, (0., float(period)),
        np.r_[initial, np.eye(3).ravel(), 0.], args=(current,),
        method='DOP853', dense_output=dense_output, events=events, **settings)
    if not result.success or not np.isfinite(result.y).all():
        raise RuntimeError(f'Independent DOP853 failed: {result.message}')
    return result


def check_jacobian():
    states = np.array([IC, [1., 0., 0.], [-1.30626732, -7.53167157, 1.17493072],
        [-.5, -1., 3.], [1.7, -4., 3.2], [-1.5, -10., 2.]])
    errors, numerical = [], []
    epsilon = 1e-30
    for state in states:
        columns = []
        for j in range(3):
            perturbed = state.astype(complex)
            perturbed[j] += epsilon*1j
            columns.append(model_rhs(0., perturbed, 3.25).imag/epsilon)
        derivative = np.column_stack(columns)
        numerical.append(derivative)
        errors.append(np.max(np.abs(derivative-model_jacobian(state))))
    maximum = float(max(errors))
    if maximum >= 2e-13:
        raise AssertionError(f'Independent analytic Jacobian failed: {maximum}')
    return {'method':'complex-step derivative of separately spelled RHS',
            'epsilon':epsilon, 'states':states.tolist(),
            'max_abs_error':maximum, 'per_state_max_abs_error':errors,
            '_arrays':{'states':states, 'complex_step_jacobians':np.array(numerical),
                       'max_abs_errors':np.array(errors), 'epsilon':epsilon}}


def equilibrium_state(current):
    x = brentq(lambda x:x**3+2*x*x+4*x+5.4-current,
               -3., 1., xtol=5e-15)
    return np.array([x, 1-5*x*x, 4*(x+1.6)])


def locate_hopf():
    """Routh-Hurwitz quartic and an 80-digit decimal Newton refinement."""
    a1 = np.poly1d([3., -6., 1.006])
    a2 = np.poly1d([3.018, 3.964, .030])
    a3 = np.poly1d([.018, .024, .024])
    polynomial = a1*a2-a3
    float_x = brentq(polynomial, -1.5, -1.1, xtol=5e-15)
    with localcontext() as context:
        context.prec = 80
        D = Decimal
        x = D(str(float_x))
        for iteration in range(1, 31):
            c1 = 3*x*x-6*x+D('1.006')
            c2 = D('3.018')*x*x+D('3.964')*x+D('.030')
            c3 = D('.006')*(3*x*x+4*x+4)
            value = c1*c2-c3
            derivative = (6*x-6)*c2+c1*(D('6.036')*x+D('3.964'))-(D('.036')*x+D('.024'))
            change = value/derivative
            x -= change
            if abs(change) < D('1e-65'):
                break
        else:
            raise AssertionError('Decimal Hopf refinement did not converge')
        current = x**3+2*x*x+4*x+D('5.4')
        decimal_residual = ((3*x*x-6*x+D('1.006'))
            *(D('3.018')*x*x+D('3.964')*x+D('.030'))
            -D('.006')*(3*x*x+4*x+4))
        decimal_x, decimal_current = str(x), str(current)
        if abs(decimal_residual) >= D('1e-65'):
            raise AssertionError('Decimal Routh-Hurwitz residual too large')
    current, x = float(decimal_current), float(decimal_x)
    omega = float(np.sqrt(a2(x)))
    eigenvalues = np.linalg.eigvals(model_jacobian(equilibrium_state(current)))
    lam = 1j*omega
    dx_dI = 1/(3*x*x+4*x+4)
    derivative = -dx_dI*(np.polyder(a1)(x)*lam**2
        +np.polyder(a2)(x)*lam+np.polyder(a3)(x))/(3*lam**2+2*a1(x)*lam+a2(x))
    branch = sorted([equilibrium_state(i)[0] for i in [0., 4.]])
    real_roots = sorted(float(v.real) for v in polynomial.r if abs(v.imag)<1e-10)
    in_range = [v for v in real_roots if branch[0] <= v <= branch[1]]
    unique = len(in_range)==1 and abs(in_range[0]-x)<1e-10
    signs = [float(np.max(np.linalg.eigvals(model_jacobian(
             equilibrium_state(current+delta))).real)) for delta in [-1e-6, 1e-6]]
    if not unique or not (signs[0]<0<signs[1]) or derivative.real<=0:
        raise AssertionError('Independent Hopf checks failed')
    return {'method':'Routh-Hurwitz a1*a2=a3; stdlib decimal Newton',
        'decimal_precision':80, 'decimal_iterations':iteration,
        'decimal_x':decimal_x, 'decimal_current':decimal_current,
        'decimal_routh_residual':str(decimal_residual), 'I':current,
        'quartic_coefficients':polynomial.c.tolist(), 'quartic_real_roots':real_roots,
        'equilibrium_x_range':branch, 'unique_hopf_in_range':unique,
        'omega':omega, 'negative_eigenvalue':float(-a1(x)),
        'crossing_derivative':float(derivative.real),
        'max_real_eigenvalue_at_I_minus_plus_1e-6':signs,
        'eigenvalues_real_imag':np.c_[eigenvalues.real,eigenvalues.imag].tolist(),
        'equilibrium_rhs_residual':float(np.max(np.abs(model_rhs(
            0., equilibrium_state(current), current))))}


def refine_cycle(name):
    """Solve flow_T(0,y,z)=(0,y,z), with a positive Poincare crossing."""
    current, crossings = {'spiking':(4.,1), 'bursting':(2.5,3)}[name]
    source = DATA/'convergence'/f'{name}_h0.01.npz'
    with np.load(source) as seed:
        initial_guess = seed['section'][0].copy()
        period_guess = float(seed['spikes'][crossings]-seed['spikes'][0])
    initial_flow = augmented_flow(initial_guess,period_guess,current,CYCLE_SETTINGS)
    initial_closure = float(np.linalg.norm(initial_flow.y[:3,-1]-initial_guess))
    cached_parameters, cached_solution = None, None

    def evaluate(parameters):
        nonlocal cached_parameters, cached_solution
        if cached_parameters is None or not np.array_equal(parameters,cached_parameters):
            start = np.array([0., parameters[0], parameters[1]])
            cached_solution = augmented_flow(start,parameters[2],current,CYCLE_SETTINGS)
            cached_parameters = parameters.copy()
        return cached_solution.y[:,-1]

    def residual(parameters):
        return evaluate(parameters)[:3]-np.array([0.,parameters[0],parameters[1]])

    def derivative(parameters):
        end = evaluate(parameters)
        monodromy = end[3:12].reshape(3,3)
        return np.column_stack([monodromy[:,1]-np.eye(3)[:,1],
            monodromy[:,2]-np.eye(3)[:,2], model_rhs(0.,end[:3],current)])

    seed_parameters = np.r_[initial_guess[1:],period_guess]
    solution = root(residual,seed_parameters,jac=derivative,method='hybr',tol=1e-11)
    start = np.array([0., solution.x[0], solution.x[1]])
    period = float(solution.x[2])
    shooting_residual = float(np.linalg.norm(residual(solution.x)))
    if shooting_residual>=1e-9 or period<=0 or model_rhs(0.,start,current)[0]<=0:
        raise AssertionError(f'Independent shooting failed for {name}: {solution.message}')

    def crossing(t, packed, current):
        return packed[0]
    crossing.direction = 1
    checked = augmented_flow(start,period,current,VERIFY_SETTINGS,
                             dense_output=True,events=crossing)
    end = checked.y[:,-1]
    closure = float(np.linalg.norm(end[:3]-start))
    monodromy = end[3:12].reshape(3,3)
    velocity = model_rhs(0.,end[:3],current)
    # Event-time correction projects onto the x=0 section, then restricts to y,z.
    projection = np.eye(3)-np.outer(velocity,np.array([1.,0.,0.]))/velocity[0]
    poincare = (projection@monodromy)[1:,1:]
    transverse = np.linalg.eigvals(poincare)
    weak_index = int(np.argmax(np.abs(transverse)))
    weak = transverse[weak_index]
    if abs(weak.imag)>1e-8:
        raise AssertionError('Expected a real dominant transverse multiplier')
    weak = float(weak.real)
    flow_multipliers = np.linalg.eigvals(monodromy)
    neutral_error = float(min(abs(flow_multipliers-1)))
    divergence_integral = float(end[12])
    # Liouville gives the product of transverse multipliers, since the autonomous
    # flow multiplier is exactly one. The strongly contracting multiplier cannot
    # be resolved by a direct double-precision monodromy eigenvalue calculation.
    strong_log_abs = divergence_integral-np.log(abs(weak))
    strong_value = (float(np.copysign(np.exp(strong_log_abs),weak))
                    if strong_log_abs>np.log(np.nextafter(0.,1.)) else None)
    times = checked.t_events[0]
    count = int(np.count_nonzero((times>=0.) & (times<period-1e-6)))
    period_residuals = []
    for delta in [-1e-4,-1e-5,0.,1e-5,1e-4]:
        final = (checked.y[:3,-1] if delta==0. else augmented_flow(
            start,period+delta,current,VERIFY_SETTINGS).y[:3,-1])
        period_residuals.append({'delta_T':delta,'period':period+delta,
                                'closure_norm':float(np.linalg.norm(final-start))})
    if closure>=1e-9 or neutral_error>=1e-7 or abs(weak)>=1 or count!=crossings:
        raise AssertionError(f'Refined-cycle validation failed for {name}')
    samples = int(np.ceil(period/.02))+1
    times = np.linspace(0.,period,samples)
    trajectory = checked.sol(times)[:3].T
    return {'regime':name, 'I':current, 'method':'DOP853 + full-state shooting on x=0',
        'seed_source':str(source.relative_to(ROOT)),
        'seed_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'initial_guess':initial_guess.tolist(), 'initial_period':period_guess,
        'initial_closure_norm':initial_closure, 'refined_state':start.tolist(),
        'refined_period':period, 'period_change':period-period_guess,
        'shooting_closure_norm':shooting_residual, 'refined_closure_norm':closure,
        'phase_rhs_x':float(model_rhs(0.,start,current)[0]),
        'root_reported_success':bool(solution.success), 'root_message':str(solution.message),
        'root_function_evaluations':int(solution.nfev),
        'expected_crossings':crossings, 'positive_crossings_per_period':count,
        'monodromy_eigenvalues_real_imag':np.c_[flow_multipliers.real,flow_multipliers.imag].tolist(),
        'poincare_eigenvalues_real_imag':np.c_[transverse.real,transverse.imag].tolist(),
        'neutral_multiplier_error':neutral_error, 'dominant_transverse_multiplier':weak,
        'dominant_transverse_exponent':float(np.log(abs(weak))/period),
        'divergence_integral':divergence_integral,
        'strong_multiplier_log_abs_from_liouville':float(strong_log_abs),
        'strong_multiplier_sign_from_liouville':int(np.sign(weak)),
        'strong_multiplier_from_liouville':strong_value,
        'strong_transverse_exponent_from_liouville':float(strong_log_abs/period),
        'strong_direct_eigenvalue_resolution':'unresolved; report raw values and Liouville log instead',
        'period_residuals':period_residuals,
        'shooting_settings':CYCLE_SETTINGS, 'verification_settings':VERIFY_SETTINGS,
        'observation_dt':float(times[1]-times[0]),
        '_arrays':{'t':times,'state':trajectory,'initial_guess':initial_guess,
            'refined_state':start,'initial_period':period_guess,'period':period,
            'I':current,'monodromy':monodromy,'poincare_matrix':poincare,
            'flow_multipliers':flow_multipliers,'transverse_multipliers':transverse,
            'divergence_integral':divergence_integral,
            'strong_multiplier_log_abs_from_liouville':strong_log_abs,
            'period_perturbations':np.array([r['delta_T'] for r in period_residuals]),
            'period_closure_norms':np.array([r['closure_norm'] for r in period_residuals]),
            'rtol':VERIFY_SETTINGS['rtol'],'atol':VERIFY_SETTINGS['atol'],
            'max_step':VERIFY_SETTINGS['max_step'],'dt':times[1]-times[0]}}


def chaotic_lyapunov():
    """Independent physical warmup, then DOP853 and Householder QR of the frame."""
    settings = CHAOTIC_SETTINGS.copy()
    current = settings['I']
    solver_settings = {key:settings[key] for key in ['rtol','atol','max_step']}
    warmup = solve_ivp(model_rhs,(0.,settings['state_transient']),IC,
        args=(current,),method='DOP853',**solver_settings)
    if not warmup.success or not np.isfinite(warmup.y).all():
        raise RuntimeError('Independent physical warmup failed')
    state = warmup.y[:,-1]
    warmed_state = state.copy()
    frame = np.eye(3)
    qr = settings['qr_interval']
    n_warm = int(round(settings['tangent_transient']/qr))
    n_observe = int(round(settings['duration']/qr))
    n_block = int(round(settings['block']/qr))
    sums, block_sum = np.zeros(3), np.zeros(3)
    divergence_sum, divergence_block = 0., 0.
    cumulative, blocks, states, divergences = [], [], [], []
    samples, sample_times = [], []
    tangent_warmed_state = None
    for k in range(n_warm+n_observe):
        result = solve_ivp(augmented_rhs,(0.,qr),
            np.r_[state,frame.ravel(),0.],args=(current,),method='DOP853',**solver_settings)
        if not result.success or not np.isfinite(result.y).all():
            raise RuntimeError('Independent tangent integration failed')
        final = result.y[:,-1]
        state = final[:3]
        frame, triangular = np.linalg.qr(final[3:12].reshape(3,3))
        norms = np.abs(np.diag(triangular))
        if np.any(norms<=0):
            raise RuntimeError('Independent QR frame lost rank')
        if k==n_warm-1:
            tangent_warmed_state = state.copy()
        if k<n_warm:
            continue
        logarithms = np.log(norms)
        sums += logarithms
        block_sum += logarithms
        divergence_sum += final[12]
        divergence_block += final[12]
        n = k-n_warm+1
        samples.append(state.copy())
        sample_times.append(n*qr)
        if n % n_block==0:
            cumulative.append(sums/(n*qr))
            blocks.append(block_sum/settings['block'])
            states.append(state.copy())
            divergences.append([divergence_sum/(n*qr),divergence_block/settings['block']])
            block_sum[:] = 0.
            divergence_block = 0.
    cumulative, blocks, divergences = map(np.asarray,[cumulative,blocks,divergences])
    spectrum = sums/settings['duration']
    mismatch = float(np.sum(spectrum)-divergence_sum/settings['duration'])
    if not (0<spectrum[0]<.05) or abs(mismatch)>1e-5:
        raise AssertionError(f'Independent short LE check failed: {spectrum}, sum error={mismatch}')
    return {'regime':'chaotic','method':'independent DOP853 + numpy.linalg.qr (Householder)',
        'settings':settings,'initial_state':IC.tolist(),
        'physical_warmup_final':warmed_state.tolist(),
        'tangent_warmup_final':tangent_warmed_state.tolist(),
        'lambda1':float(spectrum[0]),'lambda2':float(spectrum[1]),'lambda3':float(spectrum[2]),
        'block_lambda1_min':float(blocks[:,0].min()),
        'block_lambda1_max':float(blocks[:,0].max()),
        'sum_minus_divergence':mismatch,
        'interpretation':'short independent sign/scale evidence; not an asymptotic estimate to the printed digits',
        '_arrays':{'t':np.array(sample_times),'state':np.array(samples),
            'block_end_time':np.arange(1,len(blocks)+1)*settings['block'],
            'cumulative_exponents':cumulative,'block_exponents':blocks,
            'block_end_states':np.array(states),'cumulative_and_block_divergence':divergences,
            'initial':IC,'physical_warmup_final':warmed_state,
            'tangent_warmup_final':tangent_warmed_state,**settings}}


def save_json(path, value):
    def convert(item):
        if isinstance(item,np.generic):
            return item.item()
        if isinstance(item,np.ndarray):
            return item.tolist()
        raise TypeError(type(item).__name__)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,default=convert)+'\n')


def main():
    DATA.mkdir(exist_ok=True)
    TABLES.mkdir(exist_ok=True)
    results = {'jacobian':check_jacobian(),'hopf':locate_hopf()}
    for name in ['spiking','bursting']:
        results[name] = refine_cycle(name)
        record = results[name]
        print(f'{name}: T={record["refined_period"]:.12g}, closure={record["refined_closure_norm"]:.3g}, '
              f'transverse multiplier={record["dominant_transverse_multiplier"]:.9g}',flush=True)
    results['chaotic'] = chaotic_lyapunov()
    settings = {'model':'Hindmarsh-Rose','parameters':{'a':1.,'b':3.,'c':1.,'d':5.,
        'r':.006,'s':4.,'x_R':-1.6},'units':'HR model units',
        'independence':'no imports from cnm.py; separately spelled equations and Jacobian; DOP853 and Householder QR',
        'cycle_settings':CYCLE_SETTINGS,'verification_settings':VERIFY_SETTINGS,
        'chaotic_settings':CHAOTIC_SETTINGS,'hopf_arithmetic':'stdlib decimal, precision80',
        'versions':{'numpy':np.__version__,'scipy':scipy.__version__,
                    'python':sys.version,'platform':platform.platform()},
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for name,value in results.items():
        arrays = value.pop('_arrays',None)
        if arrays is not None:
            np.savez_compressed(DATA/f'independent_hr_{name}.npz',**arrays)
    save_json(DATA/'independent_hr_results.json',results)
    save_json(DATA/'independent_hr_settings.json',settings)
    rows = [
        {'check':'jacobian_complex_step','regime':'all','passed':True,
         'residual':results['jacobian']['max_abs_error']},
        {'check':'hopf_routh_decimal','regime':'equilibrium','I':results['hopf']['I'],
         'decimal_I':results['hopf']['decimal_current'],'passed':True,
         'residual':results['hopf']['equilibrium_rhs_residual'],
         'crossing_derivative':results['hopf']['crossing_derivative']}
    ]
    for name in ['spiking','bursting']:
        value = results[name]
        rows.append({'check':'periodic_orbit_shooting','regime':name,'I':value['I'],
            'passed':True,'initial_period':value['initial_period'],
            'period':value['refined_period'],'initial_residual':value['initial_closure_norm'],
            'residual':value['refined_closure_norm'],
            'neutral_multiplier_error':value['neutral_multiplier_error'],
            'dominant_transverse_multiplier':value['dominant_transverse_multiplier'],
            'dominant_transverse_exponent':value['dominant_transverse_exponent'],
            'strong_log_abs_multiplier_liouville':value['strong_multiplier_log_abs_from_liouville'],
            'strong_exponent_liouville':value['strong_transverse_exponent_from_liouville'],
            'positive_crossings':value['positive_crossings_per_period']})
        for record in value['period_residuals']:
            rows.append({'check':'period_perturbation','regime':name,'I':value['I'],
                'delta_T':record['delta_T'],'period':record['period'],'residual':record['closure_norm']})
    value = results['chaotic']
    rows.append({'check':'short_independent_lyapunov','regime':'chaotic','I':3.25,
        'passed':True,'lambda1':value['lambda1'],'lambda2':value['lambda2'],
        'lambda3':value['lambda3'],'duration':CHAOTIC_SETTINGS['duration'],
        'residual':value['sum_minus_divergence'],
        'interpretation':value['interpretation']})
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (TABLES/'independent_hr_checks.csv').open('w',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print('chaotic short independent spectrum:',[value[f'lambda{j}'] for j in [1,2,3]],flush=True)
    print('Independent HR evidence saved; no existing data/source files were updated.',flush=True)


if __name__=='__main__':
    main()
