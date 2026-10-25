#!/usr/bin/env python3
"""Regenerate the study's simulations, analyses, figures and synthesis tables."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent


def run(*args):
    print('+',sys.executable,*map(str,args),flush=True)
    subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,check=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    os.environ.setdefault('MPLCONFIGDIR','/tmp/cnm_mpl')
    os.environ.setdefault('NUMBA_CACHE_DIR','/tmp/cnm_numba')
    os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
    os.environ.setdefault('OMP_NUM_THREADS','1')
    run('run.py','simulate')
    run('run.py','analyze')
    run('independent_hr_checks.py')
    run('reconstruction_refinement.py')
    run('optional/run.py','--core-only')
    run('optional/forced_lyapunov.py')
    run('ablation_checks.py')
    run('noise_statistics.py')
    run('library_validation.py')
    run('synthesis.py')
    run('publication_figures.py')
    run('optional/plot_reliability.py')
    print('Scientific calculations regenerated. Rerun research.ipynb to display the current results.')


if __name__=='__main__':main()
