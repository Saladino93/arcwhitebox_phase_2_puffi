#!/bin/zsh
# one variant: 30-net (600-629) numpy-shim raw MSE, 2 BLAS threads. Usage: worker_numpy30.sh <ablation dir> <variant>
A=$1; v=$2; W=/Users/omard/Documents/projects/AI_projects/whest-ship; cd $W
env -u VIRTUAL_ENV uv run --project $W python $W/scripts/numpy_harness.py --root $A/roots/$v --nets 600-629 --threads 2 --out $A/np30/$v.json > $A/np30/$v.out 2>&1
echo "done $v $(grep -E '^POOLED' $A/np30/$v.out)"
