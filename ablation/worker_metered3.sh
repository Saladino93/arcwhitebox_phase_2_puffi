#!/bin/zsh
# one variant: metered (FlopScope 0.12.3 / whestbench 0.16.2) raw + GF on nets 600-602. Usage: worker_metered3.sh <ablation dir> <variant>
A=$1; v=$2; W=/Users/omard/Documents/projects/AI_projects/whest-ship; cd $W
env -u VIRTUAL_ENV uv run --project $W python $W/scripts/run_nets.py --estimator $A/roots/$v/estimator.py --nets 600-602 --no-residual-limit --out $A/met3/$v.json > $A/met3/$v.out 2>&1
echo "done $v $(grep -E '^POOLED' $A/met3/$v.out)"
