#!/bin/zsh
# all variants, at most PAR workers at a time (xargs -P). Usage: run_numpy30.sh <ablation dir> [PAR]
A=$1; PAR=${2:-3}
ls $A/roots | while read v; do [ -s $A/np30/$v.json ] || echo $v; done | xargs -P $PAR -n 1 zsh $A/worker_numpy30.sh $A
echo ALL_DONE
