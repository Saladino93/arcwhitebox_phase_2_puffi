#!/bin/zsh
# all variants, at most PAR metered workers at a time. Usage: run_metered3.sh <ablation dir> [PAR]
A=$1; PAR=${2:-2}
ls $A/roots | while read v; do [ -s $A/met3/$v.json ] || echo $v; done | xargs -P $PAR -n 1 zsh $A/worker_metered3.sh $A
echo ALL_DONE
