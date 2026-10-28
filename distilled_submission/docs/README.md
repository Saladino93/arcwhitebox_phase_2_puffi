# Documents behind the distilled program

- `SPEC.md`: the estimator's equations in loop order. Line references of the form `M:123`, `B:456`, `K:78`, `MF:`, `MM:` point
  into the readable non-Strassen rewrite of ex_37 (`estimator_plain/`: moments.py, bank.py, pk4.py, moment_formulas.py,
  moment_math.py), which is archived in the team's research repository; `E:` references point into `../estimator.py`.
- `CENSUS.md`: FLOP and memory census of the dense design, and the reconciliation of the measured tallies.
- `REVIEW.md`: line-by-line review of `../estimator.py` against SPEC.md and the graded source.
- `DISCREPANCIES.md`: places where the program, SPEC.md and the graded source disagree, and which one was followed.
- `SOURCE_MAP.md`: mechanism -> file:function:line map of the readable rewrite, with the classification of its code into
  exact economy, lossy economy, statistics and guards that guided the distillation.
- `ABLATION_EVIDENCE.md`: recorded effect on MSE and FLOPs of removing each mechanism (from the team's development notes).
- `K3_REFERENCE_RUNS.md`: runs of `../k3_reference.py`, including the memory fix history.
- `k3_dense_oracle_width64.py`: exact dense numpy implementation of ARC's K=3 SIMPLE (holds the n^3 tensor; width <= 64),
  the oracle `../k3_reference.py` was validated against.
