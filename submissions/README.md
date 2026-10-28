# Graded Phase 2 submissions

Each folder is the exact source of one graded upload, named by its AIcrowd submission number: `estimator.py` (the moment
engine), `bank.py` (the third-cumulant source bank and its fourth-cumulant reads), `pk4.py` (tree-level fourth cumulants),
`fmm.py` (the Strassen-Winograd matrix-multiplication engine), plus `README.md` (audit), `manifest.json`, `SHA256.txt`
(hashes of the uploaded files) and `platform_receipt.json`. Graded on whestbench 0.16.x with FlopScope 0.12.3, 100 public
networks each, score = raw final-layer MSE x max(0.1, FLOPs / 2^41).

## MAIN submissions: the two selected for the final score

| selected | official score (adjusted) | raw final-layer MSE | FLOP budget used | uploaded | role |
|---|---:|---:|---:|---|---|
| **[337631](337631/)** | **1.8608e-9** | 1.2106e-8 | 15.36 % | 21 Oct 2026, 21:44 | **best adjusted score** (the competition metric) |
| **[337577](337577/)** | 1.8671e-9 | **1.1741e-8** | 15.89 % | 21 Oct 2026, 17:26 | **lowest raw MSE** (most accurate estimate) |

Both completed 100/100 networks with no failures. They differ by exactly one line of `bank.py`, the rank cap of the shared
elder basis that compresses old third-cumulant sources: 320 in 337577, 256 in 337631. `estimator.py`, `pk4.py` and `fmm.py`
are byte-identical between them. The lower cap saves 0.5 % of the FLOP budget and costs 3 % raw MSE, which is a net gain under
the adjusted metric, so 337631 is the better score and 337577 the better estimator.

## The other two graded uploads of the same family

| submission | official score | raw final-layer MSE | FLOP budget used | what it is |
|---|---:|---:|---:|---|
| [337479](337479/) | 1.8841e-9 | 1.1742e-8 | 16.04 % | the base: shared elder basis (cap 320), slice row repair, exact switch restructuring |
| [337563](337563/) | 1.8783e-9 | 1.2106e-8 | 15.50 % | 337479 with the elder cap at 256 |

337577 is 337479 with a cheaper final Strassen combine (10 lines of `fmm.py`); 337631 is 337577 with 337563's cap. The
readable, trick-free rewrite of this family is in [`../distilled_submission/`](../distilled_submission/).
