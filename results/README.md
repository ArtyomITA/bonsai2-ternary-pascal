# Runs

`results.tsv`: every run of the research, as recorded by the research kernel (descriptions in Italian).
Runs 1-18 are attempts (keep = new best); runs 100001+ are evidence runs measured against the best.

Invalid (stale binary, the build did not recompile ggml-cuda.cu): 100002, 100004, 100006, 100009.
Valid ablations: 100001 (without #3), 100003 (without #5), 100005 (without #2), 100013 (without #4, measured with the
machine in use, not comparable). The contribution of #1 comes from an A/B outside the evaluator (same thermal state):
69.48 vs 59.35 tok/s.
Baselines: 2 (morning), 100007 (cold), 100008 (warm). Control: 100010 (attempt 9 re-measured).
Runs 1, 3, 5, 11, 12, 13, 17, 100011, 100012 crashed on the prefill guard or the quality gate (see the README).
