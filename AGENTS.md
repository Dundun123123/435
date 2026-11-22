# Project instructions for Codex

Read README.md, docs/HANDOFF.md and docs/EXPERIMENT_PROTOCOL.md before changing code. Read docs/MPI_TWO_NODES.md or docs/GPU_BACKEND.md for the selected extension.

- Verified code includes CPU v3, a single-host MPI implementation, the CPU-tokenization/CUDA-numeric GPU backend, and a two-host CPU backend using authenticated TCP. The two-host result is not an MPI measurement; use the measured results and correctness checks as evidence of execution.
- Preserve existing sequential/parallel commands and compute_tfidf's three-value return contract. Reuse count_batch and weight_batch wherever applicable. Optional MPI/GPU imports must not break CPU-only machines.
- Preserve regex [a-z]+, lowercase handling, raw TF, smoothed global IDF and full-vector L2 normalization. Empty documents count toward N; all-empty vocabulary raises a clear error. Restore input order and alphabetic top-k tie breaking.
- Data and experiments: no fabricated timings, no artificial sequential delays, no silently replicated corpora. Existing published CSVs and figures are historical evidence: never overwrite them when developing a new backend. New results belong under results/local/ until reviewed for publication.
- Timing must include relevant communication, transfer and reconstruction. GPU kernels are asynchronous; synchronize correctly. Report CPU+GPU hybrid preprocessing honestly. Kernel-only and end-to-end measurements are different experiments.
- A two-node MPI result needs distinct actual host identities and verified placements. Multiple ranks on one laptop or VMs on one laptop do not constitute two physical machines.
- Never commit .venv, raw corpora, download caches, tokens, private keys, real host inventories or unreviewed machine diagnostics. No hardcoded original-machine paths. Use committed download scripts and manifests for data reproduction.
- Validate CPU compatibility with `python -m unittest discover -s tests -v`. Run `python verify.py` after preparing Newsgroups. Hardware-dependent checks must be explicitly skipped/unavailable when hardware is missing, not reported as passed.
- Use a separate branch for each extension. Do not add a project LICENSE that purports to relicense dataset content. Keep sources and citations.
- Only make changes necessary for the selected task. Do not change OS firewall, credentials or install system-wide drivers without the user's applicable authorization. Routine project-local code and environment work remains within the task scope.

