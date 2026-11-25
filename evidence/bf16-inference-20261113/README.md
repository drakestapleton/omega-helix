# Mixed BF16 inference qualification on GB10
Product source tested: 8ed0711f234797c62414e78b882ce1ba979e41e2; baseline 7b4445ecaa2719cea458d591e4a90d9310bbdc17.
One NVIDIA GB10 on Spark, shared window 20261113T014127Z-50a301, quiet=none.
This is a correctness run, not a timing comparison. No services were stopped.
The build was repeated from a fresh clone and produced identical source and executable bytes.

Commands and input hashes: result.json. Compiler hash is pinned by tests/omegallm/build.sh.
Run: SPARK_WINDOW supplied by spark-window run, CPU3 nice19, python3 tests/omegallm/bf16-inference.py with candidate, baseline, compiler, concatenated source, both checkpoints and tokenizer named in result.json.
26 checks passed, including 8 comparisons of 250 steps, exact f32 and legacy BF16 regressions, dispatch counts, sampler repeatability and a compiled no-BF16 control.
Full numerical outputs are retained here; failures abort the gate. Existing 0.25-nat tolerance is unchanged.

Serving: VERB=gserveb PORT=18591 TPORT=18592 CORES=3 sh tests/omegallm/serve-check.sh, with OLLM=the same candidate, CK=golden and MERGES=the same tokenizer. All 122 checks passed. Sanitized summary and original log hash are retained; full local log is in $WORK/llm-checks/serve-r1.log.

The new path uses BF16 operands for attention-output and FFN projections; accumulation, attention and output logits retain f32. This evidence covers the supplied checkpoints and prompts only. It does not prove a universal error bound, performance improvement, all-BF16 storage, or final CPU qualification of the newer checkpoint. Existing production golden weights remain the deployment input.

Public path aliases: $WORK is the qualification workspace; $MODELS is the local model store; $R11B is the rung-11b training workspace. Original machine paths are omitted. The complete original logs and result.json remain in the qualification workspace; artifact content hashes are unchanged. Path substitutions are the only edits to the published raw result and build transcript.

The published serving summary omits process IDs and listener details; its loopback check result and the original full-log hash are retained.
