# BF16 with checkpoint-growth integration

Product source tested: 7f3680fb2eb7031872c08108783d6e2d766a3a17; baseline 3472cf15aceb0c7474046bb6a51c7f4892f937d9.
Hardware: one NVIDIA GB10 on Spark. Correctness window: 20261113T023656Z-b1e470.
The shared chat model was paused for these checks and restored afterward; other GPU
contexts could remain. This is not a timing or speed comparison.

The numerical gate passed 26 checks. The serving gate passed
122 checks with zero failures. The serving wrapper was invoked with a deliberately
inherited SAMPLE_VERB=gsample while VERB=gserveb; it selected gsampb as required.
Commands, input hashes, exits and output hashes are in result.json. The serving
summary retains its outcomes and original full-log digest without process IDs.

This reruns the qualified mixed-BF16 policy after incorporating the landed
checkpoint-growth work. WO and FFN projections use BF16 operands; sensitive
operations and accumulation retain f32. No numerical threshold was changed.
Legacy full-BF16 tolerance failures are retained as expected controls.
The earlier evidence under bf16-inference-20261113 remains historical evidence.

This does not promote the newer trained checkpoint, establish a universal error
bound, prove a speedup, or convert stored model weights to BF16. The existing
golden checkpoint remains the deployment input.

Public aliases: $WORK is the qualification workspace, $MODELS the model store,
and $R11B the new-checkpoint workspace. Only paths are substituted in result.json
and repro.log. Binary compiler stdout is omitted; its hash remains recorded.
Original logs and artifacts are retained in the qualification workspace.
