# Pilot environment observation — concurrent GPU job

Recorded during the Stage 4.8 pilot run, at the time of the third run. This is an
observability note, not a deviation from the frozen protocol.

## Observation

A **separate, pre-existing GPU job belonging to the operator** was already running when
the pilot started:

```text
PID 21484   C:\Users\CSLAB\anaconda3\envs\249gpu\python.exe
            run_stabilization.py --steps 20000
              --variants policy_memory_detection policy_memory_gpm_detection
              --reference-seed-size 50 --seed 123
started     2026-10-01 01:11:53 (≈ 3 h before the pilot)
activity    ~98 % of one CPU core continuously; GPU utilisation 41 %, memory 17 %
footprint   ~0.6 GiB VRAM
```

It is not part of this phase and was **not** started, stopped or modified by the pilot.

## Effect on the pilot

- **VRAM is not contended**: the pilot's attacker needs ~16.7 GiB, the other job holds
  ~0.6 GiB, total ≈ 17.3 GiB of 23.99 GiB. Both fit; no OOM occurred and the pipeline's
  own VRAM guards passed on every run.
- **Wall-clock is affected**: the first two pilot runs took 659.8 s and 618.0 s, versus
  the 536.0 / 509.3 / 454.2 s measured for the same configuration in the Stage 4.7 smoke
  test (mean 499.8 s). That is roughly a 25-30 % slowdown from GPU compute contention,
  and it is why the live ETA opened at ~16 h rather than the ~12.5 h projected from the
  smoke runs.

## Why this does not affect the results

Run duration is an engineering measurement, not an experimental variable. The frozen
protocol — model, revision, quantization, temperature, top_p, prompt, seeds, attacks,
target, judge, NBF state, max_turns, success criterion — is untouched, and the sampling
per run is unaffected by how long the GPU takes to produce it. The effect is confined to
the wall-clock columns of the analysis and to the ETA.

## Consequence for interpretation

Mean/median run duration and throughput in the Stage 4.8 analysis must be read as
**measured under contention**, not as the configuration's standalone speed. The Stage 4.7
smoke numbers (499.8 s/run) remain the uncontended reference.
