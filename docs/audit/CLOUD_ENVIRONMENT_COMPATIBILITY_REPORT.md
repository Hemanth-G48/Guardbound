# CLOUD_ENVIRONMENT_COMPATIBILITY_REPORT

Diagnosis and repair of the PyTorch / torchvision incompatibility that blocked
`transformers`, `sentence-transformers` and therefore the NBF embedder.

Measured on 2026-11-10. Everything below is re-measured for this audit; nothing
is carried over from `CLOUD_READINESS_REPORT.md`.

---

## 1. Initial environment (measured)

| Item | Value |
|---|---|
| OS / arch | Windows-11-10.0.26200-SP0 / AMD64 |
| Python | 3.13.9 (Anaconda, MSC v.1929 64-bit) |
| Executable | `C:\Users\CSLAB\anaconda3\python.exe` |
| Environment | conda **base**, no venv (`sys.prefix == sys.base_prefix`) |
| GPU present | NVIDIA RTX 4500 Ada Generation, 24570 MiB, driver 595.79, CUDA 13.2 |
| **torch** | **2.14.1+cpu** — `torch.version.cuda = None`, `torch.backends.cuda.is_built() = False`, `torch.cuda.is_available() = False` |
| **torchvision** | **0.22.1+cu118** |
| torchaudio | 2.7.1+cu118 |
| transformers | 5.16.1 |
| sentence-transformers | 6.0.1 |
| numpy | 2.3.5 |
| openai / httpx | 2.54.0 / 0.28.1 |
| bitsandbytes | 0.50.2 (requires `torch`, unpinned) |

Each import was probed in a **separate interpreter**, so no failure could mask
another.

| Probe | Result |
|---|---|
| `import torch` | OK |
| `import torchvision` | **FAIL** — `RuntimeError: operator torchvision::nms does not exist` |
| `torchvision.ops.nms(...)` | **FAIL** — same error |
| `import transformers` | OK (top-level only) |
| `from transformers import PreTrainedModel` | **FAIL** — `ModuleNotFoundError: Could not import module 'PreTrainedModel'` |
| `import sentence_transformers` | **FAIL** — same, via `transformers.PreTrainedModel` |

---

## 2. Root cause

Not an "ABI mismatch" in the abstract — a **declared, unmet version pin plus a
CPU/CUDA build mismatch**. `python -m pip check` states it directly:

```text
torchaudio  2.7.1+cu118 has requirement torch==2.7.1+cu118, but you have torch 2.14.1.
torchvision 0.22.1+cu118 has requirement torch==2.7.1+cu118, but you have torch 2.14.1.
```

Wheel metadata confirms it:

```text
torchvision-0.22.1+cu118.dist-info  -->  Requires-Dist: torch (==2.7.1+cu118)
torchaudio-2.7.1+cu118.dist-info    -->  Requires-Dist: torch==2.7.1+cu118
torch-2.14.1.dist-info              -->  installed, built for CPU only
```

Failure chain:

1. `torchvision/__init__.py` imports `_meta_registrations`, which registers a fake
   implementation for the op `torchvision::nms`.
2. `torchvision::nms` is provided by the compiled extension `torchvision._C`,
   built against `torch==2.7.1+cu118`.
3. The installed `torch` is `2.14.1+cpu`: wrong version **and** no CUDA runtime.
   The compiled extension cannot register its ops, so the dispatcher reports the
   operator as non-existent.
4. `transformers.is_torchvision_available()` probes `import torchvision`, which
   now raises. `transformers` degrades `PreTrainedModel` to a lazy placeholder,
   so `from transformers import PreTrainedModel` raises.
5. `sentence-transformers` imports `PreTrainedModel` at module load → the NBF
   embedder cannot be constructed.

**Also established:** CUDA is **not** available to PyTorch at all. The frozen
local-model configurations specify `device: cuda`, so local HF model serving is
already non-functional *before* any change made here. Nothing in this repair can
remove GPU capability that currently exists.

---

## 3. Proposed repair — recorded BEFORE any package change

Resolution evidence gathered without installing (pip dry-run against the CPU
index):

```text
$ python -m pip install --dry-run --report - "torchvision" \
      --index-url https://download.pytorch.org/whl/cpu
Would install torchvision-0.29.2+cpu
  torchvision 0.29.2+cpu  requires: ['torch (>=2.14.0)']   <-- satisfied by torch 2.14.1
```

The installed `0.22.1+cu118` requires `torch==2.7.1+cu118` (**unsatisfied**).
The available `0.29.2+cpu` requires `torch>=2.14.0` (**satisfied**) and is a CPU
build, matching a CPU-only torch. pip proposes to install **only** torchvision —
it does not want to change torch.

| Package | Current | Proposed | Reason |
|---|---|---|---|
| torch | 2.14.1+cpu | **keep** | Changing torch would be a large, shared-environment change. Nothing declares `torch==2.14.1` as a hard requirement except this install itself; `torchvision 0.29.2` accepts `>=2.14.0`. |
| torchvision | 0.22.1+cu118 | **0.29.2+cpu** | 0.22.1 pins `torch==2.7.1+cu118` (unmet, and cu118 against a CPU torch). 0.29.2+cpu satisfies the installed torch and matches its CPU build. This is the minimal change that restores `torchvision::nms` and therefore `transformers`. |
| torchaudio | 2.7.1+cu118 | **remove** (amendment below) | See §3.1. Its native library cannot load at all, and it is not required by this project or by `transformers` / `sentence-transformers`. |
| transformers | 5.16.1 | **keep** | Fails only because torchvision fails. No version change needed. |
| sentence-transformers | 6.0.1 | **keep** | Fails only via `transformers.PreTrainedModel`. No version change needed. |

### 3.1 Amendment after the torchvision repair (second blocker found)

Once `torchvision` was repaired, the next failure was previously hidden behind it:

```text
$ python -c "import torchaudio"
  File "...\torchaudio\_extension\utils.py", line 60, in _load_lib
    torch.ops.load_library(path)
FileNotFoundError: Could not find module
  'C:\Users\CSLAB\anaconda3\Lib\site-packages\torchaudio\lib\libtorchaudio.pyd'
  (or one of its dependencies)
```

Import chain (measured):

```text
sentence_transformers.base.modules.transformer
  -> transformers (lazy attribute)
  -> transformers.models.auto.processing_auto
  -> transformers.image_processing_utils
  -> torchaudio  <-- native library fails to load
```

`libtorchaudio.pyd` is a `cu118` extension; its dependencies include the CUDA
torch DLLs, which do not exist in a `+cpu` build. `transformers` probes
torchaudio as an *optional* dependency and handles `ImportError`, but a missing
DLL surfaces as `FileNotFoundError` (an `OSError`), so the probe does not catch
it and the whole import chain aborts.

Why removal rather than repair:

* the CPU index has no `torchaudio` build for torch 2.14.1 — its latest CPU
  release is `2.11.1+cpu`, which pins `torch==2.11.1` and would **downgrade
  torch** (a much larger change);
* `torchaudio` is not imported by Guardbound anywhere (`grep` over `src/`,
  `scripts/`, `tests/`: zero matches), is not a dependency of `transformers`
  (`pip show transformers` lists none), and is not a dependency of
  `sentence-transformers`;
* the installed copy is already 100 % non-functional — its library cannot be
  loaded by any code path. Removing it costs no capability;
* it is trivially reversible (`pip install torchaudio==2.7.1+cu118` restores the
  current, broken state; the working state requires the CUDA stack in §3.3).

Scope note: this touches the conda `base` environment only. The five other conda
environments on this machine have their own `site-packages` and are unaffected.

### 3.2 Consequence of the CPU-only stack

CUDA remains unavailable (`torch` stays `+cpu`). NBF inference is unaffected — a
768-d MLP plus embeddings runs correctly on CPU. Local HF model serving stays
impossible until a CUDA stack is restored (§3.3).


### Command to be run

```bash
python -m pip install "torchvision==0.29.2+cpu" \
    --index-url https://download.pytorch.org/whl/cpu
```

### Explicitly NOT done (requires separate approval)

Restoring GPU capability is a different, much larger decision — it replaces the
framework in a shared conda `base` environment (~2.5 GB) that other projects use:

```bash
# NOT RUN. For approval only.
python -m pip install "torch==2.7.1+cu118" "torchvision==0.22.1+cu118" \
    "torchaudio==2.7.1+cu118" --index-url https://download.pytorch.org/whl/cu118
```

That would restore the CUDA stack the frozen configs assume and satisfy pip check
for all three packages, at the cost of replacing the framework in a shared
environment (~2.5 GB). §6.2 supersedes this sketch with the *measured* set of
available CUDA builds; read it before acting on anything here.

---

## 4. Exact repair, as executed

Two changes, applied in this order. Both are recorded here before/with the
evidence that motivated them.

### 4.1 Install the CPU torchvision that matches the installed torch

```bash
python -m pip install "torchvision==0.29.2+cpu" \
    --index-url https://download.pytorch.org/whl/cpu
```

Observed output: `Successfully uninstalled torchvision-0.22.1+cu118` →
`Successfully installed torchvision-0.29.2+cpu`. No other package changed; pip
reported every `torch` requirement as already satisfied.

### 4.2 Remove the unloadable torchaudio

```bash
python -m pip uninstall -y torchaudio
```

Reasoning is in §3.1. `torchaudio` was already 100 % non-functional (its own
library could not load), is not imported anywhere in Guardbound, and is not a
dependency of `transformers` or `sentence-transformers`.

### 4.3 What was NOT changed

`torch` (2.14.1+cpu), `transformers` (5.16.1), `sentence-transformers` (6.0.1),
`numpy`, `openai`, `httpx`, and every other package. The NBF architecture,
checkpoint, embedding model, scoring formula, threshold and filtering logic are
untouched.

---

## 5. Post-repair validation (measured)

### 5.1 Package consistency

```text
$ python -m pip check
aiobotocore 2.25.0 has requirement botocore<1.40.50,>=1.40.46, but you have botocore 1.43.116.
streamlit 1.51.0 has requirement pillow<13,>=7.1.0, but you have pillow 13.0.0.
streamlit 1.51.0 has requirement protobuf<7,>=3.20, but you have protobuf 7.36.2.
```

The `torchvision` conflict is **gone**. The `torchaudio` conflict is **gone**
(package removed). The three remaining lines are pre-existing, unrelated to this
project's dependency set (aws/streamlit), and were present before the repair.

### 5.2 Imports (each probed in a separate interpreter)

| Probe | Before | After |
|---|---|---|
| `import torch` | OK | OK (`2.14.1+cpu`) |
| `import torchvision` | **FAIL** `operator torchvision::nms does not exist` | **OK** (`0.29.2+cpu`) |
| `torchvision.ops.nms(...)` | **FAIL** | **OK** → `keep=[0]` |
| `from transformers import PreTrainedModel` | **FAIL** `ModuleNotFoundError` | **OK** |
| `import sentence_transformers` | **FAIL** via `PreTrainedModel`, then via `torchaudio` | **OK** (`6.0.1`) |
| `import torchaudio` | FAIL (unloadable native lib) | n/a — removed |
| PyTorch CUDA | unavailable | unavailable (unchanged; see §6) |

### 5.3 NBF embedder — real 768-dimensional embedding

```text
model                    sentence-transformers/all-mpnet-base-v2
max_seq_length           384
embedding shape          [1, 768]
embedding dtype          float32
all values finite        True
L2 norm                  1.0
deterministic (same input twice)   True
batch == single                    True
repo get_embed_fn shape            [1, 768]
```

### 5.4 NBF checkpoint and inference

```text
path            nbf_original_stuff/.../models/models_best_nbf_released.pth
bytes           11,757,138
sha256          cea1a75bcef4fc515814b69c42541c95114f587abcc4505c9b096bbfa2a136fe
sha256 matches  True   (unchanged from the validated identity)
predictor       SafetyPredictor  state_dim=768  embedding_dim=768  params=50,405
dynamics        params=2,887,168
h(zeros, embed) = 0.0003733634948730469   (deterministic across repeated calls)
class probs     [0.144451, 0.054219, 0.322583, 0.155790, 0.322956]  sum = 1.0
score formula   p(class5) - max(p(class1..4)) == h   -> True
state after advance (real dynamics): shape [1,768], norm 0.581158
filter_query(eta=0) for that query -> allowed=False  (h = +0.00037 >= 0)
```

The barrier is not degenerate: it returns a finite, deterministic score in
`[-1, 1]` and advances the state through the real dynamics.

### 5.5 Test suites that were previously blocked

| Suite | Before | After |
|---|---|---|
| `tests/parity` | 36 failed (import error) | **41 passed** |
| `tests/test_nbf_checkpoint.py` + `tests/test_crescendo_integration.py` + `tests/test_phase17_stage2_infrastructure.py` | 4 failed, 4 errors | **98 passed** |
| Whole suite | 758 passed / 44 failed / 4 errors | **1009 passed / 4 failed / 0 skipped** (1013 collected) |

Failures that disappeared: 40 (36 parity + 4 integration) plus the 4 collection
errors. The 4 remaining are analysed in §6.

> Counts here are the post-repair measurement. The authoritative, reconciled
> timeline (806 → 926 → 926 → 1013 collected) and the per-node-ID comparison are
> in `CLOUD_FINAL_READINESS_RECONCILIATION.md` §1–2.

---

## 6. Remaining failures, and the GPU decision

### 6.1 The 4 remaining failures are environmental, not code defects

| Test | Cause | Would a GPU fix it? |
|---|---|---|
| `test_phase14_2_rotation.py::TestPreRunGuardReleasesCacheFirst::test_releases_cache_before_measuring` | `ModelManager.pre_run_check` gates its cache release behind `if torch.cuda.is_available():`. With a CPU-only torch the release never runs, so the ordering assertion fails. | **Yes** |
| `...::test_effective_free_after_release_is_what_is_measured` | Same gate, same cause. | **Yes** |
| `test_phase8_architecture.py::TestThreeModelConfigValidation::test_default_config_valid` | `configs/reproduction_three_model.yaml` names `Qwen/Qwen3.5-4B`, which is not in the local HF cache, so `validate_config` correctly reports it. | No — needs the weights |
| `...::TestConfiguredModelsPresentOnDisk::test_attacker_present` | Asserts that model's weights are on disk. | No — needs the weights |

Neither cause is related to the cloud work, and neither was introduced here: all
four were in the 48-item Phase-0 baseline.

### 6.2 Restoring CUDA requires a downgrade — decision required

Measured availability:

```text
download.pytorch.org/whl/cu128   torch        -> latest 2.11.0+cu128
                                 torchvision  -> latest 0.26.0+cu128
```

There is **no CUDA build of torch 2.14.1**. Restoring GPU therefore means
downgrading `torch` from 2.14.1 to at most 2.11.0 and `torchvision` from 0.29.2
to 0.26.0 — a change to the framework version for every project in the shared
conda `base` environment (~2.5 GB download).

```bash
# NOT RUN. Requires approval.
python -m pip install "torch==2.11.0+cu128" "torchvision==0.26.0+cu128" \
    --index-url https://download.pytorch.org/whl/cu128
```

**Effect:** restores CUDA for the RTX 4500 Ada, makes the two
`test_phase14_2_rotation` guards evaluable, and makes local HF model serving
possible again. **Cost:** a three-minor-version downgrade of `torch` in a shared
environment, and a re-validation of every local-model experiment path.

**Recommendation:** not required for the cloud qualification or the cloud
experiment — the NBF runs correctly on CPU (verified in §5.4) and the cloud
provider does not use torch at all. Defer this until local model serving is
needed again, and prefer a dedicated conda environment over mutating `base`.
