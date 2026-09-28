# Phase 14 — System Restart: Root-Cause Investigation

**Investigation date:** 2026-09-23
**Machine:** `DESKTOP-PVCRPVT` — HP Z2 Tower G1i Workstation
**Trigger:** system rebooted ~2 minutes after starting the Phase 14 local inference / performance-audit workload
**Method:** diagnostic only — no production code was modified
**Related:** `PHASE14_CODE_FIDELITY_AUDIT.md`, `results/phase14/`, `diagnose_crash.ps1`

---

## 0. TL;DR

The machine is **not** running out of memory, VRAM, disk, CPU, or power, and Python is **not**
crashing. It is taking a **kernel bugcheck**:

```
0x00020001  HYPERVISOR_ERROR
```

Microsoft: *"the hypervisor has encountered a fatal error."*
WER's own bucket names identify two distinct detection paths of **one underlying fault**:

| param1 | WER bucket (verbatim) | Count |
|---|---|---|
| `0x28` | `INTEL_IOMMU_TIMEOUT_IMAGE_GenuineIntel.sys` | 7 |
| `0x13` | `NOBLOB_HYPERVISOR_ERROR_EXCESSIVE_WAIT_WHILE_SPINNING_IMAGE_hvix64.exe` | 4 |

**Failure chain:**

```
igdkmdn64.sys   (Intel Arrow Lake iGPU KMD, v32.0.101.8132, dated 2025-09-19 ≈ 1 year old)
  └─ 177 recoverable GPU stalls logged since 2026-05-31
     (148 TDR:6 KMD · 14 TRINITY11_BBHANG · 10 TDR:3 DISPLAY · 2 RINGHANG · 2 GUC_SCHEDULER_ERROR · 1 ENGINE)
     └─ the hung Intel DMA path stalls the Intel IOMMU (VT-d)
        └─ VBS has "Iommu Protection" ENABLED ⇒ the Hyper-V hypervisor owns the IOMMU
           └─ hvix64.exe detects the stall and dies fatally:
              ├─ INTEL_IOMMU_TIMEOUT                    (param1 = 0x28)
              └─ EXCESSIVE_WAIT_WHILE_SPINNING          (param1 = 0x13)   ← the 2026-09-22 crash
                 └─ BSOD 0x00020001 HYPERVISOR_ERROR → reboot
```

The Phase 14 workload is a **trigger/amplifier, not the cause**: it puts heavy load on the
same graphics subsystem (NVIDIA CUDA + the Intel iGPU driving display/compositor/browser),
raising the probability of hitting the hang → IOMMU stall → hypervisor death.

**The defect predates Phase 14 by ~2 months.** First hypervisor crash: **2026-05-21**.
OS install date: **2026-04-24**. First GPU stall: **2026-05-31**.

**Primary fix:** update the Intel graphics driver `32.0.101.8132` → current Intel release
(`32.0.101.9030`, 2026-09-17, Arrow Lake-S supported).

> **UPDATE 2026-09-23 — see §14.** The BIOS, NVIDIA, Wi-Fi/Bluetooth/Ethernet and chipset updates
> landed, but **the primary fix did not**: `32.0.101.9030` is staged in the driver store yet the
> iGPU is still bound to `oem24.inf` = `32.0.101.8132`, because the installer's "Intel Graphics
> Software" component failed (exit `3017`) and Windows Update had already ranked the older `8132`
> package for the device. §15 records which remaining items can be automated and which need the
> operator.

---

## 1. Did the system actually reboot? — YES

| Evidence | Value |
|---|---|
| `LastBootUpTime` | 2026-09-22 23:57:33 (uptime was 6m34s at investigation start) |
| Event **1001** WER-SystemErrorReporting | *"The computer has rebooted from a bugcheck… 0x00020001"* |
| Event **41** Kernel-Power (Critical) | *"rebooted without cleanly shutting down first"* |
| Event **6008** EventLog | *"The previous system shutdown … was unexpected"* |
| Event **161** volmgr | *"Dump file creation failed … BugCheckProgress was: 0x00000053"* |

This is a full OS restart via kernel bugcheck — **not** a terminal/session/Python death.

### Failure-mode classification

* **Case A (Python crashed, OS alive)** — RULED OUT: no traceback; OS restarted.
* **Case B (GPU driver crashed/reset, OS alive)** — RULED OUT: no TDR recovery events (4101/4102).
* **Case C (entire OS restarted)** — **CONFIRMED**: bugcheck + minidump + Events 41/6008/1001.
* **Case D (freeze + watchdog restart)** — partially consistent: the kernel dump write *failed*,
  characteristic of dying on the way to the dump.
* **Case E (power loss)** — NOT SUPPORTED: a power cut cannot produce a bugcheck code + minidump.
  (Exception: the separate **2026-09-18 00:23** shutdown had Event 41/6008 but **no bugcheck** — that
  one has a power/hang signature and remains unresolved; see §10.)

---

## 2. Failure taxonomy — all 202 WER kernel reports (archive: 2026-09)

### By bugcheck code

| Code | Name | Count | First | Last |
|---|---|---|---|---|
| `0x141` | VIDEO_ENGINE_TIMEOUT_DETECTED | 174 | 2026-05-31 17:06 | 2026-09-17 12:29 |
| `0x117` | VIDEO_TDR_TIMEOUT_DETECTED | 11 | 2026-05-31 17:07 | 2026-09-16 12:37 |
| **`0x20001`** | **HYPERVISOR_ERROR** | **11** | **2026-05-21 19:16** | **2026-09-22 23:57** |
| `0x113` | VIDEO_DXGKRNL_FATAL_ERROR | 3 | 2026-05-14 09:47 | 2026-06-21 15:13 |
| `0x116` | VIDEO_TDR_FAILURE | 1 | 2026-09-16 11:23 | — |
| `0x1e` | KMODE_EXCEPTION_NOT_HANDLED | 1 | 2026-09-14 15:48 | — |
| `0x15e` | FATAL_MINIPORT_ERROR | 1 | 2026-06-06 00:09 | — |

`0x141`/`0x117` are **LiveKernelEvent** (`LKD_`) reports — GPU stalls that *recovered*.
So the machine has had **185 recoverable GPU stalls** plus **11 fatal hypervisor crashes**.

### By WER bucket (culprit attribution)

| Count | Bucket | Culprit |
|---|---|---|
| **148** | `LKD_0x141_Tdr:6_IMAGE_igdkmdn64.sys_ARL_0_KMD` | **Intel iGPU** |
| **14** | `LKD_0x141_Tdr:6_IMAGE_igdkmdn64.sys_ARL_TRINITY11_BBHANG` | **Intel iGPU** |
| **10** | `LKD_0x117_Tdr:3_TdrVTR:0_IMAGE_igdkmdn64.sys_ARL_0_DISPLAY` | **Intel iGPU** |
| **7** | `INTEL_IOMMU_TIMEOUT_IMAGE_GenuineIntel.sys` | **Intel IOMMU (VT-d)** |
| 5 | `LKD_0x141_Tdr:6_IMAGE_nvlddmkm.sys_Ada` | NVIDIA |
| **4** | `NOBLOB_HYPERVISOR_ERROR_EXCESSIVE_WAIT_WHILE_SPINNING_IMAGE_hvix64.exe` | **Hyper-V** |
| 2 | `LKD_0x141_..._igdkmdn64.sys_ARL_TRINITY11_RINGHANG` | **Intel iGPU** |
| 2 | `LKD_0x141_..._igdkmdn64.sys_ARL_TRINITY11_GUC_SCHEDULER_ERROR` | **Intel iGPU** |
| 2 | `0x113_2C_dxgkrnl!DXGCONTEXT::_DXGCONTEXT` | dxgkrnl |
| 2 | `LKD_0x141_Tdr:C_AppFault_IMAGE_nvlddmkm.sys_Ada` | NVIDIA |
| 1 | `0x116_TdrBCR:3:C0000001_Tdr:9_IMAGE_igdkmdn64.sys_ARL_0_ENGINE` | **Intel iGPU** |
| 1 | `LKD_0x117_Tdr:9_IMAGE_nvlddmkm.sys_Ada` | NVIDIA |
| 1 | `LKD_0x15E_FATAL_MINIPORT_ERROR_..._IMAGE_Netwtw14.sys` | **Intel Wi-Fi** |
| 1 | `AV_nt!KeContextFromKframes` | kernel AV |
| 1 | `BAD_DUMPFILE` | n/a |

**Intel platform drivers outnumber NVIDIA ~22:1 (177 vs 8).**

Bucket key: `igdkmdn64` = Intel graphics KMD · `ARL` = **Arrow Lake** (matches Core Ultra 9 285) ·
`TRINITY11` = Intel GPU IP block · `BBHANG` = breadcrumb hang · `KMD` = kernel-mode driver ·
`LKD_` = LiveKernelDump (recovered, no BSOD).

---

## 3. All 11 HYPERVISOR_ERROR crashes

| # | Time | param1 | param4 | BootId | WER bucket |
|---|---|---|---|---|---|
| 1 | 2026-05-21 19:16:49 | `0x13` | `0xffffe80001a27c80` | 13 | `EXCESSIVE_WAIT_WHILE_SPINNING_hvix64.exe` |
| 2 | 2026-05-22 00:36:52 | `0x28` | `0xfc810000` | 14 | `INTEL_IOMMU_TIMEOUT_GenuineIntel.sys` |
| 3 | 2026-05-22 00:41:37 | `0x28` | `0xfc810000` | 15 | `INTEL_IOMMU_TIMEOUT` |
| 4 | 2026-06-06 13:14:06 | `0x28` | `0xfc810000` | 16 | `INTEL_IOMMU_TIMEOUT` |
| 5 | 2026-06-17 13:01:37 | `0x28` | `0xfc800000` | 21 | `INTEL_IOMMU_TIMEOUT` |
| 6 | 2026-06-17 23:37:51 | `0x28` | `0xfc810000` | 22 | `INTEL_IOMMU_TIMEOUT` |
| 7 | 2026-09-14 19:21:05 | `0x28` | `0xfc810000` | 37 | `INTEL_IOMMU_TIMEOUT` |
| 8 | 2026-09-14 21:24:12 | `0x13` | `0xffffe80001a2cc80` | 38 | `EXCESSIVE_WAIT_WHILE_SPINNING` |
| 9 | 2026-09-17 00:34:10 | `0x13` | `0xffffe80001a2cc80` | 47 | `EXCESSIVE_WAIT_WHILE_SPINNING` |
| 10 | 2026-09-17 12:31:50 | `0x28` | `0xfc810000` | 52 | `INTEL_IOMMU_TIMEOUT` |
| 11 | **2026-09-22 23:57:54** | **`0x13`** | `0xffffe80001a2cc80` | **55** | **`EXCESSIVE_WAIT_WHILE_SPINNING`** |

Notes:
* `param3 = 0x29b92701` is **identical in all 11 events** across 4 months — a fixed structure/build
  signature, not a per-crash value.
* `param4 = 0xfc810000` / `0xfc800000` on **every** IOMMU variant — a physical DMA address in the
  ~4 GB window. `0xffffe800…` on the spinning variant is a kernel virtual address.
* Microsoft documents the `0x20001` parameters as *Reserved*; the meaning above comes from WER's
  bucket naming, which is the authoritative local attribution.
* `GenuineIntel.sys` **does not exist anywhere on disk** (`C:\Windows\System32\drivers\` and a full
  `C:\Windows` search both empty). It is a **synthetic WER attribution to the Intel CPU/IOMMU
  subsystem**, not a loadable driver. Confirm via `!analyze -v` on a kernel dump.

---

## 4. Timeline of the 2026-09-22 crash

Reconstructed from filesystem mtimes + Command Code session transcript (times local, UTC+5:30):

```
23:52:29  reside.txt   → 3 models loaded: alloc=24.13GB reserved=24.15GB free=0.18GB
                         prefill ctx=4087 → peak=27.23GB
23:54:43  spill_alone.txt      (attacker alone, peak 11.51GB)      [probe] done
23:55:38  spill_att_target.txt (att+target,  peak 22.33GB)         [probe] done
23:55:47  spill_all3.txt       CREATED, 0 BYTES   ← 3-model spill test launches
~23:56–57 *** BUGCHECK 0x20001 / param1=0x13 / EXCESSIVE_WAIT_WHILE_SPINNING ***
23:57:33  boot
23:57:47  Event 1001 records the bugcheck
23:57:50  WHEA-Logger 17: corrected PCIe AER on PCI\VEN_10DE&DEV_27B1 (the NVIDIA card)
00:04     investigation begins
```

Event 6008's *"previous shutdown at 23:23:33"* is a **stale periodic registry stamp** — direct
filesystem writes prove the machine was alive at 23:55:47.

Corroborating pattern: crash #9 on **2026-09-17 00:34:10** landed ~2.5 min after the GPU probe
stream in `results/phase14/probe_run3.log` stopped mid-flight at **00:31:43**.

---

## 5. Environment baseline BEFORE any updates

Capture this again after updating and diff it (see `baseline_before_update.json`).

### Hardware / firmware

| Item | Value |
|---|---|
| System | HP Z2 Tower G1i Workstation Desktop PC |
| SKU / board | `B04F2AV` / HP `8D3C`, KBC Version 14.22.00 |
| CPU | Intel Core Ultra 9 285 (**Arrow Lake-S**), 24 cores / 24 threads, 2.5 GHz base |
| RAM | 127.4 GiB usable — 4 × 32 GB Samsung `M323R4GA3EB0-CWMOL` DDR5-5600, **no ECC** |
| GPU0 | NVIDIA RTX 4500 Ada Generation, 24570 MiB, PCI `00000000:02:00.0`, WDDM |
| GPU1 | Intel(R) Graphics (Arrow Lake iGPU) |
| BIOS | **`X51 Ver. 01.08.03`**, released **2025-12-12** |
| Power plan | `HP High Performance` |
| Disks | Samsung `MZVL81T0HFLB-00BH1` NVMe 954 GB (C:) · 2 × HGST `HUS722T2TALA604` 2 TB (D:, E:) — all Healthy |
| Pagefile | `C:\pagefile.sys` 8192 MB, `AutomaticManagedPagefile = True` |
| HF cache | `C:\Users\CSLAB\.cache\huggingface` = 135.82 GB |

### OS

| Item | Value |
|---|---|
| Edition | Windows 11 Pro 25H2, build **26200.9457** (registry `ProductName` says "Windows 10 Pro" — cosmetic artifact) |
| VBS policies | `VBS Enabled, VSM Required, Secure Boot, Iommu Protection, Mmio Nx, Strong MSR Filtering, HVCI (Strict), Measured Launch` |
| Hypervisors | **Microsoft Hyper-V** (`hvix64.exe`, `Vid.sys`, `hvservice.sys`) **+ Bromium uXen** (`uxen.sys`, HP Sure Click) |

### Python / ML stack

| Item | Value |
|---|---|
| Python | 3.13.9 (resolved from `C:\Users\CSLAB\anaconda3`) |
| PyTorch | 2.7.1+cu118 |
| Transformers | 5.16.1 |
| CUDA | 11.8 (torch build); driver reports CUDA 13.0 capable |
| Known issue | `OMP: Error #15` duplicate `libiomp5md.dll`; masked by `KMP_DUPLICATE_LIB_OK=TRUE` in `phase14_full_reproduction.py` |

### Driver inventory (BEFORE — the diff target)

| Device | Driver version | Date | Provider |
|---|---|---|---|
| **Intel(R) Graphics** | **`32.0.101.8132`** | **2025-09-19** | Intel Corporation |
| NVIDIA RTX 4500 Ada Generation | `32.0.15.8208` (= 582.08) | 2025-07-12 | NVIDIA |
| Intel(R) Wi-Fi 6E AX211 160MHz | `24.0.2.1` | 2025-10-21 | Intel |
| Intel(R) Wireless Bluetooth | `24.0.1.1` | 2025-09-23 | Intel |
| Intel(R) Ethernet Connection I219-LM | `20.0.3.16` | 2025-05-10 | Intel |
| Intel(R) AI Boost (NPU) | `32.0.100.4778` | 2026-04-28 | Intel |
| Intel(R) Management Engine Interface #1 | `2552.8.10.0` | 2025-12-23 | Intel |
| Intel(R) Dynamic Tuning Technology | `9.1.10002.281` | 2025-10-17 | Intel |
| Intel(R) Innovation Platform Framework (×4) | `2.2.10204.8` | 2024-12-16 | Intel |
| Intel(R) Platform Monitoring Technology (PMT) | `3.1.2.6` | 2024-05-24 | Intel |
| Intel(R) TXT Authenticated Code Module | `20.20.19.28` | 2025-07-03 | Intel |
| Intel(R) SPI (flash) Controller — AE23 | `10.1.47.12` | — | Intel |
| Intel Processor (`intelppm.sys`) | `10.0.26100.9278` | — | Microsoft |
| NVIDIA High Definition Audio | `1.4.5.6` | 2025-07-12 | NVIDIA |
| NVIDIA Virtual Audio Device | `4.92.0.0` | 2026-04-08 | NVIDIA |

**Target:** Intel Graphics → `32.0.101.9030` (released 2026-09-17, supports Arrow Lake-S).
Verify the current version at Intel's download center before installing.

### Crash-dump configuration

| Value | BEFORE | **AFTER (changed this session)** |
|---|---|---|
| `CrashDumpEnabled` | `3` (small minidump **only**) | **`7` (Automatic)** |
| `DedicatedDumpFile` | *(empty)* | **`D:\dedicateddump.sys`** |
| `DumpFileSize` | *(empty)* | **`40960` MB** |
| `DumpFile` | `C:\Windows\MEMORY.DMP` | unchanged |
| `AutoReboot` | `1` | unchanged (`1`) |

**Why:** `CrashDumpEnabled = 3` meant **no kernel dump was ever written** — that is why
`C:\Windows\MEMORY.DMP` does not exist and why a 12 MB minidump was all we had. The next bugcheck
will now produce a full kernel dump on `D:\`.

---

## 6. What was RULED OUT (and why)

| Candidate | Verdict | Evidence |
|---|---|---|
| **System RAM exhaustion** | RULED OUT | 127.4 GiB total, 113 GiB free; pagefile **peak usage 0 MB**; zero OOM / `oom-killer` / allocation-failure events |
| **Disk / storage exhaustion** | RULED OUT | C: 364 GB free, D: 1861 GB, E: 1863 GB; all 3 volumes + all 3 physical disks Healthy/OK |
| **CPU overload** | RULED OUT | 3–4 % load; no runaway processes |
| **CPU thermal shutdown** | RULED OUT | **Zero** `Kernel-Processor-Power` thermal events (86/87/88/89/90/35/37/38) ever logged |
| **GPU thermal shutdown** | NO EVIDENCE | 43 °C idle vs 84 °C target; `HW Thermal Slowdown` / `SW Thermal Slowdown` both **Not Active**; all slowdown counters **0 µs**. See caveat below |
| **Power loss / PSU** | NOT SUPPORTED for this event | A bugcheck code + minidump cannot be produced by a power cut |
| **Python / transformers / torch crash** | RULED OUT | No traceback in the fatal run — `spill_all3.txt` is **0 bytes** (earlier failing probes *did* leave tracebacks) |
| **CUDA out of memory** | RULED OUT | **Zero** `CUDA out of memory` anywhere in the repo. (7 × `CUDA error: unknown error` in `probe_run2.log` / `probe_json_compliance.log` — a dead context, explicitly *not* OOM) |
| **Infinite loop / runaway process** | RULED OUT | No runaway processes; the probes completed normally before the last one |
| **Actual Phase 14 code bug** | RULED OUT | The fault is in the kernel/hypervisor path, not user space |
| **Kernel-level crash** | **CONFIRMED** | bugcheck 0x20001 |
| **Hardware/firmware instability** | **CONFIRMED** | 202 kernel reports over 4 months; 177 attributed to `igdkmdn64.sys` |

**Thermal caveat — stated honestly:** there is **no GPU/CPU temperature telemetry for the crash
window**. No HWiNFO-style logger was running, and Windows does not retain temperature history.
The closest sample (43 °C) is benign and all throttle counters are zero, but the counters reset on
driver load/reboot, so they do **not** cover the crash window. Thermal is therefore *unproven
rather than excluded*. **Installing a temperature logger is recommended** so the next crash has data.

---

## 7. The Phase 14 workload's role — trigger, not cause

Phase 14's `make_models()` deliberately bypasses `ModelManager` and keeps every model resident:

> *"We bypass the ModelManager eviction/swapping entirely… both models stay resident on GPU."*
> — `scripts/phase14_full_reproduction.py`

Measured VRAM (from the pre-crash probe scripts in the session scratchpad):

| Configuration | Static | Peak measured |
|---|---|---|
| attacker `Qwen/Qwen3.5-4B` alone | 8.41 GB | 11.51 GB @ ctx 4087 |
| + target `microsoft/Phi-4-mini-instruct` | 16.08 GB | **22.33 GB @ ctx 6013** (survived) |
| + evaluator `Qwen/Qwen3-4B-Instruct-2507` | **24.13 GB** (reserved 24.15, **free 0.18**) | **27.23 GB @ ctx 4087** ← crashed |
| Card usable | 24570 MiB = **25.77 GB** | |

So the next-phase model stack — attacker `Qwen3.5-4B-Instruct` + target `Phi-4-mini-instruct` +
evaluator `Qwen3-4B-Instruct-2507`, three **distinct** physical models — **exceeds the card by
~1.5 GB**. It does not raise CUDA OOM because on WDDM the driver satisfies the excess from host
memory over PCIe (CUDA Sysmem Fallback). That is the difference between "slow" and "unstable".

This is a **real, separate hazard** from the Intel-driver defect, and must be fixed regardless:

1. **Never keep three distinct LLMs co-resident.** Recommended plan: attacker + target resident
   (measured safe at 22.33 GB), evaluator swapped via the existing `ModelManager`.
2. **NVIDIA Control Panel → Manage 3D Settings → Program Settings → `python.exe` →
   "CUDA - Sysmem Fallback Policy" = "Prefer No Sysmem Fallback".** Converts an over-budget
   allocation into a catchable `CUDA out of memory` — which
   `classify_exception()` already maps to `INFRASTRUCTURE_ERROR` and records — instead of a
   driver-level host-memory spill.
3. **Pre-flight VRAM gate** per run: abort the batch if `torch.cuda.mem_get_info()[0]` is below
   static footprint + KV reserve.
4. **Promote `peak_vram_gb` to a hard stop** (abort batch above ~22 GB on this card).
5. **One GPU process at a time** — the 7 × `CUDA error: unknown error` came from overlapping runs
   (`util.txt` shows 9.1 tok/s purely from contention).
6. Install a GPU/CPU **thermal logger**.

---

## 8. Corrections to earlier hypotheses in this investigation

Recorded for honesty so a future session does not chase them:

* **HP Wolfe Security / Bromium `uXen` as the faulting hypervisor — RETRACTED as primary.**
  uXen *is* a second Type-1 hypervisor on this box (`uxen.sys`, HP Sure Click, with live micro-VMs
  `Br-uxendm.exe -n uVM0001`), which made it a reasonable inference from the bare `HYPERVISOR_ERROR`
  code. But **no WER bucket names `uxen.sys`** — the named images are `hvix64.exe` (Microsoft Hyper-V)
  and Intel's IOMMU path. uXen may still add VM-exit / IOMMU pressure, but it is **not** the
  attributed culprit.
* **NVIDIA as the dominant failing GPU — RETRACTED.** NVIDIA accounts for 8 of 202 reports; the
  Intel iGPU accounts for 177.
* **`nvidia-smi` thermal limit fields are unreliable on this driver.** It reports
  `GPU T.Limit Temp: 44 C` with `GPU Current Temp: 43 C` and `Shutdown T.Limit: -7 C` —
  self-contradictory nonsense. `GPU Target Temperature: 84 C` is the credible figure. Do not read
  those fields as a 44 °C shutdown threshold.
* **`CrashDumpEnabled = 3` was the reason `MEMORY.DMP` was missing** — not a crash symptom.

---

## 9. Evidence locations

| What | Where |
|---|---|
| **Preserved crash evidence** | `C:\Users\CSLAB\Desktop\Guardbound_crash_dumps\` — 5 minidumps, **202** `Kernel_*` WER report dirs, `bugcheck_history.txt` |
| The 2026-09-22 minidump | `C:\Windows\Minidump\092226-14609-01.dmp` (also in the folder above) |
| Live kernel reports | `C:\Windows\LiveKernelReports` — was empty |
| WER archive (originals) | `C:\ProgramData\Microsoft\Windows\WER\ReportArchive` (ACL-protected; needs elevation) |
| Phase 14 run logs | `results/phase14/batch00.log`, `probe_run2.log`, `probe_run3.log`, `probe_json_compliance.log`, `BATCH0_STATUS.md`, `ABORTED_PRE_PHASE14.marker` |
| Fidelity audit | `PHASE14_CODE_FIDELITY_AUDIT.md`, `results/phase14/phase14_1_structured_output_report.md` |
| Helper script | `diagnose_crash.ps1` (repo root) — read-only by default, `-Apply` for dump config |
| Machine baseline (this doc) | `crash_investigation/baseline_before_update.json` |

Minidump header parsed directly (confirms the dump matches the event log):

```
Signature PAGE · ValidDump DU64 · Version 15.26100
BugCheckCode 0x00020001
Param1 0x0000000000000013   Param2 0x0000000000000000
Param3 0x0000000029B92701   Param4 0xFFFFE80001A2CC80
```

---

## 10. Open questions — NOT yet proven

1. **Exact faulting image / stack.** Needs `!analyze -v` on a **kernel** dump. The 12 MB minidump
   gives us the bugcheck code, parameters, and WER's bucket, but not a reliable stack.
2. **Whether uXen contributes.** Not named by any bucket; needs the kernel dump (look for
   `uxen.sys` in the stack) and, optionally, a bounded test with Sure Click disabled.
3. **Whether VBS "Iommu Protection" must be relaxed as a fallback.** It is the feature that turns an
   Intel GPU/IOMMU stall into a hypervisor death. Disabling it (or VBS) would likely stop the
   crashes immediately, but it is a genuine security control on an HVCI-*Strict* machine. Escalate
   to HP/IT; do not change unilaterally.
4. **The 2026-09-18 00:23 unexpected shutdown** — Event 41 + 6008 but **no bugcheck**. A different
   (power/hang) signature. Run vendor hardware + memory diagnostics.
5. **Memory has never been tested.** `Microsoft-Windows-MemoryDiagnostics-Results` is empty. 4 × 32 GB
   non-ECC. Run `mdsched.exe` or MemTest86 overnight.
6. **Root cause of the failed dump write** (`volmgr` 161, `BugCheckProgress 0x53`) — now worked
   around with a dedicated dump file on `D:\`.
7. **Why `GenuineIntel.sys` is named but absent from disk** — assume synthetic WER attribution to the
   Intel CPU/IOMMU subsystem; confirm via `!analyze -v`.

---

## 11. POST-UPDATE VERIFICATION CHECKLIST

> **STATUS 2026-09-23 — this checklist has been run; results are in §14.** Nothing below needs
> re-running until the Intel iGPU is actually bound to `32.0.101.9030` (it is staged but unbound).
> §14.3 also holds the post-update crash counters to diff against, and §15 records who can action
> each remaining item.

Run these after the driver/BIOS updates and reboot, then compare against §5.

### A. Confirm the updates actually landed

```powershell
# Intel iGPU must be newer than 32.0.101.8132 (2025-09-19)
Get-CimInstance Win32_PnPSignedDriver |
  Where-Object { $_.DeviceName -match 'Intel\(R\) Graphics|NVIDIA' } |
  Select-Object DeviceName, DriverVersion, DriverDate, DriverProviderName | Format-Table -AutoSize

Get-CimInstance Win32_BIOS | Select-Object SMBIOSBIOSVersion, ReleaseDate

# Windows build
(Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion').UBR
```

### B. Confirm the machine booted cleanly (no crash during the update)

```powershell
# Uptime should be short and LastBootUpTime should match the intentional reboot
Get-CimInstance Win32_OperatingSystem | Select-Object LastBootUpTime

# Should list ONLY pre-update crashes. Any NEW row = the update did not fix it.
Get-WinEvent -FilterHashtable @{LogName='System';
  ProviderName='Microsoft-Windows-WER-SystemErrorReporting'} |
  Sort-Object TimeCreated | Select-Object TimeCreated, Id, Message | Format-Table -Wrap

# Any LogName -eq 'System', Id 41/6008 after the reboot = still crashing
Get-WinEvent -FilterHashtable @{LogName='System'; Id=@(41,6008,1001);
  StartTime=(Get-Date).AddHours(-6)} | Format-Table TimeCreated, Id, ProviderName
```

### C. Check whether GPU stalls are still accumulating (the leading indicator)

```powershell
# Count NEW Kernel_* WER reports since the update
(Get-ChildItem 'C:\ProgramData\Microsoft\Windows\Windows Error Reporting\ReportArchive' -Directory -EA 0).Count
(Get-ChildItem 'C:\ProgramData\Microsoft\Windows\WER\ReportArchive' -Directory -EA 0 |
  Where-Object { $_.Name -like 'Kernel_*' }).Count   # needs elevation

# NVIDIA/Intel driver complaints
Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='nvlddmkm'} -EA 0 |
  Group-Object Id | Select-Object Count, Name | Format-Table -AutoSize

# WHEA corrected PCIe errors on the NVIDIA card (VEN_10DE&DEV_27B1)
Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='Microsoft-Windows-WHEA-Logger'} -EA 0 |
  Select-Object TimeCreated, Id | Format-Table -AutoSize
```

### D. Reproduce under controlled load, watching for the leading indicators

1. Start a GPU/CPU **thermal logger first** (HWiNFO or equivalent) — we currently have no
   crash-window telemetry.
2. Set **CUDA Sysmem Fallback = "Prefer No Sysmem Fallback"** for `python.exe`.
3. Run the **attacker + target only** configuration (peak measured 22.33 GB — safe). Confirm it
   stays well under 25.77 GB.
4. Only then attempt the full three-model path via the `ModelManager` swap plan.
5. Watch `Get-WinEvent` for `0x141` / `0x117` / `0x20001` while it runs. `0x141` is the early-warning
   signal; `0x20001` is the fatal one.
6. If it crashes again, a **full kernel dump** will now be at `D:\dedicateddump.sys` (or
   `C:\Windows\MEMORY.DMP`). Analyze with:

```powershell
winget install Microsoft.WinDbg
kd -z <dump> -c "!analyze -v; q"
```

**What to look for in `!analyze -v`:**
* `FAULTING_MODULE` — expect `hvix64.exe` or the Intel IOMMU path
* The decoded meaning of `param1 = 0x13` vs `0x28`
* Whether `uxen.sys`, `igdkmdn64.sys`, or `nvlddmkm.sys` appear in the stack

### E. Success criteria

| Signal | Fixed | Not fixed |
|---|---|---|
| New `0x20001` after the update | none | any |
| `LKD_0x141_*igdkmdn64.sys*` rate | drops sharply / stops | continues at ~150/month |
| `WHEA-Logger` 17 on `VEN_10DE&DEV_27B1` | fewer / absent | unchanged at every boot |
| `nvlddmkm` Event 13 count | drops from ~1700/2 days | unchanged |

If `0x141` events keep accruing after a confirmed Intel driver update, escalate to HP — that would
point at the platform (BIOS/IOMMU/firmware) rather than the driver, and the `Iommu Protection`
question in §10.3 becomes the next lever.

---

## 12. Changes made during this investigation

* **No production code was modified.** No optimisation, no config change, no generation-parameter
  change, no NLA/attack/prompt/history change, no VRAM limit, no batch-size change.
* **No Phase 14 / Batch 0 run was started.**
* The system was **not** restarted by this investigation.
* Files created: `diagnose_crash.ps1`, `crash_investigation/PHASE14_CRASH_ROOT_CAUSE.md`,
  `crash_investigation/baseline_before_update.json`.
* **System change (approved, via `diagnose_crash.ps1 -Apply`):** `CrashDumpEnabled` 3 → 7,
  `DedicatedDumpFile` = `D:\dedicateddump.sys`, `DumpFileSize` = 40960 MB.
* Evidence copied out of ACL-protected locations to `C:\Users\CSLAB\Desktop\Guardbound_crash_dumps\`.

**Addendum, 2026-09-23 01:28 —** §14 (post-update state) and §15 (remediation ownership) were
appended, and `baseline_before_update.json` gained matching `post_update_snapshot` and
`remediation_ownership` blocks plus `status` / `owner` fields on all nine `next_actions_priority`
entries. Producing them was **read-only** — no system change, and still no production code modified.

---

## 13. Sources

* [Bug Check 0x20001 HYPERVISOR_ERROR — Microsoft Learn](https://learn.microsoft.com/en-us/windows-hardware/drivers/debugger/bug-check-0x20001--hypervisor-error)
* [Bug Check 0x20001 on Windows 11 — Microsoft Q&A](https://learn.microsoft.com/en-us/answers/questions/4005852/bug-check-0x20001-on-windows-11)
* [Intel Core Ultra 9 Processor 285 — downloads](https://www.intel.com/content/www/us/en/products/sku/241061/intel-core-ultra-9-processor-285-36m-cache-up-to-5-60-ghz/downloads.html)
* [Latest Intel Graphics Driver for Windows 11 (32.0.101.9030, 2026-09-17)](https://www.elevenforum.com/t/latest-intel-graphics-driver-for-windows-11.563/)
* [Intel Download Center](https://www.intel.com/content/www/us/en/download-center/home.html)
* [HYPERVISOR_ERROR (0x20001) causes](https://knowledgebase.bison.co.in/view_article.php?id=2298)

Added with §14/§15 (2026-09-23) — the `0x1A8` / `0x1B8` livedump codes:

* [LiveKernelEvent Code 1a8 & 1b8: How to Fix These Hardware Errors](https://windowsreport.com/hardware-error-1a8-1b8/)
* [`LKD_0x1A8_KEYBD_HOTKEY_GraphicsUnknown_NV_dxgkrnl!DISPLAYSTATECHECKER::CreateBlackScreenLiveDump` — bucket decoded in the wild](https://forums.tomshardware.com/threads/livekernelevent-error.3732756/)
* [Live Kernel Event 1a8 and 1b8 — Microsoft Q&A](https://learn.microsoft.com/en-us/answers/questions/4157310/live-kernel-event-1a8-and-1b8)
* [Intel Arc Graphics — Windows driver 32.0.101.9030](https://www.intel.com/content/www/us/en/download/785597/intel-arc-graphics-windows.html)

---

## 14. POST-UPDATE STATE (verified 2026-09-23 01:28)

Re-run of §11 on **2026-09-23 at 01:16–01:28 local**, after the BIOS / NVIDIA / Intel driver
updates and two reboots (`2026-09-23 00:58:13`, `2026-09-23 01:10:54`). Machine-readable diff
target: `baseline_before_update.json` → `post_update_snapshot`.

### 14.1 What actually landed

| Item | BEFORE (§5) | AFTER (verified) | How it arrived |
|---|---|---|---|
| HP BIOS | `X51 Ver. 01.08.03` (2025-12-12) | **`X51 Ver. 02.02.02` (2026-06-15)** | WU "HP Firmware Driver Update (2.2.2.0)" 2026-09-22 19:30:51 |
| NVIDIA RTX 4500 Ada | `32.0.15.8208` (= 582.08) | **`32.0.15.9579` (= 595.79)** | WU "NVIDIA Display Driver Update" 2026-09-22 19:30:49 |
| NVIDIA High Definition Audio | `1.4.5.6` | `1.4.5.7` | with the GPU package |
| Intel Wi-Fi 6E AX211 | `24.0.2.1` | **`24.30.1.1`** | WU 2026-09-22 19:29:45 |
| Intel Wireless Bluetooth | `24.0.1.1` | **`24.30.1.1`** | WU 2026-09-22 19:29:49 |
| Intel Ethernet I219-LM | `20.0.3.16` | **`20.0.3.24`** | WU 2026-09-22 19:29:41 |
| Intel chipset (LPC/eSPI 7F08, SPI 7F24) | `10.1.47.12` | **`10.1.51.10`** | — |
| Crash-dump configuration | `3`, no dedicated file | `7` + `D:\dedicateddump.sys` + 40960 MB | earlier in this investigation |
| Windows build | `26200.9457` | `26200.9457` (unchanged) | no cumulative update applied |

### 14.2 The P0 did NOT take — the Intel iGPU is still on `32.0.101.8132`

**This is the primary fix from §5 and it is not in effect.**

* Active binding (authoritative, `Get-PnpDeviceProperty` on
  `PCI\VEN_8086&DEV_7D67&SUBSYS_8D3C103C&REV_06\3&11583659\0&10`):
  `DEVPKEY_Device_DriverVersion = 32.0.101.8132`, driver date `2025-09-19`,
  `InfPath = oem24.inf`, `LastArrivalDate = 2026-09-23 01:10:55`.
* But `32.0.101.9030` **is staged** in the driver store:
  `C:\Windows\System32\DriverStore\FileRepository\iigd_dch.inf_amd64_873dcf47953be7ee\igdkmdn64.sys`
  → FileVersion `32.0.101.9030`, INF `DriverVer=09/17/2026,32.0.101.9030`, staged 2026-09-23 00:55:46.
  (The old package is still staged alongside it: `iigd_dch.inf_amd64_f753fe2f699ef44c` → `8132`.)
* Verdict: **the file is on disk, the device never bound to it.**

Why — from `C:\ProgramData\Intel\GFXInstaller\Installer\IntelGFX_20260923_004927_Install.log`:

```
00:51:20  INFO|Installing (DiInstallDriver) "...\RarSFX0\Graphics\iigd_dch.inf"
              Flag = "0x00000002" (DIIRFLAG_FORCE_INF)      <- INF install itself took 74 s, succeeded
00:54:32  INFO|Exit code: 3017
00:54:32  ERROR|Executing operation for the component "Intel(R) Graphics Software" failed.
00:57:10  INFO|Rebooting the system...
00:57:10  INFO|Exiting application with exit code: 2. Success, but installer had to trigger system restart.
```

The bootstrapper reported success-with-reboot, the machine rebooted at 00:58:13 — and the device
came back up on `oem24.inf` (8132) anyway, per `DEVPKEY_Device_LastArrivalDate`.

Contributing factor — **Windows Update had pushed the OLD driver on the same day**, before the
9030 installer ran:

```
2026-09-22 19:24:03  Intel Corporation - Extension - 32.0.101.6629       Succeeded
2026-09-22 19:24:08  Intel Corporation Driver Update (32.0.101.8132)      Succeeded
2026-09-22 19:24:45  Intel Corporation Driver Update (32.0.101.8132)      Succeeded
```

Two `8132` packages were therefore already ranked for this device; after the reboot PnP re-picked
the older INF.

**Action (elevated):**

```powershell
# 1. Bind the staged 9030. GUI equivalent:
#    Device Manager -> Intel(R) Graphics -> Update driver -> Browse my computer
#    -> Let me pick from a list -> "Intel(R) Graphics 32.0.101.9030"
pnputil /add-driver `
  'C:\Windows\System32\DriverStore\FileRepository\iigd_dch.inf_amd64_873dcf47953be7ee\iigd_dch.inf' `
  /install

# 2. Verify — must read 32.0.101.9030 / 2026-09-17
Get-PnpDeviceProperty -InstanceId 'PCI\VEN_8086&DEV_7D67&SUBSYS_8D3C103C&REV_06\3&11583659\0&10' `
  -KeyName DEVPKEY_Device_DriverVersion, DEVPKEY_Device_DriverDate

# 3. THEN stop Windows Update from re-reverting (policy equivalent of
#    Settings -> Device installation settings -> "Do not include drivers with Windows Updates"):
#    HKLM\SOFTWARE\Policies\Microsoft\Windows\WindowsUpdate\ExcludeWUDriversInQualityUpdate = 1
#    (or wushowhide / hide the "Intel Corporation Driver Update (32.0.101.8132)" package)
```

**Corroborating detail:** the Intel iGPU is the adapter actually driving the display —
`1920 x 1080 @ 60 Hz` on the HP 324pf, while the NVIDIA RTX 4500 exposes no active mode. That is
consistent with `igdkmdn64.sys` leading the attribution at 177/202 kernel reports.

### 14.3 Crash activity since the update

| Signal | Value |
|---|---|
| New `0x20001` HYPERVISOR_ERROR | **none** — last 2026-09-22 23:57:54; current boot 2026-09-23 01:10:54 |
| New `0x141` / `0x117` GPU stalls | **none** — last `0x141` was 2026-09-17 12:29. *Weak evidence:* no GPU workload has run since, so it only means the trigger has been absent |
| `Kernel_*` WER reports | 206 total (was 202), archived per day: `09-16: 37`, `09-17: 58`, `09-23: 5` |
| **`Kernel_1a8` x2, `Kernel_1b8` x2** | **NEW CODES**, 2026-09-23 00:51–00:55 — inside the driver-install window (not in the §2 taxonomy) |
| `WHEA-Logger` 17 (PCIe AER, `VEN_10DE&DEV_27B1`) | 2026-09-22 23:57:50, 2026-09-23 00:58:29, **01:11:08 (immediately after boot)** — unchanged behaviour |
| `nvlddmkm` events | Event 13 x **1864**, Event 153 x 112 |
| Dump files | **none** — correct: nothing has bugchecked. `D:\dedicateddump.sys` will appear at the next one |

`0x1A8` / `0x1B8` are the **DXGKRNL black-screen livedumps**
(`LKD_0x1A8_..._dxgkrnl!DISPLAYSTATECHECKER::CreateBlackScreenLiveDump`) — a plausible side-effect of
swapping a display driver under a live session, not a new fault class. Their `Report.wer` bodies are
ACL-blocked; read them from an elevated prompt to classify properly.

### 14.4 Everything else from §11 that is still outstanding

| Priority | Item | Status |
|---|---|---|
| **P0** | Intel iGPU `32.0.101.8132` -> `32.0.101.9030` | **staged but not bound** — see §14.2 |
| **P0** | CUDA Sysmem Fallback = "Prefer No Sysmem Fallback" for `python.exe` | **not set** — zero `python` / `sysmem` strings in `C:\ProgramData\NVIDIA Corporation\Drs\nvdrsdb0.bin`; NVIDIA App 11.0.9.251 is installed, the setting is GUI-only |
| **P0** | Never keep 3 distinct LLMs co-resident | **not changed** — `scripts/phase14_full_reproduction.py::make_models()` still bypasses `ModelManager` eviction/swapping |
| **P2** | Pre-flight VRAM gate + peak-VRAM hard stop + one GPU process | **not implemented** — no `mem_get_info` / VRAM ceiling anywhere in `scripts/` or `src/` (only `peak_vram_gb` *reporting* at lines 707–841) |
| **P1** | Thermal logger | **not installed** — no HWiNFO / HWMonitor / LibreHardwareMonitor present |
| **P1** | Memory diagnostic | **never run** — `Microsoft-Windows-MemoryDiagnostics-Results` RecordCount = 0 (4 x 32 GB non-ECC) |
| **P2** | VBS "Iommu Protection" escalate to HP/IT | **no ticket** — VBS still running, `SecurityServicesRunning = {2,3,4,7}` |
| — | Intel ME `2552.8.10.0`, ME-WMI `2544.8.3.0`, DTT `9.1.10002.281`, IPF `2.2.10204.8`, PMT `3.1.2.6`, TXT ACM `20.20.19.28`, SPI-AE23 `10.1.47.12`, AI Boost `32.0.100.4778` | unchanged |

### 14.5 Phase 14 workload

Still halted. Newest artifact `results/phase14/phase14_1_structured_output_report.md`
(2026-09-17 02:52); `results/phase14/ABORTED_PRE_PHASE14.marker` (2026-09-16 21:57). No run has been
started since the crash, and the code-side guards in §7 are still absent — so a re-run today would
reproduce the same trigger conditions.

---

## 15. REMEDIATION OWNERSHIP — automatable vs operator-required

Measured on this box, 2026-09-23: the assistant session runs **non-elevated**. HKLM writes are
denied (`New-ItemProperty HKLM:\...` -> access denied), `bcdedit /enum` -> "The boot configuration
data store could not be opened. Access is denied.", and the WER `ReportArchive` cannot be read.
`C:\Windows\System32\sudo.exe` exists (policy `Enabled = 1`) but each invocation raises a UAC
prompt a human must accept, so it does not make elevation unattended. `winget` v1.29.380 is present.

| Item | Can be done unattended | Requires the operator |
|---|---|---|
| VRAM pre-flight gate, peak-VRAM hard stop, evaluator swap via `ModelManager` | **YES** — repo-only change, testable fully offline | review/merge |
| Thermal logger (winget HWiNFO portable, plus an `nvidia-smi` temp/power/clock logger as fallback) | **YES**, user-scope | confirm the install; caveats: may not expose an *Intel iGPU* sensor, and no logger captures the final seconds before a bugcheck |
| Memory diagnostic | **schedule only** (`mdsched` / `bcdedit` both need elevation) | reboot + 30 min–2 h downtime + physical presence; MemTest86 needs a USB stick |
| CUDA Sysmem Fallback | **NO** — GUI toggle in NVIDIA App. An unsupported NVAPI write into `nvdrsdb0.bin` was rejected as too risky on an already-unstable machine (`Drs` is `Everyone:(F)` writable, so it *is* scriptable — deliberately not done) | ~30 s: NVIDIA App -> Manage 3D Settings -> Program Settings -> `python.exe` -> "CUDA - Sysmem Fallback Policy" = "Prefer No Sysmem Fallback" |
| Intel iGPU `9030` bind | **NO** — needs elevation, a display-driver rebind (screen blacks out) and a reboot | UAC + Device Manager / `pnputil` (§14.2) |
| Intel ME / DTT / IPF / PMT / TXT ACM driver updates | **download only** | elevated install each; firmware-adjacent, defer until the box is stable |
| VBS "Iommu Protection" escalation | **IMPOSSIBLE** — an organisational / security-policy action on a live HVCI-*Strict* control | HP/IT ticket (§10.3); do not change unilaterally |
| Controlled-load re-run of the 3-model path | can be launched | accepts that a bugcheck reboots the machine and kills the session |

Rationale for declining the CUDA setting and the Intel platform drivers: both mean writing to a
driver-settings store or installing firmware-adjacent packages on a machine whose kernel is still
crashing, with no one-line undo if it goes wrong. Both stay operator-initiated.

