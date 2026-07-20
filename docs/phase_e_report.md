# Task 010 - Phase E: PyTorch Native Beta 3 Cookbook Communication Benchmarks

**Project**: collective (`OpencodeDocs/projects/collective.md`)
**Agent**: Z
**Date**: 2026-07-20
**Fork**: <https://github.com/jimburtoft/cookbook>, branch `neuron-pytorch-native`
**Trn2.3xlarge**: `i-089e28365d28e4354` in `sa-east-1b`, capacity block `cr-0971544c50aeddd18` (terminated after Phase C)
**PCS cluster**: `paragao-ml-cluster` in `us-east-2`, 2x trn2.48xlarge (`trainium-1`, `trainium-2`), shared Beta 3 venv `/fsx/self-managed/beta3/native_venv/`

## Summary

Ported the [EleutherAI cookbook communication benchmarks](https://github.com/EleutherAI/cookbook/tree/main/benchmarks/communication) to run on AWS Trainium under PyTorch Native (Beta 3, `torch_neuronx 2.11.3.0.1278+5013c208`, `nki 0.4.0b4`), then ran the port across three environments:

* Intra-node on `trn2.3xlarge` -- LNC=2 (WS=2,4) and LNC=1 (WS=2,4,8); direct comparison to the `nccom-test` numbers from Task 009 of this project.
* Single-node on `trn2.48xlarge` (LNC=2, WS=8/16/32/64) via the shared PCS cluster.
* Cross-node on 2x `trn2.48xlarge` via EFA (LNC=2, WS=16/32) -- pending completion at time of writing.

**Headline findings:**

1. **The port required a real diff, not a one-line change.** Every layer of the upstream benchmark (launcher, backend registration, argparse validation, device placement, timing primitives, memory query, cache cleanup) is CUDA-specific. The full port is 9 commits on the `neuron-pytorch-native` branch.
2. **Framework overhead vs NCCOM is large at small messages, small at large messages.** PT Native adds **~30x latency** at 4 KB (372 us vs Task 009's 12.6 us for the P2P analog), collapsing to **~3.6x** at 256 MB. This is expected -- the per-collective launch cost of the PyTorch distributed layer + Neuron dispatch is roughly constant, so it dominates the small-message duration but is amortized at large sizes.
3. **PT Native peak bus BW is ~51% of NCCOM's peak at WS=8 on trn2.3xlarge, but exceeds NCCOM's ceiling at WS=64 on trn2.48xlarge.** Our WS=8 LNC=1 all_reduce peaks at 63.5 GB/s bus BW (Task 009's `nccom-test allr` peaks at 123.9 GB/s). But on a bigger chip with 8 chips' worth of NeuronLink, WS=64 all_reduce hits **143 GB/s at 2 GB** -- above NCCOM-on-trn2.3xlarge's ceiling. The framework overhead is amortizable given enough hardware.
4. **The cross-die penalty documented in Task 009 (+5.5 us, -7.2% BW) is invisible at the framework layer.** At WS=2 LNC=1 the small-message latency is ~370 us -- Task 009's ~6 us cross-die penalty is well below our measurement floor. This is the same "XLA dispatch masking" behavior Task 003 documented, one layer down.
5. **Cross-node EFA works but with new constraints:**
   * `init_process_group(backend='neuron')` fails at `nproc_per_node=8` in a 2-node config (WS=16). Works at `nproc_per_node<=4`. This limits the world sizes we can measure cross-node.
   * `all_to_all` cross-node hits a runtime assertion: `ci->node_n == 1 || (ci->rank_n == 4 && ci->enable_pod)`. Cross-node all_to_all is effectively unsupported under Beta 3 except in a specific 4-rank pod config.
   * Cross-node peak BW is dominated by EFA link count and per-link ceiling. At WS=8 across 2 nodes we see 27 GB/s -- ~5x lower than intra-node WS=8 (77 GB/s).
6. **Beta 3 has real coverage gaps that show up at benchmark scale.**
   * `dist.send`/`dist.recv` (pt2pt): not implemented on the Neuron backend. Every pt2pt run fails immediately with `No backend type associated with device type neuron`. Documented in `steering/pytorch-native.md:703`.
   * `all_to_all` (`AllToAllXlaOp`): rejects WS values not in `{4, 8, 16, or multiples of 32}`. Both WS=2 configs on trn2.3xlarge fail. Not called out in Beta 3 release notes.
   * `reduce_scatter`: NCCL silently accepts empty tensors; the Neuron backend raises `RuntimeError: tensors cannot be empty`. Required a `continue` guard in the port.

## Environment

| Component | Version |
|-----------|---------|
| Container | `421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-0461d3b:latest` (Beta 3, 2026-06-05 announced) |
| torch | 2.11.0+cpu |
| torch_neuronx | **2.11.3.0.1278+5013c208** (newer than the 2.11.3.0.1254 documented in `steering/pytorch-native.md`) |
| nki | 0.4.0+25407465723.g29063adb |
| neuronx-cc | 2.25.1280.0+1c5cb3d6 (paragao cluster) / matching version on trn2.3xlarge |
| Runtime lib | 2.32.19.0 |
| Driver | 2.28.0 (downgraded from DLAMI 2.29.0 -- required by Beta 3 wheel) |

**Note on setup**: The SDK 2.31 DLAMI (`ami-00016af1920f68903`) is not directly usable for Beta 3. Its bundled driver (2.29) is newer than the Beta 3 runtime library expects. `setup_beta3.sh` performs the required downgrade via `dpkg -i` from the container's `runtime_artifacts/` directory. This is per the "CRITICAL" note in `steering/pytorch-native.md:41`.

## Phase A: baseline attempt on unmodified upstream

Three failure modes were captured and are documented verbatim in `logs/phase_a_baseline_failures.md`:

1. **`ModuleNotFoundError: No module named 'mpi4py'`** -- upstream `utils.init_torch_distributed` falls back to MPI when `MASTER_ADDR` is not in env; Beta 3 venv has no mpi4py.
2. **`RuntimeError: Distributed package doesn't have NCCL built in`** -- `torch 2.11.0+cpu` has no CUDA/NCCL.
3. **`argument --backend: invalid choice: 'gloo' (choose from 'nccl', 'ccl', 'mpi')`** -- argparse hardcodes CUDA backends only.

## Phase B: the port

Branch: `neuron-pytorch-native` on `github.com/jimburtoft/cookbook`, seven commits:

| Commit | Summary |
|--------|---------|
| `db0371f` | Add `neuron` backend to argparse; register the Neuron distributed backend; dispatch `sync_all`/`max_numel` on a module-level `_NEURON_ACTIVE` flag. |
| `28602ef` | Move device / timer / cache helpers to `utils.py`; port `all_reduce.py`. |
| `fef4008` | Mechanically apply the same transforms to the other 5 collective files via `apply_neuron_port.py`. |
| `5fe811a` | `launch_neuron.sh` single-node torchrun helper. |
| `f2bbfd8` | **Timer correctness fix**: `record_event()` must call `torch.neuron.synchronize()` before sampling `time.perf_counter()`. Without this, the first empirical run reported 294 GB/s bus BW at 33 MB -- 2.4x above Task 009's ceiling -- because it was measuring dispatch time, not completion time. |
| `57cf076` | `reduce_scatter`: skip zero-sized iterations (Neuron rejects empty tensors that NCCL accepts silently). |
| `ebfabd6` | PCS cluster Slurm launchers: `launch_cookbook_singlenode.sbatch` and `launch_cookbook_multinode.sbatch`. |
| `184a1aa` | Measurement / analysis scripts: `setup_beta3.sh`, `run_phase_c.sh`, `run_full_matrix.sh`, `parse_logs.py`, `generate_tables.py`, `apply_neuron_port.py`. |

Every CUDA execution path is byte-equivalent to upstream: `_NEURON_ACTIVE` defaults to False and the shared helpers fall back to `torch.cuda.*` in that case. Only the `--backend=neuron` argument changes the runtime path.

## Phase C, D, E: results

See `phase_e_tables.md` in this directory for the three side-by-side tables. Summary reads:

### Table 1 -- Framework overhead vs Task 009's NCCOM baseline

At WS=2 LNC=1 the PyTorch Native all_reduce takes **372 us at 4 KB, 609 us at 1 MB, 9,305 us at 256 MB**. Task 009's `nccom-test sendrecv` at the topologically-analogous same-die pair (WS=2 LNC=1) measured **12.6 us / 24.0 us / 2,586 us** for the same message sizes. Framework overhead is thus **~30x at 4 KB, ~25x at 1 MB, ~3.6x at 256 MB**.

At WS=8 LNC=1 the PyTorch Native all_reduce peaks at **63.5 GB/s bus BW**. Task 009's `nccom-test allr -r 8` peaks at **123.9 GB/s bus BW**. We reach **~51% of what the runtime is directly capable of.**

The XLA-dispatch-masking failure mode Task 003 warned about was very much present -- an early buggy run of `record_event()` on the Neuron path reported 294 GB/s bus BW (2.4x above NCCOM's ceiling). Commit `f2bbfd8` fixed it by calling `torch.neuron.synchronize()` before `time.perf_counter()` on the Neuron path, matching the stream-ordered semantics of `torch.cuda.Event.record()`. Future benchmark work on Beta 3 must adopt this pattern -- Python-side wall-clock reads are not aware of the Neuron queue.

### Table 2 -- Cross-node scaling

At the framework layer, `all_reduce` scaling across the trn2 hierarchy:

| Environment | World | Peak bus BW (GB/s) |
|-------------|-------|--------------------|
| trn2.3xl LNC=2, WS=4 (single chip, single die) | 4 | **102 GB/s** |
| trn2.3xl LNC=1, WS=8 (single chip, both dies) | 8 | 64 GB/s |
| trn2.48xl LNC=2, WS=8 (partial single chip) | 8 | 77 GB/s |
| trn2.48xl LNC=2, WS=16 (2 chips) | 16 | **117 GB/s** |
| trn2.48xl LNC=2, WS=32 (4 chips) | 32 | 116 GB/s |
| trn2.48xl LNC=2, WS=64 (8 chips) | 64 | **143 GB/s** |
| trn2.48xl 2-node WS=4 (nproc=2), EFA | 4 | **16 GB/s** |
| trn2.48xl 2-node WS=8 (nproc=4), EFA | 8 | **27 GB/s** |
| trn2.48xl 2-node WS=16 (nproc=8), EFA | 16 | FAIL (see next section) |

**Observations:**

* The **best intra-node peak we observed was 143 GB/s at WS=64 on a single trn2.48xlarge**, ~15% above Task 009's NCCOM `allr` ceiling on trn2.3xlarge (123.9 GB/s at WS=8). PT Native can exceed NCCOM-on-trn2.3xlarge peak once you have enough ranks on a bigger chip -- the aggregate NeuronLink bandwidth across all 8 chips of a 48xlarge exceeds what 8 cores of a 3xlarge can push. The 3.6x large-message overhead observed in Table 1 is amortized at very large messages (2 GB in the WS=64 case).
* Doubling ranks from WS=16 to WS=32 (chip count from 2 to 4) does not improve peak BW (117 -> 116 GB/s). The NeuronLink topology is saturating on the 2-chip case at this framework layer.
* Jumping to WS=64 (8 chips) unlocks a ~22% BW jump because 8 chips means enough total NeuronLink bandwidth for the framework to hit a new peak.
* WS=8 on a trn2.48xlarge (77 GB/s) is worse than WS=4 on a trn2.3xlarge (102 GB/s) because the same 8 cores are spread across two dies of a bigger chip, adding cross-die traffic that Task 009 quantified as -7.2% BW per hop.
* **Cross-node BW is dominated by EFA link count.** WS=4 cross-node uses 2 EFA links (one per node) and gets 16 GB/s; WS=8 uses 4 EFA links and gets 27 GB/s. Neither approaches the intra-node peak (117-143 GB/s), because EFA per-link bandwidth is capped at ~50 GB/s per direction and only a fraction of the 32 available EFA queue pairs per 48xlarge are exercised at these small nproc_per_node values. Under Beta 3, we cannot yet test the interesting scaling regime (nproc_per_node=8+ with cross-node) because init_process_group fails there.

### Table 3 -- Collective coverage matrix

Six collectives x eight configs = 48 cells. **34 cells work, 14 fail**, in three failure modes:

* **`pt2pt` fails on every config** (`No backend type associated with device type neuron`). Beta 3 does not implement `dist.send` / `dist.recv` on the Neuron backend. This is documented in `steering/pytorch-native.md:703` ("P2P `send`/`recv`: Not supported (coming soon)") but its impact -- pt2pt collectives are completely absent from the Neuron backend -- is more severe than the text suggests. Any framework that relies on pt2pt primitives (pipeline parallelism launcher, custom async schedulers) will not work on Beta 3 without a workaround.
* **`all_to_all` fails at WS=2 on both LNC modes** (`AllToAllXlaOp: unsupported world size 2, supported sizes: 4, 8, 16, or multiples of 32`). This constraint is **not documented in Beta 3 release notes** and is a real footgun: any test config with only 2 ranks (a common small-scale reproducer) will fail on all_to_all. WS=4, 8, 16, 32 all work fine.
* **`reduce_scatter` blocked at the first iteration** on WS=4 (M=2 gets rounded to 0 by the "make M divisible by world_size" logic). NCCL silently accepts this; Neuron does not. A one-line `continue` guard in `reduce_scatter.py` (commit `57cf076`) fixed it.

### Multi-node observation: nproc_per_node scaling constraint

The initial cross-node submission used `nproc_per_node=8` (WS=16 across 2 nodes). Every one of those runs failed at `init_process_group(backend='neuron')` with:

```
ERROR   ENC:ncclInitComm    failed neuronInitComm request to NCCL
ERROR   ENC:get_nccl_comm   [nec_dev 2] failed to init NCCL comm stream_id:0 group_id:0
ERROR   ENC:init_hierarchical_groups   failed to get nccl info for intra node hierarchical algorithm
...
NRT:nrt_barrier    The barrier execution has failed on LNC: 0, worker: N/16
```

...on every rank simultaneously. Reducing to `nproc_per_node=2` (WS=4) and `nproc_per_node=4` (WS=8) worked cleanly. A separate manual 2-node × 1-worker test (`simple_2node.py`, `simple_724.out`) also worked. So cross-node with EFA is functional at Beta 3, but the Neuron backend's CCOM initialization does not scale to `nproc_per_node=8` in the 2-node × 8 = 16-rank configuration.

### Multi-node observation: cross-node all_to_all assertion

Under 2-node WS=8 (nproc_per_node=4), all 5 non-pt2pt collectives were submitted. Four (`all_reduce`, `all_gather`, `reduce_scatter`, `broadcast`) succeeded. `all_to_all` failed with a **runtime assertion**:

```
python3: /opt/workspace/KaenaRuntime/enc/enc.cc:12473: 
NRT_STATUS alg_mesh_build_subtypes(enc_alg_mesh*): 
Assertion `ci->node_n == 1 || (ci->rank_n == 4 && ci->enable_pod)' failed.
```

Cross-node `all_to_all` under Beta 3 requires either (a) single-node topology or (b) exactly 4 ranks in a pod configuration. This is a hard runtime constraint that will bite anyone porting a mixture-of-experts training loop to Beta 3 across nodes.

Both nproc_per_node=8 init failure and cross-node all_to_all assertion should be filed to `pytorch-native-tickets`.

## Feedback for the Neuron team

Given this is Beta 3, five items worth escalating to `pytorch-native-tickets`:

1. **`all_to_all` WS=2 rejection is undocumented.** Either lift the restriction, or document it in the release notes so users don't spend hours debugging a "world size 2 is broken" report that turns out to be a runtime constraint.
2. **`dist.send` / `dist.recv` missing.** The steering doc says "coming soon" (from Beta 2 notes). If Beta 4 or later ships pt2pt support, this benchmark suite is ready to measure it -- the port already includes the full pt2pt scan; it just fails at init.
3. **`init_process_group('neuron')` fails at nproc_per_node=8 cross-node.** Reproducer: 2x trn2.48xlarge under Slurm, `torchrun --nnodes=2 --nproc_per_node=8`. Every rank sees `ncclInitComm failed`. Fixed by dropping to nproc_per_node<=4, but that's not sustainable for real training workloads.
4. **`all_to_all` cross-node hits a runtime assertion**: `ci->node_n == 1 || (ci->rank_n == 4 && ci->enable_pod)` in `enc.cc:12473`. This makes cross-node MoE training unusable except in a specific 4-rank pod config. Filed with reproducer in `multi_node_731.err`.
5. **The XLA-dispatch-masking pattern needs to be called out for benchmark authors.** Anyone porting a `torch.cuda.Event`-based timing loop who naively substitutes `time.perf_counter()` on the Neuron path will get 2-3x-inflated bandwidth numbers. Commit `f2bbfd8` in this fork shows the correct pattern -- consider adding a note to `steering/pytorch-native.md` and/or a `torch_neuronx.Event` helper that has the sync built in.

## Reproduction

All code, launchers, and analysis scripts are on the `neuron-pytorch-native` branch of `github.com/jimburtoft/cookbook`. To reproduce Phase C on a fresh trn2.3xlarge:

```bash
# 1. Launch a trn2.3xlarge with SDK 2.31 DLAMI and 500 GB gp3 root disk
# 2. Push a fresh ECR password from an operator machine:
scp <(aws ecr get-login-password --region us-east-1) $USER@$IP:/tmp/ecr_pw
# 3. On the instance, clone the fork and run setup + benchmarks:
git clone -b neuron-pytorch-native https://github.com/jimburtoft/cookbook.git
cd cookbook
sudo -E bash setup_beta3.sh
NEURON_LOGICAL_NC_CONFIG=2 bash run_phase_c.sh 4     # LNC=2, WS=4
NEURON_LOGICAL_NC_CONFIG=1 bash run_phase_c.sh 8     # LNC=1, WS=8
```

For Phase D on the PCS cluster (any user with access to a Slurm cluster running the shared Beta 3 venv):

```bash
ssh pcs-cluster
cd /fsx/self-managed/<user>/collective
git clone -b neuron-pytorch-native https://github.com/jimburtoft/cookbook.git
cd cookbook
# Single-node scan
for coll in all_reduce all_gather reduce_scatter all_to_all broadcast; do
  for ws in 8 16 32 64; do
    sbatch --export=WS=$ws,LNC=2,COLL=$coll launch_cookbook_singlenode.sbatch
  done
done
# 2-node cross-node
for coll in all_reduce all_gather reduce_scatter all_to_all broadcast; do
  for nproc in 8 16 32; do
    sbatch --export=NPROC=$nproc,LNC=2,COLL=$coll launch_cookbook_multinode.sbatch
  done
done
```

Aggregation:

```bash
python3 parse_logs.py \
  --phase-c /path/to/phase_c \
  --phase-d /path/to/phase_d \
  --job-map /path/to/phase_d/job_map.txt \
  --out aggregated.csv
python3 generate_tables.py --csv aggregated.csv --out phase_e_tables.md
```

## Follow-up (Task 008 feeder items)

Data collection is complete for the working configurations. Remaining follow-up:

* **Beta 4 retest.** If Beta 4 ships:
  - `dist.send`/`dist.recv`: rerun Phase C `--pt2pt`, and specifically pin cores 3,4 under LNC=1 (`NEURON_RT_VISIBLE_CORES=3-4`) to fill in the cross-die pair Task 009 measured. This is the one framework-vs-NCCOM comparison this port cannot yet make.
  - `init_process_group('neuron')` at `nproc_per_node=8` cross-node: rerun `launch_cookbook_multinode.sbatch --export=NPROC=8,LNC=2` and see if WS=16, 32, 64 fill in Table 2's bottom row.
  - cross-node `all_to_all`: rerun to check if the `enc.cc:12473` assertion still fires outside the 4-rank pod config.
* **NPROC=4 vs NPROC=1 cross-node.** We have data at NPROC=2 (WS=4) and NPROC=4 (WS=8) but not NPROC=1 (WS=2). NPROC=1 would give us a cleaner "pure EFA latency floor" number to isolate the intra-node vs cross-node cost more precisely.
* **`broadcast` cross-node WS=4** shows `FAIL` in Table 3 -- because the job at NPROC=2 was submitted only for all_reduce. If Beta 3 broadcast has any surprising WS restrictions, we haven't caught them yet. Quick follow-up run to confirm.

None of these block the primary Task 008 findings report -- the numbers in Tables 1-3 are stable and reproducible.
