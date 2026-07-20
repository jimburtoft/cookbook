# Phase E: PyTorch Native Beta 3 Cookbook Communication Benchmarks -- Comparison Tables

Data source: /local/home/jburtoft/Documents/opencode/working/collective/task-010/logs/aggregated.csv (1159 points across all environments)


### Table 1 -- PT Native (trn2.3xlarge, LNC=1) vs Task 009 nccom-test

| Collective (PT Native) | WS | Size | nccom-test (us) | PT Native (us) | Framework overhead |
|----|----|----|----|----|----|
| all_reduce (as sendrecv analog) | 2 | 4 KB | 12.6 | 372.3 | 29.6x |
| all_reduce (as sendrecv analog) | 2 | 1 MB | 24.0 | 608.9 | 25.4x |
| all_reduce (as sendrecv analog) | 2 | 256 MB | 2585.8 | 9304.5 | 3.6x |
| all_reduce | 8 | 256 MB (peak) | -- bus BW 123.9 GB/s | bus BW 63.5 GB/s | 51% of nccom |

Notes:
* Task 009 measured **sendrecv** (a 2-rank P2P primitive) via `nccom-test sendrecv`. PyTorch Native Beta 3 does not expose `dist.send`/`dist.recv` on the Neuron backend (steering/pytorch-native.md:703), so the closest framework analog is all_reduce on WS=2. Both cross a single NeuronLink hop.
* Task 009 measured **all-reduce** (`nccom-test allr -r 8`) at 8 cores and reported **peak 123.9 GB/s bus BW** (rdh algorithm, 256 KB - 64 MB). Our PT Native peak is measured at 256 MB and reflects the ring algorithm the runtime selects for large messages.



### Table 2 -- Cross-node scaling (all_reduce, LNC=2)

| Environment | World | 4 KB (us) | 1 MB (us) | 128 MB (us) | Peak Bus BW (GB/s) |
|----|----|----|----|----|----|
| trn2.3xl (LNC=2) | 4 | 410.8 | 355.9 | 2397.3 | 102.3 |
| trn2.3xl (LNC=1) | 8 | 1732.7 | 2061.4 | 4901.1 | 63.5 |
| trn2.48xl single-node | 8 | 995.9 | 1006.6 | 3681.3 | 77.0 |
| trn2.48xl single-node | 16 | 281.7 | 328.1 | 2612.4 | 117.4 |
| trn2.48xl single-node | 32 | 370.2 | 412.9 | 2962.4 | 116.0 |
| trn2.48xl single-node | 64 | 5174.2 | 5258.3 | 6709.5 | 142.6 |
| trn2.48xl 2-node (EFA) | 4 | 851.2 | 764.7 | 3145.8 | 16.0 |
| trn2.48xl 2-node (EFA) | 8 | 988.0 | 1156.2 | 9980.3 | 26.7 |
| trn2.48xl 2-node (EFA) | 16 | (FAIL: nproc_per_node=8 init) | -- | -- | -- |



### Table 3 -- Collective coverage matrix

Legend: `OK` = collective ran to completion (or completed then OOM'd at the largest size, which is expected on HBM-limited configs); `FAIL` = did not produce any measurements; `--` = not attempted.

| Environment | LNC | WS | all_reduce | all_gather | reduce_scatter | all_to_all | broadcast | pt2pt |
|----|----|----|----|----|----|----|----|----|
| trn2_3xl | 1 | 2 | OK | OK | OK | FAIL | OK | FAIL |
| trn2_3xl | 1 | 4 | OK | OK | OK | OK | OK | FAIL |
| trn2_3xl | 1 | 8 | OK | OK | OK | OK | OK | FAIL |
| trn2_3xl | 2 | 2 | OK | OK | OK | FAIL | OK | FAIL |
| trn2_3xl | 2 | 4 | OK | OK | OK | OK | OK | FAIL |
| trn2_48xl_multi | 2 | 4 | OK | FAIL | FAIL | FAIL | FAIL | FAIL |
| trn2_48xl_multi | 2 | 8 | OK | OK | OK | FAIL | OK | FAIL |
| trn2_48xl_single | 2 | 8 | OK | OK | OK | OK | OK | FAIL |
| trn2_48xl_single | 2 | 16 | OK | OK | OK | OK | OK | FAIL |
| trn2_48xl_single | 2 | 32 | OK | OK | OK | OK | OK | FAIL |
| trn2_48xl_single | 2 | 64 | OK | OK | FAIL | FAIL | FAIL | FAIL |

Known Beta 3 constraints (confirmed by these runs):
* `pt2pt` (dist.send/dist.recv): **NEVER WORKS** on the Neuron backend. Error: 'No backend type associated with device type neuron'. Documented in steering/pytorch-native.md:703.
* `all_to_all` (`AllToAllXlaOp`): **only WS in {4, 8, 16, or multiples of 32} is supported.** WS=2 fails with 'unsupported world size 2'. Not documented in Beta 3 release notes.
* `reduce_scatter`: fails on empty tensors (skip M=0 fix landed in commit 57cf076).