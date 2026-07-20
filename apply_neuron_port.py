#!/usr/bin/env python3
"""Mechanical port of the 5 remaining EleutherAI cookbook communication
benchmark files to the Neuron backend.

Applies exactly four transformations:
  1. .cuda(local_rank)                 ->  .to(device_str)
  2. torch.cuda.empty_cache()          ->  empty_cache()
  3. Ran out of GPU memory             ->  Ran out of device memory
  4. torch.cuda.Event(enable_timing=True)  (two consecutive lines that make the
     start/end pair)                   ->  make_timer_events() (one line)
  5. Add:  device_str = device_for(local_rank)  after that pair.
  6. start_event.record()              ->  start_event = record_event(start_event)
  7. end_event.record()                ->  end_event = record_event(end_event)
  8. Replace 'duration = start_event.elapsed_time(end_event) / 1000' with
     'duration = elapsed_seconds(start_event, end_event)'

Every CUDA-path behavior is preserved because make_timer_events/record_event/
elapsed_seconds/device_for/empty_cache all fall back to torch.cuda.* when
_NEURON_ACTIVE is False (module-level flag in utils.py, set only by
init_torch_distributed(backend='neuron')).
"""
from pathlib import Path
import re
import sys

FILES = [
    "all_gather.py",
    "all_to_all.py",
    "reduce_scatter.py",
    "broadcast.py",
    "pt2pt.py",
]

BENCH_DIR = Path("/Users/jburtoft/Documents/opencode/working/collective/task-010/cookbook/benchmarks/communication")


def port_file(path: Path) -> None:
    src = path.read_text()
    original = src

    # 1. Tensor placement:  .cuda(local_rank)  ->  .to(device_str)
    src = src.replace(".cuda(local_rank)", ".to(device_str)")

    # 2. Cache flush: torch.cuda.empty_cache()  ->  empty_cache()
    src = src.replace("torch.cuda.empty_cache()", "empty_cache()")

    # 3. Warning text: GPU -> device
    src = src.replace(
        "WARNING: Ran out of GPU memory. Exiting comm op.",
        "WARNING: Ran out of device memory. Exiting comm op.",
    )
    src = src.replace(
        "WARNING: Ran out of GPU memory. Try to reduce the --mem-factor argument!",
        "WARNING: Ran out of device memory. Try to reduce the --mem-factor argument!",
    )

    # 4. Timer creation pair -> single helper call + device_str resolution
    # Original pattern (with any leading indent):
    #     start_event = torch.cuda.Event(enable_timing=True)
    #     end_event = torch.cuda.Event(enable_timing=True)
    pair_pattern = re.compile(
        r"^(?P<indent>[ \t]*)start_event = torch\.cuda\.Event\(enable_timing=True\)\n"
        r"(?P=indent)end_event = torch\.cuda\.Event\(enable_timing=True\)\n",
        flags=re.MULTILINE,
    )
    src, n_pairs = pair_pattern.subn(
        r"\g<indent>start_event, end_event = make_timer_events()\n"
        r"\g<indent>device_str = device_for(local_rank)\n",
        src,
    )
    if n_pairs != 1:
        print(f"WARNING: {path.name} had {n_pairs} timer pair replacements (expected 1)",
              file=sys.stderr)

    # 5. Record calls:
    src = src.replace("start_event.record()", "start_event = record_event(start_event)")
    src = src.replace("end_event.record()", "end_event = record_event(end_event)")

    # 6. Elapsed time: start_event.elapsed_time(end_event) / 1000
    #    -> elapsed_seconds(start_event, end_event)
    src = re.sub(
        r"start_event\.elapsed_time\(end_event\)\s*/\s*1000",
        "elapsed_seconds(start_event, end_event)",
        src,
    )

    if src == original:
        print(f"WARNING: {path.name} unchanged after port -- check patterns", file=sys.stderr)
    else:
        path.write_text(src)
        print(f"ported: {path.name}")


def main() -> None:
    for name in FILES:
        p = BENCH_DIR / name
        if not p.exists():
            print(f"MISSING: {p}", file=sys.stderr)
            sys.exit(1)
        port_file(p)


if __name__ == "__main__":
    main()
