import torch
import sys, os, time

COMMS_BENCH_DIR = os.path.join(os.path.dirname(__file__), "../")
sys.path.append(COMMS_BENCH_DIR)

from communication.utils import *
from communication.constants import *


def timed_all_reduce(input, start_event, end_event, args):
    if args.dist == 'torch':
        import torch.distributed as dist
    elif args.dist == 'deepspeed':
        import deepspeed.comm as dist

    # --- Decide framework vs NKI-kernel path once, outside the timed loop ---
    world_size = dist.get_world_size()
    nki_path = use_nki(args)
    if nki_path:
        # Reshape to (128, N/128) so the NKI HBM kernel accepts it.
        # nki_dispatch handles caching the wrapped kernel per shape.
        _call = lambda t: nki_dispatch('all_reduce', t, world_size)
    else:
        _call = lambda t: dist.all_reduce(t, async_op=args.async_op)

    sync_all()
    # Warmups, establish connections, etc.
    for i in range(args.warmups):
        _call(input)
    sync_all()

    # time the actual comm op trials times and average it
    start_event = record_event(start_event)
    for i in range(args.trials):
        _call(input)
    end_event = record_event(end_event)
    sync_all()
    duration = elapsed_seconds(start_event, end_event)

    # maintain and clean performance data
    avg_duration = duration / args.trials
    size = input.element_size() * input.nelement()
    n = dist.get_world_size()
    tput, busbw = get_bw('all_reduce', size, avg_duration, args)
    tput_str, busbw_str, duration_str = get_metric_strings(args, tput, busbw, avg_duration)
    desc = f'{input.nelement()}x{input.element_size()}'

    if not args.raw:
        size = convert_size(size)

    print_rank_0(f"{size:<20} {desc:25s} {duration_str:20s} {tput_str:20s} {busbw_str:20s}")


def run_all_reduce(local_rank, args):
    if args.dist == 'torch':
        import torch.distributed as dist
    elif args.dist == 'deepspeed':
        import deepspeed.comm as dist

    # Prepare benchmark header
    print_header(args, 'all_reduce')

    world_size = dist.get_world_size()
    global_rank = dist.get_rank()

    start_event, end_event = make_timer_events()
    device_str = device_for(local_rank)

    if args.scan:
        M_LIST = []
        for x in (2**p for p in range(1, args.maxsize)):
            M_LIST.append(x)

        sync_all()
        # loop over various tensor sizes
        for M in M_LIST:
            global_rank = dist.get_rank()
            try:
                mat = torch.ones(world_size, M,
                                 dtype=getattr(torch, args.dtype)).to(device_str)
                sync_all()
                input = ((mat.mul_(float(global_rank))).view(-1))
                del mat
                empty_cache()
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    if dist.get_rank() == 0:
                        print('WARNING: Ran out of device memory. Exiting comm op.')
                    sync_all()
                    break
                else:
                    raise e
            sync_all()
            timed_all_reduce(input, start_event, end_event, args)
    else:
        # Send the biggest message size our devices can fit. If you're facing OOM errors, reduce the mem_factor
        # Don't need output tensor, so we double mem_factor
        elements_per_gpu = max_numel(comm_op='all_reduce',
                                     dtype=getattr(torch, args.dtype),
                                     mem_factor=args.mem_factor * 2,
                                     local_rank=local_rank,
                                     args=args)
        try:
            mat = torch.ones(elements_per_gpu, dtype=getattr(torch,
                                                             args.dtype)).to(device_str)
            input = ((mat.mul_(float(global_rank))).view(-1))
        except RuntimeError as e:
            if 'out of memory' in str(e):
                if dist.get_rank() == 0:
                    print('WARNING: Ran out of device memory. Try to reduce the --mem-factor argument!')
                sync_all()
                return
            else:
                raise e
        sync_all()
        timed_all_reduce(input, start_event, end_event, args)


if __name__ == "__main__":
    args = benchmark_parser().parse_args()
    rank = args.local_rank
    init_processes(local_rank=rank, args=args)
    run_all_reduce(local_rank=rank, args=args)
