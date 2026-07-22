DEFAULT_WARMUPS = 5
DEFAULT_TRIALS = 50
DEFAULT_TYPE = 'float'
DEFAULT_BACKEND = 'nccl'
DEFAULT_UNIT = 'Gbps'
DEFAULT_DIST = 'torch'
DEFAULT_MAXSIZE = 24
TORCH_DISTRIBUTED_DEFAULT_PORT = 29500

# --- Neuron additions ---
# Backend string for torch.distributed on AWS Trainium under PyTorch Native.
# torch_neuronx registers a distributed backend under the name 'neuron' when
# imported.  This is NOT nccl, xla, or gloo.
NEURON_BACKEND = 'neuron'

# HBM budget per logical NeuronCore (bytes).
# trn2 LNC=2: 24 GB per logical core (each logical core spans 2 physical cores + 2 HBM banks)
# trn2 LNC=1: 12 GB per logical core (8 logical cores sharing 4 HBM banks)
# We keep a conservative default (12 GB) that works for both configurations and
# leave headroom for the runtime, NEFF cache, and activation memory.
NEURON_HBM_PER_LOGICAL_CORE_BYTES = 12 * 1024 * 1024 * 1024
