import math
import os
import sys
from typing import Union

with open(sys.argv[0]) as f:
    code = f.read()  # read the code of this file ASAP, for logging
import uuid
import time
import glob
from dataclasses import dataclass
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch

torch.empty(1, device="cuda", requires_grad=True).backward()  # prevents a bug on some systems
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Import CMA model
from cma_model import CMAModel, CascadeMemoryAttention
from cma_components import CMAConfig


# -----------------------------------------------------------------------------
# Muon optimizer (reusing from reference)

@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2  # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A  # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iteration steps to use.
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, rank=0, world_size=1):
        self.rank = rank
        self.world_size = world_size
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        params: list[Tensor] = [*params]
        param_groups = []
        for size in {p.numel() for p in params}:
            b = torch.empty(world_size, size, dtype=torch.bfloat16, device="cuda")
            group = dict(params=[p for p in params if p.numel() == size],
                         update_buffer=b, update_buffer_views=[b[i] for i in range(world_size)])
            param_groups.append(group)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            update_buffer: Tensor = group["update_buffer"]
            update_buffer_views: list[Tensor] = group["update_buffer_views"]
            # generate weight updates in distributed fashion
            params: list[Tensor] = group["params"]
            handle = None
            params_world = None

            def update_prev():  # optimized Muon implementation contributed by @YouJiacheng
                handle.wait()
                for p_world, g_world in zip(params_world, update_buffer_views):
                    p_world.add_(g_world.view_as(p_world),
                                 alpha=-group["lr"] * max(1, p_world.size(-2) / p_world.size(-1)) ** 0.5)

            for base_i in range(len(params))[::self.world_size]:
                if base_i + self.rank < len(params):
                    p = params[base_i + self.rank]
                    g = p.grad
                    assert g is not None
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf: Tensor = state["momentum_buffer"]
                    buf.lerp_(g, 1 - group["momentum"])
                    g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                    g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"]).flatten()
                else:
                    g = update_buffer_views[self.rank]
                if base_i > 0:
                    update_prev()  # async all_gather instead of sync all_reduce by @YouJiacheng
                handle = dist.all_gather_into_tensor(update_buffer, g, async_op=True)
                params_world = params[base_i: base_i + self.world_size]
            update_prev()


# -----------------------------------------------------------------------------
# CMA-specific data handler

class CMADataHandler:
    """Handles data loading for CMA training with chunked processing"""

    def __init__(self, train_files: str, val_files: str, train_tokens: int, val_tokens: int, rank: int, world_size: int):
        self.train_files = train_files
        self.val_files = val_files
        self.train_tokens = train_tokens
        self.val_tokens = val_tokens
        self.rank = rank
        self.world_size = world_size


        # For CMA, we don't use batch sequences like GPT, we process one sequence at a time
        self.batch_size = 1

        self.chunk_size = 0

    def set_chunk_size(self, chunk_size: int):
        self.chunk_size = chunk_size

    def _load_data_shard(self, file: Path):
        header = torch.from_file(str(file), False, 256, dtype=torch.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        num_tokens = int(header[2])
        with file.open("rb", buffering=0) as f:
            tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
            f.seek(256 * 4)
            nbytes = f.readinto(tokens.numpy())
            assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
        return tokens

    def get_training_data_generator(self):
        """Generate training data in chunks for CMA processing"""
        files = [Path(file) for file in sorted(glob.glob(self.train_files))]
        file_iter = iter(files)
        tokens, pos = self._load_data_shard(next(file_iter)), 0

        while True:
            # For training, we need sequences longer than chunk_size to allow full update cycles
            min_seq_len = self.train_tokens #chunk_size * self.train_seq_len_multiplier
            max_seq_len = self.train_tokens #chunk_size * (self.train_seq_len_multiplier * 2)

            # Check if we need to load next shard
            if pos + min_seq_len >= len(tokens):
                try:
                    tokens, pos = self._load_data_shard(next(file_iter)), 0
                except StopIteration:
                    if master_process:  # Optional: Log epoch completion/restart
                        print0(f"Data loader: Reached end of training files, restarting.", console=True)
                    file_iter = iter(files)  # Restart from beginning
                    tokens, pos = self._load_data_shard(next(file_iter)), 0

            # Get a sequence that's at least min_seq_len
            actual_seq_len = min(len(tokens) - pos, max_seq_len)

            # Make sure we have at least min_seq_len tokens
            if actual_seq_len < min_seq_len:
                continue

            # Extract sequence
            seq = tokens[pos:pos + actual_seq_len]

            # Send to device as int32 for input and int64 for targets
            inputs = seq[:-1].to(dtype=torch.int32, device="cuda", non_blocking=True)
            targets = seq[1:].to(dtype=torch.int64, device="cuda", non_blocking=True)

            pos += actual_seq_len
            yield inputs, targets

    def get_validation_data_generator(self):
        """Generate validation data in chunks"""
        files = [Path(file) for file in sorted(glob.glob(self.val_files))]
        val_batch_size = self.world_size * self.train_tokens

        #assert self.val_tokens % val_batch_size == 0
        val_steps = math.ceil(float(self.val_tokens) / val_batch_size)
        processed_tokens = 0

        file_iter = iter(files)
        tokens, pos = self._load_data_shard(next(file_iter)), 0

        for _ in range(val_steps):
            seq_len = min(self.train_tokens, self.val_tokens - processed_tokens)
            # Check if we need to load next shard
            if pos + seq_len + 1 >= len(tokens):
                tokens, pos = self._load_data_shard(next(file_iter)), 0

            # Extract validation sequence
            seq = tokens[pos + self.rank * seq_len:pos + (self.rank + 1) * seq_len + 1]

            # Send to device
            inputs = seq[:-1].to(dtype=torch.int32, device="cuda", non_blocking=True)
            targets = seq[1:].to(dtype=torch.int64, device="cuda", non_blocking=True)

            pos += self.world_size * seq_len
            processed_tokens += self.world_size * seq_len
            yield inputs, targets


# -----------------------------------------------------------------------------
# Training helper

class CMATrainingHelper:
    """Helper class to manage CMA training mechanics"""

    def __init__(self, model: nn.Module, ddp_active: bool):
        self.model = model # This is the potentially DDP-wrapped model
        self.ddp_active = ddp_active
        self.underlying_model = model.module if ddp_active else model

    def train_step(self, inputs_token_ids: Union[list, torch.Tensor], targets_token_ids: Union[list, torch.Tensor],
                   step: int, total_steps: int) -> tuple[Tensor, Tensor]:
        """
        Perform a single training step for CMA.
        inputs_token_ids: The full list of token IDs for the input sequence.
        targets_token_ids: The full list of token IDs for the target sequence (shifted input).
        Returns: (total_loss_tensor, prediction_loss_normalized_tensor) - UN SCALED
        """
        self.model.train()
        self.underlying_model.set_training_step(step, total_steps)

        # CMAModel.forward expects List[int] or similar for input_ids when not pre-chunked
        # The `inputs_token_ids` from data_handler is already List[int] effectively (from Tensor.tolist())

        # Logits from model.forward are now potentially concatenated from all Pass 2 chunks
        # Gate_loss is also aggregated from all Pass 2 chunks
        logits, gate_loss = self.model(inputs_token_ids, training_mode=True)

        pred_loss_normalized = torch.tensor(0.0, device=logits.device)  # Match logits device

        if logits.numel() > 0 and logits.size(1) > 0:
            # targets_token_ids should correspond to the sequence that generated the logits.
            # If logits are from all Pass 2 chunks, targets_token_ids should be the
            # concatenation of targets for those chunks.
            # The data loader provides one long `inputs` and `targets`.
            # `inputs` is `seq[:-1]`, `targets` is `seq[1:]`.
            # `CMAModel.forward` uses `inputs` (as `input_ids`) to run the cycle.
            # The concatenated logits from Pass 2 will correspond to the full sequence processed in Pass 2.
            # So, `targets_token_ids` (which is `seq[1:]`) should be used directly,
            # but sliced to match the length of the concatenated logits.

            num_predicted_tokens = logits.size(1)

            # Convert targets_token_ids to tensor for cross_entropy
            targets_tensor = torch.tensor(targets_token_ids, dtype=torch.long, device=logits.device)

            if targets_tensor.numel() >= num_predicted_tokens:
                actual_targets_for_loss = targets_tensor[:num_predicted_tokens].contiguous()
            else:
                # This case implies an issue, logits are longer than available original targets
                print0(f"Warning: Logits length ({num_predicted_tokens}) > original targets length ({targets_tensor.numel()}). Truncating logits.")
                logits = logits[:, :targets_tensor.numel(), :]
                actual_targets_for_loss = targets_tensor.contiguous()
                num_predicted_tokens = logits.size(1)  # Update to new length

            if num_predicted_tokens > 0:
                pred_loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    actual_targets_for_loss.view(-1),
                    reduction='sum'
                )
                pred_loss_normalized = pred_loss / max(num_predicted_tokens, 1)
            # else pred_loss_normalized remains 0.0
        # else pred_loss_normalized remains 0.0

        total_loss = pred_loss_normalized
        if gate_loss is not None:
            total_loss = total_loss + gate_loss  # gate_loss is already scaled and aggregated

        return total_loss, pred_loss_normalized

    def val_step(self, inputs_token_ids: Union[list, torch.Tensor], targets_token_ids: Union[list, torch.Tensor], step: int, total_steps: int) -> float:
        self.model.eval()
        self.underlying_model.set_training_step(step, total_steps)

        val_loss_value = 0.0
        with torch.no_grad():
            # training_mode=False, gate_loss is ignored
            #print(inputs_token_ids)
            logits, _gate_loss_ignored = self.model(inputs_token_ids, training_mode=False)

            if logits.numel() > 0 and logits.size(1) > 0:
                num_predicted_tokens = logits.size(1)
                targets_tensor = torch.tensor(targets_token_ids, dtype=torch.long, device=logits.device)

                if targets_tensor.numel() >= num_predicted_tokens:
                    actual_targets_for_loss = targets_tensor[:num_predicted_tokens].contiguous()
                else:
                    print0(f"Warning (VAL): Logits length ({num_predicted_tokens}) > original targets length ({targets_tensor.numel()}). Truncating logits.")
                    logits = logits[:, :targets_tensor.numel(), :]
                    actual_targets_for_loss = targets_tensor.contiguous()
                    num_predicted_tokens = logits.size(1)

                if num_predicted_tokens > 0:
                    pred_loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        actual_targets_for_loss.view(-1),
                        reduction='sum'
                    )
                    val_loss_value = (pred_loss / max(num_predicted_tokens, 1)).item()
            # else val_loss_value remains 0.0

        return val_loss_value


# -----------------------------------------------------------------------------
# Main training script

@dataclass
class Hyperparameters:
    DEBUG_LEVEL = 2  # 0=warnings/errors/etc. only, 1=debug, 2=info

    # data
    train_files = "data/fineweb10B/fineweb_train_*.bin"
    val_files = "data/fineweb10B/fineweb_val_*.bin"
    val_tokens = 20480 #10485760
    train_seq_len = 10240  # Should be ~10240??

    # CMA specific
    chunk_size_min = 256 #128 #256  # Chunk size starts here and...
    chunk_size_max = 256 #768  # Will be progressively increased to this max
    max_memory_size = 128 #3072
    reverse_memory_size = 32 #320
    memory_cap_length = 49152

    # model architecture  
    vocab_size = 50257
    embed_dim = 512 #768
    n_heads = 4 #6
    head_dim = 128 #128  # Typically embed_dim // n_heads
    n_layers = 6 #12

    # optimization
    num_iterations = 1770
    cooldown_frac = 0.4
    gradient_accumulation_steps = 1  # Accumulate gradients over multiple sequences before update

    # evaluation and logging
    val_loss_every = 125
    save_checkpoint = False


args = Hyperparameters()

assert args.train_seq_len > args.chunk_size_min

# Setup distributed training
try:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
except KeyError:
    rank = 0
    world_size = 1
    device = torch.device("cuda", 0)
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['LOCAL_RANK'] = '0'
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '8080'
assert torch.cuda.is_available()
torch.cuda.set_device(device)
dist.init_process_group(backend="nccl", device_id=device)
dist.barrier()
master_process = (rank == 0)

# Logging setup
logfile = None
if master_process:
    run_id = uuid.uuid4()
    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{run_id}.txt"
    print(logfile)


def print0(s, console=False):
    if master_process:
        with open(logfile, "a") as f:
            if console:
                print(s, flush=True)
            print(s, file=f, flush=True)


# Log initial information
#print0(code)
print0("=" * 100)
print0(f"Running Python {sys.version}")
print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}")
print0("=" * 100)

# Create CMA configuration
config = CMAConfig(
    chunk_size=args.chunk_size_max,  # This will be updated during training
    max_memory_size=args.max_memory_size,
    memory_cap_length=args.memory_cap_length,
    reverse_memory_size=args.reverse_memory_size,
    embed_dim=args.embed_dim,
    n_heads=args.n_heads,
    head_dim=args.head_dim,
    n_layers=args.n_layers,
    share_initial_memory=True,
    reset_memory_on_cycle=True,
    enable_mask_future_dropout=False,
    gate_regularization_type="l1",  # Enable gate regularization as per training methodology
    gate_regularization_strength=0.001,
    # Use simple layer structure for now
    layer_structure=[
        {"type": "local_only"} if i % args.n_layers != args.n_layers-1 else {"type": "memory_update"}
        for i in range(args.n_layers)
    ],
    DEBUG_LEVEL=args.DEBUG_LEVEL,
    logfile=logfile
)

# Create model
model_without_ddp = CMAModel(config, args.vocab_size).cuda() # Create the base model
total_params = sum(p.numel() for p in model_without_ddp.parameters())
print0(f"Model created with {total_params} parameters", console=True)

# DDP Setup
ddp_active = False
if world_size > 1:
    # Broadcast initial parameters of the base model before wrapping with DDP
    for param in model_without_ddp.parameters():
        dist.broadcast(param.detach(), 0)

    # Wrap model with DDP
    # find_unused_parameters can be True if some layers/outputs are conditional
    # and might not be used in every forward pass, which can happen with CMA's
    # complex paths (e.g. empty logits, different pass types).
    # Set to False if confident all parameters are used in every training forward pass.
    # For CMA, True is safer initially.
    model = DDP(model_without_ddp, device_ids=device, find_unused_parameters=True)
    ddp_active = True
    print0("DDP is active.")
else:
    model = model_without_ddp  # Use the base model if not distributed
    print0("DDP is not active (single GPU or world_size=1).")

# Set up data handlers (pass the underlying model if DDP is active for config access)
# The helper will call methods on `model`, which will be the DDP-wrapped one if active.
# Model methods like reset_state, set_training_step, config access should be on the module.
# If DDP is active, model.module gives access to the original CMAModel instance.
underlying_model = model.module if ddp_active else model

data_handler = CMADataHandler(
    args.train_files, args.val_files, args.train_seq_len, args.val_tokens, rank, world_size
)
train_loader = data_handler.get_training_data_generator()
# Pass the underlying_model to CMATrainingHelper if it needs to access specific
# attributes of CMAModel not exposed by DDP wrapper (like .config directly).
# However, CMATrainingHelper as written primarily calls model.train(), model.eval(), model.forward(),
# model.set_training_step(), model.reset_state() which are fine with DDP wrapper.
# Let's adjust CMATrainingHelper to expect the potentially DDP-wrapped model and access .module if needed.
training_helper = CMATrainingHelper(model, ddp_active)  # Pass model and ddp_active flag
print0("Data handler and training helper created.")

# Collect parameters for different optimizers
#print0("Parameter Grouping...", console=True)
# --- Adam Parameters ---
adam_params_dict = {}  # Use a dictionary to avoid duplicate additions by name

# 1. LM Head (Explicit)
# lm_head.weight is typically a 2D matrix.
# If lm_head has a bias, it's 1D.
for name, p in underlying_model.lm_head.named_parameters():
    if p.requires_grad:
        adam_params_dict[f"lm_head.{name}"] = p
#print0(f"Collected {len(adam_params_dict)} params for lm_head initially for Adam.", console=True)

# 2. Embeddings (Token embedding primarily)
# token_embedding.weight is 2D
current_adam_count = len(adam_params_dict)
for name, p in underlying_model.token_embedding.named_parameters():
    full_name = f"token_embedding.{name}"
    if p.requires_grad and full_name not in adam_params_dict:
        adam_params_dict[full_name] = p
#print0(f"Collected {len(adam_params_dict) - current_adam_count} params for token_embedding for Adam.", console=True)

# 3. Initial Memory Parameters (Explicit by name)
# These are 3D but often treated with Adam-like optimizers due to their unique role.
current_adam_count = len(adam_params_dict)
if hasattr(underlying_model, 'initial_fwd_params'):
    for i, p in enumerate(underlying_model.initial_fwd_params):
        if p.requires_grad and f"initial_fwd_params.{i}" not in adam_params_dict:
            adam_params_dict[f"initial_fwd_params.{i}"] = p
if hasattr(underlying_model, 'initial_rev_s_params'):
    for i, p in enumerate(underlying_model.initial_rev_s_params):
        if p.requires_grad and f"initial_rev_s_params.{i}" not in adam_params_dict:
            adam_params_dict[f"initial_rev_s_params.{i}"] = p
if hasattr(underlying_model, 'initial_rev_p_params'):
    for i, p in enumerate(underlying_model.initial_rev_p_params):
        if p.requires_grad and f"initial_rev_p_params.{i}" not in adam_params_dict:
            adam_params_dict[f"initial_rev_p_params.{i}"] = p
#print0(f"Collected {len(adam_params_dict) - current_adam_count} params for initial_memories for Adam.", console=True)

# --- Iterate All Parameters for Remaining Adam Groups and Muon ---
# Keep track of parameters assigned to Adam to give the rest to Muon
adam_assigned_param_ids = {id(p) for p in adam_params_dict.values()}

# Parameters specifically for Adam (usually 1D or special 2D like gates)
other_adam_params_by_group = {
    "cma_attn_gates": [],  # CascadeMemoryAttention.gate_proj (weights & biases)
    "cma_control_proj": [],  # CascadeMemoryAttention.control_proj (weights)
    "cma_update_gates": [],  # fwd_memory_gate_proj, rev_memory_gate_proj (weights)
    "other_biases_and_scalars": [],  # All other 1D params
}

# Parameters for Muon (2D+ matrices not explicitly in Adam)
muon_candidate_params = []

for name, param in underlying_model.named_parameters():
    if not param.requires_grad or id(param) in adam_assigned_param_ids:
        continue  # Skip if no grad or already explicitly handled for Adam

    # CMA Specific Components that often go to Adam
    if "control_proj" in name:  # control_proj.weight (2D)
        other_adam_params_by_group["cma_control_proj"].append(param)
        adam_assigned_param_ids.add(id(param))
    elif "gate_proj" in name:  # CascadeMemoryAttention.gate_proj (attn gate W+B)
        other_adam_params_by_group["cma_attn_gates"].append(param)
        adam_assigned_param_ids.add(id(param))
    elif "memory_gate_proj" in name:  # fwd/rev_memory_gate_proj (update gate W)
        other_adam_params_by_group["cma_update_gates"].append(param)
        adam_assigned_param_ids.add(id(param))
    elif param.ndim < 2:  # All other 1D params (biases, single norms if any)
        other_adam_params_by_group["other_biases_and_scalars"].append(param)
        adam_assigned_param_ids.add(id(param))
    elif param.ndim >= 2:  # Candidate for Muon
        muon_candidate_params.append(param)
    else:
        print0(f"Warning: Unhandled parameter during iteration: {name}", console=True)

# Construct Adam param groups
adam_param_groups_list = [
    dict(params=list(underlying_model.lm_head.parameters()), lr=0.20, name="lm_head"),
    dict(params=list(underlying_model.token_embedding.parameters()), lr=0.6, name="token_embedding")
]
if hasattr(underlying_model, 'initial_fwd_params') and underlying_model.initial_fwd_params:
    adam_param_groups_list.append(
        dict(params=list(underlying_model.initial_fwd_params), lr=0.06, name="initial_fwd_mem"))
if hasattr(underlying_model, 'initial_rev_s_params') and underlying_model.initial_rev_s_params:
    adam_param_groups_list.append(
        dict(params=list(underlying_model.initial_rev_s_params), lr=0.06, name="initial_rev_s_mem"))
if hasattr(underlying_model, 'initial_rev_p_params') and underlying_model.initial_rev_p_params:
    adam_param_groups_list.append(
        dict(params=list(underlying_model.initial_rev_p_params), lr=0.06, name="initial_rev_p_mem"))

if other_adam_params_by_group["cma_control_proj"]:
    adam_param_groups_list.append(
        dict(params=other_adam_params_by_group["cma_control_proj"], lr=0.03, name="cma_control_proj"))
if other_adam_params_by_group["cma_attn_gates"]:
    adam_param_groups_list.append(
        dict(params=other_adam_params_by_group["cma_attn_gates"], lr=0.03, name="cma_attn_gates"))
if other_adam_params_by_group["cma_update_gates"]:
    adam_param_groups_list.append(
        dict(params=other_adam_params_by_group["cma_update_gates"], lr=0.03, name="cma_update_gates"))
if other_adam_params_by_group["other_biases_and_scalars"]:
    adam_param_groups_list.append(
        dict(params=other_adam_params_by_group["other_biases_and_scalars"], lr=0.04, name="other_biases_scalars"))

adam_param_groups_filtered = [pg for pg in adam_param_groups_list if pg['params']]

#print0("--- Adam Groups ---", console=True)
total_adam_params = 0
for pg in adam_param_groups_filtered:
    current_pg_params = sum(p.numel() for p in pg['params'])
    total_adam_params += current_pg_params
    #print0(f"Adam group: {pg['name']}, Num params: {current_pg_params}, LR: {pg['lr']}", console=True)
#print0(f"Total Adam params: {total_adam_params}", console=True)

# Muon gets all remaining 2D+ params
final_muon_params = []
for p_muon_cand in muon_candidate_params:
    # Double check it wasn't somehow added to Adam (shouldn't be necessary with current logic, but safe)
    if id(p_muon_cand) not in adam_assigned_param_ids:
        final_muon_params.append(p_muon_cand)
    else:  # This case means a 2D+ param was explicitly assigned to Adam (e.g. control_proj if it was 2D)
        pass  # Already handled by Adam

#print0("--- Muon Group ---", console=True)
total_muon_params = sum(p.numel() for p in final_muon_params)
#print0(f"Muon params: Num params: {total_muon_params}", console=True)

# Final Verification
total_optimized_params = 0
all_optimized_param_ids = set()

for pg in adam_param_groups_filtered:
    for p in pg['params']:
        if id(p) in all_optimized_param_ids:
            print0(f"CRITICAL WARNING: Param {p.shape} assigned to multiple Adam groups!", console=True)
        all_optimized_param_ids.add(id(p))
        total_optimized_params += p.numel()

for p in final_muon_params:
    if id(p) in all_optimized_param_ids:
        print0(f"CRITICAL WARNING: Param {p.shape} assigned to both Adam and Muon!", console=True)
    all_optimized_param_ids.add(id(p))
    total_optimized_params += p.numel()  # This line was missing in my prev total count

#print0(f"Total params in optimizer groups: {total_optimized_params}", console=True)
total_model_params_requiring_grad = sum(p.numel() for p in underlying_model.parameters() if p.requires_grad)
#print0(f"Total model params requiring grad: {total_model_params_requiring_grad}", console=True)

if total_optimized_params != total_model_params_requiring_grad:
    print0("CRITICAL WARNING: Optimizer parameter count MISMATCH!", console=True)
    all_model_param_ids = {id(p) for p in underlying_model.parameters() if p.requires_grad}

    missing_from_optimizer_ids = all_model_param_ids - all_optimized_param_ids
    if missing_from_optimizer_ids:
        print0(f"Missing from optimizer ({len(missing_from_optimizer_ids)} params):", console=True)
        for name, p_model in underlying_model.named_parameters():
            if id(p_model) in missing_from_optimizer_ids:
                print0(f"  - {name} (Shape: {p_model.shape}, Numel: {p_model.numel()})", console=True)

    extra_in_optimizer_ids = all_optimized_param_ids - all_model_param_ids
    if extra_in_optimizer_ids:  # Should be empty
        print0(f"CRITICAL WARNING: Extra in optimizer ({len(extra_in_optimizer_ids)} params) - THIS IS A BUG IN SCRIPT", console=True)
else:
    #print0("Parameter coverage check: OKAY", console=True)
    pass

optimizer1 = torch.optim.Adam(adam_param_groups_filtered, betas=(0.8, 0.95), eps=1e-10, fused=True)
optimizer2 = Muon(final_muon_params, lr=0.05, momentum=0.95, rank=rank, world_size=world_size)
optimizers = [optimizer1, optimizer2]

for opt in optimizers:
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]
adam_params = {p for g in optimizer1.param_groups for p in g['params']}
print0(f"Adam ▶ {len(adam_params)} tensors, {total_adam_params:,} scalars", console=True)
print0(f"Muon ▶ {len(final_muon_params)} tensors, {total_muon_params:,} scalars", console=True)
#print0(f"Combined ▶ {total_optimized_params:,} scalars", console=True)
assert total_params == total_optimized_params


# Learning rate schedule
def get_lr(step: int):
    x = step / args.num_iterations
    assert 0 <= x < 1
    if x < 1 - args.cooldown_frac:
        return 1.0
    else:
        w = (1 - x) / args.cooldown_frac
        return w * 1.0 + (1 - w) * 0.1


# Curriculum learning: progressively increase chunk size
def get_chunk_size(step: int):
    x = step / (args.num_iterations * 0.8)
    # Linearly increase from 256 to 768 tokens
    min_chunk_size = args.chunk_size_min
    max_chunk_size = args.chunk_size_max
    current_size = int(min_chunk_size + x * (max_chunk_size - min_chunk_size))
    return min(current_size, max_chunk_size)


# Compile model (if using PyTorch 2.0+)
print0("Compiling model...", console=True)
try:
    model = torch.compile(model, dynamic=False)
    print0("Compile complete.", console=True)
except:
    print0("Warning: torch.compile not available, running uncompiled model", console=True)

# Training loop
training_time_ms = 0
torch.cuda.synchronize()
t0 = time.perf_counter()

print0("Starting main training loop.", console=True)
train_steps = args.num_iterations
for step in range(train_steps + 1):
    last_step = (step == train_steps)

    # Get current chunk size for curriculum learning
    current_chunk_size = get_chunk_size(step)
    data_handler.set_chunk_size(current_chunk_size)

    # Update model's chunk_size (Fix #1: Chunk Size Curriculum)
    current_model_instance = model.module if ddp_active else model
    current_model_instance.config.chunk_size = current_chunk_size
    current_model_instance.chunk_processor.config.chunk_size = current_chunk_size

    # --------------- VALIDATION SECTION -----------------
    if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
        # Stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.perf_counter() - t0)

        val_steps = math.ceil(float(args.val_tokens) / args.train_seq_len)
        print0(f"Running validation ({val_steps} steps).", console=True)
        model.eval()
        val_loader = data_handler.get_validation_data_generator()
        val_loss = 0
        val_count = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                model.reset_state()
                inputs_list = inputs.cpu().tolist()
                targets_list = targets.cpu().tolist()
                loss = training_helper.val_step(inputs_list, targets_list, step, train_steps)
                val_loss += loss
                val_count += 1
                print(".", end='', flush=True)
                print0(f"########## VALIDATION STEP {val_count} of {val_steps} COMPLETE ##########")

        if val_count > 0:
            val_loss /= val_count
        else:
            val_loss = 0.0

        # Properly handle distributed validation loss
        val_loss_tensor = torch.tensor(val_loss, device=device)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
        val_loss = val_loss_tensor.item()

        print0("\nValidation complete.", console=True)
        print0(f"step:{step}/{train_steps} val_loss:{val_loss:.4f} chunk_size:{current_chunk_size} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms", console=True)

        model.train()

        # Start the clock again
        torch.cuda.synchronize()
        t0 = time.perf_counter()

    if last_step:
        if master_process and args.save_checkpoint:
            save_model_state = model.module.state_dict() if ddp_active else model.state_dict()
            log = dict(step=step, code=code, model=save_model_state,  # Save underlying model
                       config=underlying_model.config.to_dict() if hasattr(underlying_model.config,
                                                                           'to_dict') else vars(
                           underlying_model.config),  # Save config
                       optimizers=[opt.state_dict() for opt in optimizers])
            os.makedirs(f"logs/{run_id}", exist_ok=True)
            torch.save(log, f"logs/{run_id}/state_step{step:06d}.pt")
        break

    # --------------- TRAINING SECTION -----------------
    # Zero gradients once before accumulation
    model.zero_grad(set_to_none=True)  # Moved here

    accumulated_total_loss_for_logging = 0.0
    accumulated_pred_loss_for_logging = 0.0

    for accum_step in range(args.gradient_accumulation_steps):
        # Reset model state at the start of each sequence
        # DDP forwards this to the underlying module if the method exists there
        model.reset_state()

        # Get next sequence (inputs and targets are full sequence tensors)
        raw_inputs_tensor, raw_targets_tensor = next(train_loader)

        # Convert to lists for CMATrainingHelper and CMAModel
        # inputs_for_model is seq[:-1], targets_for_loss_calc is seq[1:]
        # CMAModel.forward will use inputs_for_model.
        # CMATrainingHelper.train_step will use targets_for_loss_calc to align with returned logits.
        inputs_for_model_list = raw_inputs_tensor.cpu().tolist()
        targets_for_loss_calc_list = raw_targets_tensor.cpu().tolist()

        total_loss_tensor, pred_loss_normalized_tensor = training_helper.train_step(
            inputs_for_model_list,
            targets_for_loss_calc_list,  # Pass the corresponding targets
            step,
            train_steps  # Corrected from total_steps to train_steps for consistency
        )

        # Accumulate for logging (using .item() for Python floats)
        accumulated_total_loss_for_logging += total_loss_tensor.item()
        accumulated_pred_loss_for_logging += pred_loss_normalized_tensor.item()

        # Scale loss for gradient accumulation before backward pass
        scaled_loss_for_backward = total_loss_tensor / args.gradient_accumulation_steps

        # Backward pass on the scaled loss tensor
        # The DDP hook will handle allreduce during backward if model is DDP wrapped.
        # If not DDP wrapped, manual allreduce is needed after accumulation.
        scaled_loss_for_backward.backward()

    # Average the accumulated losses for logging
    avg_total_loss_for_logging = accumulated_total_loss_for_logging / args.gradient_accumulation_steps
    avg_pred_loss_for_logging = accumulated_pred_loss_for_logging / args.gradient_accumulation_steps

    # Gradient reduction across all workers (ONLY if not using DDP's automatic gradient sync)
    if not ddp_active and world_size > 1:
        for param in model.parameters():  # model.parameters() is fine here
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)

    # --- START GRADIENT NORM LOGGING (for step % log_every_n_steps == 0) ---
    if master_process and (step + 1) % 10 == 0:  # Log every 10 steps, or adjust frequency
        print0(f"--- Grad Norms for Step {step + 1} ---")
        # Access the underlying model if DDP is active
        model_to_inspect = model.module if ddp_active else model

        # 1. Initial Memory Parameters
        if hasattr(model_to_inspect, 'initial_fwd_params') and model_to_inspect.initial_fwd_params:
            for i, param in enumerate(model_to_inspect.initial_fwd_params):
                if param.grad is not None:
                    print0(f"  initial_fwd_params[{i}]: {param.grad.norm().item():.4e}")
                else:
                    print0(f"  initial_fwd_params[{i}]: No Grad")
        if hasattr(model_to_inspect,
                   'initial_rev_s_params') and model_to_inspect.initial_rev_s_params:  # For completeness
            for i, param in enumerate(model_to_inspect.initial_rev_s_params):
                if param.grad is not None:
                    print0(f"  initial_rev_s_params[{i}]: {param.grad.norm().item():.4e}")
                else:
                    print0(f"  initial_rev_s_params[{i}]: No Grad")
        if hasattr(model_to_inspect, 'initial_rev_p_params') and model_to_inspect.initial_rev_p_params:
            for i, param in enumerate(model_to_inspect.initial_rev_p_params):
                if param.grad is not None:
                    print0(f"  initial_rev_p_params[{i}]: {param.grad.norm().item():.4e}")
                else:
                    print0(f"  initial_rev_p_params[{i}]: No Grad")

        # 2. CascadeMemoryAttention Layer Parameters (iterate through layers)
        for layer_idx, block in enumerate(model_to_inspect.layers):
            if hasattr(block, 'attn') and isinstance(block.attn, CascadeMemoryAttention):
                # Attention Gating Proj
                if block.attn.gate_proj.weight.grad is not None:
                    print0(f"  Layer {layer_idx} AttnGate W: {block.attn.gate_proj.weight.grad.norm().item():.4e}")
                if block.attn.gate_proj.bias.grad is not None:
                    print0(f"  Layer {layer_idx} AttnGate B: {block.attn.gate_proj.bias.grad.norm().item():.4e}")

                # Memory Update Components (if they exist on this layer instance)
                if block.attn.is_memory_update:
                    # Forward Memory Update Gate
                    if hasattr(block.attn,
                               'fwd_memory_gate_proj') and block.attn.fwd_memory_gate_proj.weight.grad is not None:
                        print0(
                            f"  Layer {layer_idx} FwdMemUpdGate W: {block.attn.fwd_memory_gate_proj.weight.grad.norm().item():.4e}")

                    # Reverse Memory Update Gate
                    if hasattr(block.attn,
                               'rev_memory_gate_proj') and block.attn.rev_memory_gate_proj.weight.grad is not None:
                        print0(
                            f"  Layer {layer_idx} RevMemUpdGate W: {block.attn.rev_memory_gate_proj.weight.grad.norm().item():.4e}")

                    # QKV for Forward Memory Update
                    if hasattr(block.attn,
                               'fwd_memory_q_proj') and block.attn.fwd_memory_q_proj.weight.grad is not None:
                        print0(
                            f"  Layer {layer_idx} FwdMemUpd_Q W: {block.attn.fwd_memory_q_proj.weight.grad.norm().item():.4e}")
                    if hasattr(block.attn,
                               'fwd_memory_k_proj') and block.attn.fwd_memory_k_proj.weight.grad is not None:
                        print0(
                            f"  Layer {layer_idx} FwdMemUpd_K W: {block.attn.fwd_memory_k_proj.weight.grad.norm().item():.4e}")
                    if hasattr(block.attn,
                               'fwd_memory_v_proj') and block.attn.fwd_memory_v_proj.weight.grad is not None:
                        print0(
                            f"  Layer {layer_idx} FwdMemUpd_V W: {block.attn.fwd_memory_v_proj.weight.grad.norm().item():.4e}")

                    # QKV for Reverse Memory Update (shared)
                    if hasattr(block.attn,
                               'rev_memory_q_proj') and block.attn.rev_memory_q_proj.weight.grad is not None:
                        print0(
                            f"  Layer {layer_idx} RevMemUpd_Q W: {block.attn.rev_memory_q_proj.weight.grad.norm().item():.4e}")
                    # ... K and V for reverse, if you want to be exhaustive, but Q is indicative

            # You could also log norms for block.mlp if desired
            # if block.mlp[0].weight.grad is not None: # First linear layer of MLP
            #     print0(f"  Layer {layer_idx} MLP1 W: {block.mlp[0].weight.grad.norm().item():.4e}")

        print0(f"--- End Grad Norms for Step {step + 1} ---")
    # --- END GRADIENT NORM LOGGING ---

    # Set learning rates
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * get_lr(step)

    # Momentum warmup for Muon
    for group in optimizer2.param_groups:
        frac = min(step / 300, 1)
        group["momentum"] = (1 - frac) * 0.85 + frac * 0.95

    # Step optimizers
    for opt in optimizers:
        opt.step()

    # Logging
    approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
    print0(f"########## TRAINING STEP {step + 1} of {train_steps} COMPLETE ##########")
    print0(f"step:{step + 1}/{train_steps} pred_loss:{avg_pred_loss_for_logging:.4f} total_loss:{avg_total_loss_for_logging:.4f} chunk_size:{current_chunk_size} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / (step + 1):.2f}ms", console=True)

print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
       f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)
dist.destroy_process_group()