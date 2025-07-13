from typing import Tuple, Optional, List, Dict
import re
import tiktoken
import torch.nn as nn
import wandb
from triton.language import bfloat16

from cma_components import *


# -----------------------------------------------------------------------------
# Configuration

@dataclass
class LayerGroup:
    """Represents a group of layers and their associated memory update layer."""
    group_idx: int
    layer_indices: List[int] = field(default_factory=list)
    memory_update_layer_idx: Optional[int] = None
    read_only_layer_indices: List[int] = field(default_factory=list)
    local_only_layer_indices: List[int] = field(default_factory=list)
    has_memory: bool = False


class CanonConv(nn.Module):
    """Causal 1D depth-wise convolution for Canon."""

    def __init__(self, dim: int, kernel_size: int = 4):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size, groups=dim, padding=kernel_size - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.size(1) == 0:
            return x
        
        x_transposed = x.transpose(1, 2)
        y = self.conv(x_transposed)
        y_sliced = y[:, :, :-(self.conv.padding[0])] if self.conv.padding[0] > 0 else y
        return x + y_sliced.transpose(1, 2)


# -----------------------------------------------------------------------------
# Attention layers

class CausalSelfAttention(nn.Module):
    """Standard causal self-attention for local-only layers"""

    def __init__(self, config: CMAConfig, layer_idx: int):
        super().__init__()
        assert config.embed_dim % config.n_heads == 0

        self.config = config
        self.layer_idx = layer_idx
        self.n_heads = config.n_heads
        self.head_dim = config.embed_dim // config.n_heads

        self.qkv = nn.Linear(config.embed_dim, 3 * config.embed_dim, bias=False)
        self.out_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        self.canon_b = CanonConv(config.embed_dim, config.canon_kernel_size) if config.enable_canon else None

        if config.output_proj_zero_init:
            nn.init.zeros_(self.out_proj.weight)

    def forward(self, x: Tensor, causal_mask: Optional[Tensor] = None) -> Tensor:
        B, T, C = x.size()
        
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [tensor.view(B, T, self.n_heads, self.head_dim).transpose(1, 2) 
                   for tensor in (q, k, v)]

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        if causal_mask is None:
            causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn_scores.masked_fill_(causal_mask, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)

        if self.canon_b is not None:
            attn_output = self.canon_b(attn_output)

        return self.out_proj(attn_output)


class CascadeMemoryAttention(nn.Module):
    """CMA attention layer with memory integration and optional memory update"""

    def __init__(self, config: CMAConfig, layer_idx: int, is_memory_update: bool = False):
        super().__init__()
        assert config.embed_dim % config.n_heads == 0

        self.config = config
        self.layer_idx = layer_idx
        self.is_memory_update = is_memory_update
        self.n_heads = config.n_heads
        self.head_dim = config.embed_dim // config.n_heads

        # QKV projections
        self.q_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        self.k_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        self.v_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        self.out_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=False)

        # Canon-B (only for memory_read layers)
        self.canon_b = (CanonConv(config.embed_dim, config.canon_kernel_size) 
                       if config.enable_canon and not is_memory_update else None)

        # Control token integration
        if config.integration_method == "query_fusion":
            self.control_proj = nn.Linear(5, config.embed_dim, bias=False)
            nn.init.normal_(self.control_proj.weight, std=config.ctrl_init_scale)

        # Adaptive gating
        self.gate_proj = nn.Linear(config.embed_dim, config.n_heads, bias=True)
        nn.init.constant_(self.gate_proj.bias, config.gate_bias_init)

        # Memory update components
        if is_memory_update:
            for prefix in ['fwd_memory', 'rev_memory']:
                setattr(self, f'{prefix}_q_proj', nn.Linear(config.embed_dim, config.embed_dim, bias=False))
                setattr(self, f'{prefix}_k_proj', nn.Linear(config.embed_dim, config.embed_dim, bias=False))
                setattr(self, f'{prefix}_v_proj', nn.Linear(config.embed_dim, config.embed_dim, bias=False))
                setattr(self, f'{prefix}_gate_proj', nn.Linear(2 * config.embed_dim, config.embed_dim, bias=False))

        if config.output_proj_zero_init:
            nn.init.zeros_(self.out_proj.weight)

    def forward(
            self,
            x: Tensor,
            forward_memory: Optional[Tensor] = None,
            reverse_memory: Optional[Tensor] = None,
            control_tokens: Optional[Dict[str, float]] = None,
            do_memory_update: bool = False,
            write_mask: Optional[Tensor] = None,
            decay_weights: Optional[Tensor] = None,
            is_reverse_update: bool = False
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        B, T, C = x.size()

        # Apply control tokens to queries
        q = self.q_proj(x)
        if control_tokens is not None and self.config.integration_method == "query_fusion":
            ctrl_vec = torch.tensor([
                control_tokens["generation_flag"],
                control_tokens["memory_mode_flag"],
                control_tokens["memory_usage_ratio"],
                control_tokens["memory_density_ratio"],
                control_tokens["chunk_position_ratio"]
            ], device=x.device).unsqueeze(0).unsqueeze(0).expand(B, T, -1).to(x.dtype)
            q = q + self.control_proj(ctrl_vec)

        if T == 0:
            return x, None, None, None

        # Integrate memory into keys and values
        k, v = self.k_proj(x), self.v_proj(x)
        if forward_memory is not None or reverse_memory is not None:
            if self.config.DEBUG_LEVEL > 0: 
                print0(f"DEBUG: CMA Layer {self.layer_idx}: Fwd mem shape: {forward_memory.shape if forward_memory is not None else None}, Rev mem shape: {reverse_memory.shape if reverse_memory is not None else None}", self.config.logfile)
            k, v = self._integrate_memory(k, v, forward_memory, reverse_memory)

        # Attention computation
        q, k, v = [tensor.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2) 
                   for tensor in (q, k, v)]

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Apply causal mask
        S = k.size(-2)
        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        full_mask = torch.zeros(T, S, device=x.device, dtype=torch.bool)
        full_mask[:, :T] = causal
        attn_scores.masked_fill_(full_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)

        # Apply gating if memory is present
        gate_reg_loss = None
        if forward_memory is not None or reverse_memory is not None:
            attn_output, gate_reg_loss = self._apply_gate(q, attn_weights, v, T)
        else:
            attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)

        if self.canon_b is not None:
            attn_output = self.canon_b(attn_output)

        output = self.out_proj(attn_output)

        # Handle memory updates
        updated_forward_memory = None
        updated_reverse_memory = None
        if do_memory_update and self.is_memory_update:
            if forward_memory is not None and not is_reverse_update:
                updated_forward_memory = self._update_memory(
                    forward_memory, x, write_mask=write_mask, is_forward=True
                )
            if reverse_memory is not None and is_reverse_update:
                updated_reverse_memory = self._update_memory(
                    reverse_memory, x, write_mask=write_mask, 
                    decay_weights=decay_weights, is_forward=False
                )

        return output, updated_forward_memory, updated_reverse_memory, gate_reg_loss

    def _integrate_memory(self, k: Tensor, v: Tensor, forward_memory: Optional[Tensor], 
                         reverse_memory: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        """Integrate memory states into keys and values"""
        memory_k_list, memory_v_list = [k], [v]

        for memory in [forward_memory, reverse_memory]:
            if memory is not None:
                memory_k_list.append(self.k_proj(memory.to(k.dtype)))
                memory_v_list.append(self.v_proj(memory.to(k.dtype)))

        return torch.cat(memory_k_list, dim=1), torch.cat(memory_v_list, dim=1)

    def _apply_gate(self, q: torch.Tensor, attn_weights: torch.Tensor, 
                   v: torch.Tensor, chunk_len: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply adaptive gating between chunk and memory attention outputs"""
        B, h, T, _ = q.shape

        if T == 0:
            return torch.zeros_like(q), None

        S_keys = v.size(-2)
        mem_len = S_keys - chunk_len
        
        if self.config.DEBUG_LEVEL > 0: 
            print0(f"DEBUG: CMA Layer {self.layer_idx} _apply_gate: chunk_len={chunk_len}, S_keys={S_keys}, mem_len={mem_len}", self.config.logfile)
        
        if mem_len == 0:
            return torch.matmul(attn_weights, v), None

        # Split attention weights and values
        v_chunk, v_mem = torch.split(v, [chunk_len, mem_len], dim=-2)
        w_chunk, w_mem = torch.split(attn_weights, [chunk_len, mem_len], dim=-1)

        Y_chunk = torch.matmul(w_chunk, v_chunk)
        Y_mem = torch.matmul(w_mem, v_mem)

        # Compute adaptive gate
        gate_logits = self.gate_proj(q.permute(0, 2, 1, 3).reshape(B, T, -1))
        g = torch.sigmoid(gate_logits).permute(0, 2, 1).unsqueeze(-1)

        # Debug logging
        if mem_len > 0:
            v_chunk_std = v_chunk.std(dim=(-1, -2)).mean().item() if v_chunk.numel() > 0 else 0.0
            v_mem_std = v_mem.std(dim=(-1, -2)).mean().item() if v_mem.numel() > 0 else 0.0
            g_mean = g.mean().item()
            if self.config.DEBUG_LEVEL > 0 and (torch.rand(1).item() < 0.05 or not self.training):
                print0(f"DEBUG CMA Layer {self.layer_idx} _apply_gate details (training={self.training}): Step {getattr(self.config, 'training_step', 'N/A')}, g_mean={g_mean:.4f}, v_chunk_std={v_chunk_std:.4f}, v_mem_std={v_mem_std:.4f}", self.config.logfile)
            if self.config.log_wandb:
                wandb.log({f"gate_stats/layer_{self.layer_idx}_g_mean":g_mean, f"gate_stats/layer_{self.layer_idx}_v_chunk_std":v_chunk_std, f"gate_stats/layer_{self.layer_idx}_v_mem_std":v_mem_std}, step=self.config.training_step)

        Y = Y_chunk + g * Y_mem

        # Regularization
        reg_loss = None
        if self.config.gate_regularization_type == "l1":
            reg_loss = self.config.gate_regularization_strength * torch.mean(torch.abs(g))
        elif self.config.gate_regularization_type == "entropy":
            ent = -(g * torch.log(g + 1e-8) + (1 - g) * torch.log(1 - g + 1e-8))
            reg_loss = self.config.gate_regularization_strength * torch.mean(ent)
        
        if self.training and self.config.gate_regularization_type is not None and self.config.gate_saturation_penalty:
            gate_saturation_penalty = torch.mean(torch.square(g - 0.5)) * 0.01
            reg_loss = gate_saturation_penalty if reg_loss is None else reg_loss + gate_saturation_penalty

        return Y, reg_loss

    def _update_memory(self, memory_old: Tensor, chunk_tokens: Tensor, 
                      write_mask: Optional[Tensor] = None,
                      decay_weights: Optional[Tensor] = None,
                      is_forward: bool = True) -> Tensor:
        """Update memory state based on chunk tokens"""
        B, M, C = memory_old.size()

        # Select appropriate projections
        prefix = 'fwd_memory' if is_forward else 'rev_memory'
        q_proj = getattr(self, f'{prefix}_q_proj')
        k_proj = getattr(self, f'{prefix}_k_proj')
        v_proj = getattr(self, f'{prefix}_v_proj')
        gate_proj = getattr(self, f'{prefix}_gate_proj')

        # Compute attention update
        memory_old = memory_old.to(chunk_tokens.dtype)
        memory_q = q_proj(memory_old).view(B, M, self.n_heads, self.head_dim).transpose(1, 2)
        chunk_k = k_proj(chunk_tokens).view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        chunk_v = v_proj(chunk_tokens).view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = torch.matmul(memory_q, chunk_k.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        delta = torch.matmul(attn_weights, chunk_v)
        delta = delta.transpose(1, 2).contiguous().view(B, M, C)

        if decay_weights is not None:
            delta = delta * decay_weights

        # Apply gated update
        gate = torch.sigmoid(gate_proj(torch.cat([memory_old, delta], dim=-1).to(attn_weights.dtype)))
        memory_new = gate * memory_old + (1 - gate) * delta

        if write_mask is not None:
            write_mask = write_mask.unsqueeze(-1).expand_as(memory_new)
            memory_new = torch.where(write_mask, memory_new, memory_old)

        return memory_new


# -----------------------------------------------------------------------------
# Block definition

class Block(nn.Module):
    """Transformer block with group-aware memory handling."""

    def __init__(self, config: CMAConfig, layer_idx: int, layer_type: str = "local_only"):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = layer_type

        # Attention layer
        if layer_type == "skip":
            self.attn = None
        elif layer_type in ["memory_read", "memory_update"]:
            self.attn = CascadeMemoryAttention(config, layer_idx, is_memory_update=(layer_type == "memory_update"))
        else:  # local_only
            self.attn = CausalSelfAttention(config, layer_idx)

        # MLP
        self.mlp_fc1 = nn.Linear(config.embed_dim, 4 * config.embed_dim, bias=False)
        self.mlp_act = nn.GELU()
        self.mlp_fc2 = nn.Linear(4 * config.embed_dim, config.embed_dim, bias=False)

        # Canon layers
        if config.enable_canon and layer_type != "skip":
            self.canon_a = CanonConv(config.embed_dim, config.canon_kernel_size)
            self.canon_c = CanonConv(config.embed_dim, config.canon_kernel_size)
            self.canon_d = CanonConv(4 * config.embed_dim, config.canon_kernel_size) if layer_type == "local_only" else None
        else:
            self.canon_a = self.canon_c = self.canon_d = None

    def forward(
            self,
            x: Tensor,
            M_fwd_dict: Dict[int, Tensor],
            M_rev_ahead_dict: Dict[int, Tensor],
            M_rev_persist_dict: Dict[int, Tensor],
            group_id: int,
            mode: str,
            control_tokens: Optional[Dict[str, float]] = None,
            write_mask: Optional[Tensor] = None,
            decay_weights: Optional[Tensor] = None,
            total_logical_sequence_length: int = 0
    ) -> Tuple[Tensor, Optional[Tuple[int, Tensor]], Optional[Tuple[int, Tensor]], Optional[Tensor]]:
        
        # Determine memory inputs and update settings
        fwd_mem_in = rev_mem_in = None
        do_memory_update = is_reverse_update = False
        
        if self.layer_type not in ["local_only", "skip"]:
            if mode == "forward":
                fwd_mem_in = M_fwd_dict.get(group_id)
                rev_mem_in = M_rev_persist_dict.get(group_id)
            elif mode == "lookahead_reverse":
                rev_mem_in = M_rev_ahead_dict.get(group_id)
                is_reverse_update = True
            elif mode == "persistent_reverse":
                rev_mem_in = M_rev_persist_dict.get(group_id)
                is_reverse_update = True
            elif mode == "generate":
                fwd_mem_in = M_fwd_dict.get(group_id)
                rev_mem_in = M_rev_persist_dict.get(group_id)
            else:
                raise ValueError(f"Unknown mode: {mode}")
            
            do_memory_update = (self.layer_type == "memory_update" and 
                              total_logical_sequence_length > self.config.chunk_size)

        # Debug logging
        if self.layer_type == "memory_update" and self.config.DEBUG_LEVEL > 0:
            print0(f"DEBUG Block L{self.layer_idx} (mode={mode}, training={self.training}): do_memory_update eval: total_log_seq_len={total_logical_sequence_length}, chunk_size={self.config.chunk_size}, final_do_update_decision={do_memory_update}", self.config.logfile)

        # Apply canon_a
        if self.canon_a is not None:
            x = self.canon_a(x)
        
        # Attention
        updated_fwd_result = updated_rev_result = gate_reg_loss = None
        if self.attn is not None:
            residual = x
            x_norm = norm(x)

            if isinstance(self.attn, CascadeMemoryAttention):
                attn_output, fwd_mem_out, rev_mem_out, gate_reg_loss = self.attn(
                    x_norm, forward_memory=fwd_mem_in, reverse_memory=rev_mem_in,
                    control_tokens=control_tokens, do_memory_update=do_memory_update,
                    write_mask=write_mask if not is_reverse_update else None,
                    decay_weights=decay_weights if is_reverse_update else None,
                    is_reverse_update=is_reverse_update
                )
                if fwd_mem_out is not None: 
                    updated_fwd_result = (group_id, fwd_mem_out)
                if rev_mem_out is not None: 
                    updated_rev_result = (group_id, rev_mem_out)
            else:
                attn_output = self.attn(x_norm)
            
            x = residual + attn_output

        # Apply canon_c
        if self.canon_c is not None:
            x = self.canon_c(x)

        # MLP
        residual = x
        x_norm = norm(x)
        mlp_hidden = self.mlp_act(self.mlp_fc1(x_norm))
        if self.canon_d is not None:
            mlp_hidden = self.canon_d(mlp_hidden)
        x = residual + self.mlp_fc2(mlp_hidden)

        return x, updated_fwd_result, updated_rev_result, gate_reg_loss


# -----------------------------------------------------------------------------
# Main model class

class CMAModel(nn.Module):
    """CMA-based language model"""

    def __init__(self, config: CMAConfig, vocab_size: int, tokenizer=None):
        super().__init__()
        self.config = config
        if config.log_wandb:
            import wandb
        self.vocab_size = vocab_size
        self.tokenizer = tokenizer or tiktoken.get_encoding("gpt2")

        # Initialize components
        self.chunk_processor = ChunkProcessor(config, self.tokenizer)
        self.memory_manager = MemoryManager(config)
        self.control_token_generator = ControlTokenGenerator(config)
        self.token_embedding = nn.Embedding(vocab_size, config.embed_dim)

        # Parse layers and groups
        self._parse_layer_structure(config)
        self.lm_head = nn.Linear(config.embed_dim, vocab_size, bias=False)

        # Initialize memory parameters
        self._initialize_memory_parameters()

        # State tracking
        self.M_fwd: Dict[int, Tensor] = {}
        self.M_rev_ahead: Dict[int, Tensor] = {}
        self.M_rev_persist: Dict[int, Tensor] = {}
        self.closed_chunks: List[List[int]] = []
        self.current_chunk_tokens: List[int] = []
        self.current_chunk_text: str = ""
        self.total_tokens_processed = 0
        self.tokens_since_persistent_update = 0
        self.training_step = 0
        self.total_training_steps = 10000
        self.total_seq_len = 0

        # Initialize weights
        self.apply(self._init_weights)
        #self.lm_head.weight.data.zero_()
        self._finalize_gate_bias_init()
        self._print_parameter_count()

    def _parse_layer_structure(self, config):
        """Parse layer structure and create layer groups"""
        self.layers = nn.ModuleList()
        self.layer_groups: List[LayerGroup] = []
        self.layer_idx_to_group_idx: Dict[int, int] = {}
        
        layer_idx = group_idx = 0

        for group_spec in config.layer_structure:
            if "group" in group_spec:
                # Handle grouped layers
                group_config = group_spec["group"]
                layer_types = group_config["layers"]
                repeat = group_config.get("repeat", 1)
                
                for _ in range(repeat):
                    current_group = LayerGroup(group_idx=group_idx)
                    update_layers = []
                    
                    for layer_type in layer_types:
                        actual_type = "skip" if layer_idx in config.skip_attention_layers else layer_type
                        layer = Block(config, layer_idx, actual_type)
                        self.layers.append(layer)
                        
                        current_group.layer_indices.append(layer_idx)
                        self.layer_idx_to_group_idx[layer_idx] = group_idx
                        
                        if layer_type == "memory_update":
                            update_layers.append(layer_idx)
                            current_group.has_memory = True
                        elif layer_type == "memory_read":
                            current_group.read_only_layer_indices.append(layer_idx)
                            current_group.has_memory = True
                        elif layer_type in ["local_only", "skip"]:
                            current_group.local_only_layer_indices.append(layer_idx)
                        
                        layer_idx += 1
                    
                    if len(update_layers) > 1:
                        raise ValueError(f"Group {group_idx}: >1 update layer")
                    if current_group.read_only_layer_indices and not update_layers:
                        raise ValueError(f"Group {group_idx}: read-only layers require an update layer")
                    
                    if update_layers:
                        current_group.memory_update_layer_idx = update_layers[0]
                    
                    self.layer_groups.append(current_group)
                    group_idx += 1
            else:
                # Handle individual layers
                layer_type = group_spec.get("type", "local_only")
                actual_type = "skip" if layer_idx in config.skip_attention_layers else layer_type
                layer = Block(config, layer_idx, actual_type)
                self.layers.append(layer)
                
                current_group = LayerGroup(group_idx=group_idx)
                current_group.layer_indices.append(layer_idx)
                self.layer_idx_to_group_idx[layer_idx] = group_idx
                
                if layer_type == "memory_update":
                    current_group.memory_update_layer_idx = layer_idx
                    current_group.has_memory = True
                elif layer_type == "memory_read":
                    current_group.read_only_layer_indices.append(layer_idx)
                    current_group.has_memory = True
                elif layer_type in ["local_only", "skip"]:
                    current_group.local_only_layer_indices.append(layer_idx)
                
                self.layer_groups.append(current_group)
                group_idx += 1
                layer_idx += 1

        # Validate read-only layers have update layers
        for group in self.layer_groups:
            if group.read_only_layer_indices and group.memory_update_layer_idx is None:
                raise ValueError(f"Group {group.group_idx}: read-only layers lacks update layer")

        self.num_layers = layer_idx
        self.num_groups = group_idx
        self.num_memory_groups = sum(1 for g in self.layer_groups if g.has_memory)
        print0(f"Parsed {self.num_layers} layers into {self.num_groups} groups ({self.num_memory_groups} with memory).", self.config.logfile)

    def _initialize_memory_parameters(self):
        """Initialize memory parameters for groups"""
        self.group_id_to_memory_idx: Dict[int, int] = {}
        
        if self.num_memory_groups == 0:
            print0("No memory groups found. Initial memory parameters will not be created.", self.config.logfile)
            self.initial_fwd_params = self.initial_rev_la_params = self.initial_rev_p_params = nn.ParameterList()
            return

        if self.config.share_initial_memory:
            print0("Using shared initial memory parameters.", self.config.logfile)
            shared_fwd = nn.Parameter(torch.randn(1, self.config.max_memory_size, self.config.embed_dim) * self.config.memory_init_scale)
            shared_rev_la = nn.Parameter(torch.randn(1, self.config.reverse_memory_size, self.config.embed_dim) * self.config.memory_init_scale)
            shared_rev_p = nn.Parameter(torch.randn(1, self.config.reverse_memory_size, self.config.embed_dim) * self.config.memory_init_scale)
            
            self.initial_fwd_params = nn.ParameterList([shared_fwd] * self.num_memory_groups)
            self.initial_rev_la_params = nn.ParameterList([shared_rev_la] * self.num_memory_groups)
            self.initial_rev_p_params = nn.ParameterList([shared_rev_p] * self.num_memory_groups)
            
            for group in self.layer_groups:
                if group.has_memory:
                    self.group_id_to_memory_idx[group.group_idx] = 0
        else:
            print0("Using dedicated initial memory parameters per group.", self.config.logfile)
            fwd_params, rev_la_params, rev_p_params = [], [], []
            memory_idx = 0
            
            for group in self.layer_groups:
                if group.has_memory:
                    self.group_id_to_memory_idx[group.group_idx] = memory_idx
                    fwd_params.append(nn.Parameter(torch.randn(1, self.config.max_memory_size, self.config.embed_dim) * self.config.memory_init_scale))
                    rev_la_params.append(nn.Parameter(torch.randn(1, self.config.reverse_memory_size, self.config.embed_dim) * self.config.memory_init_scale))
                    rev_p_params.append(nn.Parameter(torch.randn(1, self.config.reverse_memory_size, self.config.embed_dim) * self.config.memory_init_scale))
                    memory_idx += 1
            
            self.initial_fwd_params = nn.ParameterList(fwd_params)
            self.initial_rev_la_params = nn.ParameterList(rev_la_params)
            self.initial_rev_p_params = nn.ParameterList(rev_p_params)

        # Ensure parameters are on correct device
        device = self.token_embedding.weight.device
        for param_list in [self.initial_fwd_params, self.initial_rev_la_params, self.initial_rev_p_params]:
            for param in param_list:
                param.data = param.data.to(device)

        print0(f"Created {memory_idx if not self.config.share_initial_memory else 1} sets of initial memory parameters.", self.config.logfile)

    def _finalize_gate_bias_init(self):
        """Explicitly set gate bias after generic initialization"""
        for layer_block in self.layers:
            if isinstance(layer_block.attn, CascadeMemoryAttention):
                if hasattr(layer_block.attn.gate_proj, 'bias') and layer_block.attn.gate_proj.bias is not None:
                    nn.init.constant_(layer_block.attn.gate_proj.bias, self.config.gate_bias_init)
                    if self.config.DEBUG_LEVEL > 0 and self.config.logfile:
                        print0(f"DEBUG: Explicitly set gate_proj.bias for layer {layer_block.layer_idx} to {self.config.gate_bias_init}", self.config.logfile)

    def _print_parameter_count(self):
        """Print parameter count breakdown"""
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Calculate memory parameter counts
        initial_memory_params = 0
        for param_list in [self.initial_fwd_params, self.initial_rev_la_params, self.initial_rev_p_params]:
            initial_memory_params += sum(p.numel() for p in param_list if p.requires_grad)
        
        if self.config.share_initial_memory and self.num_memory_groups > 1:
            params_per_set = initial_memory_params // (self.num_memory_groups * 3) if self.num_memory_groups > 0 else 0
            initial_memory_params = 3 * params_per_set

        print0(f"Model initialized with {total_params:,} total parameters", self.config.logfile)
        print0(f"  Initial memory parameters: {initial_memory_params:,}", self.config.logfile)
        print0(f"  Non-memory parameters: {total_params - initial_memory_params:,}", self.config.logfile)

    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def _initialize_memory_states(self, force_reset: bool = False):
        """Initialize or reset memory states"""
        if not self.layer_groups or not any(g.has_memory for g in self.layer_groups):
            return

        device = self.token_embedding.weight.device
        B = 1

        if force_reset or self.config.reset_memory_on_cycle:
            if self.config.DEBUG_LEVEL > 0:
                print0("DEBUG: Resetting memory states.", self.config.logfile)
            self.M_fwd.clear()
            self.M_rev_ahead.clear()
            self.M_rev_persist.clear()

        # Initialize missing memory states
        for group in self.layer_groups:
            if group.has_memory:
                group_idx = group.group_idx
                mem_idx = self.group_id_to_memory_idx[group_idx]

                if group_idx not in self.M_fwd:
                    self.M_fwd[group_idx] = self.initial_fwd_params[mem_idx].clone().to(device).repeat(B, 1, 1)
                if group_idx not in self.M_rev_ahead:
                    self.M_rev_ahead[group_idx] = self.initial_rev_la_params[mem_idx].clone().to(device).repeat(B, 1, 1)
                if group_idx not in self.M_rev_persist:
                    self.M_rev_persist[group_idx] = self.initial_rev_p_params[mem_idx].clone().to(device).repeat(B, 1, 1)

    def forward(self, input_data: Union[str, List[int], List[List[int]], torch.Tensor, None] = None,
                training_mode: bool = True) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Forward pass with chunking and memory update cycles"""

        self.training = training_mode
        if training_mode:
            super().train()
        else:
            super().eval()

        self._initialize_memory_states(force_reset=False)
        dev = self.token_embedding.weight.device

        final_logits: Optional[torch.Tensor] = None
        final_gate_loss: Optional[torch.Tensor] = None

        new_tokens_list_for_call: List[int] = []  # Renamed for clarity
        num_new_tokens_for_this_call = 0
        input_processing_mode: str
        trigger_update_cycle = False
        all_cycle_logits: Optional[torch.Tensor] = None
        chunks_for_cycle: Optional[List[List[int]]] = None
        logits_full_buffer: Optional[torch.Tensor] = None

        if (input_data is None or input_data == "" or input_data == [] or
                (isinstance(input_data, torch.Tensor) and input_data.numel() == 0)):
            self.current_total_logical_sequence_length = sum(len(c) for c in self.closed_chunks) + len(
                self.current_chunk_tokens)
            if self.config.DEBUG_LEVEL > 0:
                print0(
                    f"DEBUG CMAModel.forward: Empty/None input. Logical seq_len for current buffer set to {self.current_total_logical_sequence_length}",
                    self.config.logfile)

            if self.current_chunk_tokens:
                logits_full_buffer, gate_loss_chunk = self._process_current_chunk(
                    generation_mode_hint=not self.training)
                final_logits = logits_full_buffer
                # final_gate_loss remains None for eval, or if training this path (unlikely for None input)
                if self.training:
                    final_gate_loss = gate_loss_chunk
            else:
                final_logits = torch.zeros(1, 0, self.vocab_size, device=dev)
            # num_new_tokens_for_this_call remains 0
        else:
            new_chunks_from_input: Optional[List[List[int]]] = None
            if isinstance(input_data, str):
                new_tokens_list_for_call = self.tokenizer.encode(input_data)
                input_processing_mode = "semantic"
            elif isinstance(input_data, list) and input_data and isinstance(input_data[0], list):
                new_chunks_from_input = [list(c) for c in input_data]
                new_tokens_list_for_call = [t for c_list in new_chunks_from_input for t in c_list]
                input_processing_mode = "caller_exact"
            else:
                new_tokens_list_for_call = input_data.tolist() if isinstance(input_data, torch.Tensor) else list(
                    input_data)
                input_processing_mode = "fixed"

            num_new_tokens_for_this_call = len(new_tokens_list_for_call)
            if self.config.DEBUG_LEVEL > 0: print0(
                f"DEBUG CMAModel.forward: Mode={input_processing_mode}, new_tokens length={num_new_tokens_for_this_call}",
                self.config.logfile)

            full_tokens_for_cycle: List[int] = []

            if input_processing_mode == "caller_exact":
                trigger_update_cycle = True
                full_tokens_for_cycle = [t for c in self.closed_chunks for t in c] + \
                                        self.current_chunk_tokens + new_tokens_list_for_call
            elif len(self.current_chunk_tokens) + len(new_tokens_list_for_call) >= self.config.chunk_size:
                trigger_update_cycle = True
                full_tokens_for_cycle = [t for c in self.closed_chunks for t in c] + \
                                        self.current_chunk_tokens + new_tokens_list_for_call
            elif self.training and self.current_chunk_tokens and num_new_tokens_for_this_call > 0 and input_processing_mode in {
                "semantic", "fixed"}:
                trigger_update_cycle = True
                full_tokens_for_cycle = [t for c in self.closed_chunks for t in c] + \
                                        self.current_chunk_tokens + new_tokens_list_for_call

            if trigger_update_cycle:
                if self.config.reset_memory_on_cycle:
                    self._initialize_memory_states(force_reset=True)

                self.current_total_logical_sequence_length = len(full_tokens_for_cycle)
                if self.config.DEBUG_LEVEL > 0:
                    print0(
                        f"DEBUG CMAModel.forward: Cycle triggered. Logical seq_len set to {self.current_total_logical_sequence_length}",
                        self.config.logfile)

                if input_processing_mode == "caller_exact":
                    assert new_chunks_from_input is not None
                    chunks_for_cycle = self.closed_chunks + \
                                       ([self.current_chunk_tokens] if self.current_chunk_tokens else []) + \
                                       new_chunks_from_input
                else:
                    if not full_tokens_for_cycle:
                        chunks_for_cycle = []
                    elif input_processing_mode == "semantic":
                        try:
                            #full_text = self.tokenizer.decode(full_tokens_for_cycle)
                            #chunks_for_cycle = self.chunk_processor.semantic_chunk_reverse_with_gap(full_text)
                            chunks_for_cycle = self.chunk_processor.semantic_chunk_reverse_with_gap(input_data)
                        except Exception as e:
                            print0(f"Error during semantic re-chunking: {e}. Falling back to fixed.",
                                   self.config.logfile)
                            chunks_for_cycle = self.chunk_processor.fixed_size_chunk_reverse_with_gap(
                                full_tokens_for_cycle)
                    else:
                        chunks_for_cycle = self.chunk_processor.fixed_size_chunk_reverse_with_gap(full_tokens_for_cycle)

                self.current_chunk_tokens = []
                self.current_chunk_text = ""

                if not chunks_for_cycle:
                    final_logits = torch.zeros(1, 0, self.vocab_size, device=dev)
                else:
                    all_cycle_logits, cycle_gate_loss = self._trigger_memory_update_cycle(chunks_for_cycle)

                    if self.training:
                        final_logits = all_cycle_logits
                        final_gate_loss = cycle_gate_loss
                    else:  # Evaluation/Generation after cycle
                        if all_cycle_logits is not None:
                            # CRITICAL: For evaluation, slice for the original new tokens of *this call*.
                            if num_new_tokens_for_this_call > 0:
                                if all_cycle_logits.size(1) >= num_new_tokens_for_this_call:
                                    final_logits = all_cycle_logits[:, -num_new_tokens_for_this_call:, :]
                                else:
                                    print0(
                                        f"Warning (EVAL CYCLE): All cycle logits len {all_cycle_logits.size(1)} < "
                                        f"num_new_tokens_for_this_call {num_new_tokens_for_this_call}. "
                                        f"Using all available cycle logits.",
                                        self.config.logfile
                                    )
                                    final_logits = all_cycle_logits
                            elif len(
                                    self.current_chunk_tokens) > 0:  # If no new tokens but cycle occurred and resulted in a current_chunk
                                final_logits = all_cycle_logits[:, -len(self.current_chunk_tokens):, :]
                            else:  # No new tokens and no current_chunk from cycle either
                                final_logits = torch.zeros(1, 0, self.vocab_size, device=dev)

                        else:  # all_cycle_logits is None
                            final_logits = torch.zeros(1, 0, self.vocab_size, device=dev)
                        # final_gate_loss is None for eval

            else:  # No cycle triggered
                self.current_chunk_tokens.extend(new_tokens_list_for_call)
                try:
                    decoded_new = self.tokenizer.decode(new_tokens_list_for_call) if new_tokens_list_for_call else ""
                    self.current_chunk_text += decoded_new
                except Exception as e:
                    print0(f"Warning: Error decoding appended tokens: {new_tokens_list_for_call}. Error: {e}",
                           self.config.logfile)

                self.current_total_logical_sequence_length = sum(len(c) for c in self.closed_chunks) + len(
                    self.current_chunk_tokens)
                if self.config.DEBUG_LEVEL > 0:
                    print0(
                        f"DEBUG CMAModel.forward: No cycle. Logical seq_len for current chunk set to {self.current_total_logical_sequence_length}",
                        self.config.logfile)

                if self.current_chunk_tokens:
                    logits_full_buffer, gate_loss_chunk = self._process_current_chunk(
                        generation_mode_hint=not self.training)

                    if self.training:
                        if logits_full_buffer is not None and logits_full_buffer.size(
                                1) > 0 and num_new_tokens_for_this_call > 0:
                            final_logits = logits_full_buffer[:, -num_new_tokens_for_this_call:, :]
                        else:
                            final_logits = torch.zeros(1, 0, self.vocab_size, device=dev)
                        final_gate_loss = gate_loss_chunk
                    else:  # Evaluation/Generation, no cycle
                        # Return logits for the *entire current buffer*
                        final_logits = logits_full_buffer
                        # final_gate_loss is None for eval
                else:
                    final_logits = torch.zeros(1, 0, self.vocab_size, device=dev)

        if final_logits is None:
            print0("CRITICAL Warning: final_logits was None at end of CMAModel.forward. Defaulting to empty.",
                   self.config.logfile)
            final_logits = torch.zeros(1, 0, self.vocab_size, device=dev)

        # Debug prints (can be adjusted or made conditional on self.config.DEBUG_LEVEL)
        if final_logits is not None:
            actual_len = final_logits.shape[1]
            log_prefix = "[TRAIN]" if self.training else "[EVAL]"
            expected_len = 0

            if self.training:
                if trigger_update_cycle:  # Cycle occurred
                    if all_cycle_logits is not None: expected_len = all_cycle_logits.shape[1]
                elif num_new_tokens_for_this_call > 0:  # No cycle, new tokens added to buffer
                    if logits_full_buffer is not None and logits_full_buffer.size(1) > 0:
                        expected_len = num_new_tokens_for_this_call  # Slice for new tokens
                elif logits_full_buffer is not None:  # No cycle, no new input tokens, but buffer had content
                    expected_len = logits_full_buffer.size(1)  # Entire buffer
            else:  # Evaluation
                if trigger_update_cycle:  # Cycle occurred
                    if num_new_tokens_for_this_call > 0:
                        expected_len = num_new_tokens_for_this_call
                        if all_cycle_logits is not None and all_cycle_logits.size(1) < num_new_tokens_for_this_call:
                            expected_len = all_cycle_logits.size(1)  # If cycle output shorter
                    elif len(
                            self.current_chunk_tokens) > 0 and all_cycle_logits is not None:  # No new input, but cycle happened and left a current_chunk
                        expected_len = len(self.current_chunk_tokens)


                elif self.current_chunk_tokens and logits_full_buffer is not None:  # No cycle, buffer processed
                    expected_len = len(self.current_chunk_tokens)  # Entire buffer
                elif num_new_tokens_for_this_call == 0 and not self.current_chunk_tokens:  # No input, no buffer
                    expected_len = 0

            status_msg = "OK" if actual_len == expected_len else "MISMATCH"
            if self.config.DEBUG_LEVEL > 0 and (
                    status_msg == "MISMATCH" or self.training_step < 5 or not self.training):
                print0(
                    f"DEBUG CMAModel.forward {log_prefix}: Step {self.training_step}, {status_msg} final_logits len {actual_len} vs expected len {expected_len}. "
                    f"Cycle={trigger_update_cycle}. Input num_new_tokens_for_this_call={num_new_tokens_for_this_call}. "
                    f"current_chunk_tokens_len_AFTER_OP={len(self.current_chunk_tokens)}.",
                    self.config.logfile)

        return final_logits, final_gate_loss if self.training else None

    def _process_current_chunk(self, generation_mode_hint: bool = False) -> Tuple[
        Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Process current chunk without full memory update cycle"""
        if not self.current_chunk_tokens:
            return torch.zeros(1, 0, self.vocab_size, device=self.token_embedding.weight.device), None

        device = self.token_embedding.weight.device
        chunk_tensor = torch.tensor(self.current_chunk_tokens, dtype=torch.long, device=device).unsqueeze(0)
        x = self.token_embedding(chunk_tensor)

        ctrl_mode = "generate" if not self.training or generation_mode_hint else "forward"

        approx_processed_len_for_ctrl_tokens = sum(len(c) for c in self.closed_chunks)
        current_mem_size_for_ctrl = self.memory_manager.get_effective_size(
            approx_processed_len_for_ctrl_tokens,
            current_chunk_length=len(self.current_chunk_tokens)
        )

        ctrl = self.control_token_generator.generate_control_tokens(
            mode=ctrl_mode,
            current_chunk_idx=len(self.closed_chunks),
            total_chunks=len(self.closed_chunks) + 1,
            current_mem_size=current_mem_size_for_ctrl,
            max_mem_size=self.config.max_memory_size,
            seq_len=approx_processed_len_for_ctrl_tokens
        )

        gate_loss_accumulator: Optional[torch.Tensor] = None

        for layer_idx, layer_block in enumerate(self.layers):
            group_idx = self.layer_idx_to_group_idx[layer_idx]

            # In _process_current_chunk, memory is only read, not updated via cycle mechanics.
            # The `do_memory_update` flag within Block.forward will be False because total_logical_sequence_length
            # might not meet the criteria, or the mode is "generate".
            x, _, _, layer_gate_loss = layer_block(
                x, M_fwd_dict=self.M_fwd, M_rev_ahead_dict=self.M_rev_ahead,
                M_rev_persist_dict=self.M_rev_persist, group_id=group_idx,
                mode=ctrl_mode,
                control_tokens=ctrl,
                # write_mask, decay_weights are not relevant here as do_memory_update will be false
                total_logical_sequence_length=self.current_total_logical_sequence_length
            )

            if layer_gate_loss is not None:
                current_loss = layer_gate_loss.mean() if layer_gate_loss.numel() > 1 else layer_gate_loss
                if gate_loss_accumulator is None:
                    gate_loss_accumulator = current_loss
                else:
                    gate_loss_accumulator += current_loss

        logits = self.lm_head(norm(x))
        # Return aggregated gate_loss; CMAModel.forward will decide if it's used based on self.training
        return logits, gate_loss_accumulator

    def _print_memory_stats(self, memory_dict: Dict[int, Tensor], pass_name: str, chunk_idx: Optional[int] = None):
        """Debug utility to print memory statistics"""
        if not memory_dict or not (self.config.logfile and self.config.DEBUG_LEVEL > 1):
            return

        log_prefix = f"DEBUG MEMORY: Pass='{pass_name}'"
        if chunk_idx is not None:
            log_prefix += f", ChunkIdx={chunk_idx}"

        for group_id, mem_tensor in memory_dict.items():
            if mem_tensor is not None and mem_tensor.numel() > 0:
                mean_val = mem_tensor.mean().item()
                std_val = mem_tensor.std().item()
                abs_max_val = mem_tensor.abs().max().item()
                print0(f"{log_prefix} - Group {group_id}: Shape={mem_tensor.shape}, Mean={mean_val:.4f}, Std={std_val:.4f}, AbsMax={abs_max_val:.4f}", self.config.logfile)
                if self.config.log_wandb:
                    wandb.log({f"mem_stats/{pass_name}_{group_id}_chunk{chunk_idx}_mean":mean_val, f"mem_stats/{pass_name}_{group_id}_chunk{chunk_idx}_std":std_val, f"mem_stats/{pass_name}_{group_id}_chunk{chunk_idx}_abs_max":abs_max_val}, step=self.training_step)
            elif mem_tensor is not None:
                print0(f"{log_prefix} - Group {group_id}: Shape={mem_tensor.shape} (Empty Tensor)", self.config.logfile)
            else:
                print0(f"{log_prefix} - Group {group_id}: None", self.config.logfile)

    def _trigger_memory_update_cycle(self, chunks: List[List[int]]) -> Tuple[
        Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Execute the three-pass memory update cycle"""
        if self.config.DEBUG_LEVEL > 0:
            print0(f"DEBUG: Starting _trigger_memory_update_cycle with {len(chunks)} chunks", self.config.logfile)

        if not chunks:
            return None, None

        dev = self.token_embedding.weight.device
        B = 1  # Assuming batch size 1 for stateful processing

        # Pass 1: Lookahead Reverse
        try:
            M_rev_ahead_computed = self._run_lookahead_reverse_pass(chunks, B, dev,
                                                                    self.current_total_logical_sequence_length)
            self._print_memory_stats(M_rev_ahead_computed, "LookaheadReverse (Output)")
        except Exception as e:
            print0(f"ERROR in _run_lookahead_reverse_pass: {e}", self.config.logfile)
            if self.config.DEBUG_LEVEL > 1:
                import traceback
                traceback.print_exc()
            raise

        # Pass 2: Forward with memory updates
        pass2_logits_list: List[torch.Tensor] = []
        pass2_gate_loss_aggregated: Optional[torch.Tensor] = None
        tokens_processed_in_pass2_before_current_chunk = 0  # Corrected name

        self._print_memory_stats(self.M_fwd, "Forward (Input to Pass 2)")
        self._print_memory_stats(self.M_rev_persist, "PersistentReverse (Input to Pass 2 for attention)")

        for chunk_idx, chunk_tokens in enumerate(chunks):
            if not chunk_tokens:
                continue

            chunk_tensor = torch.tensor(chunk_tokens, dtype=torch.long, device=dev).unsqueeze(0)
            x = self.token_embedding(chunk_tensor)

            current_mem_size_for_ctrl = self.memory_manager.get_effective_size(
                self.total_tokens_processed + tokens_processed_in_pass2_before_current_chunk)
            ctrl = self.control_token_generator.generate_control_tokens(
                mode="forward", current_chunk_idx=chunk_idx, total_chunks=len(chunks),
                current_mem_size=current_mem_size_for_ctrl, max_mem_size=self.config.max_memory_size,
                seq_len=self.total_tokens_processed + tokens_processed_in_pass2_before_current_chunk
            )

            M_rev_persist_for_chunk = self.M_rev_persist
            p_drop_for_debug = 0.0
            if self.training and self.config.enable_mask_future_dropout:
                p_drop = get_mask_future_schedule(self.config, self.training_step, self.total_training_steps)
                if self.config.log_wandb:
                    wandb.log({"curriculum/p_future_drop": p_drop}, step=self.training_step)
                p_drop_for_debug = p_drop
                if self.config.DEBUG_LEVEL > 0:
                    print0(f"DEBUG MASKING: Training step {self.training_step}, M_rev_persist p_drop: {p_drop:.4f}",
                           self.config.logfile)
                M_rev_persist_for_chunk = self._mask_persistent_memory(self.M_rev_persist, p_drop)
                if chunk_idx == 0 and self.config.DEBUG_LEVEL > 1:
                    self._print_memory_stats(M_rev_persist_for_chunk, "PersistentReverse (Masked for Pass 2 Attn)",
                                             chunk_idx=chunk_idx)

            chunk_gate_loss_accumulator_for_layers: Optional[torch.Tensor] = None

            for layer_idx, layer_block in enumerate(self.layers):
                group_idx = self.layer_idx_to_group_idx[layer_idx]
                group = self.layer_groups[group_idx]

                wmask = None
                if group.has_memory and layer_idx == group.memory_update_layer_idx:
                    wmask = self.memory_manager.get_write_mask(
                        current_chunk_idx_in_pass=chunk_idx, total_chunks_in_pass=len(chunks),
                        total_tokens_processed_before_chunk=self.total_tokens_processed + tokens_processed_in_pass2_before_current_chunk,
                        current_chunk_len=len(chunk_tokens), batch_size=B, device=dev
                    )

                if not self.training and chunk_idx == 0 and wmask is not None and self.config.DEBUG_LEVEL > 0:
                    print0(
                        f"DEBUG VAL First Write Mask (Pass 2, Chunk 0): true_frac={wmask.float().mean().item():.4f}, shape={wmask.shape}, total_tokens_processed_before_chunk={self.total_tokens_processed + tokens_processed_in_pass2_before_current_chunk}",
                        self.config.logfile)

                x, updated_fwd, _, layer_gate_loss = layer_block(
                    x, M_fwd_dict=self.M_fwd, M_rev_ahead_dict=M_rev_ahead_computed,
                    M_rev_persist_dict=M_rev_persist_for_chunk, group_id=group_idx,
                    mode="forward", control_tokens=ctrl, write_mask=wmask,
                    total_logical_sequence_length=self.current_total_logical_sequence_length
                )

                if chunk_idx == 0 and self.config.DEBUG_LEVEL > 0:
                    self._print_memory_stats(M_rev_persist_for_chunk,
                                             f"PersistentReverse ({'Masked' if self.training and self.config.enable_mask_future_dropout and p_drop_for_debug > 0 else 'Unmasked'} for Pass 2 Attn)",
                                             chunk_idx=chunk_idx)

                if updated_fwd is not None:
                    g_id, mem = updated_fwd
                    self.M_fwd[g_id] = mem
                    if layer_idx == group.memory_update_layer_idx and \
                            (chunk_idx == 0 or chunk_idx == len(chunks) - 1) and \
                            self.config.DEBUG_LEVEL > 1:
                        self._print_memory_stats({g_id: mem}, f"Forward (Updated by L{layer_idx})", chunk_idx=chunk_idx)

                if layer_gate_loss is not None:
                    current_loss = layer_gate_loss.mean() if layer_gate_loss.numel() > 1 else layer_gate_loss
                    if chunk_gate_loss_accumulator_for_layers is None:
                        chunk_gate_loss_accumulator_for_layers = current_loss
                    else:
                        chunk_gate_loss_accumulator_for_layers += current_loss

            logits_for_chunk = self.lm_head(norm(x))
            pass2_logits_list.append(logits_for_chunk)

            if chunk_gate_loss_accumulator_for_layers is not None:
                if pass2_gate_loss_aggregated is None:
                    pass2_gate_loss_aggregated = chunk_gate_loss_accumulator_for_layers
                else:
                    pass2_gate_loss_aggregated += chunk_gate_loss_accumulator_for_layers

            tokens_processed_in_pass2_before_current_chunk += len(chunk_tokens)

        self._print_memory_stats(self.M_fwd, "Forward (Output of Pass 2)")
        if 'M_rev_ahead_computed' in locals():
            del M_rev_ahead_computed
        self.M_rev_ahead = {}

        # Pass 3: Persistent Reverse
        self._print_memory_stats(self.M_rev_persist, "PersistentReverse (Input to Pass 3)")
        self._run_persistent_reverse_pass(chunks, B, dev, self.current_total_logical_sequence_length)
        self._print_memory_stats(self.M_rev_persist, "PersistentReverse (Output of Pass 3)")

        self.closed_chunks = chunks[:-1]
        self.current_chunk_tokens = chunks[-1] if chunks else []
        self.total_tokens_processed = sum(len(c) for c in chunks)
        self.tokens_since_persistent_update = 0

        concatenated_pass2_logits: Optional[torch.Tensor] = None
        if pass2_logits_list:
            concatenated_pass2_logits = torch.cat(pass2_logits_list, dim=1)
        elif chunks:
            concatenated_pass2_logits = torch.zeros(B, 0, self.vocab_size, device=dev)

        # Return ALL concatenated logits from the cycle.
        # The gate loss is aggregated across all chunks in the cycle.
        # CMAModel.forward will decide whether to use the gate_loss (e.g., only if self.training).
        return concatenated_pass2_logits, pass2_gate_loss_aggregated

    def _run_lookahead_reverse_pass(self, all_chunks: List[List[int]], B: int, dev: torch.device,
                                   total_logical_sequence_length: int) -> Dict[int, Tensor]:
        """Execute lookahead reverse pass"""
        try:
            n_chunks = len(all_chunks)
            window_start_idx = max(0, n_chunks - self.config.reverse_max_chunks)
            reverse_window_chunks = all_chunks[window_start_idx:]
            window_size = len(reverse_window_chunks)

            if self.config.DEBUG_LEVEL > 0:
                print0(f"DEBUG: Entering _run_lookahead_reverse_pass. n_chunks={n_chunks}, window_size={window_size}", self.config.logfile)

            # Initialize M_rev_ahead for this pass
            current_M_rev_ahead: Dict[int, Tensor] = {}
            for group in self.layer_groups:
                if group.has_memory:
                    group_idx = group.group_idx
                    mem_idx = self.group_id_to_memory_idx[group_idx]
                    param = self.initial_rev_la_params[mem_idx]
                    current_M_rev_ahead[group_idx] = param.clone().to(dev).repeat(B, 1, 1)

            for i, chunk_tokens in enumerate(reversed(reverse_window_chunks)):
                if not chunk_tokens:
                    continue

                chunk_tensor = torch.tensor(chunk_tokens, dtype=torch.long, device=dev).unsqueeze(0)
                x = self.token_embedding(chunk_tensor)
                
                global_chunk_idx = window_start_idx + (window_size - 1 - i)
                reverse_chunk_idx = i

                # Generate control tokens
                ctrl = self.control_token_generator.generate_control_tokens(
                    mode="lookahead_reverse", current_chunk_idx=global_chunk_idx, total_chunks=n_chunks,
                    current_mem_size=self.memory_manager.get_effective_size(self.total_tokens_processed),
                    max_mem_size=self.config.max_memory_size, seq_len=self.total_tokens_processed,
                    reverse_chunk_idx=reverse_chunk_idx, reverse_window_size=window_size
                )

                for layer_idx, layer_block in enumerate(self.layers):
                    group_idx = self.layer_idx_to_group_idx[layer_idx]
                    group = self.layer_groups[group_idx]

                    decay = None
                    if group.has_memory and layer_idx == group.memory_update_layer_idx:
                        mem_shape = current_M_rev_ahead[group_idx].shape
                        decay = self.memory_manager.calculate_reverse_decay_weights(
                            reverse_chunk_index=reverse_chunk_idx, window_size=window_size,
                            is_persistent=False, memory_shape=mem_shape, device=dev
                        )

                    x, _, updated_rev, _ = layer_block(
                        x, M_fwd_dict=self.M_fwd, M_rev_ahead_dict=current_M_rev_ahead,
                        M_rev_persist_dict=self.M_rev_persist, group_id=group_idx,
                        mode="lookahead_reverse", control_tokens=ctrl,
                        decay_weights=decay, total_logical_sequence_length=total_logical_sequence_length
                    )

                    if updated_rev is not None:
                        g_id, mem = updated_rev
                        current_M_rev_ahead[g_id] = mem

            return current_M_rev_ahead

        except Exception as e:
            print0(f"Error in _run_lookahead_reverse_pass: {type(e).__name__}: {e}", self.config.logfile)
            import traceback
            traceback.print_exc()
            raise

    def _run_persistent_reverse_pass(self, all_chunks: List[List[int]], B: int, dev: torch.device, 
                                    total_logical_sequence_length: int):
        """Execute persistent reverse pass"""
        n_chunks = len(all_chunks)
        eligible_chunks = all_chunks[:-1]  # All except last chunk
        
        if not eligible_chunks:
            print0("Skipping Persistent Reverse Pass: No eligible preceding chunks.", self.config.logfile)
            return

        window_start_idx = max(0, len(eligible_chunks) - self.config.reverse_max_chunks)
        reverse_window_chunks = eligible_chunks[window_start_idx:]
        window_size = len(reverse_window_chunks)

        if not reverse_window_chunks:
            print0("Skipping Persistent Reverse Pass: Window is empty after eligibility check.", self.config.logfile)
            return

        # Initialize M_rev_persist for this pass
        current_M_rev_persist: Dict[int, Tensor] = {}
        for group in self.layer_groups:
            if group.has_memory:
                group_idx = group.group_idx
                mem_idx = self.group_id_to_memory_idx[group_idx]
                current_M_rev_persist[group_idx] = self.initial_rev_p_params[mem_idx].clone().to(dev).repeat(B, 1, 1)

        for i, chunk_tokens in enumerate(reversed(reverse_window_chunks)):
            chunk_tensor = torch.tensor(chunk_tokens, dtype=torch.long, device=dev).unsqueeze(0)
            x = self.token_embedding(chunk_tensor)
            
            global_chunk_idx_in_all = window_start_idx + (window_size - 1 - i)
            reverse_chunk_idx = i

            ctrl = self.control_token_generator.generate_control_tokens(
                mode="persistent_reverse", current_chunk_idx=global_chunk_idx_in_all, total_chunks=n_chunks,
                current_mem_size=self.memory_manager.get_effective_size(self.total_tokens_processed),
                max_mem_size=self.config.max_memory_size, seq_len=self.total_tokens_processed,
                reverse_chunk_idx=reverse_chunk_idx, reverse_window_size=window_size
            )

            for layer_idx, layer_block in enumerate(self.layers):
                group_idx = self.layer_idx_to_group_idx[layer_idx]
                group = self.layer_groups[group_idx]

                decay = None
                if group.has_memory and layer_idx == group.memory_update_layer_idx:
                    if group_idx in current_M_rev_persist:
                        mem_shape = current_M_rev_persist[group_idx].shape
                        decay = self.memory_manager.calculate_reverse_decay_weights(
                            reverse_chunk_index=reverse_chunk_idx, window_size=window_size,
                            is_persistent=True, memory_shape=mem_shape, device=dev
                        )

                x, _, updated_rev, _ = layer_block(
                    x, M_fwd_dict=self.M_fwd, M_rev_ahead_dict=self.M_rev_ahead,
                    M_rev_persist_dict=current_M_rev_persist, group_id=group_idx,
                    mode="persistent_reverse", control_tokens=ctrl,
                    decay_weights=decay, total_logical_sequence_length=total_logical_sequence_length
                )

                if updated_rev is not None:
                    g_id, mem = updated_rev
                    current_M_rev_persist[g_id] = mem

        self.M_rev_persist = current_M_rev_persist

    def _mask_persistent_memory(self, memory_dict: Dict[int, Tensor], p_drop: float) -> Dict[int, Tensor]:
        """Apply mask-future dropout to persistent memory"""
        if p_drop <= 0.0:
            return memory_dict

        masked_memory_dict = {}
        for group_id, mem_tensor in memory_dict.items():
            if mem_tensor is None or mem_tensor.numel() == 0:
                masked_memory_dict[group_id] = mem_tensor
                continue

            B, M, D = mem_tensor.shape
            if M == 0:
                masked_memory_dict[group_id] = mem_tensor.clone()
                continue

            keep_prob = 1.0 - p_drop
            mask = torch.bernoulli(torch.full((B, M, 1), keep_prob, device=mem_tensor.device)).to(mem_tensor.dtype)
            masked_memory_dict[group_id] = mem_tensor * mask

        return masked_memory_dict

    @torch.no_grad()
    def generate(self, prompt: Union[str, List[int], List[List[int]], torch.Tensor, None] = None, *,
                 max_new_tokens: int = 128, temperature: float = 1.0, top_k: Optional[int] = None,
                 stop_token_id: Optional[int] = None, reset_state: bool = False) -> List[int]:
        """Autoregressive generation"""
        if reset_state:
            print0("Resetting model state for generation.", self.config.logfile)
            self.reset_state()
        
        self._initialize_memory_states(force_reset=False)
        dev = self.token_embedding.weight.device

        def sample_token(logits: torch.Tensor) -> int:
            logits = logits.float()
            if top_k is not None and top_k > 0:
                actual_k = min(top_k, logits.shape[-1])
                if actual_k > 0:
                    thresh = torch.topk(logits, actual_k).values[-1]
                    logits = torch.where(logits < thresh, torch.full_like(logits, float("-inf")), logits)
            
            temp = max(float(temperature), 1e-5)
            probs = torch.nn.functional.softmax(logits / temp, dim=-1)
            return torch.multinomial(probs, 1).item()

        generated: List[int] = []

        # Process initial prompt
        if prompt not in (None, "", [], torch.tensor([], dtype=torch.long)):
            if self.config.DEBUG_LEVEL > 0:
                print0(f"DEBUG: Processing initial prompt ({type(prompt)})", self.config.logfile)
            self.forward(prompt, training_mode=False)

        try:
            self.current_chunk_text = self.tokenizer.decode(self.current_chunk_tokens)
        except Exception as e:
            print0(f"Warning: Error decoding initial buffer tokens: {e}", self.config.logfile)
            self.current_chunk_text = ""

        current_token_input = None

        for i in range(max_new_tokens):
            if self.config.DEBUG_LEVEL > 0:
                print0(f"DEBUG: Generation step {i}, input: {current_token_input}", self.config.logfile)
            
            logits, _ = self.forward(current_token_input, training_mode=False)

            if logits is None or logits.shape[1] == 0:
                if self.config.DEBUG_LEVEL > 1:
                    print0("INFO: Generation stopped - empty logits.", self.config.logfile)
                break

            next_id = sample_token(logits[0, -1, :])
            generated.append(next_id)

            if stop_token_id is not None and next_id == stop_token_id:
                if self.config.DEBUG_LEVEL > 1:
                    print0(f"INFO: Stop token {stop_token_id} generated. Stopping.", self.config.logfile)
                break

            try:
                self.current_chunk_text += self.tokenizer.decode([next_id])
            except Exception as e:
                print0(f"Warning: Error decoding token {next_id}: {e}", self.config.logfile)

            # Periodic persistent reverse update
            self.tokens_since_persistent_update += 1
            if self._should_update_persistent_memory():
                self._perform_periodic_persistent_update(dev)

            current_token_input = [next_id]

        # Final processing if needed
        if current_token_input is not None:
            if self.config.DEBUG_LEVEL > 0:
                print0(f"DEBUG: Processing final generated token {current_token_input} to update state.", self.config.logfile)
            self.forward(current_token_input, training_mode=False)

        return generated

    def _should_update_persistent_memory(self) -> bool:
        """Check if periodic persistent reverse update should be triggered"""
        if (self.config.persistent_reverse_update_freq_tokens is not None and 
            self.tokens_since_persistent_update >= self.config.persistent_reverse_update_freq_tokens):
            return True

        if self.config.persistent_reverse_update_freq_semantic and len(self.current_chunk_text) > 1:
            try:
                boundary_level = self.config.persistent_reverse_update_freq_semantic
                if boundary_level in self.config.boundary_types:
                    search_window = self.current_chunk_text[-10:]
                    for boundary_type in self.config.boundary_types[boundary_level]:
                        pattern = BOUNDARY_PATTERNS.get(boundary_type)
                        if pattern and re.search(pattern, search_window):
                            return True
            except Exception as e:
                print0(f"Warning: Error during semantic boundary check: {e}", self.config.logfile)

        return False

    def _perform_periodic_persistent_update(self, dev: torch.device):
        """Perform periodic persistent reverse update during generation"""
        if self.config.DEBUG_LEVEL > 0:
            print0("Triggering periodic persistent reverse update during generation.", self.config.logfile)
        
        full_chunks = self.closed_chunks + ([self.current_chunk_tokens] if self.current_chunk_tokens else [])
        if full_chunks:
            logical_len = sum(len(c) for c in full_chunks)
            if self.config.DEBUG_LEVEL > 0:
                print0(f"DEBUG Generate: Logical seq_len for persist pass: {logical_len}", self.config.logfile)
            self._run_persistent_reverse_pass(full_chunks, B=1, dev=dev, total_logical_sequence_length=logical_len)
        else:
            if self.config.DEBUG_LEVEL > 0:
                print0("Skipping periodic update: No history or current chunks available.", self.config.logfile)
        
        self.tokens_since_persistent_update = 0

    def reset_state(self) -> None:
        """Clear all sequence-level state and reset memory dictionaries"""
        if self.config.DEBUG_LEVEL > 0:
            print0("DEBUG: Resetting CMAModel state.", self.config.logfile)
        
        self.closed_chunks = []
        self.current_chunk_tokens = []
        self.current_chunk_text = ""
        self.total_tokens_processed = 0
        self.tokens_since_persistent_update = 0
        self.M_fwd = {}
        self.M_rev_ahead = {}
        self.M_rev_persist = {}

    def set_training_step(self, step: int, total_steps: int):
        """Set current training step for mask-future scheduling"""
        self.training_step = step
        self.config.training_step = step
        self.total_training_steps = total_steps
        if self.config.DEBUG_LEVEL > 0:
            print0(f"DEBUG: Training step set to {step}, total steps {total_steps}", self.config.logfile)
