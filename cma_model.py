from typing import Tuple, Optional, List, Dict
import tiktoken
import torch.nn as nn
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
    has_memory: bool = False  # True if memory_update_layer_idx is not None


class CanonConv(nn.Module):
    """Causal 1D depth-wise convolution for Canon."""

    def __init__(self, dim: int, kernel_size: int = 4):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            groups=dim,
            padding=kernel_size - 1  # This padding + slicing makes it causal
        )
        # Default Kaiming uniform initialization for Conv1d is fine, as requested.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        if x.ndim != 3 or x.size(1) == 0:  # (Batch, Seq, Dim)
            # Handle cases with no sequence length or not 3D
            return x

        # Transpose for Conv1d: (B, D, T)
        x_transposed = x.transpose(1, 2)

        # Apply convolution
        # Output of conv will have length T_in + (kernel_size-1) due to padding
        # T_out = T_in + 2*padding - (kernel_size-1) = T_in + 2*(kernel_size-1) - (kernel_size-1) = T_in + kernel_size-1
        y = self.conv(x_transposed)

        # Slice to make output length same as input T and ensure causality
        # Remove (kernel_size-1) elements from the end
        y_sliced = y[:, :, :-(self.conv.padding[0])] if self.conv.padding[0] > 0 else y

        # Transpose back and add residual: (B, T, D)
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

        # QKV projections
        self.qkv = nn.Linear(config.embed_dim, 3 * config.embed_dim, bias=False)
        self.out_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=False)

        # Canon-B (for local_only layers)
        self.canon_b: Optional[CanonConv] = None
        if config.enable_canon:
            # CausalSelfAttention is used for "local_only" layers, which get Canon-A, B, C, D.
            # So, Canon-B is active here.
            self.canon_b = CanonConv(config.embed_dim, config.canon_kernel_size)

        # Initialize output projection to zero if configured
        if config.output_proj_zero_init:
            nn.init.zeros_(self.out_proj.weight)

    def forward(self, x: Tensor, causal_mask: Optional[Tensor] = None) -> Tensor:
        B, T, C = x.size()

        # QKV projections
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head attention
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Apply causal mask
        if causal_mask is None:
            causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn_scores.masked_fill_(causal_mask, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)

        # Apply Canon-B if active
        if self.canon_b is not None:
            attn_output = self.canon_b(attn_output)

        output = self.out_proj(attn_output)

        return output


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

        # Canon-B
        self.canon_b: Optional[CanonConv] = None
        if config.enable_canon:
            if not self.is_memory_update:  # memory_read layers get Canon-B
                self.canon_b = CanonConv(config.embed_dim, config.canon_kernel_size)
            # memory_update layers do NOT get Canon-B

        # Control token integration
        if config.integration_method == "query_fusion":
            self.control_proj = nn.Linear(5, config.embed_dim, bias=False)
            nn.init.normal_(self.control_proj.weight, std=config.ctrl_init_scale)

        # Adaptive gating
        self.gate_proj = nn.Linear(config.embed_dim, config.n_heads, bias=True)
        nn.init.constant_(self.gate_proj.bias, config.gate_bias_init)

        # Memory update components (if this is an update layer)
        if is_memory_update:
            # Forward memory update parameters
            self.fwd_memory_q_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
            self.fwd_memory_k_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
            self.fwd_memory_v_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
            self.fwd_memory_gate_proj = nn.Linear(2 * config.embed_dim, config.embed_dim, bias=False)

            # Reverse memory update parameters (shared between lookahead and persistent)
            self.rev_memory_q_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
            self.rev_memory_k_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
            self.rev_memory_v_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
            self.rev_memory_gate_proj = nn.Linear(2 * config.embed_dim, config.embed_dim, bias=False)

        # Initialize output projection to zero if configured
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

        # -------- Q projection (+ optional control-token fusion) -----------------
        q = self.q_proj(x)
        if control_tokens is not None and self.config.integration_method == "query_fusion":
            ctrl_vec = torch.tensor([
                control_tokens["generation_flag"],
                control_tokens["memory_mode_flag"],
                control_tokens["memory_usage_ratio"],
                control_tokens["memory_density_ratio"],
                control_tokens["chunk_position_ratio"]
            ], device=x.device).unsqueeze(0).unsqueeze(0).expand(B, T, -1)
            q = q + self.control_proj(ctrl_vec)

        # Handle empty sequence
        if T == 0:
            return x, None, None, None

        # -------- K / V projection (plus memory tokens) --------------------------
        k = self.k_proj(x)
        v = self.v_proj(x)
        if (forward_memory is not None) or (reverse_memory is not None):
            if self.config.DEBUG_LEVEL > 0: print0(f"DEBUG: CMA Layer {self.layer_idx}: Fwd mem shape: {forward_memory.shape if forward_memory is not None else None}, Rev mem shape: {reverse_memory.shape if reverse_memory is not None else None}", self.config.logfile)
            k, v = self._integrate_memory(k, v, forward_memory, reverse_memory)

        # reshape to (B , h , T , d)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)

        # -------- scaled dot-product attention -----------------------------------
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # causal mask (applies only to chunk → chunk positions)
        S = k.size(-2)
        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        full_mask = torch.zeros(T, S, device=x.device, dtype=torch.bool)
        full_mask[:, :T] = causal
        attn_scores.masked_fill_(full_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)

        # -------- output & adaptive memory gate ----------------------------------
        if (forward_memory is not None) or (reverse_memory is not None):
            attn_output, gate_reg_loss = self._apply_gate(q, attn_weights, v, T)
        else:  # no memory tokens → plain attention
            attn_output = torch.matmul(attn_weights, v)
            gate_reg_loss = None

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)

        # Apply Canon-B if active
        if self.canon_b is not None:
            attn_output = self.canon_b(attn_output)

        output = self.out_proj(attn_output)

        # -------- optional memory update ----------------------------------------
        updated_forward_memory = None
        updated_reverse_memory = None
        if do_memory_update and self.is_memory_update:
            if forward_memory is not None and not is_reverse_update:
                updated_forward_memory = self._update_memory(
                    forward_memory, x, write_mask=write_mask, is_forward=True
                )
            if reverse_memory is not None and is_reverse_update:
                updated_reverse_memory = self._update_memory(
                    reverse_memory, x,
                    write_mask=write_mask,
                    decay_weights=decay_weights,
                    is_forward=False
                )

        return output, updated_forward_memory, updated_reverse_memory, gate_reg_loss

    def _integrate_memory(
            self,
            k: Tensor,
            v: Tensor,
            forward_memory: Optional[Tensor],
            reverse_memory: Optional[Tensor]
    ) -> Tuple[Tensor, Tensor]:
        """Integrate memory states into keys and values"""
        memory_k_list = [k]
        memory_v_list = [v]

        if forward_memory is not None:
            fwd_k = self.k_proj(forward_memory)
            fwd_v = self.v_proj(forward_memory)
            memory_k_list.append(fwd_k)
            memory_v_list.append(fwd_v)

        if reverse_memory is not None:
            rev_k = self.k_proj(reverse_memory)
            rev_v = self.v_proj(reverse_memory)
            memory_k_list.append(rev_k)
            memory_v_list.append(rev_v)

        combined_k = torch.cat(memory_k_list, dim=1)
        combined_v = torch.cat(memory_v_list, dim=1)

        return combined_k, combined_v

    def _apply_gate(
            self,
            q: torch.Tensor,  # (B , h , T , d_head)
            attn_weights: torch.Tensor,  # (B , h , T , S = T+M)
            v: torch.Tensor,  # (B , h , S , d_head)
            chunk_len: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Split the attention output into Y_chunk (keys ∈ chunk) and
        Y_mem   (keys ∈ memories), then fuse with a learnable gate.

        Returns
        -------
        Y : (B , h , T , d_head)
        reg_loss : optional scalar
        """
        B, h, T, _ = q.shape

        # Handle empty sequence
        if T == 0:
            return torch.zeros_like(q), None

        S_keys = v.size(-2)  # T + mem_len
        mem_len = S_keys - chunk_len
        if self.config.DEBUG_LEVEL > 0: print0(f"DEBUG: CMA Layer {self.layer_idx} _apply_gate: chunk_len={chunk_len}, S_keys={S_keys}, mem_len={mem_len}", self.config.logfile)
        if mem_len == 0:  # nothing to gate
            out = torch.matmul(attn_weights, v)
            return out, None

        # split keys/values
        v_chunk, v_mem = torch.split(v, [chunk_len, mem_len], dim=-2)
        w_chunk, w_mem = torch.split(attn_weights, [chunk_len, mem_len], dim=-1)

        Y_chunk = torch.matmul(w_chunk, v_chunk)  # (B , h , T , d)
        Y_mem = torch.matmul(w_mem, v_mem)  # (B , h , T , d)

        # gate g for every query token
        gate_logits = self.gate_proj(q.permute(0, 2, 1, 3).reshape(B, T, -1))  # (B,T,h)
        g = torch.sigmoid(gate_logits).permute(0, 2, 1).unsqueeze(-1)  #
        if mem_len > 0 and self.config.DEBUG_LEVEL > 0 and (
                torch.rand(1).item() < 0.05 or not self.training):  # Sample, or always for val
            # v_chunk and v_mem are (B, h, T_chunk/T_mem, d_head)
            # q is (B, h, T_query, d_head)
            # To get overall std, might need to reshape or take norm per head then average.
            # For simplicity, let's take mean std across heads.
            v_chunk_std = v_chunk.std(dim=(-1, -2)).mean().item() if v_chunk.numel() > 0 else 0.0
            v_mem_std = v_mem.std(dim=(-1, -2)).mean().item() if v_mem.numel() > 0 else 0.0
            g_mean = g.mean().item()

            print0(
                f"DEBUG CMA Layer {self.layer_idx} _apply_gate details (training={self.training}): "  # Pass training status
                f"Step {self.config.training_step if hasattr(self.config, 'training_step') else 'N/A'}, "  # Access training_step if available
                f"g_mean={g_mean:.4f}, v_chunk_std={v_chunk_std:.4f}, v_mem_std={v_mem_std:.4f}",
                self.config.logfile)

        if self.config.DEBUG_LEVEL > 0 and abs(g.mean() - 0.99) > 0.05:
            print0(
                f"DEBUG CMA Layer {self.layer_idx} _apply_gate: gate_logits min/max/mean, g min/max/mean: {gate_logits.min().item()}, {gate_logits.max().item()}, {gate_logits.mean().item()}, {g.min().item()}, {g.max().item()}, {g.mean().item()}", self.config.logfile)
        if self.config.DEBUG_LEVEL > 0 and torch.rand(1).item() < 0.05:  # Sample 5% of calls
            local_contrib_norm = Y_chunk.norm().item()
            g_Y_mem_norm = (g * Y_mem).norm().item()  # Norm of the gated memory contribution
            g_mean = g.mean().item()
            print0(f"DEBUG CMA Layer {self.layer_idx} _apply_gate contributions: "
                   f"Y_chunk_norm={local_contrib_norm:.4f}, (g*Y_mem)_norm={g_Y_mem_norm:.4f}, "
                   f"ratio_mem_to_local={(g_Y_mem_norm / (local_contrib_norm + 1e-9)):.2f}, "
                   f"g_mean={g_mean:.4f}, training_mode_in_model={self.training}",
                   # Assuming self.training reflects current model.training state
                   self.config.logfile)

        Y = Y_chunk + g * Y_mem  # final fused output

        # optional regularisation
        reg_loss = None
        if self.config.gate_regularization_type == "l1":
            reg_loss = self.config.gate_regularization_strength * torch.mean(torch.abs(g))
        elif self.config.gate_regularization_type == "entropy":
            ent = -(g * torch.log(g + 1e-8) + (1 - g) * torch.log(1 - g + 1e-8))
            reg_loss = self.config.gate_regularization_strength * torch.mean(ent)
        if self.training and self.config.gate_regularization_type is not None and self.config.gate_saturation_penalty:
            # Add saturation penalty to encourage more balanced gating
            gate_saturation_penalty = torch.mean(torch.square(g - 0.5)) * 0.01
            if reg_loss is None:
                reg_loss = gate_saturation_penalty
            else:
                reg_loss = reg_loss + gate_saturation_penalty

        #if self.config.DEBUG_LEVEL > 1 and reg_loss is not None: print0(f"DEBUG CMA Layer {self.layer_idx} _apply_gate: reg_loss: {reg_loss.item()}", self.config.logfile)

        return Y, reg_loss

    def _update_memory(
            self,
            memory_old: Tensor,
            chunk_tokens: Tensor,
            write_mask: Optional[Tensor] = None,
            decay_weights: Optional[Tensor] = None,
            is_forward: bool = True
    ) -> Tensor:
        """Update memory state based on chunk tokens"""
        B, M, C = memory_old.size()

        # Select appropriate update parameters
        if is_forward:
            q_proj = self.fwd_memory_q_proj
            k_proj = self.fwd_memory_k_proj
            v_proj = self.fwd_memory_v_proj
            gate_proj = self.fwd_memory_gate_proj
        else:
            q_proj = self.rev_memory_q_proj
            k_proj = self.rev_memory_k_proj
            v_proj = self.rev_memory_v_proj
            gate_proj = self.rev_memory_gate_proj

        # Project memory and chunk tokens for attention
        memory_q = q_proj(memory_old)
        chunk_k = k_proj(chunk_tokens)
        chunk_v = v_proj(chunk_tokens)

        # Reshape for attention
        memory_q = memory_q.view(B, M, self.n_heads, self.head_dim).transpose(1, 2)
        chunk_k = chunk_k.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        chunk_v = chunk_v.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)

        # Compute attention (memory queries attending to chunk keys/values)
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = torch.matmul(memory_q, chunk_k.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        delta = torch.matmul(attn_weights, chunk_v)

        # Reshape delta back
        delta = delta.transpose(1, 2).contiguous().view(B, M, C)

        # Apply decay weights if provided (for reverse memory)
        if decay_weights is not None:
            delta = delta * decay_weights

        # Compute gated update
        gate_input = torch.cat([memory_old, delta], dim=-1)
        gate = torch.sigmoid(gate_proj(gate_input))

        memory_new = gate * memory_old + (1 - gate) * delta

        # Apply write mask if provided
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

        if self.layer_type == "skip":
            self.attn = None
        elif layer_type in ["memory_read", "memory_update"]:
            is_update = (layer_type == "memory_update")
            self.attn = CascadeMemoryAttention(config, layer_idx, is_memory_update=is_update)
        else:  # local_only
            self.attn = CausalSelfAttention(config, layer_idx)

        self.mlp_fc1 = nn.Linear(config.embed_dim, 4 * config.embed_dim, bias=False)
        self.mlp_act = nn.GELU()
        self.mlp_fc2 = nn.Linear(4 * config.embed_dim, config.embed_dim, bias=False)

        self.canon_a: Optional[CanonConv] = None
        self.canon_c: Optional[CanonConv] = None
        self.canon_d: Optional[CanonConv] = None

        if config.enable_canon and self.layer_type != "skip":
            canon_dim = config.embed_dim
            canon_kernel = config.canon_kernel_size
            self.canon_a = CanonConv(canon_dim, canon_kernel)
            self.canon_c = CanonConv(canon_dim, canon_kernel)
            if self.layer_type == "local_only":
                self.canon_d = CanonConv(4 * config.embed_dim, canon_kernel)

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
            total_logical_sequence_length: int = 0 # MODIFIED: Parameter name
    ) -> Tuple[Tensor, Optional[Tuple[int, Tensor]], Optional[Tuple[int, Tensor]], Optional[Tensor]]:
        updated_fwd_result = None
        updated_rev_result = None
        gate_reg_loss = None

        fwd_mem_in: Optional[Tensor] = None
        rev_mem_in: Optional[Tensor] = None
        do_memory_update: bool = False
        is_reverse_update: bool = False

        can_update = (self.layer_type == "memory_update")
        seq_len_cond_met = (total_logical_sequence_length > self.config.chunk_size)
        do_memory_update = can_update and seq_len_cond_met
        if self.layer_type not in ["local_only", "skip"]:
            chunk_size = self.config.chunk_size
            if mode == "forward": # Pass 2 of cycle, or _process_current_chunk
                fwd_mem_in = M_fwd_dict.get(group_id)
                rev_mem_in = M_rev_persist_dict.get(group_id)
                is_reverse_update = False
            elif mode == "lookahead_reverse": # Pass 1
                rev_mem_in = M_rev_ahead_dict.get(group_id)
                is_reverse_update = True
            elif mode == "persistent_reverse": # Pass 3
                rev_mem_in = M_rev_persist_dict.get(group_id)
                is_reverse_update = True
            elif mode == "generate":
                fwd_mem_in = M_fwd_dict.get(group_id)
                rev_mem_in = M_rev_persist_dict.get(group_id)
                do_memory_update = False
                is_reverse_update = False
            else:
                raise ValueError(f"Unknown mode in Block.forward: {mode}")

        if can_update and self.config.DEBUG_LEVEL > 0:  # Log for all memory update layers
            print0(
                f"DEBUG Block L{self.layer_idx} (mode={mode}, training={self.training}): "
                f"do_memory_update eval: total_log_seq_len={total_logical_sequence_length}, "
                f"chunk_size={self.config.chunk_size}, seq_len_cond_met={seq_len_cond_met}, "
                f"final_do_update_decision={do_memory_update}",
                self.config.logfile
            )

        if self.canon_a is not None:
            x = self.canon_a(x)
        x_after_canon_a = x

        if self.attn is not None:
            residual = x_after_canon_a
            x_norm = norm(x_after_canon_a)

            if isinstance(self.attn, CascadeMemoryAttention):
                attn_output_val, fwd_mem_out, rev_mem_out, gate_reg_loss = self.attn(
                    x_norm,
                    forward_memory=fwd_mem_in,
                    reverse_memory=rev_mem_in, # Will use M_rev_persist if mode="forward"
                    control_tokens=control_tokens,
                    do_memory_update=do_memory_update, # Flag uses corrected total_logical_sequence_length
                    write_mask=write_mask if not is_reverse_update else None,
                    decay_weights=decay_weights if is_reverse_update else None,
                    is_reverse_update=is_reverse_update
                )
                if fwd_mem_out is not None: updated_fwd_result = (group_id, fwd_mem_out)
                if rev_mem_out is not None: updated_rev_result = (group_id, rev_mem_out)
            elif isinstance(self.attn, CausalSelfAttention):
                attn_output_val = self.attn(x_norm)
            else:
                attn_output_val = torch.zeros_like(x_after_canon_a)
            x = residual + attn_output_val
        else:
            x = x_after_canon_a

        if self.canon_c is not None:
            x = self.canon_c(x)
        x_after_canon_c = x

        residual = x_after_canon_c
        x_norm = norm(x_after_canon_c)
        mlp_hidden = self.mlp_fc1(x_norm)
        mlp_hidden = self.mlp_act(mlp_hidden)
        if self.canon_d is not None:
            mlp_hidden = self.canon_d(mlp_hidden)
        mlp_output = self.mlp_fc2(mlp_hidden)
        x = residual + mlp_output

        return x, updated_fwd_result, updated_rev_result, gate_reg_loss


# -----------------------------------------------------------------------------
# Main model class

class CMAModel(nn.Module):
    """CMA-based language model"""

    def __init__(self, config: CMAConfig, vocab_size: int, tokenizer=None):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.tokenizer = tokenizer or tiktoken.get_encoding("gpt2")

        # Initialize components
        self.chunk_processor = ChunkProcessor(config, self.tokenizer)
        self.memory_manager = MemoryManager(config)
        self.control_token_generator = ControlTokenGenerator(config)

        # Model components
        self.token_embedding = nn.Embedding(vocab_size, config.embed_dim)

        # --- Layer and Group Parsing ---
        self.layers = nn.ModuleList()
        self.layer_groups: List[LayerGroup] = []
        self.layer_idx_to_group_idx: Dict[int, int] = {}
        layer_idx_counter = 0
        group_idx_counter = 0
        all_read_only_indices = []
        update_layer_group_map: Dict[int, int] = {}

        for group_spec in config.layer_structure:
            if "group" in group_spec:
                group_config = group_spec["group"]
                layer_types = group_config["layers"]
                repeat = group_config.get("repeat", 1)
                for _ in range(repeat):
                    current_group = LayerGroup(group_idx=group_idx_counter)
                    group_update_layers = []
                    group_read_only_layers = []
                    group_local_only_layers = []
                    for layer_type in layer_types:
                        if layer_idx_counter in config.skip_attention_layers:
                            layer = Block(config, layer_idx_counter, layer_type="skip")
                        else:
                            layer = Block(config, layer_idx_counter, layer_type)
                        self.layers.append(layer)
                        current_group.layer_indices.append(layer_idx_counter)
                        self.layer_idx_to_group_idx[layer_idx_counter] = group_idx_counter
                        if layer_type == "memory_update":
                            group_update_layers.append(layer_idx_counter)
                            current_group.has_memory = True
                        elif layer_type == "memory_read":
                            group_read_only_layers.append(layer_idx_counter)
                            all_read_only_indices.append(layer_idx_counter)
                            current_group.has_memory = True
                        elif layer_type == "local_only" or layer_type == "skip":
                            group_local_only_layers.append(layer_idx_counter)
                        layer_idx_counter += 1
                    if len(group_update_layers) > 1: raise ValueError(f"Group {group_idx_counter}: >1 update layer")
                    if group_read_only_layers and not group_update_layers: raise ValueError(
                        f"Group {group_idx_counter}: read-only layers require an update layer")
                    if group_update_layers:
                        current_group.memory_update_layer_idx = group_update_layers[0]
                        update_layer_group_map[group_update_layers[0]] = group_idx_counter
                    current_group.read_only_layer_indices = group_read_only_layers
                    current_group.local_only_layer_indices = group_local_only_layers
                    self.layer_groups.append(current_group)
                    group_idx_counter += 1
            else:
                layer_type = group_spec.get("type", "local_only")
                if layer_idx_counter in config.skip_attention_layers:
                    layer = Block(config, layer_idx_counter, layer_type="skip")
                else:
                    layer = Block(config, layer_idx_counter, layer_type)
                self.layers.append(layer)
                current_group = LayerGroup(group_idx=group_idx_counter)
                current_group.layer_indices.append(layer_idx_counter)
                self.layer_idx_to_group_idx[layer_idx_counter] = group_idx_counter
                group_update_layers = []
                group_read_only_layers = []
                group_local_only_layers = []
                if layer_type == "memory_update":
                    current_group.memory_update_layer_idx = layer_idx_counter
                    current_group.has_memory = True
                    update_layer_group_map[layer_idx_counter] = group_idx_counter
                    group_update_layers.append(layer_idx_counter)
                elif layer_type == "memory_read":
                    all_read_only_indices.append(layer_idx_counter)
                    group_read_only_layers.append(layer_idx_counter)
                    current_group.has_memory = True
                elif layer_type == "local_only" or layer_type == "skip":
                    group_local_only_layers.append(layer_idx_counter)
                current_group.read_only_layer_indices = group_read_only_layers
                current_group.local_only_layer_indices = group_local_only_layers
                self.layer_groups.append(current_group)
                group_idx_counter += 1
                layer_idx_counter += 1

        for read_only_idx in all_read_only_indices:
            group_idx = self.layer_idx_to_group_idx.get(read_only_idx)
            if group_idx is None: raise ValueError(f"Internal error: Read-only layer {read_only_idx} has no group.")
            group = self.layer_groups[group_idx]
            if group.memory_update_layer_idx is None: raise ValueError(
                f"Config error: Read-only layer {read_only_idx} in group {group_idx} lacks update layer.")

        self.num_layers = layer_idx_counter
        self.num_groups = group_idx_counter
        self.num_memory_groups = sum(1 for g in self.layer_groups if g.has_memory)
        print0(f"Parsed {self.num_layers} layers into {self.num_groups} groups ({self.num_memory_groups} with memory).", self.config.logfile)
        # --- End Layer and Group Parsing ---

        self.lm_head = nn.Linear(config.embed_dim, vocab_size, bias=False)

        # --- Initialize Memory Parameters ---
        self.group_id_to_memory_idx: Dict[int, int] = {}
        memory_param_idx_counter = 0
        if self.num_memory_groups > 0:
            if config.share_initial_memory:
                print0("Using shared initial memory parameters.", self.config.logfile)
                shared_fwd = nn.Parameter(
                    torch.randn(1, config.max_memory_size, config.embed_dim) * config.memory_init_scale)
                shared_rev_la = nn.Parameter(
                    torch.randn(1, config.reverse_memory_size, config.embed_dim) * config.memory_init_scale)
                shared_rev_p = nn.Parameter(
                    torch.randn(1, config.reverse_memory_size, config.embed_dim) * config.memory_init_scale)
                self.initial_fwd_params = nn.ParameterList([shared_fwd] * self.num_memory_groups)
                self.initial_rev_la_params = nn.ParameterList([shared_rev_la] * self.num_memory_groups)
                self.initial_rev_p_params = nn.ParameterList([shared_rev_p] * self.num_memory_groups)
                for group in self.layer_groups:
                    if group.has_memory: self.group_id_to_memory_idx[group.group_idx] = 0
                memory_param_idx_counter = 1
            else:
                print0("Using dedicated initial memory parameters per group.", self.config.logfile)
                fwd_params, rev_la_params, rev_p_params = [], [], []
                for group in self.layer_groups:
                    if group.has_memory:
                        self.group_id_to_memory_idx[group.group_idx] = memory_param_idx_counter
                        fwd_params.append(nn.Parameter(
                            torch.randn(1, config.max_memory_size, config.embed_dim) * config.memory_init_scale))
                        rev_la_params.append(nn.Parameter(
                            torch.randn(1, config.reverse_memory_size, config.embed_dim) * config.memory_init_scale))
                        rev_p_params.append(nn.Parameter(
                            torch.randn(1, config.reverse_memory_size, config.embed_dim) * config.memory_init_scale))
                        memory_param_idx_counter += 1
                self.initial_fwd_params = nn.ParameterList(fwd_params)
                self.initial_rev_la_params = nn.ParameterList(rev_la_params)
                self.initial_rev_p_params = nn.ParameterList(rev_p_params)

            # Ensure all parameters are on the correct device
            device = self.token_embedding.weight.device
            for param_list in [self.initial_fwd_params, self.initial_rev_la_params, self.initial_rev_p_params]:
                for param in param_list:
                    param.data = param.data.to(device)
        else:
            print0("No memory groups found. Initial memory parameters will not be created.", self.config.logfile)
            self.initial_fwd_params = nn.ParameterList()
            self.initial_rev_la_params = nn.ParameterList()
            self.initial_rev_p_params = nn.ParameterList()
        print0(f"Created {memory_param_idx_counter} sets of initial memory parameters.", self.config.logfile)
        # --- End Initialize Memory Parameters ---

        # --- State Tracking ---
        # Runtime memory states are dictionaries keyed by group_idx
        self.M_fwd: Dict[int, Tensor] = {}
        self.M_rev_ahead: Dict[int, Tensor] = {}
        self.M_rev_persist: Dict[int, Tensor] = {}

        self.closed_chunks: List[List[int]] = []
        self.total_tokens_processed = 0
        self.tokens_since_persistent_update = 0
        self.current_chunk_tokens: List[int] = []
        self.current_chunk_text: str = ""
        self.training_step = 0
        self.total_training_steps = 10000
        self.total_seq_len = 0

        # Initialize weights
        self.apply(self._init_weights)

        # Explicitly re-apply gate_bias_init after generic initialization
        for layer_block in self.layers:
            if isinstance(layer_block.attn, CascadeMemoryAttention):
                if hasattr(layer_block.attn.gate_proj, 'bias') and layer_block.attn.gate_proj.bias is not None:
                    nn.init.constant_(layer_block.attn.gate_proj.bias, self.config.gate_bias_init)
                    if self.config.DEBUG_LEVEL > 0 and self.config.logfile:  # Ensure logfile exists
                        print0(f"DEBUG: Explicitly set gate_proj.bias for layer {layer_block.layer_idx} to {self.config.gate_bias_init}", self.config.logfile)

        # --- Parameter Count ---
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Count initial memory parameters
        initial_memory_params = 0
        if hasattr(self, 'initial_fwd_params'): initial_memory_params += sum(
            p.numel() for p in self.initial_fwd_params if p.requires_grad)
        if hasattr(self, 'initial_rev_la_params'): initial_memory_params += sum(
            p.numel() for p in self.initial_rev_la_params if p.requires_grad)
        if hasattr(self, 'initial_rev_p_params'): initial_memory_params += sum(
            p.numel() for p in self.initial_rev_p_params if p.requires_grad)
        if config.share_initial_memory and self.num_memory_groups > 1:
            num_shared_sets = 3
            params_per_set = initial_memory_params // (
                    self.num_memory_groups * num_shared_sets) if self.num_memory_groups > 0 else 0
            initial_memory_params = num_shared_sets * params_per_set

        # Initialize counters for different parameter types
        memory_qkv_params = 0
        memory_update_qkv_params = 0
        memory_update_gating_params = 0
        attention_gating_params = 0
        control_token_params = 0

        # Count parameters by category
        for layer in self.layers:
            if isinstance(layer.attn, CascadeMemoryAttention):
                # 1. Memory QKV projections (standard q/k/v used in memory layers)
                if hasattr(layer.attn, 'q_proj'):
                    memory_qkv_params += sum(p.numel() for p in layer.attn.q_proj.parameters() if p.requires_grad)
                if hasattr(layer.attn, 'k_proj'):
                    memory_qkv_params += sum(p.numel() for p in layer.attn.k_proj.parameters() if p.requires_grad)
                if hasattr(layer.attn, 'v_proj'):
                    memory_qkv_params += sum(p.numel() for p in layer.attn.v_proj.parameters() if p.requires_grad)

                # 2. Memory update QKV projections
                if hasattr(layer.attn, 'fwd_memory_q_proj'):
                    memory_update_qkv_params += sum(
                        p.numel() for p in layer.attn.fwd_memory_q_proj.parameters() if p.requires_grad)
                if hasattr(layer.attn, 'fwd_memory_k_proj'):
                    memory_update_qkv_params += sum(
                        p.numel() for p in layer.attn.fwd_memory_k_proj.parameters() if p.requires_grad)
                if hasattr(layer.attn, 'fwd_memory_v_proj'):
                    memory_update_qkv_params += sum(
                        p.numel() for p in layer.attn.fwd_memory_v_proj.parameters() if p.requires_grad)
                if hasattr(layer.attn, 'rev_memory_q_proj'):
                    memory_update_qkv_params += sum(
                        p.numel() for p in layer.attn.rev_memory_q_proj.parameters() if p.requires_grad)
                if hasattr(layer.attn, 'rev_memory_k_proj'):
                    memory_update_qkv_params += sum(
                        p.numel() for p in layer.attn.rev_memory_k_proj.parameters() if p.requires_grad)
                if hasattr(layer.attn, 'rev_memory_v_proj'):
                    memory_update_qkv_params += sum(
                        p.numel() for p in layer.attn.rev_memory_v_proj.parameters() if p.requires_grad)

                # 3. Memory update gating
                if hasattr(layer.attn, 'fwd_memory_gate_proj'):
                    memory_update_gating_params += sum(
                        p.numel() for p in layer.attn.fwd_memory_gate_proj.parameters() if p.requires_grad)
                if hasattr(layer.attn, 'rev_memory_gate_proj'):
                    memory_update_gating_params += sum(
                        p.numel() for p in layer.attn.rev_memory_gate_proj.parameters() if p.requires_grad)

                # 4. Attention gating
                if hasattr(layer.attn, 'gate_proj'):
                    attention_gating_params += sum(
                        p.numel() for p in layer.attn.gate_proj.parameters() if p.requires_grad)

                # 5. Control token integration
                if hasattr(layer.attn, 'control_proj') and config.integration_method == "query_fusion":
                    control_token_params += sum(
                        p.numel() for p in layer.attn.control_proj.parameters() if p.requires_grad)

        # Calculate totals
        core_memory_params = (
                initial_memory_params +
                memory_update_qkv_params +
                memory_update_gating_params +
                attention_gating_params +
                control_token_params
        )
        total_memory_params = core_memory_params + memory_qkv_params
        standard_params = total_params - total_memory_params

        print0(f"--- CMA Model Parameter Count ---", self.config.logfile)
        print0(f"Initial memory parameters: {initial_memory_params:,} ({memory_param_idx_counter} sets, {'shared' if config.share_initial_memory else 'dedicated'})", self.config.logfile)
        print0(f"Memory QKV projections: {memory_qkv_params:,}", self.config.logfile)
        print0(f"Memory update QKV projections: {memory_update_qkv_params:,}", self.config.logfile)
        print0(f"Memory update gating: {memory_update_gating_params:,}", self.config.logfile)
        print0(f"Attention gating: {attention_gating_params:,}", self.config.logfile)
        print0(f"Control token integration: {control_token_params:,}", self.config.logfile)
        print0(f"Core memory parameters (no read QKV proj): {core_memory_params:,} ({core_memory_params / total_params * 100:.2f}% of total)", self.config.logfile)
        print0(f"Total memory parameters: {total_memory_params:,} ({total_memory_params / total_params * 100:.2f}% of total)", self.config.logfile)
        print0(f"Standard parameters: {standard_params:,} ({standard_params / total_params * 100:.2f}% of total)", self.config.logfile)
        print0(f"Total trainable parameters: {total_params:,}", self.config.logfile)
        print0(f"------------------------------------------", self.config.logfile)
        # --- End Parameter Count ---

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _initialize_memory_states(self, force_reset: bool = False):
        """Initializes or resets runtime memory state dictionaries."""
        dev = self.token_embedding.weight.device
        for group in self.layer_groups:
            if group.has_memory:
                group_idx = group.group_idx
                mem_idx = self.group_id_to_memory_idx[group_idx]

                # Initialize if None or if force_reset is True
                if force_reset or group_idx not in self.M_fwd:
                    self.M_fwd[group_idx] = self.initial_fwd_params[mem_idx].clone().to(dev)
                # M_rev_ahead is always reset/recomputed in the cycle, initialize empty or reset
                # We initialize it here mainly to ensure the key exists if accessed before cycle
                self.M_rev_ahead[group_idx] = torch.zeros(  # Placeholder, will be overwritten
                    1, self.config.reverse_memory_size, self.config.embed_dim, device=dev
                )
                if force_reset or group_idx not in self.M_rev_persist:
                    self.M_rev_persist[group_idx] = self.initial_rev_p_params[mem_idx].clone().to(dev)

    def forward(
            self,
            input_ids: Union[str, List[int], List[List[int]], torch.Tensor, None],
            *, training_mode: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.config.DEBUG_LEVEL > 0: print0(f"DEBUG: CMAModel.forward called with input_ids type={type(input_ids)}, training_mode={training_mode}", self.config.logfile)

        self.training = training_mode
        if training_mode:
            super().train(True)
        else:
            super().eval()
        self._initialize_memory_states(force_reset=False)
        dev = self.token_embedding.weight.device

        final_logits: Optional[torch.Tensor] = None
        final_gate_loss: Optional[torch.Tensor] = None
        num_new_tokens_for_this_call = 0 # Initialize

        if (input_ids is None or input_ids == "" or input_ids == [] or
                (isinstance(input_ids, torch.Tensor) and input_ids.numel() == 0)):
            # ---- SET current_total_logical_sequence_length for potentially empty input but existing buffer ----
            self.current_total_logical_sequence_length = sum(len(c) for c in self.closed_chunks) + len(self.current_chunk_tokens)
            if self.config.DEBUG_LEVEL > 0:
                 print0(f"DEBUG CMAModel.forward: Empty/None input. Logical seq_len for current buffer set to {self.current_total_logical_sequence_length}", self.config.logfile)

            if self.current_chunk_tokens:
                final_logits, final_gate_loss = self._process_current_chunk(generation_mode=not self.training)
            else:
                final_logits = torch.zeros(1, 0, self.vocab_size, device=dev)
        else:
            new_tokens: List[int]
            input_processing_mode: str
            if isinstance(input_ids, str):
                new_tokens = self.tokenizer.encode(input_ids)
                input_processing_mode = "semantic"
            elif isinstance(input_ids, list) and input_ids and isinstance(input_ids[0], list):
                new_chunks_from_input = [list(c) for c in input_ids]
                new_tokens = [t for c in new_chunks_from_input for t in c]
                input_processing_mode = "caller_exact"
            else:
                new_tokens = input_ids.tolist() if isinstance(input_ids, torch.Tensor) else list(input_ids)
                input_processing_mode = "fixed"

            num_new_tokens_for_this_call = len(new_tokens) # Set it here after parsing new_tokens

            # self.total_seq_len = len(new_tokens) # This line was identified as problematic, effectively removed
            if self.config.DEBUG_LEVEL > 0: print0(f"DEBUG: Mode={input_processing_mode}, new_tokens length={num_new_tokens_for_this_call}", self.config.logfile)


            trigger_update_cycle = False
            full_tokens_for_cycle: List[int] = []
            chunks_for_cycle: List[List[int]] = []

            is_likely_generation_step = isinstance(input_ids, list) and len(input_ids) == 1 and not isinstance(
                input_ids[0], list) and not self.training # Added not self.training for clarity

            if input_processing_mode == "caller_exact":
                trigger_update_cycle = True
                full_tokens_for_cycle = [t for c in self.closed_chunks for t in
                                         c] + self.current_chunk_tokens + new_tokens
                chunks_for_cycle = self.closed_chunks + \
                                   ([self.current_chunk_tokens] if self.current_chunk_tokens else []) + \
                                   new_chunks_from_input
                self.current_chunk_tokens = []
                self.current_chunk_text = ""
            elif len(self.current_chunk_tokens) + len(new_tokens) >= self.config.chunk_size:
                trigger_update_cycle = True
                full_tokens_for_cycle = [t for c in self.closed_chunks for t in
                                         c] + self.current_chunk_tokens + new_tokens
                self.current_chunk_tokens = []
                self.current_chunk_text = ""
            # This condition was for training, to force cycle even if buffer not full.
            # Let's ensure it only applies in training.
            elif self.training and self.current_chunk_tokens and not is_likely_generation_step and input_processing_mode in {"semantic", "fixed"}:
                trigger_update_cycle = True
                full_tokens_for_cycle = [t for c in self.closed_chunks for t in
                                         c] + self.current_chunk_tokens + new_tokens
                self.current_chunk_tokens = []
                self.current_chunk_text = ""
            else:
                self.current_chunk_tokens.extend(new_tokens)
                try:
                    decoded_new = self.tokenizer.decode(new_tokens)
                    self.current_chunk_text += decoded_new
                except Exception as e:
                    print0(f"Warning: Error decoding appended tokens: {new_tokens}. Error: {e}", self.config.logfile)

            if trigger_update_cycle:
                if self.config.reset_memory_on_cycle:
                    self._initialize_memory_states(force_reset=True)

                # ---- SET current_total_logical_sequence_length for the cycle ----
                self.current_total_logical_sequence_length = len(full_tokens_for_cycle)
                if self.config.DEBUG_LEVEL > 0:
                    print0(f"DEBUG CMAModel.forward: Cycle triggered. Logical seq_len set to {self.current_total_logical_sequence_length}", self.config.logfile)

                if input_processing_mode != "caller_exact":
                    if not full_tokens_for_cycle:
                        chunks_for_cycle = []
                    elif input_processing_mode == "semantic":
                        try:
                            full_text = self.tokenizer.decode(full_tokens_for_cycle)
                            chunks_for_cycle = self.chunk_processor.semantic_chunk_reverse_with_gap(full_text)
                        except Exception as e:
                            print0(f"Error during semantic re-chunking: {e}. Falling back to fixed.", self.config.logfile)
                            chunks_for_cycle = self.chunk_processor.fixed_size_chunk_reverse_with_gap(
                                full_tokens_for_cycle)
                    else: # fixed
                        chunks_for_cycle = self.chunk_processor.fixed_size_chunk_reverse_with_gap(full_tokens_for_cycle)

                if not chunks_for_cycle:
                    final_logits = torch.zeros(1, 0, self.vocab_size, device=dev)
                    final_gate_loss = None
                else:
                    all_cycle_logits, cycle_gate_loss = self._trigger_memory_update_cycle(chunks_for_cycle)
                    if all_cycle_logits is not None:
                        # num_new_tokens_for_this_call was defined earlier based on input_ids length
                        if all_cycle_logits.size(1) >= num_new_tokens_for_this_call:
                            final_logits = all_cycle_logits[:, -num_new_tokens_for_this_call:, :]
                        else:
                            # This case means the cycle produced fewer logits than the input sequence length.
                            # This could happen if chunking resulted in fewer total tokens than original input,
                            # or if input was very short.
                            # Use all available logits and pad targets if necessary (or error, or log warning).
                            # For now, just use what's available and let the outer loss handling deal with it,
                            # but a warning is good.
                            print0(
                                f"Warning (CYCLE): All cycle logits len {all_cycle_logits.size(1)} < "
                                f"num_new_tokens_for_this_call {num_new_tokens_for_this_call}. "
                                f"Using all available cycle logits. Mode: {'TRAIN' if self.training else 'EVAL'}",
                                self.config.logfile
                            )
                            final_logits = all_cycle_logits
                            # If training, ensure targets are also sliced if this path is taken.
                            # If eval, the targets are already sliced by the caller based on these logits.
                    elif chunks_for_cycle:  # Cycle was triggered, chunks existed, but no logits (should be rare)
                        final_logits = torch.zeros(1, 0, self.vocab_size, device=dev)
                        print0(
                            f"Warning (CYCLE): all_cycle_logits is None despite having chunks. "
                            f"Mode: {'TRAIN' if self.training else 'EVAL'}",
                            self.config.logfile
                        )
                    else:  # No chunks for cycle (e.g. empty input to cycle)
                        final_logits = torch.zeros(1, 0, self.vocab_size, device=dev)

                    if self.training:
                        final_gate_loss = cycle_gate_loss  # Keep gate loss from cycle for training
                    else:
                        final_gate_loss = None  # No gate loss for eval output after cycle

            else: # No cycle
                # ---- SET current_total_logical_sequence_length for current chunk processing ----
                self.current_total_logical_sequence_length = sum(len(c) for c in self.closed_chunks) + len(self.current_chunk_tokens)
                if self.config.DEBUG_LEVEL > 0:
                    print0(f"DEBUG CMAModel.forward: No cycle. Logical seq_len for current chunk set to {self.current_total_logical_sequence_length}", self.config.logfile)

                if self.current_chunk_tokens:
                    logits_full_buffer, gate_loss_chunk = self._process_current_chunk(
                        generation_mode=(not self.training and is_likely_generation_step)
                    )
                    if self.training:
                        if logits_full_buffer.size(1) > 0 and num_new_tokens_for_this_call > 0:
                            if logits_full_buffer.size(1) >= num_new_tokens_for_this_call:
                                final_logits = logits_full_buffer[:, -num_new_tokens_for_this_call:, :]
                            else:
                                print0(
                                    f"Warning (TRAIN NO CYCLE): Buffer logits len {logits_full_buffer.size(1)} vs num_new_tokens {num_new_tokens_for_this_call}. Slicing with -{num_new_tokens_for_this_call}:", self.config.logfile)
                                final_logits = logits_full_buffer[:, -num_new_tokens_for_this_call:, :]
                        elif num_new_tokens_for_this_call == 0:
                            final_logits = torch.zeros(1, 0, self.vocab_size, device=dev)
                        else:
                            final_logits = torch.zeros(1, 0, self.vocab_size, device=dev)
                        final_gate_loss = gate_loss_chunk
                    else: # Evaluation, no cycle
                        final_logits = logits_full_buffer
                        final_gate_loss = None # No gate loss for eval output
                else:
                    final_logits = torch.zeros(1, 0, self.vocab_size, device=dev)
                    final_gate_loss = None

        # Ensure final_logits is not None (safeguard, should be set by one of the paths)
        if final_logits is None:
            # This state should ideally not be reached if logic is exhaustive
            print0("CRITICAL Warning: final_logits was None at end of CMAModel.forward. Defaulting to empty.",
                  self.config.logfile)
            final_logits = torch.zeros(1, 0, self.vocab_size, device=dev)
            # final_gate_loss might also be None here, which is acceptable if logits are empty.

        if self.training:
            # Existing slicing logic for cycle or no-cycle path leads to final_logits
            if self.config.DEBUG_LEVEL > 0 and self.training_step < 5:
                expected_len = num_new_tokens_for_this_call  # This was len(new_tokens)
                if final_logits.shape[1] != expected_len:
                    print0(
                        f"DEBUG CMAModel.forward (TRAIN): Step {self.training_step}, MISMATCH final_logits len {final_logits.shape[1]} vs expected new_tokens len {expected_len}. Input new_tokens len was {len(new_tokens)}. Cycle was {trigger_update_cycle}.",
                        logfile=self.config.logfile)
                else:
                    print0(
                        f"DEBUG CMAModel.forward (TRAIN): Step {self.training_step}, OK final_logits len {final_logits.shape[1]} == expected new_tokens len {expected_len}. Cycle was {trigger_update_cycle}.",
                        logfile=self.config.logfile)

        if not self.training and final_logits is not None:
            expected_len_val = num_new_tokens_for_this_call
            actual_len_val = final_logits.shape[1]
            is_mismatch = expected_len_val != actual_len_val
            status_msg = "MISMATCH" if is_mismatch else "OK"
            len_last_chunk_val = len(
                self.current_chunk_tokens) if trigger_update_cycle and self.current_chunk_tokens else 'N/A'

            print0(
                f"DEBUG CMAModel.forward (VALIDATION): Step {self.training_step}, {status_msg} final_logits len {actual_len_val} vs. input_ids len {expected_len_val}. "
                f"Cycle: {trigger_update_cycle}. len_last_chunk: {len_last_chunk_val}. Closed chunks: {len(self.closed_chunks)}.",
                logfile=self.config.logfile
            )
        if final_logits is not None:
            log_msg_prefix = "[TRAIN]" if self.training else "[VAL]"
            if not self.training and final_logits is not None:
                print0(f"DEBUG CMAModel.forward {log_msg_prefix}: "
                       f"Logits len={final_logits.shape[1]}, num_new_tokens_for_this_call (original input len for this call)={num_new_tokens_for_this_call}. "
                       f"Cycle triggered={trigger_update_cycle}. If cycle, current_chunk_tokens len={len(self.current_chunk_tokens) if self.current_chunk_tokens else 'N/A'}",
                       self.config.logfile)
            elif self.training and final_logits is not None and self.config.DEBUG_LEVEL > 0:
                print0(
                    f"DEBUG CMAModel.forward (TRAIN): Step {self.training_step}, OK final_logits len {final_logits.shape[1]} == expected new_tokens len {num_new_tokens_for_this_call}. Cycle was {trigger_update_cycle}.",
                    logfile=self.config.logfile)

        # final_gate_loss is now the aggregated gate loss from the relevant path
        # (either cycle Pass 2 if training, or _process_current_chunk if training and no cycle)
        return final_logits, final_gate_loss

    def _process_current_chunk(self, generation_mode: bool = False) -> Tuple[
        torch.Tensor, Optional[torch.Tensor]]:
        if not self.current_chunk_tokens:
            return torch.zeros(1, 0, self.vocab_size, device=self.token_embedding.weight.device), None
        dev = self.token_embedding.weight.device

        chunk_idx = len(self.closed_chunks)
        total_chunks_in_history = chunk_idx + 1  # For control token ratio

        # current_total_logical_sequence_length is set by the caller (CMAModel.forward or generate)

        chunk_tensor = torch.tensor(self.current_chunk_tokens, device=dev, dtype=torch.long).unsqueeze(0)
        x = self.token_embedding(chunk_tensor)

        approx_processed_len_for_ctrl_tokens = sum(len(c) for c in self.closed_chunks)  # History before current chunk
        current_mem_size_for_ctrl_tokens = self.memory_manager.get_effective_size(approx_processed_len_for_ctrl_tokens)

        mode = "generate" if generation_mode else "forward"
        ctrl = self.control_token_generator.generate_control_tokens(
            mode=mode,
            current_chunk_idx=chunk_idx, total_chunks=total_chunks_in_history,
            current_mem_size=current_mem_size_for_ctrl_tokens, max_mem_size=self.config.max_memory_size,
            seq_len=approx_processed_len_for_ctrl_tokens
        )

        M_fwd = self.M_fwd
        M_rev_ahead = self.M_rev_ahead
        M_rev_persist = self.M_rev_persist
        aggregated_gate_loss: Optional[torch.Tensor] = None

        for layer_idx, layer_block in enumerate(self.layers):  # Renamed 'layer' to 'layer_block'
            group_idx = self.layer_idx_to_group_idx[layer_idx]

            x, updated_fwd, updated_rev, gate_loss = layer_block(  # Use 'layer_block'
                x,
                M_fwd_dict=M_fwd, M_rev_ahead_dict=M_rev_ahead, M_rev_persist_dict=M_rev_persist,
                group_id=group_idx, mode=mode, control_tokens=ctrl,
                write_mask=None, decay_weights=None,
                # ---- PASS THE CORRECT logical sequence length ----
                total_logical_sequence_length=self.current_total_logical_sequence_length
            )

            if gate_loss is not None:
                current_loss = gate_loss.mean() if gate_loss.numel() > 1 else gate_loss
                if aggregated_gate_loss is None:
                    aggregated_gate_loss = current_loss
                else:
                    aggregated_gate_loss += current_loss

        x = norm(x)
        logits = self.lm_head(x)
        return logits, aggregated_gate_loss

    def _print_memory_stats(self, memory_dict: Dict[int, Tensor], pass_name: str, chunk_idx: Optional[int] = None):
        if not (self.config.DEBUG_LEVEL > 1):  # Only print if DEBUG_LEVEL is high enough
            return

        log_prefix = f"DEBUG MEMORY: Pass='{pass_name}'"
        if chunk_idx is not None:
            log_prefix += f", ChunkIdx={chunk_idx}"

        if not memory_dict:
            print0(f"{log_prefix} - Memory dict is empty.", self.config.logfile)
            return

        for group_id, mem_tensor in memory_dict.items():
            if mem_tensor is not None and mem_tensor.numel() > 0:
                mean_val = mem_tensor.mean().item()
                std_val = mem_tensor.std().item()
                abs_max_val = mem_tensor.abs().max().item()
                print0(
                    f"{log_prefix} - Group {group_id}: Shape={mem_tensor.shape}, Mean={mean_val:.4f}, Std={std_val:.4f}, AbsMax={abs_max_val:.4f}",
                    self.config.logfile)
            elif mem_tensor is not None:  # Empty tensor
                print0(f"{log_prefix} - Group {group_id}: Shape={mem_tensor.shape} (Empty Tensor)", self.config.logfile)
            else:  # None
                print0(f"{log_prefix} - Group {group_id}: None", self.config.logfile)

    def _trigger_memory_update_cycle(
            self,
            chunks: List[List[int]]
    ) -> Tuple[
        Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.config.DEBUG_LEVEL > 0: print0(
            f"DEBUG: Starting _trigger_memory_update_cycle with {len(chunks)} chunks", self.config.logfile)
        if not chunks:
            return None, None

        dev = self.token_embedding.weight.device
        B = 1

        # self.current_total_logical_sequence_length is set by CMAModel.forward before calling this.
        # It represents sum(len(c) for c in chunks).

        # --- Pass 1: Lookahead Reverse ---
        try:
            M_rev_ahead_computed = self._run_lookahead_reverse_pass(chunks, B, dev,
                                                                    self.current_total_logical_sequence_length)
            self._print_memory_stats(M_rev_ahead_computed, "LookaheadReverse (Output)")
        except Exception as e:
            print0(f"ERROR in _run_lookahead_reverse_pass: {e}", self.config.logfile)
            raise

        pass2_new_tokens_logits_list: List[torch.Tensor] = []
        pass2_gate_loss_aggregated: Optional[torch.Tensor] = None

        # For control tokens and write masks in Pass 2, we need tokens processed *before* the current chunk in *this pass*
        tokens_processed_in_pass2_before_current_chunk = 0

        self._print_memory_stats(self.M_fwd, "Forward (Input to Pass 2)")
        # M_rev_persist is used for attention in Pass 2. M_rev_ahead_computed is also passed but Block logic will select M_rev_persist.
        self._print_memory_stats(self.M_rev_persist, "PersistentReverse (Input to Pass 2 for attention)")

        for chunk_idx, chunk_tokens in enumerate(chunks):
            if not chunk_tokens: continue

            chunk_tensor = torch.tensor(chunk_tokens, dtype=torch.long, device=dev).unsqueeze(0)
            x = self.token_embedding(chunk_tensor)

            # Control tokens for Pass 2
            # seq_len for control tokens should be tokens processed *before* this chunk in this pass.
            current_mem_size_for_ctrl = self.memory_manager.get_effective_size(
                self.total_tokens_processed + tokens_processed_in_pass2_before_current_chunk)
            ctrl = self.control_token_generator.generate_control_tokens(
                mode="forward", current_chunk_idx=chunk_idx, total_chunks=len(chunks),
                current_mem_size=current_mem_size_for_ctrl, max_mem_size=self.config.max_memory_size,
                seq_len=self.total_tokens_processed + tokens_processed_in_pass2_before_current_chunk
            )

            M_rev_persist_for_chunk: Dict[int, Tensor]
            if self.training and self.config.enable_mask_future_dropout:
                p_drop = get_mask_future_schedule(self.config, self.training_step, self.total_training_steps)
                if self.config.DEBUG_LEVEL > 0:
                    print0(f"DEBUG MASKING: Training step {self.training_step}, M_rev_persist p_drop: {p_drop:.4f}",
                           self.config.logfile)
                M_rev_persist_for_chunk = self._mask_persistent_memory(self.M_rev_persist, p_drop)
                if chunk_idx == 0 and self.config.DEBUG_LEVEL > 1:
                    self._print_memory_stats(M_rev_persist_for_chunk, "PersistentReverse (Masked for Pass 2)",
                                             chunk_idx=chunk_idx)
            else:
                p_drop = 0.0
                M_rev_persist_for_chunk = self.M_rev_persist
                if not self.training and self.config.enable_mask_future_dropout:
                    print0(
                        f"DEBUG MASKING: Validation (step {self.training_step}), M_rev_persist p_drop: N/A (masking disabled for eval)",
                        self.config.logfile)

            chunk_gate_loss_accumulator_for_layers: Optional[torch.Tensor] = None

            for layer_idx, layer_block in enumerate(self.layers):
                group_idx = self.layer_idx_to_group_idx[layer_idx]
                group = self.layer_groups[group_idx]
                wmask = None
                if group.has_memory and layer_idx == group.memory_update_layer_idx:
                    wmask = self.memory_manager.get_write_mask(
                        current_chunk_idx_in_pass=chunk_idx,
                        total_chunks_in_pass=len(chunks),
                        # total_tokens_processed_before_chunk for write_mask should be global history + progress in current pass
                        total_tokens_processed_before_chunk=self.total_tokens_processed + tokens_processed_in_pass2_before_current_chunk,
                        current_chunk_len=len(chunk_tokens),
                        batch_size=B, device=dev
                    )
                if not self.training and chunk_idx == 0 and wmask is not None and self.config.DEBUG_LEVEL > 0:
                    print0(f"DEBUG VAL First Write Mask (Pass 2, Chunk 0): "
                           f"true_frac={wmask.float().mean().item():.4f}, "
                           f"shape={wmask.shape}, "
                           f"total_tokens_processed_before_chunk={self.total_tokens_processed + tokens_processed_in_pass2_before_current_chunk}",
                           self.config.logfile)

                x, updated_fwd, _, layer_gate_loss = layer_block(
                    x, M_fwd_dict=self.M_fwd, M_rev_ahead_dict=M_rev_ahead_computed,
                    # M_rev_ahead is passed for potential other uses
                    M_rev_persist_dict=M_rev_persist_for_chunk,  # This will be selected by Block for mode="forward"
                    group_id=group_idx, mode="forward", control_tokens=ctrl,
                    write_mask=wmask, decay_weights=None,
                    total_logical_sequence_length=self.current_total_logical_sequence_length  # Corrected
                )
                if chunk_idx == 0 and self.config.DEBUG_LEVEL > 0:  # Print for first chunk of sequence
                    self._print_memory_stats(M_rev_persist_for_chunk,
                                             f"PersistentReverse ({'Masked' if self.training and self.config.enable_mask_future_dropout and p_drop > 0 else 'Unmasked'} for Pass 2 Attn)",
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
            pass2_new_tokens_logits_list.append(logits_for_chunk)

            if chunk_gate_loss_accumulator_for_layers is not None:
                if pass2_gate_loss_aggregated is None:
                    pass2_gate_loss_aggregated = chunk_gate_loss_accumulator_for_layers
                else:
                    pass2_gate_loss_aggregated += chunk_gate_loss_accumulator_for_layers

            tokens_processed_in_pass2_before_current_chunk += len(chunk_tokens)

        self._print_memory_stats(self.M_fwd, "Forward (Output of Pass 2)")
        del M_rev_ahead_computed  # Temporary
        self.M_rev_ahead = {}  # Clear runtime state

        # --- Pass 3: Persistent Reverse ---
        self._print_memory_stats(self.M_rev_persist, "PersistentReverse (Input to Pass 3)")
        self._run_persistent_reverse_pass(chunks, B, dev, self.current_total_logical_sequence_length)
        self._print_memory_stats(self.M_rev_persist, "PersistentReverse (Output of Pass 3)")

        self.closed_chunks = chunks[:-1]
        self.current_chunk_tokens = chunks[-1] if chunks else []
        # Update model state after successful cycle
        self.closed_chunks = chunks[:-1]
        self.current_chunk_tokens = chunks[-1] if chunks else []

        # When reset_memory_on_cycle is False, 'chunks' represents the re-chunked entirety
        # of all historical tokens plus new tokens. So, its total length IS the new
        # total_tokens_processed.
        # When reset_memory_on_cycle is True, self.total_tokens_processed would have been 0
        # (or reset by _initialize_memory_states if that were called before this point,
        # but CMAModel.forward handles the _initialize_memory_states reset).
        # The key is that `chunks` argument to this function is the definitive list of all
        # tokens considered for this cycle.
        self.total_tokens_processed = sum(len(c) for c in chunks)

        self.tokens_since_persistent_update = 0  # Reset as persistent was just run (or attempted)

        concatenated_pass2_logits: Optional[torch.Tensor] = None
        if pass2_new_tokens_logits_list:
            concatenated_pass2_logits = torch.cat(pass2_new_tokens_logits_list, dim=1)
        elif chunks:
            concatenated_pass2_logits = torch.zeros(B, 0, self.vocab_size, device=dev)

        return concatenated_pass2_logits, pass2_gate_loss_aggregated

    def _process_chunk_in_cycle(
            self,
            chunk_tensor: torch.Tensor,
            chunk_idx: int,
            M_rev_ahead: torch.Tensor
    ) -> None:
        """
        Helper used inside _trigger_memory_update_cycle.
        Processes a single chunk tensor, using provided M_rev_ahead and current self.M_fwd.
        Updates self.M_fwd via memory update layers.
        Updates self.total_tokens_processed.
        """
        # Assuming B=1 for stateful processing
        if chunk_tensor.ndim == 1:
            chunk_tensor = chunk_tensor.unsqueeze(0)  # Add batch dim if needed by layers

        # Get total chunks estimate for control tokens (might need adjustment)
        # Total chunks = history + current one being processed + estimate of future?
        # Let's use history + current as the known sequence length for ratios.
        total_chunks_so_far = len(self.closed_chunks) + 1

        x = self.token_embedding(chunk_tensor)  # Shape (1, T, D)

        # Generate control tokens for the forward pass processing this chunk
        ctrl = self.control_token_generator.generate_control_tokens(
            mode="forward",  # Processing a known chunk in the forward direction
            current_chunk_idx=chunk_idx,
            total_chunks=total_chunks_so_far,  # Use current known total
            current_mem_size=self.memory_manager.get_effective_size(self.total_tokens_processed),
            max_mem_size=self.config.max_memory_size,
            seq_len=self.total_tokens_processed  # Tokens processed *before* this chunk
        )

        # Use local copies of memory states to avoid modifying them incorrectly if layer forward pass fails
        f_mem = self.M_fwd
        r_mem_std = M_rev_ahead  # Use the passed lookahead reverse memory

        for layer in self.layers:
            if layer.layer_type == "memory_update":
                # Calculate write mask based on current progress
                wmask = self.memory_manager.get_write_mask(
                    chunk_idx, total_chunks_so_far, self.total_tokens_processed, batch_size=1
                )
                # Pass M_rev_ahead for reading, but only update M_fwd (is_reverse_update=False)
                x, updated_fwd, _, _gate_loss = layer(  # Expect only updated_fwd to be non-None
                    x,
                    forward_memory=f_mem,
                    reverse_memory=r_mem_std,  # Read from lookahead reverse
                    control_tokens=ctrl,
                    do_memory_update=True,  # Enable memory update
                    write_mask=wmask,
                    is_reverse_update=False  # IMPORTANT: We are updating FORWARD memory here
                )
                if updated_fwd is not None:
                    f_mem = updated_fwd  # Update local copy for next layer
            else:
                # Read-only or local-only layer
                x, _, _, _gate_loss = layer(
                    x,
                    forward_memory=f_mem if layer.layer_type != "local_only" else None,
                    reverse_memory=r_mem_std if layer.layer_type != "local_only" else None,
                    control_tokens=ctrl
                )

        # Update the main forward memory state *after* processing the chunk through all layers
        self.M_fwd = f_mem

        # Update total tokens processed *after* successfully processing the chunk
        self.total_tokens_processed += chunk_tensor.size(1)

    def _run_lookahead_reverse_pass(self, all_chunks: List[List[int]], B: int, dev: torch.device, total_logical_sequence_length: int) -> Dict[int, Tensor]:
        try:
            """ Runs the lookahead reverse pass. """
            n_chunks = len(all_chunks)
            window_start_idx = max(0, n_chunks - self.config.reverse_max_chunks)
            reverse_window_chunks = all_chunks[window_start_idx:]
            window_size = len(reverse_window_chunks)
            approx_processed_len_for_ctrl_tokens = self.total_tokens_processed
            current_mem_size_for_ctrl = self.memory_manager.get_effective_size(approx_processed_len_for_ctrl_tokens)

            if self.config.DEBUG_LEVEL > 0: print0(f"DEBUG: Entering _run_lookahead_reverse_pass. n_chunks={n_chunks}, window_size={window_size}", self.config.logfile)

            # Check if we're dealing with large chunks
            for i, chunk in enumerate(reverse_window_chunks):
                #if self.config.DEBUG_LEVEL > 0: print0(f"DEBUG: Chunk {i} size: {len(chunk)}", self.config.logfile)
                pass

            # Initialize M_rev_ahead for this pass locally
            current_M_rev_ahead: Dict[int, Tensor] = {}
            for group in self.layer_groups:
                if group.has_memory:
                    group_idx = group.group_idx
                    mem_idx = self.group_id_to_memory_idx[group_idx]
                    #if self.config.DEBUG_LEVEL > 0: print0(f"DEBUG: Initializing memory for group {group_idx}", self.config.logfile)

                    param = self.initial_rev_la_params[mem_idx]
                    #if self.config.DEBUG_LEVEL > 0: print0(f"DEBUG: Parameter shape: {param.shape}, device: {param.device}", self.config.logfile)

                    current_M_rev_ahead[group_idx] = param.clone().to(dev).repeat(B, 1, 1)
                    #if self.config.DEBUG_LEVEL > 0: print0(f"DEBUG: Initialized memory for group {group_idx}, shape: {current_M_rev_ahead[group_idx].shape}", self.config.logfile)
            for i, chunk_tokens in enumerate(reversed(reverse_window_chunks)):
                if not chunk_tokens:
                    if self.config.DEBUG_LEVEL > 0: print0(f"DEBUG: Skipping empty chunk at index {i}", self.config.logfile)
                    continue

                if self.config.DEBUG_LEVEL > 0: print0(f"DEBUG: Processing chunk {i} with {len(chunk_tokens)} tokens", self.config.logfile)

                chunk_tensor = torch.tensor(chunk_tokens, dtype=torch.long, device=dev).unsqueeze(0)
                #if self.config.DEBUG_LEVEL > 0: print0(f"DEBUG: Created chunk tensor with shape {chunk_tensor.shape}", self.config.logfile)

                # Check if embedding lookup could be the issue
                try:
                    x = self.token_embedding(chunk_tensor)
                    #if self.config.DEBUG_LEVEL > 0: print0(f"DEBUG: Embedded chunk shape: {x.shape}", self.config.logfile)
                except Exception as e:
                    print0(f"ERROR: Embedding lookup failed: {e}", self.config.logfile)
                    exit(3)
                global_chunk_idx = window_start_idx + (window_size - 1 - i)
                reverse_chunk_idx = i

                approx_processed_len = self.total_tokens_processed
                current_mem_size = self.memory_manager.get_effective_size(approx_processed_len)
                ctrl = self.control_token_generator.generate_control_tokens(
                    mode="lookahead_reverse", current_chunk_idx=global_chunk_idx, total_chunks=n_chunks,
                    current_mem_size=current_mem_size_for_ctrl, max_mem_size=self.config.max_memory_size,
                    seq_len=approx_processed_len_for_ctrl_tokens, # Use global history for ratios
                    reverse_chunk_idx=reverse_chunk_idx, reverse_window_size=window_size
                )

                # Pass state dictionaries to layers
                M_fwd = self.M_fwd  # Pass empty or existing state (not used by layers in this mode)
                M_rev_persist = self.M_rev_persist  # Pass existing state (not used by layers in this mode)

                for layer_idx, layer_block in enumerate(self.layers):
                    group_idx = self.layer_idx_to_group_idx[layer_idx]
                    group = self.layer_groups[group_idx]

                    decay = None
                    # Calculate decay only if it's an update layer for a group with memory
                    if group.has_memory and layer_idx == group.memory_update_layer_idx:
                        mem_shape = current_M_rev_ahead[group_idx].shape
                        decay = self.memory_manager.calculate_reverse_decay_weights(
                            reverse_chunk_index=reverse_chunk_idx, window_size=window_size,
                            is_persistent=False, memory_shape=mem_shape, device=dev
                        )

                    x, _, updated_rev, _gate_loss = layer_block(  # Expect only updated_rev
                        x,
                        M_fwd_dict=M_fwd,
                        M_rev_ahead_dict=current_M_rev_ahead,
                        M_rev_persist_dict=M_rev_persist,
                        group_id=group_idx,
                        mode="lookahead_reverse",
                        control_tokens=ctrl,
                        write_mask=None,
                        decay_weights=decay,
                        total_logical_sequence_length=total_logical_sequence_length  # MODIFIED: Pass through
                    )

                    # Update the local state dict for the next layer
                    if updated_rev is not None:
                        g_id, mem = updated_rev
                        current_M_rev_ahead[g_id] = mem
            return current_M_rev_ahead

        except Exception as e:
            print0(f"Error in _run_lookahead_reverse_pass: {type(e).__name__}: {e}", self.config.logfile)
            import traceback
            traceback.print_exc()
            raise

    def _run_persistent_reverse_pass(self, all_chunks: List[List[int]], B: int, dev: torch.device, total_logical_sequence_length: int):
        """ Runs the persistent reverse pass. """
        n_chunks = len(all_chunks)
        # Eligible chunks for persistent reverse are all except the last one
        eligible_chunks = all_chunks[:-1]
        if not eligible_chunks:
            print0("  Skipping Persistent Reverse Pass: No eligible preceding chunks.", self.config.logfile)
            return  # Nothing to process

        # Determine the window of chunks to process based on reverse_max_chunks
        window_start_idx = max(0, len(eligible_chunks) - self.config.reverse_max_chunks)
        reverse_window_chunks = eligible_chunks[window_start_idx:]
        window_size = len(reverse_window_chunks)
        approx_processed_len_for_ctrl_tokens = self.total_tokens_processed
        current_mem_size_for_ctrl = self.memory_manager.get_effective_size(approx_processed_len_for_ctrl_tokens)

        if not reverse_window_chunks:
            print0("  Skipping Persistent Reverse Pass: Window is empty after eligibility check.", self.config.logfile)
            return

        # --- REMOVED MASK-FUTURE DROPOUT LOGIC FROM HERE ---
        # The dropout is now applied when *reading* this memory in the forward pass during training.

        # Pass state dictionaries (M_rev_persist will be updated directly)
        M_fwd = self.M_fwd  # Not used by layers in this mode
        M_rev_ahead = self.M_rev_ahead  # Not used by layers in this mode

        # Initialize M_rev_persist for this pass if it doesn't exist (should exist after init/reset)
        # Or reset it based on config? Let's reset it here to ensure clean state for the pass.
        # This assumes the pass *recomputes* M_rev_persist from initial state + window.
        # If M_rev_persist should accumulate across cycles (when reset_memory_on_cycle=False),
        # this reset needs to be conditional. For now, assume reset per cycle pass.
        current_M_rev_persist: Dict[int, Tensor] = {}
        for group in self.layer_groups:
            if group.has_memory:
                group_idx = group.group_idx
                mem_idx = self.group_id_to_memory_idx[group_idx]
                # Start from the learned initial state for this pass
                current_M_rev_persist[group_idx] = self.initial_rev_p_params[mem_idx].clone().to(dev).repeat(B, 1, 1)

        for i, chunk_tokens in enumerate(reversed(reverse_window_chunks)):
            chunk_tensor = torch.tensor(chunk_tokens, dtype=torch.long, device=dev).unsqueeze(0)
            x = self.token_embedding(chunk_tensor)
            # global_chunk_idx refers to index within eligible_chunks
            global_chunk_idx_in_eligible = window_start_idx + (window_size - 1 - i)
            # global_chunk_idx in all_chunks context is the same here as eligible are all but last
            global_chunk_idx_in_all = global_chunk_idx_in_eligible
            reverse_chunk_idx = i  # Index within the reversed window (0 = newest in window)

            # Use total_tokens_processed which reflects state *before* this cycle started
            approx_processed_len = self.total_tokens_processed
            current_mem_size = self.memory_manager.get_effective_size(approx_processed_len)
            ctrl = self.control_token_generator.generate_control_tokens(
                mode="persistent_reverse",
                current_chunk_idx=global_chunk_idx_in_all,
                total_chunks=n_chunks,
                current_mem_size=current_mem_size_for_ctrl, max_mem_size=self.config.max_memory_size,
                seq_len=approx_processed_len_for_ctrl_tokens,  # Use global history for ratios
                reverse_chunk_idx=reverse_chunk_idx,
                reverse_window_size=window_size
            )

            for layer_idx, layer_block in enumerate(self.layers):
                group_idx = self.layer_idx_to_group_idx[layer_idx]
                group = self.layer_groups[group_idx]

                decay = None
                if group.has_memory and layer_idx == group.memory_update_layer_idx:
                    # Ensure the group_idx exists in the dict before accessing shape
                    if group_idx in current_M_rev_persist:
                        mem_shape = current_M_rev_persist[group_idx].shape
                        decay = self.memory_manager.calculate_reverse_decay_weights(
                            reverse_chunk_index=reverse_chunk_idx, window_size=window_size,
                            is_persistent=True, memory_shape=mem_shape, device=dev
                        )
                    else:
                        # Should not happen if initialized correctly
                        print0(f"Warning: Group {group_idx} not found in current_M_rev_persist during decay calc.", self.config.logfile)

                x, _, updated_rev, _gate_loss = layer_block(
                    x,
                    M_fwd_dict=M_fwd,
                    M_rev_ahead_dict=M_rev_ahead,
                    M_rev_persist_dict=current_M_rev_persist,
                    group_id=group_idx,
                    mode="persistent_reverse",
                    control_tokens=ctrl,
                    write_mask=None,
                    decay_weights=decay,
                    total_logical_sequence_length=total_logical_sequence_length
                )

                if updated_rev is not None:
                    g_id, mem = updated_rev
                    current_M_rev_persist[g_id] = mem  # Update the state for the next layer/chunk

        # --- After processing all chunks in the window, update the main state ---
        self.M_rev_persist = current_M_rev_persist

    def _mask_persistent_memory(self, memory_dict: Dict[int, Tensor], p_drop: float) -> Dict[int, Tensor]:
        if p_drop <= 0.0:
            return memory_dict

        masked_memory_dict = {}
        for group_id, mem_tensor_original in memory_dict.items():  # Use a different name
            if mem_tensor_original is None or mem_tensor_original.numel() == 0:
                masked_memory_dict[group_id] = mem_tensor_original
                continue

            B, M, D = mem_tensor_original.shape
            if M == 0:
                masked_memory_dict[group_id] = mem_tensor_original.clone()  # Still clone if not modifying
                continue

            keep_prob = 1.0 - p_drop
            mask = torch.bernoulli(torch.full((B, M, 1), keep_prob, device=mem_tensor_original.device)).to(
                mem_tensor_original.dtype)

            # Apply mask. mem_tensor_original should be part of the graph.
            # We are creating a new tensor that depends on mem_tensor_original.
            masked_tensor = mem_tensor_original * mask
            masked_memory_dict[group_id] = masked_tensor
        return masked_memory_dict

    @torch.no_grad()
    def generate(self, prompt: Union[str, List[int], List[List[int]], torch.Tensor, None] = None, *,
                 max_new_tokens: int = 128, temperature: float = 1.0, top_k: Optional[int] = None,
                 stop_token_id: Optional[int] = None, reset_state: bool = False) -> List[int]:
        """ Autoregressive decoding. """
        if reset_state: print0("Resetting model state for generation.", self.config.logfile); self.reset_state()
        self._initialize_memory_states(force_reset=False)
        dev = self.token_embedding.weight.device  # Get device early

        def _sample_row(row: torch.Tensor) -> int:
            row = row.float()
            if top_k is not None and top_k > 0:
                # Ensure k is not larger than vocab size
                actual_k = min(top_k, row.shape[-1])
                if actual_k > 0:
                    # Get the threshold value (the k-th largest logit)
                    thresh = torch.topk(row, actual_k).values[-1]
                    # Mask out logits below the threshold
                    row = torch.where(row < thresh, torch.full_like(row, float("-inf")), row)
            # Apply temperature scaling
            temp = max(float(temperature), 1e-5)  # Avoid division by zero
            probs = torch.nn.functional.softmax(row / temp, dim=-1)
            # Sample from the distribution
            return torch.multinomial(probs, 1).item()

        # --- State Initialization ---
        generated: List[int] = []
        # Process initial prompt. CMAModel.forward will set self.current_total_logical_sequence_length internally.
        if prompt not in (None, "", [], torch.tensor([], dtype=torch.long)):
            if self.config.DEBUG_LEVEL > 0: print0(f"DEBUG: Processing initial prompt ({type(prompt)})",
                                                   self.config.logfile)
            _ = self.forward(prompt, training_mode=False)  # CMAModel.forward sets its internal logical length
            if self.config.DEBUG_LEVEL > 0: print0(
                f"DEBUG: After prompt processing, current_chunk_tokens len: {len(self.current_chunk_tokens)}",
                self.config.logfile)
        # else: self.current_total_logical_sequence_length remains 0 or its value from a previous call if not reset_state

        try:
            self.current_chunk_text = self.tokenizer.decode(self.current_chunk_tokens)
        except Exception as e:
            print0(f"Warning: Error decoding initial buffer tokens: {e}", self.config.logfile)
            self.current_chunk_text = ""

        current_token_input: Optional[List[int]] = None

        for i in range(max_new_tokens):
            # CMAModel.forward will correctly determine and use self.current_total_logical_sequence_length
            # based on current_token_input and existing state (closed_chunks, current_chunk_tokens)
            if self.config.DEBUG_LEVEL > 0: print0(f"DEBUG: Calling forward with input: {current_token_input}",
                                                   self.config.logfile)
            logits, _gate_loss = self.forward(current_token_input, training_mode=False)
            # Note: forward() internally updates self.current_chunk_tokens if current_token_input is not None
            if self.config.DEBUG_LEVEL > 0: print0(f"DEBUG: After forward, current_chunk_tokens len: {len(self.current_chunk_tokens)}", self.config.logfile)

            # --- TEMPORARILY COPYING THE SAMPLING LOGIC HERE TO ENSURE `next_id` is available ---
            if logits.shape[1] == 0:
                if self.config.DEBUG_LEVEL > 1: print0("INFO: Generation stopped - empty logits.", self.config.logfile)
                current_token_input = None  # To skip final forward
                break

            # Helper function for sampling (assuming it's defined or copied here if not accessible)
            def _sample_row_local(row: torch.Tensor) -> int:
                row = row.float()
                if top_k is not None and top_k > 0:
                    actual_k = min(top_k, row.shape[-1])
                    if actual_k > 0:
                        thresh = torch.topk(row, actual_k).values[-1]
                        row = torch.where(row < thresh, torch.full_like(row, float("-inf")), row)
                temp = max(float(temperature), 1e-5)
                probs = torch.nn.functional.softmax(row / temp, dim=-1)
                return torch.multinomial(probs, 1).item()

            next_id = _sample_row_local(logits[0, -1, :])
            # --- END OF TEMPORARY SAMPLING LOGIC COPY --
            if self.config.DEBUG_LEVEL > 0: print0(f"DEBUG: Sampled token ID: {next_id}", self.config.logfile)

            generated.append(next_id)

            # Check for stop token
            if stop_token_id is not None and next_id == stop_token_id:
                if self.config.DEBUG_LEVEL > 1: print0(f"INFO: Stop token {stop_token_id} generated. Stopping.",
                                                       self.config.logfile)
                current_token_input = None  # Skip final forward
                break

            try:
                self.current_chunk_text += self.tokenizer.decode([next_id])
            except Exception as e:
                print0(f"Warning: Error decoding token {next_id}: {e}", self.config.logfile)

            # --- Periodic Persistent Reverse Update Check ---
            self.tokens_since_persistent_update += 1
            update_persist = False
            if self.config.persistent_reverse_update_freq_tokens is not None and \
                    self.tokens_since_persistent_update >= self.config.persistent_reverse_update_freq_tokens:
                update_persist = True
            if not update_persist and self.config.persistent_reverse_update_freq_semantic and len(
                    self.current_chunk_text) > 1:
                # ... (semantic boundary check logic) ...
                try:
                    boundary_level = self.config.persistent_reverse_update_freq_semantic
                    if boundary_level in self.config.boundary_types:
                        search_window = self.current_chunk_text[-10:]
                        for boundary_type_key in self.config.boundary_types[
                            boundary_level]:  # renamed boundary_type to boundary_type_key
                            pattern = BOUNDARY_PATTERNS.get(boundary_type_key)
                            if pattern and re.search(pattern, search_window):
                                update_persist = True
                                break
                except Exception as e:
                    print0(f"Warning: Error during semantic boundary check: {e}", self.config.logfile)

            if update_persist:
                if self.config.DEBUG_LEVEL > 0: print0(
                    "Triggering periodic persistent reverse update during generation.", self.config.logfile)
                # self.current_chunk_tokens already includes the 'next_id' due to the self.forward call
                full_chunks_for_persist = self.closed_chunks + (
                    [self.current_chunk_tokens] if self.current_chunk_tokens else [])
                if full_chunks_for_persist:
                    # Determine logical length for this specific persistent reverse pass
                    _logical_len_for_persist_pass = sum(len(c) for c in full_chunks_for_persist)
                    if self.config.DEBUG_LEVEL > 0: print0(
                        f"DEBUG Generate: Logical seq_len for persist pass: {_logical_len_for_persist_pass}",
                        self.config.logfile)
                    self._run_persistent_reverse_pass(full_chunks_for_persist, B=1, dev=dev,
                                                      total_logical_sequence_length=_logical_len_for_persist_pass)
                else:
                    if self.config.DEBUG_LEVEL > 0: print0(
                        "Skipping periodic update: No history or current chunks available.", self.config.logfile)
                self.tokens_since_persistent_update = 0

            current_token_input = [next_id]  # Prepare for the next iteration's forward call

        # After the loop, if current_token_input is not None, it means the loop ended due to max_new_tokens
        # and the last generated token (in current_token_input) hasn't been fully processed to update state.
        if current_token_input is not None:  # True if loop finished by max_new_tokens, not by stop_token
            if self.config.DEBUG_LEVEL > 0: print0(
                f"DEBUG: Processing final generated token {current_token_input} to update state.",
                self.config.logfile)
            # This call updates self.current_chunk_tokens with the last token and ensures logical length is set
            _ = self.forward(current_token_input, training_mode=False)
            if self.config.DEBUG_LEVEL > 0: print0(
                f"DEBUG: After final forward, current_chunk_tokens len: {len(self.current_chunk_tokens)}",
                self.config.logfile)
        else:
            if self.config.DEBUG_LEVEL > 0: print0(
                "DEBUG: No final token processing needed (loop broke early or stop token hit).",
                self.config.logfile)

        return generated

    def reset_state(self) -> None:
        """Clear all sequence-level state and reset memory dictionaries."""
        if self.config.DEBUG_LEVEL > 0: print0("DEBUG: Resetting CMAModel state.", self.config.logfile)
        self.closed_chunks = []
        self.current_chunk_tokens = []
        self.current_chunk_text = ""
        self.total_tokens_processed = 0
        self.tokens_since_persistent_update = 0
        # Clear runtime memory state dictionaries
        self.M_fwd = {}
        self.M_rev_ahead = {}
        self.M_rev_persist = {}
        # Re-initialize from learned parameters on next use by calling _initialize_memory_states

    def set_training_step(self, step: int, total_steps: int):
        """Set current training step for mask-future scheduling"""
        self.training_step = step
        self.total_training_steps = total_steps
        if self.config.DEBUG_LEVEL > 0: print0(f"DEBUG: Training step set to {step}, total steps {total_steps}", self.config.logfile)
