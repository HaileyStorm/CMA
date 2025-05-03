import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple, Optional, Union, Dict, Any
import tiktoken
from dataclasses import dataclass
import yaml
from pathlib import Path
import re


# -----------------------------------------------------------------------------
# Configuration

@dataclass
class CMAConfig:
    """Configuration for CMA model"""
    # Chunking
    chunk_size: int = 1024
    semantic_chunking_gap_percentage: float = 25.0
    boundary_search_chars: List[int] = (256, 64, 32)
    boundary_types: Dict[str, List[str]] = None
    buffer_ratio: float = 0.1

    # Memory
    max_memory_size: int = 4096
    reverse_memory_size: int = 1024
    initial_write_fraction: float = 0.6
    memory_growth_function: str = "linear"
    memory_cap_length: int = 65536

    # Reverse pass
    reverse_max_chunks: int = 4
    standard_reverse_decay_step: float = 0.2
    standard_reverse_decay_rate: float = 0.5
    persistent_reverse_decay_step: float = 0.05
    persistent_reverse_decay_rate: float = 0.1
    persistent_reverse_update_freq_tokens: int = 128
    persistent_reverse_update_freq_semantic: Optional[str] = "secondary"

    # Model architecture
    embed_dim: int = 768
    n_heads: int = 12
    n_layers: int = 12
    head_dim: int = 64
    layer_structure: Optional[List[Dict]] = None
    skip_attention_layers: List[int] = (6,)

    # Control tokens
    integration_method: str = "query_fusion"
    ctrl_init_scale: float = 0.0001

    # Initialization
    memory_init_scale: float = 0.02
    gate_bias_init: float = -1.0
    output_proj_zero_init: bool = True

    # Adaptive gating regularization
    gate_regularization_type: Optional[str] = None  # None, "l1", or "entropy"
    gate_regularization_strength: float = 0.001

    # Future‐masking schedule: progress breakpoints and rates
    mask_future_schedule: Tuple[float, float] = (0.3, 0.7)
    mask_future_rates: Tuple[float, float, float] = (0.3, 0.5, 0.8)

    def __post_init__(self):
        if self.boundary_types is None:
            self.boundary_types = {
                "primary": ["section_break", "code_block", "paragraph_break", "double_line_break"],
                "secondary": ["sentence_end", "line_break"],
                "tertiary": ["clause_end", "code_line_end"]
            }
        if self.layer_structure is None:
            self.layer_structure = [
                {"group": {
                    "layers": ["memory_read", "local_only", "local_only", "memory_update"],
                    "repeat": 3
                }}
            ]

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'CMAConfig':
        return cls(**config_dict)

    @classmethod
    def from_yaml(cls, yaml_file: Union[str, Path]) -> 'CMAConfig':
        with open(yaml_file, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    def validate(self):
        assert 0 < self.semantic_chunking_gap_percentage < 100
        assert len(self.boundary_search_chars) == 3
        assert self.max_memory_size > 0
        assert self.reverse_memory_size > 0
        assert 0 <= self.initial_write_fraction <= 1.0
        assert self.reverse_max_chunks > 0
        assert self.embed_dim % self.n_heads == 0


# -----------------------------------------------------------------------------
# Boundary patterns

BOUNDARY_PATTERNS = {
    # Primary boundaries (most important)
    "section_break": r"\n\n+[#*=-]+\n+",
    "code_block": r"```[\s\S]*?```",
    "paragraph_break": r"\n\s*\n",
    "double_line_break": r"\n\n+",

    # Secondary boundaries
    "sentence_end": r"[.!?]+(?=\s|$)",
    "line_break": r"\n",

    # Tertiary boundaries
    "clause_end": r"[,;:](?=\s|$)",
    "code_line_end": r"[;{}]\s*(?:\n|$)",
}


# -----------------------------------------------------------------------------
# Utility functions

def norm(x: Tensor, eps: float = 1e-5) -> Tensor:
    """RMS normalization"""
    return F.rms_norm(x, (x.size(-1),))


def get_mask_future_schedule(
        config: CMAConfig,
        step: int,
        total_steps: int
) -> float:
    """Get progressive mask‐future dropout schedule for training."""
    progress = step / max(total_steps, 1)
    early_end, mid_end = config.mask_future_schedule
    r_early, r_mid, r_late = config.mask_future_rates

    if progress < early_end:
        return r_early
    elif progress < mid_end:
        # linear interp from early to mid
        return r_early + (progress - early_end) * (r_mid - r_early) / (mid_end - early_end)
    else:
        # linear interp from mid to late (1.0)
        return r_mid + (progress - mid_end) * (r_late - r_mid) / (1.0 - mid_end)


# -----------------------------------------------------------------------------
# Control token generation

class ControlTokenGenerator:
    """Generates control tokens for CMA operation modes"""

    def __init__(self, config: CMAConfig):
        self.config = config

    def generate_control_tokens(
            self,
            mode: str,
            current_chunk_idx: int,
            total_chunks: int,
            current_mem_size: int,
            max_mem_size: int,
            seq_len: int,
            reverse_chunk_idx: Optional[int] = None,
            reverse_window_size: Optional[int] = None
    ) -> Dict[str, float]:
        """Generate control token values for current operation mode"""

        # Generation flag
        generation_flag = 0.0 if mode != "generate" else 1.0

        # Memory mode flag
        if mode == "forward":
            memory_mode_flag = 0.0
        elif mode == "standard_reverse":
            memory_mode_flag = 1.0
        elif mode == "persistent_reverse":
            memory_mode_flag = 0.8
        elif mode == "generate":
            memory_mode_flag = 0.0
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Memory usage ratio
        memory_usage_ratio = current_mem_size / max_mem_size if max_mem_size > 0 else 0.0

        # Memory density ratio
        memory_density_ratio = current_mem_size / seq_len if seq_len > 0 else 0.0

        # Chunk position ratio
        if mode == "forward":
            chunk_position_ratio = current_chunk_idx / total_chunks if total_chunks > 0 else 0.0
        elif mode in ["standard_reverse", "persistent_reverse"]:
            if reverse_chunk_idx is not None and reverse_window_size is not None:
                chunk_position_ratio = (reverse_window_size - reverse_chunk_idx) / reverse_window_size
            else:
                chunk_position_ratio = 0.0
        else:  # generate
            chunk_position_ratio = current_chunk_idx / total_chunks if total_chunks > 0 else 0.0

        return {
            "generation_flag": generation_flag,
            "memory_mode_flag": memory_mode_flag,
            "memory_usage_ratio": memory_usage_ratio,
            "memory_density_ratio": memory_density_ratio,
            "chunk_position_ratio": chunk_position_ratio
        }


# -----------------------------------------------------------------------------
# Chunking module

class ChunkProcessor:
    """Handles semantic and fixed-size chunking"""

    def __init__(self, config: CMAConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

    def semantic_chunk_reverse_with_gap(self, text: str) -> List[List[int]]:
        """Semantic chunking using reverse-with-gap algorithm with tokenization checking"""
        if not text:
            return []

        chunks = []
        end_pos = len(text)

        while end_pos > 0:
            # For last chunk, use target size with gap
            if end_pos == len(text):
                gap_percentage = self.config.semantic_chunking_gap_percentage / 100.0
                target_size = int(self.config.chunk_size * (1 - gap_percentage))
                target_size = max(1, target_size)
            else:
                target_size = self.config.chunk_size

            # Initial estimate of start position
            est_start = max(0, end_pos - target_size)

            # Loop to find suitable boundary
            found_valid_chunk = False
            iteration = 0
            max_iterations = 10  # Prevent infinite loops

            while not found_valid_chunk and iteration < max_iterations:
                # Add buffer for search
                buffer_chars = int(self.config.boundary_search_chars[0] * self.config.buffer_ratio)
                search_start = max(0, est_start + buffer_chars)

                # Search for boundary within character windows
                start_pos = self._find_semantic_boundary_backward(text, search_start, end_pos)

                # Tokenize and check chunk size
                chunk_text = text[start_pos:end_pos]
                chunk_tokens = self.tokenizer.encode(chunk_text)

                if len(chunk_tokens) <= self.config.chunk_size:
                    found_valid_chunk = True
                    chunks.insert(0, chunk_tokens)
                else:
                    # Estimate new start position to reduce chunk size
                    excess_tokens = len(chunk_tokens) - self.config.chunk_size
                    chars_per_token = len(chunk_text) / len(chunk_tokens)
                    est_start = start_pos + int(excess_tokens * chars_per_token * 1.1)  # 1.1 buffer
                    iteration += 1

            # If we couldn't find a valid boundary, use the estimated start
            if not found_valid_chunk:
                chunk_text = text[est_start:end_pos]
                chunk_tokens = self.tokenizer.encode(chunk_text)
                # If still too large, truncate
                if len(chunk_tokens) > self.config.chunk_size:
                    chunk_tokens = chunk_tokens[:self.config.chunk_size]
                chunks.insert(0, chunk_tokens)
                # Adjust start_pos for next iteration
                start_pos = est_start

            end_pos = start_pos

        return chunks

    def fixed_size_chunk_reverse_with_gap(self, tokens: List[int]) -> List[List[int]]:
        total_len = len(tokens)
        if total_len == 0:
            return []

        gap_pct = self.config.semantic_chunking_gap_percentage / 100.0
        target_last = max(1, math.floor(self.config.chunk_size * (1 - gap_pct)))
        start = max(0, total_len - target_last)

        chunks, boundary = [tokens[start:]], start

        while boundary > 0:
            prev_start = max(0, boundary - self.config.chunk_size)
            chunk = tokens[prev_start:boundary]

            # --- retry loop exactly like semantic version -------------------------
            while len(chunk) > self.config.chunk_size:
                # overshoot ⇒ shorten window and retry
                prev_start += int(0.1 * self.config.chunk_size)
                chunk = tokens[prev_start:boundary]
            # ---------------------------------------------------------------------

            chunks.insert(0, chunk)
            boundary = prev_start

        return chunks

    def _find_semantic_boundary_backward(self, text: str, search_start_point: int, end_pos: int) -> int:
        """Find semantic boundary by searching backward from search_start_point"""
        if search_start_point <= 0:
            return 0

        search_distances = self.config.boundary_search_chars
        boundary_types = self.config.boundary_types

        # Try boundary types in order of priority
        for level, distance in zip(['primary', 'secondary', 'tertiary'], search_distances):
            search_start = max(0, search_start_point - distance)
            search_text = text[search_start:search_start_point]

            for boundary_type in boundary_types[level]:
                pattern = BOUNDARY_PATTERNS.get(boundary_type)
                if pattern:
                    matches = list(re.finditer(pattern, search_text))
                    if matches:
                        # Return the position of the last match
                        last_match = matches[-1]
                        boundary_pos = search_start + last_match.end()
                        if boundary_pos <= search_start_point:
                            return boundary_pos

        # If no boundary found, return the estimated start position
        return search_start_point


# -----------------------------------------------------------------------------
# Memory management

class MemoryManager:
    """Manages forward and reverse memory states"""

    def __init__(self, config: CMAConfig):
        self.config = config

    def get_effective_size(self, seq_len: int) -> int:
        """Get effective forward memory size based on sequence length"""
        if self.config.memory_growth_function == "linear":
            fraction = min(1.0, seq_len / self.config.memory_cap_length)
        elif self.config.memory_growth_function == "log":
            if seq_len <= 1:
                fraction = 0.0
            else:
                fraction = math.log(seq_len) / math.log(self.config.memory_cap_length)
                fraction = min(1.0, max(0.0, fraction))
        else:
            raise ValueError(f"Unknown growth function: {self.config.memory_growth_function}")

        return int(self.config.max_memory_size * fraction)

    def get_write_mask(
            self,
            current_chunk_idx: int,
            total_chunks: int,
            seq_len: int,
            batch_size: int = 1
    ) -> Tensor:
        """Get write mask for memory update"""
        # Sequence-based write cap
        seq_write_cap = self.get_effective_size(seq_len)

        # Progressive write access within the sequence-determined cap
        if total_chunks > 0:
            chunk_progress = (current_chunk_idx + 1) / total_chunks
            initial_fraction = self.config.initial_write_fraction
            write_fraction = initial_fraction + (1.0 - initial_fraction) * chunk_progress
        else:
            write_fraction = self.config.initial_write_fraction

        writable_size = int(seq_write_cap * write_fraction)

        # Create mask
        mask = torch.zeros(batch_size, self.config.max_memory_size, dtype=torch.bool)
        mask[:, :writable_size] = True

        return mask

    def apply_downweighting(
            self,
            memory: Tensor,
            chunk_indices: List[int],
            is_reverse: bool,
            is_persistent: bool
    ) -> Tensor:
        """Apply downweighting to reverse memory during update"""
        if not is_reverse:
            return memory

        if is_persistent:
            decay_step = self.config.persistent_reverse_decay_step
            decay_rate = self.config.persistent_reverse_decay_rate
        else:
            decay_step = self.config.standard_reverse_decay_step
            decay_rate = self.config.standard_reverse_decay_rate

        # Apply exponential decay based on chunk position
        weights = torch.ones_like(memory)

        for i, chunk_idx in enumerate(chunk_indices):
            decay_factor = decay_rate ** (i * decay_step)
            start_idx = i * (self.config.reverse_memory_size // len(chunk_indices))
            end_idx = (i + 1) * (self.config.reverse_memory_size // len(chunk_indices))
            weights[:, start_idx:end_idx] *= decay_factor

        return memory * weights


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

            # Reverse memory update parameters (shared between standard and persistent)
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

        # -------- K / V projection (plus memory tokens) --------------------------
        k = self.k_proj(x)
        v = self.v_proj(x)
        if (forward_memory is not None) or (reverse_memory is not None):
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
        S_keys = v.size(-2)  # T + mem_len
        mem_len = S_keys - chunk_len
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
        g = torch.sigmoid(gate_logits).permute(0, 2, 1).unsqueeze(-1)  # (B,h,T,1)

        Y = Y_chunk + g * Y_mem  # final fused output

        # optional regularisation
        reg_loss = None
        if self.config.gate_regularization_type == "l1":
            reg_loss = self.config.gate_regularization_strength * torch.mean(torch.abs(g))
        elif self.config.gate_regularization_type == "entropy":
            ent = -(g * torch.log(g + 1e-8) + (1 - g) * torch.log(1 - g + 1e-8))
            reg_loss = self.config.gate_regularization_strength * torch.mean(ent)

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
    """Transformer block with either CMA or standard attention"""

    def __init__(self, config: CMAConfig, layer_idx: int, layer_type: str = "local_only"):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = layer_type

        # Create attention layer
        if layer_idx in config.skip_attention_layers:
            self.attn = None
        elif layer_type in ["memory_read", "memory_update"]:
            is_update = (layer_type == "memory_update")
            self.attn = CascadeMemoryAttention(config, layer_idx, is_memory_update=is_update)
        else:  # local_only
            self.attn = CausalSelfAttention(config, layer_idx)

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, 4 * config.embed_dim, bias=False),
            nn.GELU(),
            nn.Linear(4 * config.embed_dim, config.embed_dim, bias=False)
        )

        # Skip connection weights
        self.lambdas = nn.Parameter(torch.tensor([1.0, 0.0]))

    def forward(
            self,
            x: Tensor,
            forward_memory: Optional[Tensor] = None,
            reverse_memory: Optional[Tensor] = None,
            x0: Optional[Tensor] = None,
            control_tokens: Optional[Dict[str, float]] = None,
            do_memory_update: bool = False,
            write_mask: Optional[Tensor] = None,
            decay_weights: Optional[Tensor] = None,
            is_reverse_update: bool = False
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]]:

        # Apply skip connection from input
        if x0 is not None:
            x = self.lambdas[0] * x + self.lambdas[1] * x0

        updated_forward_memory = None
        updated_reverse_memory = None
        gate_reg_loss = None

        # Attention
        if self.attn is not None:
            residual = x
            x = norm(x)  # Always use RMSNorm

            if isinstance(self.attn, CascadeMemoryAttention):
                attn_out, updated_forward_memory, updated_reverse_memory, gate_reg_loss = self.attn(
                    x,
                    forward_memory=forward_memory,
                    reverse_memory=reverse_memory,
                    control_tokens=control_tokens,
                    do_memory_update=do_memory_update,
                    write_mask=write_mask,
                    decay_weights=decay_weights,
                    is_reverse_update=is_reverse_update
                )
            else:
                attn_out = self.attn(x)

            x = residual + attn_out

        # MLP
        residual = x
        x = norm(x)  # Always use RMSNorm
        x = residual + self.mlp(x)

        return x, updated_forward_memory, updated_reverse_memory, gate_reg_loss


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
        # No positional embeddings - rely on intra-chunk causal masking + CMA memories

        # Layer structure from configuration
        self.layers = nn.ModuleList()
        layer_idx = 0

        # Parse layer structure
        for group_spec in config.layer_structure:
            if "group" in group_spec:
                group_config = group_spec["group"]
                layer_types = group_config["layers"]
                repeat = group_config.get("repeat", 1)

                for _ in range(repeat):
                    for layer_type in layer_types:
                        self.layers.append(Block(config, layer_idx, layer_type))
                        layer_idx += 1
            else:
                # Single layer specified
                layer_type = group_spec.get("type", "local_only")
                self.layers.append(Block(config, layer_idx, layer_type))
                layer_idx += 1

        self.lm_head = nn.Linear(config.embed_dim, vocab_size, bias=False)

        # Initialize memory states -- learnable parameters
        self.initial_forward_memory = nn.Parameter(
            torch.randn(1, config.max_memory_size, config.embed_dim) * config.memory_init_scale
        )
        self.initial_backward_memory = nn.Parameter(
            torch.randn(1, config.reverse_memory_size, config.embed_dim) * config.memory_init_scale
        )

        # State tracking
        self.M_fwd = None
        self.M_rev_persist = None
        self.current_chunks = []
        self.closed_chunks = []
        self.total_tokens_processed = 0
        self.tokens_since_persistent_update = 0
        self.current_chunk_tokens = []
        self.current_chunk_text: str = ""
        self.training_step = 0
        self.total_training_steps = 10000  # Default, should be set by trainer

        # Initialize weights
        self.apply(self._init_weights)

        # --- Parameter Count ---
        # Calculate total trainable parameters (includes initial memories)
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Calculate parameters specifically for initial memories
        initial_memory_params = self.initial_forward_memory.numel() + self.initial_backward_memory.numel()

        # Calculate other parameters by subtraction
        other_params = total_params - initial_memory_params

        print(f"--- CMA Model Parameter Count ---")
        print(f"Initial memory parameters: {initial_memory_params:,}")
        print(f"Other trainable parameters: {other_params:,}")
        print(f"Total trainable parameters: {total_params:,}")
        print(f"---------------------------------")
        # --- End Parameter Count ---

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
            self,
            input_ids: Union[str, List[int], List[List[int]], torch.Tensor, None],
            *, training_mode: bool = False
    ) -> torch.Tensor:
        """
        CMA forward pass (clean spec-aligned implementation).

        ─ Flush triggers ────────────────────────────────────────────────────────
        • user text / flat-token input arrives while an open chunk already exists
        • new input would make the open chunk exceed `chunk_size`
        • caller supplies pre-chunked input         (always flush)
        After deciding to flush we:
            1. build FULL_SEQUENCE = closed_chunks + open_chunk + new_input
            2. re-chunk FULL_SEQUENCE (semantic, fixed, or caller-exact)
            3. call _trigger_memory_update_cycle() once per CLOSED chunk
            4. leave the tail as the new open chunk
        Returns logits for the open chunk; memories live on self.
        """

        # ───────── state guards ────────────────────────────────────────────────
        if not hasattr(self, "closed_chunks"):
            self.closed_chunks: List[List[int]] = []  # history
        if not hasattr(self, "current_chunk_tokens"):
            self.current_chunk_tokens: List[int] = []  # open chunk

        dev = self.token_embedding.weight.device
        if self.M_fwd is None:
            self.M_fwd = self.initial_forward_memory.clone().to(dev)
        if self.M_rev_persist is None:
            self.M_rev_persist = self.initial_backward_memory.clone().to(dev)

        # ───────── early-exit on “no new tokens” ───────────────────────────────
        if input_ids in (None, "", [], torch.tensor([], dtype=torch.long)):
            return (
                self._process_current_chunk(generation_mode=not training_mode)
                if self.current_chunk_tokens
                else torch.zeros(1, 0, self.vocab_size, device=dev)
            )

        # ───────── normalise new input ─────────────────────────────────────────
        if isinstance(input_ids, str):
            new_raw_text = input_ids
            new_tokens = self.tokenizer.encode(new_raw_text)  # encode once
            mode = "semantic"
        elif isinstance(input_ids, list) and input_ids and isinstance(input_ids[0], list):
            new_chunks = [list(c) for c in input_ids]  # honour caller
            new_tokens = [t for c in new_chunks for t in c]
            mode = "caller_exact"
        else:  # flat tokens
            new_tokens = input_ids.tolist() if isinstance(input_ids, torch.Tensor) else list(input_ids)
            mode = "fixed"

        # ───────── decide whether to FLUSH before appending ────────────────────
        flush_now = (
                mode == "caller_exact" or
                (self.current_chunk_tokens and mode in {"semantic", "fixed"}) or
                (len(self.current_chunk_tokens) + len(new_tokens) > self.config.chunk_size)
        )

        if flush_now:
            # 1. full visible sequence
            full_tokens = [t for c in self.closed_chunks for t in c] + \
                          self.current_chunk_tokens + new_tokens

            # 2. global re-chunk
            if mode == "semantic":
                full_text = self.tokenizer.decode(full_tokens)
                chunk_texts = self.chunk_processor.semantic_chunk_reverse_with_gap(full_text)
                chunks = [self.tokenizer.encode(txt) for txt in chunk_texts]
            elif mode == "caller_exact":
                chunks = new_chunks  # history immutable
                # prepend history & open chunk as-is
                chunks = self.closed_chunks + (
                    [self.current_chunk_tokens] if self.current_chunk_tokens else []) + chunks
            else:  # fixed
                chunks = self.chunk_processor.fixed_size_chunk_reverse_with_gap(full_tokens)

            # 3. iterate over CLOSED chunks
            self.closed_chunks = []  # rebuild from scratch
            self.current_chunk_tokens = []
            for chunk in chunks[:-1]:
                self.current_chunk_tokens = chunk
                self._trigger_memory_update_cycle()  # updates memories, clears buffer
                self.closed_chunks.append(chunk)

            # 4. keep tail open
            self.current_chunk_tokens = chunks[-1] if chunks else []

        # ───────── append remaining tokens (only if no flush) ──────────────────
        if not flush_now:
            for t in new_tokens:
                self.current_chunk_tokens.append(t)
                self.tokens_since_persistent_update += 1

        # ───────── logits for open chunk ───────────────────────────────────────
        logits = (
            self._process_current_chunk(generation_mode=not training_mode)
            if self.current_chunk_tokens
            else torch.zeros(1, 0, self.vocab_size, device=dev)
        )
        return logits

    def _process_current_chunk(self, generation_mode: bool = False) -> torch.Tensor:
        """
        Process the **current partial (or full) chunk** that lives in
        `self.current_chunk_tokens`.

        Args
        ----
        generation_mode : bool
            • False  → we are inside a forward-update pass (enables memory writes)
            • True   → we are streaming / sampling (read-only memories)

        Returns
        -------
        logits : (1, T_chunk, vocab_size) tensor
        """
        if not self.current_chunk_tokens:  # empty edge-case
            return torch.zeros(1, 0, self.vocab_size, device=self.M_fwd.device)

        # -------- basic bookkeeping ------------------------------------------------
        chunk_idx = len(self.closed_chunks)   # 0-based index of *current* chunk
        total_chunks = chunk_idx + 1  # preceding chunks + this one
        chunk_tokens = torch.tensor(self.current_chunk_tokens,
                                    device=self.M_fwd.device, dtype=torch.long)
        T_chunk = chunk_tokens.size(0)

        # -------- embeddings & control tokens --------------------------------------
        x = self.token_embedding(chunk_tokens).unsqueeze(0)  # (1 , T , D)

        ctrl = self.control_token_generator.generate_control_tokens(
            mode="generate" if generation_mode else "forward",
            current_chunk_idx=chunk_idx,
            total_chunks=total_chunks,
            current_mem_size=self.memory_manager.get_effective_size(self.total_tokens_processed),
            max_mem_size=self.config.max_memory_size,
            seq_len=self.total_tokens_processed
        )

        # -------- run through the layer stack --------------------------------------
        f_mem = self.M_fwd
        r_mem = self.M_rev_persist

        for layer in self.layers:
            if layer.layer_type == "memory_update" and not generation_mode:
                wmask = self.memory_manager.get_write_mask(chunk_idx, total_chunks,
                                                           self.total_tokens_processed)
                x, up_f, up_r, _ = layer(
                    x,
                    forward_memory=f_mem,
                    reverse_memory=r_mem,
                    control_tokens=ctrl,
                    do_memory_update=True,
                    write_mask=wmask,
                    is_reverse_update=False
                )
                if up_f is not None:
                    self.M_fwd = f_mem = up_f
                if up_r is not None:  # unlikely in current call
                    self.M_rev_persist = r_mem = up_r
            else:  # normal / read-only layer
                x, _, _, _ = layer(
                    x,
                    forward_memory=f_mem if layer.layer_type != "local_only" else None,
                    reverse_memory=r_mem if layer.layer_type != "local_only" else None,
                    control_tokens=ctrl
                )

        # -------- projection, book-keeping, return ---------------------------------
        x = norm(x)
        logits = self.lm_head(x)  # (1 , T , V)

        return logits

    def _trigger_memory_update_cycle(self):
        """
        Processes the single, complete chunk currently in self.current_chunk_tokens,
        updates memory states (M_fwd, M_rev_persist) based on this chunk and
        the history in self.closed_chunks, adds the processed chunk to
        self.closed_chunks, and clears self.current_chunk_tokens.

        Assumes global re-chunking has already happened before this is called.
        """
        if not self.current_chunk_tokens:
            # Should not happen if called correctly, but good to check.
            print("WARNING: _trigger_memory_update_cycle called with empty buffer.")
            return

        # The chunk to process now
        chunk_to_process_list = list(self.current_chunk_tokens)
        chunk_to_process_tensor = torch.tensor(chunk_to_process_list, device=self.M_fwd.device, dtype=torch.long)
        chunk_idx = len(self.closed_chunks) # Index of the chunk *being* processed

        # --- Step 1: Compute Standard Reverse Memory ---
        # Includes history (closed_chunks) + the chunk being processed now
        chunks_for_std_rev = self.closed_chunks + [chunk_to_process_list]
        M_rev_std = self._compute_standard_reverse_memory(chunks_for_std_rev)

        # --- Step 2: Forward Pass over the *single* current chunk ---
        # This updates self.M_fwd based on chunk_to_process and M_rev_std
        # We need a function like _process_chunk_in_cycle, let's assume it exists
        # and correctly updates self.M_fwd when do_memory_update=True in CMA layers.
        # We also need to update self.total_tokens_processed here.
        self._process_chunk_in_cycle(
            chunk_tensor=chunk_to_process_tensor,
            chunk_idx=chunk_idx, # Use the index relative to the full sequence after rechunk
            M_rev_std=M_rev_std
            # This function needs to internally use self.M_fwd and update it.
        )
        # _process_chunk_in_cycle should also update self.total_tokens_processed

        # --- Step 3: Compute Persistent Reverse Memory ---
        # Based *only* on the history *before* this chunk was processed
        # Note: self.closed_chunks does NOT yet include chunk_to_process_list
        self.M_rev_persist = self._compute_persistent_reverse_memory(
            self.closed_chunks + [chunk_to_process_list],  # Pass history + current chunk
            mask_future=self.training
        )
        self.tokens_since_persistent_update = 0  # Reset counter after update

        # --- Step 4: Update History and Clear Buffer ---
        self.closed_chunks.append(chunk_to_process_list) # Add the processed chunk to history
        self.current_chunk_tokens = [] # Clear the buffer
        self.current_chunk_text = ""

    def _process_chunk_in_cycle(
            self,
            chunk_tensor: torch.Tensor,
            chunk_idx: int,
            M_rev_std: torch.Tensor
    ) -> None:
        """
        Helper used inside _trigger_memory_update_cycle.
        Processes a single chunk tensor, using provided M_rev_std and current self.M_fwd.
        Updates self.M_fwd via memory update layers.
        Updates self.total_tokens_processed.
        """
        # Assuming B=1 for stateful processing
        if chunk_tensor.ndim == 1:
            chunk_tensor = chunk_tensor.unsqueeze(0) # Add batch dim if needed by layers

        # Get total chunks estimate for control tokens (might need adjustment)
        # Total chunks = history + current one being processed + estimate of future?
        # Let's use history + current as the known sequence length for ratios.
        total_chunks_so_far = len(self.closed_chunks) + 1

        x = self.token_embedding(chunk_tensor) # Shape (1, T, D)

        # Generate control tokens for the forward pass processing this chunk
        ctrl = self.control_token_generator.generate_control_tokens(
            mode="forward", # Processing a known chunk in the forward direction
            current_chunk_idx=chunk_idx,
            total_chunks=total_chunks_so_far, # Use current known total
            current_mem_size=self.memory_manager.get_effective_size(self.total_tokens_processed),
            max_mem_size=self.config.max_memory_size,
            seq_len=self.total_tokens_processed # Tokens processed *before* this chunk
        )

        # Use local copies of memory states to avoid modifying them incorrectly if layer forward pass fails
        f_mem = self.M_fwd
        r_mem_std = M_rev_std # Use the passed standard reverse memory

        for layer in self.layers:
            if layer.layer_type == "memory_update":
                # Calculate write mask based on current progress
                wmask = self.memory_manager.get_write_mask(
                    chunk_idx, total_chunks_so_far, self.total_tokens_processed, batch_size=1
                )
                # Pass M_rev_std for reading, but only update M_fwd (is_reverse_update=False)
                x, updated_fwd, _, _ = layer( # Expect only updated_fwd to be non-None
                    x,
                    forward_memory=f_mem,
                    reverse_memory=r_mem_std, # Read from standard reverse
                    control_tokens=ctrl,
                    do_memory_update=True, # Enable memory update
                    write_mask=wmask,
                    is_reverse_update=False # IMPORTANT: We are updating FORWARD memory here
                )
                if updated_fwd is not None:
                    f_mem = updated_fwd # Update local copy for next layer
            else:
                # Read-only or local-only layer
                x, _, _, _ = layer(
                    x,
                    forward_memory=f_mem if layer.layer_type != "local_only" else None,
                    reverse_memory=r_mem_std if layer.layer_type != "local_only" else None,
                    control_tokens=ctrl
                )

        # Update the main forward memory state *after* processing the chunk through all layers
        self.M_fwd = f_mem

        # Update total tokens processed *after* successfully processing the chunk
        self.total_tokens_processed += chunk_tensor.size(1)

    def _compute_standard_reverse_memory(
            self,
            chunks: List[List[int]]
    ) -> torch.Tensor:
        """
        Compute the standard reverse memory (M_rev_std) over the given chunks list,
        including the most recently completed chunk at the end.

        Parameters
        ----------
        chunks : List[List[int]]
            Token-ID lists of each chunk, in chronological order.

        Returns
        -------
        M_rev_std : Tensor
            The reverse memory state after processing up to the last chunk.
        """

        # Initialize from the learned backward-memory parameters
        M_rev_std = self.initial_backward_memory.clone().to(self.M_fwd.device)
        total_chunks = len(chunks)

        # Select window of most-recent chunks for reverse pass
        rev_window = chunks[-self.config.reverse_max_chunks:]
        window_start = total_chunks - len(rev_window)

        # Iterate in reverse (newest→oldest)
        for rev_idx, chunk in enumerate(reversed(rev_window)):
            # Global index of this chunk
            global_idx = window_start + (len(rev_window) - 1 - rev_idx)

            # Token embedding + batch dim
            chunk_tensor = torch.tensor(chunk, device=self.M_fwd.device)
            x = self.token_embedding(chunk_tensor).unsqueeze(0)

            # Control tokens for this reverse step
            ctrl = self.control_token_generator.generate_control_tokens(
                mode="standard_reverse",
                current_chunk_idx=global_idx,
                total_chunks=total_chunks,
                current_mem_size=self.memory_manager.get_effective_size(self.total_tokens_processed),
                max_mem_size=self.config.max_memory_size,
                seq_len=self.total_tokens_processed,
                reverse_chunk_idx=rev_idx,
                reverse_window_size=len(rev_window)
            )

            # Layer-by-layer processing
            for layer in self.layers:
                if layer.layer_type == "memory_update":
                    # Downweighting curve for reverse update
                    decay = self.memory_manager.apply_downweighting(
                        torch.ones_like(M_rev_std),
                        [global_idx],
                        is_reverse=True,
                        is_persistent=False
                    )
                    x, _, updated_rev, _ = layer(
                        x,
                        reverse_memory=M_rev_std,
                        control_tokens=ctrl,
                        do_memory_update=True,
                        decay_weights=decay,
                        is_reverse_update=True
                    )
                    if updated_rev is not None:
                        M_rev_std = updated_rev
                else:
                    x, _, _, _ = layer(
                        x,
                        reverse_memory=M_rev_std if layer.layer_type != "local_only" else None,
                        control_tokens=ctrl
                    )
        return M_rev_std

    def _compute_persistent_reverse_memory(
            self,
            chunks: List[List[int]],
            *,
            mask_future: bool = False
    ) -> torch.Tensor:
        """
        Recompute the persistent reverse memory (M_rev_persist) over all chunks
        *except* the most-recent (current) one.

        Parameters
        ----------
        chunks : List[List[int]]
            Full list of chunks including the current one.
        mask_future : bool
            If True and model.training is True, apply future-dropout masking.

        Returns
        -------
        M_rev_persist : Tensor
            The persistent reverse memory state.
        """

        # Start from the learned backward-memory parameters
        M_rev = self.initial_backward_memory.clone().to(self.M_fwd.device)
        total_chunks = len(chunks)

        # Exclude the last chunk (current)
        eligible = chunks[:-1]
        if not eligible:
            return M_rev

        # Window of most-recent eligible chunks
        rev_window = eligible[-self.config.reverse_max_chunks:]
        start_idx = len(eligible) - len(rev_window)

        # Optional future-dropout mask
        if mask_future and self.training and rev_window:
            p = get_mask_future_schedule(
                self.config, self.training_step, self.total_training_steps
            )
            keep = torch.rand(len(rev_window), device=M_rev.device) > p
            rev_window = [c for c, k in zip(rev_window, keep) if k]
            # adjust start_idx if some were dropped
            # (simplest is to recompute start_idx = total_chunks - 1 - len(rev_window))
            start_idx = total_chunks - 1 - len(rev_window)

        # Process in reverse order
        for rev_idx, chunk in enumerate(reversed(rev_window)):
            global_idx = start_idx + (len(rev_window) - 1 - rev_idx)
            chunk_tensor = torch.tensor(chunk, device=M_rev.device)
            x = self.token_embedding(chunk_tensor).unsqueeze(0)

            ctrl = self.control_token_generator.generate_control_tokens(
                mode="persistent_reverse",
                current_chunk_idx=global_idx,
                total_chunks=total_chunks,
                current_mem_size=self.memory_manager.get_effective_size(self.total_tokens_processed),
                max_mem_size=self.config.max_memory_size,
                seq_len=self.total_tokens_processed,
                reverse_chunk_idx=rev_idx,
                reverse_window_size=len(rev_window)
            )

            for layer in self.layers:
                if layer.layer_type == "memory_update":
                    decay = self.memory_manager.apply_downweighting(
                        torch.ones_like(M_rev),
                        [global_idx],
                        is_reverse=True,
                        is_persistent=True
                    )
                    x, _, updated_rev, _ = layer(
                        x,
                        reverse_memory=M_rev,
                        control_tokens=ctrl,
                        do_memory_update=True,
                        decay_weights=decay,
                        is_reverse_update=True
                    )
                    if updated_rev is not None:
                        M_rev = updated_rev
                else:
                    x, _, _, _ = layer(
                        x,
                        reverse_memory=M_rev if layer.layer_type != "local_only" else None,
                        control_tokens=ctrl
                    )
        return M_rev

    @torch.no_grad()
    def generate(
            self,
            prompt: Union[str, List[int], List[List[int]], torch.Tensor, None] = None,
            *,
            max_new_tokens: int = 128,
            temperature: float = 1.0,
            top_k: Optional[int] = None,
            stop_token_id: Optional[int] = None,
            reset_state: bool = False,
    ) -> List[int]:  # Modified return type
        """
        Autoregressive decoding driven entirely by `forward()`. Caches current chunk text.

        • If `reset_state` is True, clears model state first.
        • If `prompt` is not None/empty, it is fed through `forward()`.
        • Initializes `self.current_chunk_text` based on any prompt residue.
        • Each loop iteration then:
            1.  feeds the token chosen in the previous step (or prompt on first
                iteration) via `forward(next_input)`;
            2.  receives logits for the *current* position (T > 0 guaranteed);
            3.  samples the next token `next_id`;
            4.  appends `next_id` to `generated` list;
            5.  decodes `next_id` and appends to `self.current_chunk_text`;
            6.  checks for periodic persistent reverse update using cached text;
            7.  stores `[next_id]` to feed in step 1 of the next iteration.

        Returns
        -------
        List[int], the list of generated token IDs.
        """
        if reset_state:
            self.reset_state()

        # Helper: temperature / top-k sampling for one logit row
        def _sample_row(row: torch.Tensor) -> int:
            # Ensure row is float32 for topk and softmax stability
            row = row.float()
            if top_k is not None and top_k > 0:
                # Handle potential case where vocab size < top_k
                actual_k = min(top_k, row.shape[-1])
                if actual_k > 0:
                    thresh = torch.topk(row, actual_k).values[-1]
                    row = torch.where(row < thresh,
                                      torch.full_like(row, float("-inf")), row)
            # Prevent division by zero or negative temperature
            temp = max(float(temperature), 1e-5)
            probs = torch.nn.functional.softmax(row / temp, dim=-1)
            return torch.multinomial(probs, 1).item()

        next_input = prompt  # first pass takes the prompt (may be None)
        generated: List[int] = []

        # Initialize/Sync current_chunk_text after processing prompt
        # This ensures the cache matches the buffer state before generation starts
        if prompt not in (None, "", [], torch.tensor([], dtype=torch.long)):
            # Process prompt first to update self.current_chunk_tokens
            _ = self.forward(prompt, training_mode=False)  # Ignore logits here

        # Now sync the text cache with the buffer state
        if self.current_chunk_tokens:
            try:
                self.current_chunk_text = self.tokenizer.decode(self.current_chunk_tokens)
            except Exception as e:
                print(f"Warning: Error decoding initial buffer tokens: {e}")
                self.current_chunk_text = ""  # Fallback
        else:
            self.current_chunk_text = ""

        # Set next_input to None after processing the initial prompt for the loop
        next_input = None

        for _ in range(max_new_tokens):
            # Step 1: Feed previous token (or None initially) into forward
            # This updates state and potentially triggers flush
            logits = self.forward(next_input, training_mode=False)

            # Step 2: Check if logits are valid for sampling
            T = logits.shape[1]
            if T == 0:
                # This happens if a flush left the open chunk empty.
                print("INFO: Generation stopped because flush resulted in empty chunk.")
                break

            # Step 3: Sample the next token
            next_id = _sample_row(logits[0, -1, :])

            # Check stop condition before appending
            if stop_token_id is not None and next_id == stop_token_id:
                break

            # Step 4: Store generated token ID
            generated.append(next_id)

            # Step 5: Update the text cache for the current chunk
            try:
                self.current_chunk_text += self.tokenizer.decode([next_id])
            except Exception as e:
                print(f"Warning: Error decoding token {next_id}: {e}")
                # Continue generation, but semantic check might be impaired

            # Step 6: Check for Periodic Persistent Reverse Memory Update
            self.tokens_since_persistent_update += 1
            update_persist = False
            # Check token count trigger
            if self.config.persistent_reverse_update_freq_tokens is not None and \
                    self.tokens_since_persistent_update >= self.config.persistent_reverse_update_freq_tokens:
                update_persist = True

            # Check semantic boundary trigger (using cached text)
            if not update_persist and \
                    self.config.persistent_reverse_update_freq_semantic and \
                    len(self.current_chunk_text) > 1:  # Need some text to check
                try:
                    boundary_level = self.config.persistent_reverse_update_freq_semantic
                    if boundary_level in self.config.boundary_types:
                        # Check near the end of the cached text (e.g., last 10 chars)
                        search_window = self.current_chunk_text[-10:]
                        for boundary_type in self.config.boundary_types[boundary_level]:
                            pattern = BOUNDARY_PATTERNS.get(boundary_type)
                            if pattern and re.search(pattern, search_window):
                                update_persist = True
                                break  # Stop checking patterns once one is found
                except Exception as e:
                    print(f"Warning: Error during semantic boundary check: {e}")

            if update_persist:
                # Compute persistent reverse using history + current buffer content
                full_chunks_for_persist = self.closed_chunks + (
                    [self.current_chunk_tokens] if self.current_chunk_tokens else [])
                self.M_rev_persist = self._compute_persistent_reverse_memory(
                    full_chunks_for_persist,
                    mask_future=False  # No mask-future during inference
                )
                self.tokens_since_persistent_update = 0  # Reset counter

            # Step 7: Prepare input for the next iteration
            next_input = [next_id]

        return generated

    def reset_state(self) -> None:
        """Clear all sequence-level state before starting a new dialogue."""
        self.closed_chunks = []
        self.current_chunk_tokens = []
        self.M_fwd = None
        self.M_rev_persist = None
        self.current_chunk_text: str = ""

    def set_training_step(self, step: int, total_steps: int):
        """Set current training step for mask-future scheduling"""
        self.training_step = step
        self.total_training_steps = total_steps