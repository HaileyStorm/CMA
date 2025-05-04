import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict, Any
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import yaml


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


@dataclass
class CMAConfig:
    """Configuration for CMA model"""
    # Chunking
    chunk_size: int = 768
    semantic_chunking_gap_percentage: float = 25.0
    boundary_search_chars: List[int] = (256, 64, 32)
    boundary_types: Dict[str, List[str]] = None
    buffer_ratio: float = 0.1

    # Memory
    max_memory_size: int = 3072
    reverse_memory_size: int = 320
    initial_write_fraction: float = 0.6
    memory_growth_function: str = "linear"
    memory_cap_length: int = 49152
    share_initial_memory: bool = False
    reset_memory_on_cycle: bool = True

    # Reverse pass
    reverse_max_chunks: int = 4
    lookahead_reverse_decay_step: float = 0.2
    lookahead_reverse_decay_rate: float = 0.5
    persistent_reverse_decay_step: float = 0.05
    persistent_reverse_decay_rate: float = 0.1
    persistent_reverse_update_freq_tokens: Optional[int] = None
    persistent_reverse_update_freq_semantic: Optional[str] = None # e.g. "secondary"

    # Model architecture
    embed_dim: int = 768
    n_heads: int = 6
    n_layers: int = 12  # Note: This might become redundant if layer_structure defines all layers
    head_dim: int = 128  # Typically embed_dim // n_heads
    layer_structure: Optional[List[Dict]] = None
    skip_attention_layers: List[int] = (6,)  # Note: Indices need careful handling with groups

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
    mask_future_rates: Tuple[float, float, float] = (0.333, 0.667, 1.0)
    enable_mask_future_dropout: bool = True

    def __post_init__(self):
        if self.boundary_types is None:
            self.boundary_types = {
                "primary": ["section_break", "code_block", "paragraph_break", "double_line_break"],
                "secondary": ["sentence_end", "line_break"],
                "tertiary": ["clause_end", "code_line_end"]
            }
        # Default layer structure if none provided
        if self.layer_structure is None:
            # Example: Alternate local and memory update layers implicitly grouped
            self.layer_structure = []
            for i in range(self.n_layers):
                if (i + 1) % 5 == 0:  # Make every 5th layer a memory update layer
                    self.layer_structure.append({"type": "memory_update"})
                else:
                    self.layer_structure.append({"type": "local_only"})
            print("Warning: No layer_structure provided. Using default alternating structure.")

        self.validate()

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'CMAConfig':
        # Handle potential nested dicts if loading from complex YAML/JSON
        # For now, assume flat structure matching dataclass fields
        return cls(**config_dict)

    @classmethod
    def from_yaml(cls, yaml_file: Union[str, Path]) -> 'CMAConfig':
        with open(yaml_file, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    def validate(self):
        # Basic validations (keep these)
        assert 0 < self.semantic_chunking_gap_percentage < 100
        assert len(self.boundary_search_chars) == 3
        assert self.max_memory_size > 0
        assert self.reverse_memory_size > 0
        assert 0 <= self.initial_write_fraction <= 1.0
        assert self.reverse_max_chunks > 0
        assert self.embed_dim > 0
        assert self.n_heads > 0
        assert self.embed_dim % self.n_heads == 0
        assert self.head_dim == self.embed_dim // self.n_heads  # Ensure head_dim is consistent, disable if adding variable head sizes or such.
        assert isinstance(self.enable_mask_future_dropout, bool)

        # Layer structure validation is handled during model init parsing
        # because it needs the parsed structure itself. We could add config-level
        # checks here later if needed (e.g., check layer type strings are valid).


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
        elif mode == "lookahead_reverse":
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
        elif mode in ["lookahead_reverse", "persistent_reverse"]:
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

            # Don't bypass gap logic for short texts
            # Only use this shortcut if the text is shorter than the gap-adjusted target
            if end_pos <= target_size and end_pos == len(text):
                chunk_tokens = self.tokenizer.encode(text)
                chunks.insert(0, chunk_tokens)
                break

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

                # Check if chunk respects the size limit
                max_allowed_size = target_size if end_pos == len(text) else self.config.chunk_size

                if len(chunk_tokens) <= max_allowed_size:
                    found_valid_chunk = True
                    chunks.insert(0, chunk_tokens)
                else:
                    # Estimate new start position to reduce chunk size
                    excess_tokens = len(chunk_tokens) - max_allowed_size
                    chars_per_token = len(chunk_text) / len(chunk_tokens)
                    est_start = start_pos + int(excess_tokens * chars_per_token * 1.1)  # 1.1 buffer
                    iteration += 1

            # If we couldn't find a valid boundary, use the estimated start
            if not found_valid_chunk:
                chunk_text = text[est_start:end_pos]
                chunk_tokens = self.tokenizer.encode(chunk_text)

                # If still too large, truncate to respect the limit
                max_allowed_size = target_size if end_pos == len(text) else self.config.chunk_size
                if len(chunk_tokens) > max_allowed_size:
                    chunk_tokens = chunk_tokens[:max_allowed_size]

                chunks.insert(0, chunk_tokens)
                # Adjust start_pos for next iteration
                start_pos = est_start

            # Ensure we make progress in the outer loop
            if start_pos >= end_pos:
                start_pos = end_pos - 1
            if start_pos < 0:
                start_pos = 0

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
    """Manages forward memory scaling and reverse memory decay weights"""

    def __init__(self, config: CMAConfig):
        self.config = config

    def get_effective_size(self, total_tokens_processed: int) -> int:
        seq_len = total_tokens_processed
        if self.config.memory_growth_function == "linear":
            cap_length = max(1, self.config.memory_cap_length)
            fraction = min(1.0, seq_len / cap_length)
        elif self.config.memory_growth_function == "log":
            if seq_len <= 1: fraction = 0.0
            else:
                cap_length = max(2, self.config.memory_cap_length)
                fraction = math.log(seq_len) / math.log(cap_length)
                fraction = min(1.0, max(0.0, fraction))
        else: raise ValueError(f"Unknown growth function: {self.config.memory_growth_function}")
        effective_size = max(0, round(self.config.max_memory_size * fraction))
        return effective_size

    def get_write_mask(
            self,
            current_chunk_idx_in_pass: int,
            total_chunks_in_pass: int,
            total_tokens_processed_before_chunk: int,
            batch_size: int = 1,
            device: torch.device = torch.device('cpu')
    ) -> Tensor:
        seq_write_cap = self.get_effective_size(total_tokens_processed_before_chunk)
        if total_chunks_in_pass > 0:
            chunk_progress = (current_chunk_idx_in_pass + 1) / total_chunks_in_pass
            initial_fraction = self.config.initial_write_fraction
            write_fraction = initial_fraction + (1.0 - initial_fraction) * chunk_progress
        else: write_fraction = self.config.initial_write_fraction
        target_writable_size = int(self.config.max_memory_size * write_fraction)
        writable_size = min(seq_write_cap, target_writable_size)
        writable_size = max(0, writable_size)
        mask = torch.zeros(batch_size, self.config.max_memory_size, dtype=torch.bool, device=device)
        if writable_size > 0: mask[:, :writable_size] = True
        return mask

    def calculate_reverse_decay_weights(
            self,
            reverse_chunk_index: int,  # 0 for newest chunk in window, 1 for next, ...
            window_size: int,
            is_persistent: bool,
            memory_shape: Tuple,  # e.g., (B, M, D)
            device: torch.device
    ) -> Tensor:
        """Calculate decay weights for a specific chunk in a reverse pass window."""
        if window_size <= 0 or reverse_chunk_index < 0 or reverse_chunk_index >= window_size:
            # Return uniform weights if input is invalid or window is empty
            return torch.ones(memory_shape, device=device)

        if is_persistent:
            decay_step = self.config.persistent_reverse_decay_step
            decay_rate = self.config.persistent_reverse_decay_rate
        else:  # Lookahead reverse
            decay_step = self.config.lookahead_reverse_decay_step
            decay_rate = self.config.lookahead_reverse_decay_rate

        # Calculate the decay exponent based on the relative position from the newest chunk
        # reverse_chunk_index = 0 (newest) -> exponent = 0 -> decay = 1.0
        # reverse_chunk_index = 1 (next oldest) -> exponent = decay_step
        # reverse_chunk_index = 2 -> exponent = decay_step + decay_step * decay_rate
        # reverse_chunk_index = k -> exponent = decay_step * (1 + rate + rate^2 ... + rate^(k-1))
        # Using simpler exponential decay for now as per original spec structure: rate^(index * step)
        # This means decay_step acts more like a scaling factor on the index.
        # Let's refine this based on the spec description:
        # "initial step-dow ... exponential decay rate for weighting chunks beyond the second"
        # Index 0 (newest): weight 1.0
        # Index 1 (second): weight decay_step (if step is < 1) or maybe 1.0 - decay_step? Let's assume rate^step
        # Index 2 (third): weight rate^(step) * rate^(step) = rate^(2*step) ?
        # Let's use a simple exponential decay: decay_factor = decay_rate ** (reverse_chunk_index * decay_step)
        # Ensure rate and step are reasonable (e.g., rate <= 1)
        decay_rate = min(1.0, max(0.0, decay_rate))
        decay_factor = decay_rate ** (reverse_chunk_index * decay_step)

        # Create weights tensor
        weights = torch.full(memory_shape, decay_factor, device=device)
        return weights


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

        # Handle empty sequence
        if T == 0:
            return torch.zeros_like(q), None

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
    """Transformer block with group-aware memory handling."""

    def __init__(self, config: CMAConfig, layer_idx: int, layer_type: str = "local_only"):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        # Layer type determines behavior w.r.t memory group
        self.layer_type = layer_type  # "local_only", "memory_read", "memory_update", "skip"

        # Create attention layer based on type
        if self.layer_type == "skip":
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

    def forward(
            self,
            x: Tensor,
            # Pass the *dictionaries* of memory states
            M_fwd_dict: Dict[int, Tensor],
            M_rev_std_dict: Dict[int, Tensor],
            M_rev_persist_dict: Dict[int, Tensor],
            # Pass the group ID for this layer
            group_id: int,
            # Pass the current operational mode/pass name
            mode: str,  # "forward", "lookahead_reverse", "persistent_reverse", "generate"
            control_tokens: Optional[Dict[str, float]] = None,
            write_mask: Optional[Tensor] = None,  # Forward pass only
            decay_weights: Optional[Tensor] = None  # Reverse passes only
    ) -> Tuple[Tensor, Optional[Tuple[int, Tensor]], Optional[Tuple[int, Tensor]], Optional[Tensor]]:
        """
        Forward pass for a block, aware of its group and the current pass mode.

        Args:
            x: Input tensor (B, T, D).
            M_fwd_dict: Dictionary mapping group_id to forward memory tensor.
            M_rev_std_dict: Dictionary mapping group_id to lookahead reverse memory tensor.
            M_rev_persist_dict: Dictionary mapping group_id to persistent reverse memory tensor.
            group_id: The index of the group this layer belongs to.
            mode: The current processing mode/pass.
            control_tokens: Control signals.
            write_mask: Mask for forward memory updates.
            decay_weights: Weights for reverse memory updates.

        Returns:
            Tuple:
                - output_tensor (Tensor): Result after block processing.
                - updated_fwd (Optional[Tuple[int, Tensor]]): (group_id, updated_tensor) if fwd mem updated.
                - updated_rev (Optional[Tuple[int, Tensor]]): (group_id, updated_tensor) if rev mem updated.
                - gate_reg_loss (Optional[Tensor]): Regularization loss from gating.
        """
        updated_fwd_result = None
        updated_rev_result = None
        gate_reg_loss = None
        attn_output = torch.zeros_like(x)  # Default if attention is skipped

        # --- Determine Memory Inputs and Update Flags based on Group and Mode ---
        fwd_mem_in: Optional[Tensor] = None
        rev_mem_in: Optional[Tensor] = None
        do_memory_update: bool = False
        is_reverse_update: bool = False

        # Only access memory if the layer is not local_only and not skip
        if self.layer_type not in ["local_only", "skip"]:
            # Fetch memory relevant to the current mode
            if mode == "forward":
                fwd_mem_in = M_fwd_dict.get(group_id)
                rev_mem_in = M_rev_std_dict.get(group_id)  # Forward pass uses Lookahead Reverse
                # Update forward memory only if this is the group's update layer
                do_memory_update = (self.layer_type == "memory_update")
                is_reverse_update = False
            elif mode == "lookahead_reverse":
                # No forward memory input in reverse passes
                rev_mem_in = M_rev_std_dict.get(group_id)
                do_memory_update = (self.layer_type == "memory_update")
                is_reverse_update = True
            elif mode == "persistent_reverse":
                rev_mem_in = M_rev_persist_dict.get(group_id)
                do_memory_update = (self.layer_type == "memory_update")
                is_reverse_update = True
            elif mode == "generate":
                # Generation reads Fwd and Persistent Reverse, never updates
                fwd_mem_in = M_fwd_dict.get(group_id)
                rev_mem_in = M_rev_persist_dict.get(group_id)
                do_memory_update = False
                is_reverse_update = False
            else:
                raise ValueError(f"Unknown mode in Block.forward: {mode}")

        # --- Attention + Residual ---
        if self.attn is not None:
            residual = x
            x_norm = norm(x)

            if isinstance(self.attn, CascadeMemoryAttention):
                # Pass the determined memory inputs and flags
                attn_output, fwd_mem_out, rev_mem_out, gate_reg_loss = self.attn(
                    x_norm,
                    forward_memory=fwd_mem_in,
                    reverse_memory=rev_mem_in,
                    control_tokens=control_tokens,
                    do_memory_update=do_memory_update,
                    write_mask=write_mask if not is_reverse_update else None,  # Only pass write_mask for fwd update
                    decay_weights=decay_weights if is_reverse_update else None,  # Only pass decay for rev update
                    is_reverse_update=is_reverse_update
                )
                # Package updated memory with group_id for return
                if fwd_mem_out is not None: updated_fwd_result = (group_id, fwd_mem_out)
                if rev_mem_out is not None: updated_rev_result = (group_id, rev_mem_out)

            elif isinstance(self.attn, CausalSelfAttention):
                # Local attention only operates on x_norm
                attn_output = self.attn(x_norm)

            x = residual + attn_output
        # If self.attn is None (skip layer), x remains unchanged

        # --- MLP + Residual ---
        residual = x
        x_norm = norm(x)
        mlp_output = self.mlp(x_norm)
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
        print(f"Parsed {self.num_layers} layers into {self.num_groups} groups ({self.num_memory_groups} with memory).")
        # --- End Layer and Group Parsing ---

        self.lm_head = nn.Linear(config.embed_dim, vocab_size, bias=False)

        # --- Initialize Memory Parameters ---
        self.group_id_to_memory_idx: Dict[int, int] = {}
        memory_param_idx_counter = 0
        if self.num_memory_groups > 0:
            if config.share_initial_memory:
                print("Using shared initial memory parameters.")
                shared_fwd = nn.Parameter(
                    torch.randn(1, config.max_memory_size, config.embed_dim) * config.memory_init_scale)
                shared_rev_s = nn.Parameter(
                    torch.randn(1, config.reverse_memory_size, config.embed_dim) * config.memory_init_scale)
                shared_rev_p = nn.Parameter(
                    torch.randn(1, config.reverse_memory_size, config.embed_dim) * config.memory_init_scale)
                self.initial_fwd_params = nn.ParameterList([shared_fwd] * self.num_memory_groups)
                self.initial_rev_s_params = nn.ParameterList([shared_rev_s] * self.num_memory_groups)
                self.initial_rev_p_params = nn.ParameterList([shared_rev_p] * self.num_memory_groups)
                for group in self.layer_groups:
                    if group.has_memory: self.group_id_to_memory_idx[group.group_idx] = 0
                memory_param_idx_counter = 1
            else:
                print("Using dedicated initial memory parameters per group.")
                fwd_params, rev_s_params, rev_p_params = [], [], []
                for group in self.layer_groups:
                    if group.has_memory:
                        self.group_id_to_memory_idx[group.group_idx] = memory_param_idx_counter
                        fwd_params.append(nn.Parameter(
                            torch.randn(1, config.max_memory_size, config.embed_dim) * config.memory_init_scale))
                        rev_s_params.append(nn.Parameter(
                            torch.randn(1, config.reverse_memory_size, config.embed_dim) * config.memory_init_scale))
                        rev_p_params.append(nn.Parameter(
                            torch.randn(1, config.reverse_memory_size, config.embed_dim) * config.memory_init_scale))
                        memory_param_idx_counter += 1
                self.initial_fwd_params = nn.ParameterList(fwd_params)
                self.initial_rev_s_params = nn.ParameterList(rev_s_params)
                self.initial_rev_p_params = nn.ParameterList(rev_p_params)

            # Ensure all parameters are on the correct device
            device = self.token_embedding.weight.device
            for param_list in [self.initial_fwd_params, self.initial_rev_s_params, self.initial_rev_p_params]:
                for param in param_list:
                    param.data = param.data.to(device)
        else:
            print("No memory groups found. Initial memory parameters will not be created.")
            self.initial_fwd_params = nn.ParameterList()
            self.initial_rev_s_params = nn.ParameterList()
            self.initial_rev_p_params = nn.ParameterList()
        print(f"Created {memory_param_idx_counter} sets of initial memory parameters.")
        # --- End Initialize Memory Parameters ---

        # --- State Tracking ---
        # Runtime memory states are dictionaries keyed by group_idx
        self.M_fwd: Dict[int, Tensor] = {}
        self.M_rev_std: Dict[int, Tensor] = {}
        self.M_rev_persist: Dict[int, Tensor] = {}

        self.closed_chunks: List[List[int]] = []
        self.total_tokens_processed = 0
        self.tokens_since_persistent_update = 0
        self.current_chunk_tokens: List[int] = []
        self.current_chunk_text: str = ""
        self.training_step = 0
        self.total_training_steps = 10000

        # Initialize weights
        self.apply(self._init_weights)

        # --- Parameter Count ---
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        initial_memory_params = 0
        if hasattr(self, 'initial_fwd_params'): initial_memory_params += sum(
            p.numel() for p in self.initial_fwd_params if p.requires_grad)
        if hasattr(self, 'initial_rev_s_params'): initial_memory_params += sum(
            p.numel() for p in self.initial_rev_s_params if p.requires_grad)
        if hasattr(self, 'initial_rev_p_params'): initial_memory_params += sum(
            p.numel() for p in self.initial_rev_p_params if p.requires_grad)
        if config.share_initial_memory and self.num_memory_groups > 1:
            num_shared_sets = 3
            params_per_set = initial_memory_params // (
                        self.num_memory_groups * num_shared_sets) if self.num_memory_groups > 0 else 0
            initial_memory_params = num_shared_sets * params_per_set
        other_params = total_params - initial_memory_params
        print(f"--- CMA Model Parameter Count ---")
        print(
            f"Initial memory parameters: {initial_memory_params:,} ({memory_param_idx_counter} sets, {'shared' if config.share_initial_memory else 'dedicated'})")
        print(f"Other trainable parameters: {other_params:,}")
        print(f"Total trainable parameters: {total_params:,}")
        print(f"------------------------------------------")
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
                    self.M_fwd[group_idx] = self.initial_fwd_params[mem_idx].clone().detach().to(dev)
                # M_rev_std is always reset/recomputed in the cycle, initialize empty or reset
                # We initialize it here mainly to ensure the key exists if accessed before cycle
                self.M_rev_std[group_idx] = torch.zeros(  # Placeholder, will be overwritten
                    1, self.config.reverse_memory_size, self.config.embed_dim, device=dev
                )
                if force_reset or group_idx not in self.M_rev_persist:
                    self.M_rev_persist[group_idx] = self.initial_rev_p_params[mem_idx].clone().detach().to(dev)

    def forward(
            self,
            input_ids: Union[str, List[int], List[List[int]], torch.Tensor, None],
            *, training_mode: bool = False  # Use this parameter
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        CMA forward pass. Handles input processing, chunking, triggering update cycles,
        processing the current chunk, and aggregating gate loss during training.
        """
        print(f"DEBUG: Forward called with input_ids type={type(input_ids)}, training_mode={training_mode}", flush=True)

        # --- Set training state based on parameter ---
        self.training = training_mode  # Set the nn.Module training state

        self._initialize_memory_states(force_reset=False)
        dev = self.token_embedding.weight.device
        print(f"DEBUG: Device in forward: {dev}", flush=True)

        cycle_gate_loss: Optional[torch.Tensor] = None  # Initialize cycle loss
        chunk_gate_loss: Optional[torch.Tensor] = None  # Initialize chunk loss

        if input_ids in (None, "", [], torch.tensor([], dtype=torch.long)):
            if self.current_chunk_tokens:
                logits, chunk_gate_loss = self._process_current_chunk(generation_mode=not self.training)
            else:
                logits = torch.zeros(1, 0, self.vocab_size, device=dev)
        else:
            new_tokens: List[int]
            mode: str
            if isinstance(input_ids, str):
                new_tokens = self.tokenizer.encode(input_ids)
                mode = "semantic"
            elif isinstance(input_ids, list) and input_ids and isinstance(input_ids[0], list):
                new_chunks = [list(c) for c in input_ids]
                new_tokens = [t for c in new_chunks for t in c]
                mode = "caller_exact"
            else:
                new_tokens = input_ids.tolist() if isinstance(input_ids, torch.Tensor) else list(input_ids)
                mode = "fixed"

            print(f"DEBUG: Mode={mode}, new_tokens length={len(new_tokens)}", flush=True)

            trigger_update_cycle = False
            full_tokens = []
            chunks = []
            if mode == "caller_exact":
                trigger_update_cycle = True
                full_tokens = [t for c in self.closed_chunks for t in c] + self.current_chunk_tokens + new_tokens
                chunks = self.closed_chunks + (
                    [self.current_chunk_tokens] if self.current_chunk_tokens else []) + new_chunks
            elif self.current_chunk_tokens and mode in {"semantic", "fixed"}:
                # If there's existing content in the buffer and new input arrives,
                # we need to re-chunk and run a cycle to incorporate it properly.
                trigger_update_cycle = True
                full_tokens = [t for c in self.closed_chunks for t in c] + self.current_chunk_tokens + new_tokens
                # Clear current chunk buffer as it's now part of full_tokens
                self.current_chunk_tokens = []
                self.current_chunk_text = ""
            elif len(self.current_chunk_tokens) + len(new_tokens) >= self.config.chunk_size:
                # If adding new tokens fills or exceeds the chunk size, trigger cycle.
                trigger_update_cycle = True
                full_tokens = [t for c in self.closed_chunks for t in c] + self.current_chunk_tokens + new_tokens
                # Clear current chunk buffer
                self.current_chunk_tokens = []
                self.current_chunk_text = ""
            else:
                # Append new tokens if no cycle is triggered
                self.current_chunk_tokens.extend(new_tokens)
                try:
                    self.current_chunk_text += self.tokenizer.decode(new_tokens)
                except Exception as e:
                    print(f"Warning: Error decoding appended tokens: {e}", flush=True)

            if trigger_update_cycle:
                if self.config.reset_memory_on_cycle:
                    print("Resetting memory states for update cycle.", flush=True)
                    self._initialize_memory_states(force_reset=True)

                if mode != "caller_exact":
                    if mode == "semantic":
                        full_text = self.tokenizer.decode(full_tokens)
                        chunks = self.chunk_processor.semantic_chunk_reverse_with_gap(full_text)
                    else:  # fixed
                        chunks = self.chunk_processor.fixed_size_chunk_reverse_with_gap(full_tokens)

                if not chunks:
                    print("Warning: Re-chunking resulted in zero chunks.", flush=True)
                    self.closed_chunks = []
                    self.current_chunk_tokens = []
                    self.current_chunk_text = ""
                else:
                    print(f"Triggering memory update cycle with {len(chunks)} chunks.", flush=True)
                    print(f"DEBUG: About to call _trigger_memory_update_cycle", flush=True)
                    # --- Capture gate loss from the update cycle ---
                    cycle_gate_loss = self._trigger_memory_update_cycle(chunks)
                    print(f"DEBUG: Returned from _trigger_memory_update_cycle", flush=True)
                    # After cycle, current_chunk_tokens is updated, update text buffer
                    try:
                        self.current_chunk_text = self.tokenizer.decode(self.current_chunk_tokens)
                    except Exception as e:
                        print(f"Warning: Error decoding buffer after cycle: {e}", flush=True)
                        self.current_chunk_text = ""

            # --- Process the final (potentially partial) chunk for logits ---
            # This runs regardless of whether a cycle was triggered, processing whatever is in current_chunk_tokens
            if self.current_chunk_tokens:
                logits, chunk_gate_loss = self._process_current_chunk(generation_mode=not self.training)
            else:
                # If cycle happened and last chunk was empty, or no input initially
                logits = torch.zeros(1, 0, self.vocab_size, device=dev)

        # --- Aggregate Gate Loss (only in training mode with regularization enabled) ---
        total_gate_loss: Optional[torch.Tensor] = None
        if self.training and self.config.gate_regularization_type is not None:
            if cycle_gate_loss is not None and chunk_gate_loss is not None:
                total_gate_loss = cycle_gate_loss + chunk_gate_loss
            elif cycle_gate_loss is not None:
                total_gate_loss = cycle_gate_loss
            elif chunk_gate_loss is not None:
                total_gate_loss = chunk_gate_loss
            # If both are None, total_gate_loss remains None

        # --- Return logits and the final aggregated gate loss ---
        return logits, total_gate_loss

    def _process_current_chunk(self, generation_mode: bool = False) -> Tuple[
        torch.Tensor, Optional[torch.Tensor]]:  # Updated return signature
        """
        Processes the current chunk buffer for logits.
        Returns logits and accumulated gate loss for this chunk.
        """
        if not self.current_chunk_tokens:
            return torch.zeros(1, 0, self.vocab_size,
                               device=self.token_embedding.weight.device), None  # No loss if no tokens
        self._initialize_memory_states(force_reset=False)
        dev = self.token_embedding.weight.device

        chunk_idx = len(self.closed_chunks)
        total_chunks_in_history = chunk_idx + 1
        chunk_tensor = torch.tensor(self.current_chunk_tokens, device=dev, dtype=torch.long).unsqueeze(0)  # (1, T)
        B, T = chunk_tensor.shape
        x = self.token_embedding(chunk_tensor)  # (1, T, D)

        approx_processed_len = sum(len(c) for c in self.closed_chunks)
        current_mem_size = self.memory_manager.get_effective_size(approx_processed_len)
        # Determine mode for control tokens and layer processing
        mode = "generate" if generation_mode else "forward"  # Or could be a specific training mode
        ctrl = self.control_token_generator.generate_control_tokens(
            mode=mode,
            current_chunk_idx=chunk_idx, total_chunks=total_chunks_in_history,
            current_mem_size=current_mem_size, max_mem_size=self.config.max_memory_size,
            seq_len=approx_processed_len
        )

        # Use copies of state dicts to pass to layers if needed, or pass self.M_* directly
        # Passing self.M_* directly is fine as Block.forward reads from them
        M_fwd = self.M_fwd
        M_rev_std = self.M_rev_std  # Not used in generation/forward processing chunk
        M_rev_persist = self.M_rev_persist
        aggregated_gate_loss: Optional[torch.Tensor] = None  # Initialize accumulator for this chunk

        # --- Layer processing loop (Accumulate gate loss) ---
        for layer_idx, layer in enumerate(self.layers):
            group_idx = self.layer_idx_to_group_idx[layer_idx]

            x, updated_fwd, updated_rev, gate_loss = layer(  # Capture gate_loss
                x,
                M_fwd_dict=M_fwd, M_rev_std_dict=M_rev_std, M_rev_persist_dict=M_rev_persist,
                group_id=group_idx, mode=mode, control_tokens=ctrl,
                write_mask=None, decay_weights=None  # No updates expected here
            )

            # Accumulate gate loss if returned
            if gate_loss is not None:
                # Ensure loss is a scalar or sum appropriately if per-head/token
                current_loss = gate_loss.mean() if gate_loss.numel() > 1 else gate_loss
                if aggregated_gate_loss is None:
                    aggregated_gate_loss = current_loss
                else:
                    aggregated_gate_loss += current_loss  # Simple sum across layers

            # --- Handle state updates (should be None in this mode) ---
            if updated_fwd is not None: print(
                f"Warning: Fwd memory updated unexpectedly in mode '{mode}' for group {updated_fwd[0]}")
            if updated_rev is not None: print(
                f"Warning: Rev memory updated unexpectedly in mode '{mode}' for group {updated_rev[0]}")

        x = norm(x)
        logits = self.lm_head(x)

        # Return logits and the accumulated gate loss for this chunk
        return logits, aggregated_gate_loss

    def _trigger_memory_update_cycle(self, chunks: List[List[int]]) -> Optional[torch.Tensor]:
        """
        Processes the full sequence of chunks through the 3-pass cycle.
        Returns aggregated gate loss from the forward pass if applicable.
        """
        print(f"DEBUG: Starting _trigger_memory_update_cycle with {len(chunks)} chunks", flush=True)

        if not chunks:
            print("Warning: _trigger_memory_update_cycle called with empty chunks list.", flush=True)
            return None

        print(f"Running 3-pass update cycle on {len(chunks)} chunks...", flush=True)
        dev = self.token_embedding.weight.device
        print(f"DEBUG: Device is {dev}", flush=True)

        B = 1
        cycle_gate_loss: Optional[torch.Tensor] = None  # Initialize

        print("  Starting Pass 1: Lookahead Reverse...", flush=True)
        print(f"DEBUG: About to call _run_lookahead_reverse_pass with chunks={len(chunks)}, B={B}, dev={dev}", flush=True)

        try:
            M_rev_std_computed = self._run_lookahead_reverse_pass(chunks, B, dev)
            print("DEBUG: Successfully completed _run_lookahead_reverse_pass", flush=True)
        except Exception as e:
            print(f"ERROR in _run_lookahead_reverse_pass: {type(e).__name__}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise

        print("  Finished Pass 1.", flush=True)

        print("  Starting Pass 2: Forward...", flush=True)
        # --- Capture the returned gate loss from the forward pass ---
        cycle_gate_loss = self._run_forward_pass(chunks, M_rev_std_computed, B, dev)
        print("  Finished Pass 2.", flush=True)

        del M_rev_std_computed
        self.M_rev_std = {}  # Clear the temporary lookahead reverse memory

        print("  Starting Pass 3: Persistent Reverse...", flush=True)
        self._run_persistent_reverse_pass(chunks, B, dev)
        print("  Finished Pass 3.", flush=True)

        # Update state after successful cycle
        self.closed_chunks = chunks[:-1]
        self.current_chunk_tokens = chunks[-1]
        new_total_tokens = sum(len(c) for c in chunks)
        print(
            f"  Update cycle complete. Updating total_tokens_processed from {self.total_tokens_processed} to {new_total_tokens}", flush=True)
        self.total_tokens_processed = new_total_tokens
        self.tokens_since_persistent_update = 0

        # --- Return the aggregated gate loss from the forward pass ---
        return cycle_gate_loss

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
        r_mem_std = M_rev_std  # Use the passed lookahead reverse memory

        for layer in self.layers:
            if layer.layer_type == "memory_update":
                # Calculate write mask based on current progress
                wmask = self.memory_manager.get_write_mask(
                    chunk_idx, total_chunks_so_far, self.total_tokens_processed, batch_size=1
                )
                # Pass M_rev_std for reading, but only update M_fwd (is_reverse_update=False)
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

    def _run_lookahead_reverse_pass(self, all_chunks: List[List[int]], B: int, dev: torch.device) -> Dict[int, Tensor]:
        try:
            """ Runs the lookahead reverse pass. """
            print(f"DEBUG: Entering _run_lookahead_reverse_pass", flush=True)

            n_chunks = len(all_chunks)
            window_start_idx = max(0, n_chunks - self.config.reverse_max_chunks)
            reverse_window_chunks = all_chunks[window_start_idx:]
            window_size = len(reverse_window_chunks)

            print(f"DEBUG: n_chunks={n_chunks}, window_size={window_size}", flush=True)

            # Check if we're dealing with large chunks
            for i, chunk in enumerate(reverse_window_chunks):
                print(f"DEBUG: Chunk {i} size: {len(chunk)}", flush=True)
                if len(chunk) > 150:  # Suspiciously large chunk
                    print(f"WARNING: Very large chunk detected: {len(chunk)} tokens", flush=True)

            # Initialize M_rev_std for this pass locally
            current_M_rev_std: Dict[int, Tensor] = {}
            for group in self.layer_groups:
                if group.has_memory:
                    group_idx = group.group_idx
                    mem_idx = self.group_id_to_memory_idx[group_idx]
                    print(f"DEBUG: Initializing memory for group {group_idx}", flush=True)

                    param = self.initial_rev_s_params[mem_idx]
                    print(f"DEBUG: Parameter shape: {param.shape}, device: {param.device}", flush=True)

                    current_M_rev_std[group_idx] = param.clone().detach().to(dev).repeat(B, 1, 1)
                    print(f"DEBUG: Initialized memory for group {group_idx}, shape: {current_M_rev_std[group_idx].shape}",
                          flush=True)
            for i, chunk_tokens in enumerate(reversed(reverse_window_chunks)):
                if not chunk_tokens:
                    print(f"DEBUG: Skipping empty chunk at index {i}", flush=True)
                    continue

                print(f"DEBUG: Processing chunk {i} with {len(chunk_tokens)} tokens", flush=True)

                chunk_tensor = torch.tensor(chunk_tokens, dtype=torch.long, device=dev).unsqueeze(0)
                print(f"DEBUG: Created chunk tensor with shape {chunk_tensor.shape}", flush=True)

                # Check if embedding lookup could be the issue
                try:
                    x = self.token_embedding(chunk_tensor)
                    print(f"DEBUG: Embedded chunk shape: {x.shape}", flush=True)
                except Exception as e:
                    print(f"ERROR: Embedding lookup failed: {e}", flush=True)
                    exit(3)
                global_chunk_idx = window_start_idx + (window_size - 1 - i);
                reverse_chunk_idx = i

                approx_processed_len = self.total_tokens_processed
                current_mem_size = self.memory_manager.get_effective_size(approx_processed_len)
                ctrl = self.control_token_generator.generate_control_tokens(
                    mode="lookahead_reverse", current_chunk_idx=global_chunk_idx, total_chunks=n_chunks,
                    current_mem_size=current_mem_size, max_mem_size=self.config.max_memory_size,
                    seq_len=approx_processed_len, reverse_chunk_idx=reverse_chunk_idx, reverse_window_size=window_size
                )

                # Pass state dictionaries to layers
                M_fwd = self.M_fwd  # Pass empty or existing state (not used by layers in this mode)
                M_rev_persist = self.M_rev_persist  # Pass existing state (not used by layers in this mode)

                for layer_idx, layer in enumerate(self.layers):
                    group_idx = self.layer_idx_to_group_idx[layer_idx]
                    group = self.layer_groups[group_idx]

                    decay = None
                    # Calculate decay only if it's an update layer for a group with memory
                    if group.has_memory and layer_idx == group.memory_update_layer_idx:
                        mem_shape = current_M_rev_std[group_idx].shape
                        decay = self.memory_manager.calculate_reverse_decay_weights(
                            reverse_chunk_index=reverse_chunk_idx, window_size=window_size,
                            is_persistent=False, memory_shape=mem_shape, device=dev
                        )

                    x, _, updated_rev, _gate_loss = layer(  # Expect only updated_rev
                        x,
                        M_fwd_dict=M_fwd,
                        M_rev_std_dict=current_M_rev_std,  # Pass the dict being updated
                        M_rev_persist_dict=M_rev_persist,
                        group_id=group_idx,
                        mode="lookahead_reverse",
                        control_tokens=ctrl,
                        write_mask=None,
                        decay_weights=decay  # Pass calculated decay
                    )

                    # Update the local state dict for the next layer
                    if updated_rev is not None:
                        g_id, mem = updated_rev
                        current_M_rev_std[g_id] = mem
            return current_M_rev_std

        except Exception as e:
            print(f"Error in _run_lookahead_reverse_pass: {type(e).__name__}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise

    def _run_forward_pass(self, all_chunks: List[List[int]], M_rev_std_computed: Dict[int, Tensor], B: int,
                          dev: torch.device) -> Optional[torch.Tensor]:
        """ Runs the forward pass over all chunks, returning aggregated gate loss if applicable. """
        n_chunks = len(all_chunks)
        tokens_processed_before_cycle = self.total_tokens_processed
        tokens_processed_in_pass = 0

        M_fwd = self.M_fwd  # Will be updated in-place via layer returns

        # --- Handle Persistent Reverse Memory Masking (Training Only) ---
        M_rev_persist_to_use: Dict[int, Tensor]
        if self.training and self.config.enable_mask_future_dropout:
            # Calculate dropout probability based on schedule
            p_drop = get_mask_future_schedule(self.config, self.training_step, self.total_training_steps)
            print(f"  Forward Pass (Training): Applying mask-future dropout with p={p_drop:.3f}")
            # Create a masked copy for this pass
            M_rev_persist_to_use = self._mask_persistent_memory(self.M_rev_persist, p_drop)
        else:
            # Use the original persistent memory during inference or if disabled
            M_rev_persist_to_use = self.M_rev_persist
        # --- End Masking ---

        forward_gate_loss: Optional[torch.Tensor] = None

        for chunk_idx, chunk_tokens in enumerate(all_chunks):
            chunk_tensor = torch.tensor(chunk_tokens, dtype=torch.long, device=dev).unsqueeze(0)
            x = self.token_embedding(chunk_tensor)

            current_total_processed_before_chunk = tokens_processed_before_cycle + tokens_processed_in_pass

            current_mem_size = self.memory_manager.get_effective_size(current_total_processed_before_chunk)
            ctrl = self.control_token_generator.generate_control_tokens(
                mode="forward", current_chunk_idx=chunk_idx, total_chunks=n_chunks,
                current_mem_size=current_mem_size, max_mem_size=self.config.max_memory_size,
                seq_len=current_total_processed_before_chunk
            )

            for layer_idx, layer in enumerate(self.layers):
                group_idx = self.layer_idx_to_group_idx[layer_idx]
                group = self.layer_groups[group_idx]

                wmask = None
                if group.has_memory and layer_idx == group.memory_update_layer_idx:
                    wmask = self.memory_manager.get_write_mask(
                        current_chunk_idx_in_pass=chunk_idx,
                        total_chunks_in_pass=n_chunks,
                        total_tokens_processed_before_chunk=current_total_processed_before_chunk,
                        batch_size=B,
                        device=dev
                    )

                # --- Pass the potentially masked persistent memory ---
                x, updated_fwd, _, _gate_loss = layer(
                    x,
                    M_fwd_dict=M_fwd,
                    M_rev_std_dict=M_rev_std_computed,  # Lookahead reverse (computed in Pass 1)
                    M_rev_persist_dict=M_rev_persist_to_use,  # Persistent reverse (potentially masked)
                    group_id=group_idx, mode="forward", control_tokens=ctrl,
                    write_mask=wmask, decay_weights=None
                )

                if _gate_loss is not None:
                    current_loss = _gate_loss.mean() if _gate_loss.numel() > 1 else _gate_loss
                    if forward_gate_loss is None:
                        forward_gate_loss = current_loss
                    else:
                        forward_gate_loss += current_loss

                if updated_fwd is not None:
                    g_id, mem = updated_fwd
                    self.M_fwd[g_id] = mem

            tokens_processed_in_pass += len(chunk_tokens)

        return forward_gate_loss

    def _run_persistent_reverse_pass(self, all_chunks: List[List[int]], B: int, dev: torch.device):
        """ Runs the persistent reverse pass. """
        n_chunks = len(all_chunks)
        # Eligible chunks for persistent reverse are all except the last one
        eligible_chunks = all_chunks[:-1]
        if not eligible_chunks:
            print("  Skipping Persistent Reverse Pass: No eligible preceding chunks.")
            return  # Nothing to process

        # Determine the window of chunks to process based on reverse_max_chunks
        window_start_idx = max(0, len(eligible_chunks) - self.config.reverse_max_chunks)
        reverse_window_chunks = eligible_chunks[window_start_idx:]
        window_size = len(reverse_window_chunks)

        if not reverse_window_chunks:
            print("  Skipping Persistent Reverse Pass: Window is empty after eligibility check.")
            return

        # --- REMOVED MASK-FUTURE DROPOUT LOGIC FROM HERE ---
        # The dropout is now applied when *reading* this memory in the forward pass during training.

        # Pass state dictionaries (M_rev_persist will be updated directly)
        M_fwd = self.M_fwd  # Not used by layers in this mode
        M_rev_std = self.M_rev_std  # Not used by layers in this mode

        # Initialize M_rev_persist for this pass if it doesn't exist (should exist after init/reset)
        # Or reset it based on config? Let's reset it here to ensure clean state for the pass.
        # This assumes the pass *recomputes* M_rev_persist from initial state + window.
        # If M_rev_persist should accumulate across cycles (when reset_memory_on_cycle=False),
        # this reset needs to be conditional. For now, assume reset per cycle pass.
        current_M_rev_persist: Dict[int, Tensor] = {}
        for group in self.layer_groups:
            if group.has_memory:
                group_idx = group.group_idx;
                mem_idx = self.group_id_to_memory_idx[group_idx]
                # Start from the learned initial state for this pass
                current_M_rev_persist[group_idx] = self.initial_rev_p_params[mem_idx].clone().detach().to(
                    dev).repeat(B, 1, 1)

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
                current_chunk_idx=global_chunk_idx_in_all,  # Use index relative to all chunks
                total_chunks=n_chunks,  # Total chunks in the sequence
                current_mem_size=current_mem_size, max_mem_size=self.config.max_memory_size,
                seq_len=approx_processed_len,
                reverse_chunk_idx=reverse_chunk_idx,  # Index within the processing window
                reverse_window_size=window_size
            )

            for layer_idx, layer in enumerate(self.layers):
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
                        print(f"Warning: Group {group_idx} not found in current_M_rev_persist during decay calc.")

                x, _, updated_rev, _gate_loss = layer(
                    x,
                    M_fwd_dict=M_fwd,
                    M_rev_std_dict=M_rev_std,
                    M_rev_persist_dict=current_M_rev_persist,  # Pass dict holding state to update
                    group_id=group_idx,
                    mode="persistent_reverse",
                    control_tokens=ctrl,
                    write_mask=None,
                    decay_weights=decay
                )

                if updated_rev is not None:
                    g_id, mem = updated_rev
                    current_M_rev_persist[g_id] = mem  # Update the state for the next layer/chunk

        # --- After processing all chunks in the window, update the main state ---
        self.M_rev_persist = current_M_rev_persist

    def _mask_persistent_memory(self, memory_dict: Dict[int, Tensor], p_drop: float) -> Dict[int, Tensor]:
        """
        Applies mask-future dropout to a *copy* of the persistent memory dictionary.
        Zeros out rows along the memory dimension (M) with probability p_drop.
        Returns a new dictionary with detached, cloned, masked tensors.
        """
        if p_drop <= 0.0:
            return memory_dict  # No dropout needed, return original dict

        masked_memory_dict = {}
        for group_id, mem_tensor in memory_dict.items():
            if mem_tensor is None or mem_tensor.numel() == 0:
                masked_memory_dict[group_id] = mem_tensor  # Keep None or empty tensors as is
                continue

            B, M, D = mem_tensor.shape
            if M == 0:  # Skip if memory dimension is zero
                masked_memory_dict[group_id] = mem_tensor.clone().detach()
                continue

            # Create dropout mask along the memory dimension (M)
            # Keep = 1, Drop = 0. Probability of keeping is (1 - p_drop)
            keep_prob = 1.0 - p_drop
            # Shape (B, M, 1) to broadcast across the embedding dim D
            mask = torch.bernoulli(torch.full((B, M, 1), keep_prob, device=mem_tensor.device)).to(mem_tensor.dtype)

            # Apply mask (element-wise multiplication)
            # Clone and detach *before* masking to avoid modifying original and stop gradients
            masked_tensor = mem_tensor.clone().detach() * mask
            masked_memory_dict[group_id] = masked_tensor

        return masked_memory_dict

    @torch.no_grad()
    def generate(self, prompt: Union[str, List[int], List[List[int]], torch.Tensor, None] = None, *,
                 max_new_tokens: int = 128, temperature: float = 1.0, top_k: Optional[int] = None,
                 stop_token_id: Optional[int] = None, reset_state: bool = False) -> List[int]:
        """ Autoregressive decoding. """
        if reset_state: print("Resetting model state for generation."); self.reset_state()
        self._initialize_memory_states(force_reset=False)

        def _sample_row(row: torch.Tensor) -> int:
            row = row.float()
            if top_k is not None and top_k > 0:
                actual_k = min(top_k, row.shape[-1]);
                if actual_k > 0: thresh = torch.topk(row, actual_k).values[-1]; row = torch.where(row < thresh,
                                                                                                  torch.full_like(row,
                                                                                                                  float(
                                                                                                                      "-inf")),
                                                                                                  row)
            temp = max(float(temperature), 1e-5);
            probs = torch.nn.functional.softmax(row / temp, dim=-1)
            return torch.multinomial(probs, 1).item()

        next_input = prompt
        generated: List[int] = []
        if prompt not in (None, "", [], torch.tensor([], dtype=torch.long)): _ = self.forward(prompt,
                                                                                              training_mode=False)
        try:
            self.current_chunk_text = self.tokenizer.decode(self.current_chunk_tokens)
        except Exception as e:
            print(f"Warning: Error decoding initial buffer tokens: {e}");
            self.current_chunk_text = ""
        next_input = None

        for _ in range(max_new_tokens):
            logits, _ = self.forward(next_input, training_mode=False)
            if logits.shape[1] == 0: print("INFO: Generation stopped - empty logits."); break
            next_id = _sample_row(logits[0, -1, :])
            if stop_token_id is not None and next_id == stop_token_id: break
            generated.append(next_id)
            try:
                self.current_chunk_text += self.tokenizer.decode([next_id])
            except Exception as e:
                print(f"Warning: Error decoding token {next_id}: {e}")

            self.tokens_since_persistent_update += 1
            update_persist = False
            if self.config.persistent_reverse_update_freq_tokens is not None and self.tokens_since_persistent_update >= self.config.persistent_reverse_update_freq_tokens: update_persist = True
            if not update_persist and self.config.persistent_reverse_update_freq_semantic and len(
                    self.current_chunk_text) > 1:
                try:
                    boundary_level = self.config.persistent_reverse_update_freq_semantic
                    if boundary_level in self.config.boundary_types:
                        search_window = self.current_chunk_text[-10:]
                        for boundary_type in self.config.boundary_types[boundary_level]:
                            pattern = BOUNDARY_PATTERNS.get(boundary_type)
                            if pattern and re.search(pattern, search_window): update_persist = True; break
                except Exception as e:
                    print(f"Warning: Error during semantic boundary check: {e}")

            if update_persist:
                print("Triggering periodic persistent reverse update during generation.")
                # Need to run the persistent reverse pass using current state
                # Combine history + current buffer content
                # Note: self.current_chunk_tokens is updated *after* forward returns the logit used
                # to generate next_id, but *before* next_id is added to generated list.
                # So, it represents the state *before* the latest token.
                full_chunks_for_persist = self.closed_chunks + (
                    [self.current_chunk_tokens] if self.current_chunk_tokens else [])
                if full_chunks_for_persist:  # Only run if there's history
                    self._run_persistent_reverse_pass(full_chunks_for_persist, B=1,
                                                      dev=self.token_embedding.weight.device)
                else:
                    print("Skipping periodic update: No history chunks available.")
                self.tokens_since_persistent_update = 0

            next_input = [next_id]
        return generated

    def reset_state(self) -> None:
        """Clear all sequence-level state and reset memory dictionaries."""
        print("Resetting CMAModel state.")
        self.closed_chunks = []
        self.current_chunk_tokens = []
        self.current_chunk_text = ""
        self.total_tokens_processed = 0
        self.tokens_since_persistent_update = 0
        # Clear runtime memory state dictionaries
        self.M_fwd = {}
        self.M_rev_std = {}
        self.M_rev_persist = {}
        # Re-initialize from learned parameters on next use by calling _initialize_memory_states
        # Or force re-initialization here? Let's rely on _initialize_memory_states in forward/generate.

    def set_training_step(self, step: int, total_steps: int):
        """Set current training step for mask-future scheduling"""
        self.training_step = step
        self.total_training_steps = total_steps
