import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict, Any
import torch
import torch.nn.functional as F
import yaml
from torch import Tensor


def print0(s, logfile):
    #print(s, flush=True)
    if logfile:
        with open(logfile, "a") as f:
            print(s, file=f, flush=True)

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
    memory_init_scale: float = 0.01
    gate_bias_init: float = 1.0
    output_proj_zero_init: bool = True

    # Adaptive gating regularization
    gate_regularization_type: Optional[str] = "l1"  # None, "l1", or "entropy"
    gate_regularization_strength: float = 0.001

    # Future‐masking schedule: progress breakpoints and rates
    mask_future_schedule: List[float] = field(default_factory=lambda: [0.3, 0.7])
    mask_future_rates: List[float] = field(default_factory=lambda: [0.333, 0.667, 1.0])
    enable_mask_future_dropout: bool = True

    DEBUG_LEVEL: int = 0
    logfile: str = None

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
            print0("Warning: No layer_structure provided. Using default alternating structure.", self.config.logfile)

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
        assert isinstance(self.mask_future_schedule, list) and len(self.mask_future_schedule) == 2
        assert isinstance(self.mask_future_rates, list) and len(self.mask_future_rates) == 3
        assert 0 <= self.DEBUG_LEVEL <= 2

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

        initial_tokens = self.tokenizer.encode(text)
        gap_percentage = self.config.semantic_chunking_gap_percentage / 100.0
        target_last_size = int(self.config.chunk_size * (1 - gap_percentage))
        target_last_size = max(1, target_last_size)

        if len(initial_tokens) <= target_last_size:
             return [initial_tokens]

        chunks = []
        end_pos = len(text)
        # Keep track of the last successful start_pos to prevent infinite loops if fallback keeps failing
        last_successful_start_pos = -1

        while end_pos > 0:
            if end_pos == len(text):
                current_target_size = target_last_size
            else:
                current_target_size = self.config.chunk_size

            # Heuristic estimation - use character count relative to target token count
            # Assume avg 1.5 chars/token as a rough middle ground? Or stick to *2? Let's try 1.5
            est_start = max(0, end_pos - int(current_target_size * 1.5))

            # --- Check for simple termination case ---
            # If the remaining text fits the target size, just take it.
            # Avoids unnecessary boundary search/overshoot loops at the very end.
            remaining_text = text[:end_pos]
            remaining_tokens = self.tokenizer.encode(remaining_text)
            if len(remaining_tokens) <= current_target_size:
                if remaining_tokens: # Avoid adding empty chunk if text was empty somehow
                    chunks.insert(0, remaining_tokens)
                end_pos = 0 # We are done
                continue # Exit while loop

            # --- End simple termination check ---

            found_valid_chunk = False
            iteration = 0
            max_iterations = 10 # Limit inner loop retries
            current_est_start = est_start # Use this for inner loop adjustments

            # Prevent fallback loop: if fallback used the same start_pos last time, force progress
            if est_start == last_successful_start_pos:
                 if self.config.DEBUG_LEVEL > 0: print0(f"DEBUG: semantic_chunk - Fallback might loop. Forcing progress from end_pos {end_pos}", self.config.logfile)
                 start_pos = max(0, end_pos - 1) # Take at least one char
                 chunk_text = text[start_pos:end_pos]
                 if chunk_text: # Ensure text exists
                     chunk_tokens = self.tokenizer.encode(chunk_text)
                     # Truncate if needed (highly unlikely here, but safe)
                     if len(chunk_tokens) > current_target_size:
                         chunk_tokens = chunk_tokens[:current_target_size]
                     if chunk_tokens: # Final check if tokens exist
                        chunks.insert(0, chunk_tokens)
                 else: # If text[end_pos-1:end_pos] is empty (e.g. multi-byte char issue?)
                     print0(f"WARN: semantic_chunk - Forced progress resulted in empty text slice.", self.config.logfile)
                 end_pos = start_pos # Ensure outer loop makes progress
                 last_successful_start_pos = start_pos # Update last pos
                 continue # Skip normal inner loop/fallback


            while not found_valid_chunk and iteration < max_iterations:
                # Add buffer for search
                buffer_chars = int(self.config.boundary_search_chars[0] * self.config.buffer_ratio)
                search_start_point = min(end_pos, max(0, current_est_start + buffer_chars))

                start_pos = self._find_semantic_boundary_backward(text, search_start_point, end_pos)

                # --- Prevent start_pos from exceeding end_pos ---
                start_pos = min(start_pos, end_pos)

                chunk_text = text[start_pos:end_pos]

                if not chunk_text:
                    # Empty slice usually means start_pos >= end_pos
                    if start_pos == 0 and end_pos <=0 : # Check if we are truly at the beginning and done
                         if self.config.DEBUG_LEVEL > 0: print0("DEBUG: semantic_chunk - Empty chunk text at start_pos 0, terminating.", self.config.logfile)
                         end_pos = 0
                         found_valid_chunk = True # Exit inner loop cleanly
                         continue # Go to outer loop check (will terminate)
                    else:
                        # Treat as overshoot signal if slice is empty unexpectedly mid-sequence
                        if self.config.DEBUG_LEVEL > 0: print0(f"DEBUG: semantic_chunk - Empty chunk text for start_pos {start_pos}, end_pos {end_pos}. Adjusting estimate.", self.config.logfile)
                        # Need to move current_est_start *back* slightly to try and get content
                        current_est_start = max(0, start_pos - 10) # Move back 10 chars arbitrarily
                        iteration += 1
                        continue # Retry inner loop

                chunk_tokens = self.tokenizer.encode(chunk_text)

                # If encoding results in empty tokens (e.g., only whitespace removed by tokenizer)
                if not chunk_tokens:
                     if self.config.DEBUG_LEVEL > 0: print0(f"DEBUG: semantic_chunk - Encoded tokens empty for text slice [{start_pos}:{end_pos}]. Adjusting estimate.", self.config.logfile)
                     current_est_start = max(0, start_pos - 10)
                     iteration += 1
                     continue # Retry inner loop

                # Check size limit
                max_allowed_size = current_target_size

                if len(chunk_tokens) <= max_allowed_size:
                    found_valid_chunk = True
                    chunks.insert(0, chunk_tokens)
                    end_pos = start_pos
                    last_successful_start_pos = start_pos # Record successful position
                else:
                    # Overshoot logic
                    excess_tokens = len(chunk_tokens) - max_allowed_size
                    chars_per_token = len(chunk_text) / len(chunk_tokens)
                    # Adjust estimate forward based on the *current* failed start_pos
                    current_est_start = start_pos + int(excess_tokens * chars_per_token * 1.1) # Reduced buffer slightly
                    iteration += 1
                    if self.config.DEBUG_LEVEL > 0: print0(f"DEBUG: semantic_chunk - Overshot size ({len(chunk_tokens)} > {max_allowed_size}), retrying with est_start {current_est_start}", self.config.logfile)


            # Fallback if inner loop exhausted iterations
            if not found_valid_chunk and end_pos > 0:
                if self.config.DEBUG_LEVEL > 0: print0(f"DEBUG: semantic_chunk - Fallback after {max_iterations} iterations for end_pos {end_pos}", self.config.logfile)
                # Use the original estimate for this outer loop pass
                start_pos = est_start
                # --- Prevent start_pos from exceeding end_pos in fallback ---
                start_pos = min(start_pos, end_pos)

                chunk_text = text[start_pos:end_pos]

                if chunk_text:
                    chunk_tokens = self.tokenizer.encode(chunk_text)
                    max_allowed_size = current_target_size
                    if len(chunk_tokens) > max_allowed_size:
                        if self.config.DEBUG_LEVEL > 0: print0(f"DEBUG: semantic_chunk - Truncating fallback chunk ({len(chunk_tokens)} > {max_allowed_size})", self.config.logfile)
                        chunk_tokens = chunk_tokens[:max_allowed_size]

                    if chunk_tokens:
                        chunks.insert(0, chunk_tokens)
                    else:
                         print0(f"WARN: semantic_chunk - Fallback resulted in empty tokens for slice [{start_pos}:{end_pos}]", self.config.logfile)
                else:
                     print0(f"WARN: semantic_chunk - Fallback resulted in empty text slice [{start_pos}:{end_pos}]", self.config.logfile)


                # Update end_pos based on the start_pos used for the fallback chunk
                end_pos = start_pos
                last_successful_start_pos = start_pos # Record fallback position

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

    def get_effective_size(self, total_tokens_processed: int, current_chunk_length: int = 0) -> int:
        seq_len_for_calc = total_tokens_processed

        # If total_tokens_processed is 0, let effective_size be based on initial_write_fraction directly.
        if total_tokens_processed == 0 and current_chunk_length > 0:
            #seq_len_for_calc = max(1, int(self.config.memory_cap_length * 0.01))  # Allow 1% growth from start
            seq_len_for_calc = max(1, int(self.config.chunk_size * self.config.initial_write_fraction))

        if self.config.memory_growth_function == "linear":
            cap_length = max(1, self.config.memory_cap_length)
            fraction = min(1.0, seq_len_for_calc / cap_length)
        elif self.config.memory_growth_function == "log":
            if seq_len_for_calc <= 1:
                fraction = 0.0
            else:
                cap_length = max(2, self.config.memory_cap_length)
                fraction = math.log(seq_len_for_calc) / math.log(cap_length)
                fraction = min(1.0, max(0.0, fraction))
        else:
            raise ValueError(f"Unknown growth function: {self.config.memory_growth_function}")

        effective_size = max(0, round(self.config.max_memory_size * fraction))

        if total_tokens_processed == 0 and current_chunk_length > 0:
            min_size_for_initial_write = round(self.config.max_memory_size * self.config.initial_write_fraction)
            effective_size = max(effective_size, min_size_for_initial_write)
            effective_size = min(effective_size, self.config.max_memory_size)

        return effective_size

    def get_write_mask(
            self,
            current_chunk_idx_in_pass: int,
            total_chunks_in_pass: int,
            total_tokens_processed_before_chunk: int,
            current_chunk_len: int,
            batch_size: int = 1,
            device: torch.device = torch.device('cpu')
    ) -> Tensor:
        # Effective size should consider up to the end of the current chunk for growth
        # seq_write_cap = self.get_effective_size(total_tokens_processed_before_chunk + current_chunk_len)
        # OR, more simply, ensure get_effective_size doesn't return 0 if current_chunk_len > 0
        seq_write_cap = self.get_effective_size(total_tokens_processed_before_chunk,
                                                current_chunk_length=current_chunk_len)

        if total_chunks_in_pass > 0:
            chunk_progress = (current_chunk_idx_in_pass + 1) / total_chunks_in_pass
            initial_fraction = self.config.initial_write_fraction
            write_fraction = initial_fraction + (1.0 - initial_fraction) * chunk_progress
        else:
            write_fraction = self.config.initial_write_fraction

        target_writable_size = int(self.config.max_memory_size * write_fraction)
        writable_size = min(seq_write_cap, target_writable_size)
        writable_size = max(0, writable_size)  # Should be at least 0

        # If total_tokens_processed_before_chunk is 0 and current_chunk_len > 0,
        # ensure at least some minimal write if initial_write_fraction > 0
        if total_tokens_processed_before_chunk == 0 and current_chunk_len > 0 and self.config.initial_write_fraction > 0:
            min_write_for_first_chunk = max(1, round(
                self.config.max_memory_size * self.config.initial_write_fraction * 0.1))  # e.g. 10% of initial fraction
            # or just 1 token?
            # writable_size = max(writable_size, min_write_for_first_chunk) # This might override seq_write_cap logic
            # The fix in get_effective_size using current_chunk_length should be primary.

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
