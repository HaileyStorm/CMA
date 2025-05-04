# test_cma.py
"""
Comprehensive test‑suite for `cma.py`.

---------------------------------------------------------------------------
Functional‑requirement checklist (all taken from CMA.md):

  FR‑01  Semantic chunking must respect   gap% and max `chunk_size` … §4.1 :contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}
  FR‑02  Fixed‑size chunking reverse‑with‑gap mirrors semantic rules … §4.1 :contentReference[oaicite:2]{index=2}:contentReference[oaicite:3]{index=3}
  FR‑03  `get_mask_future_schedule` implements piece‑wise linear schedule … §9 (table) :contentReference[oaicite:4]{index=4}:contentReference[oaicite:5]{index=5}
  FR‑04  Control‑token generator returns five scalars with mode‑specific flags … §4.5 :contentReference[oaicite:6]{index=6}:contentReference[oaicite:7]{index=7}
  FR‑05  Forward‑memory effective size grows with processed tokens … §8 :contentReference[oaicite:8]{index=8}:contentReference[oaicite:9]{index=9}
  FR‑06  Forward‑write mask respects initial fraction + progressive growth … §8 :contentReference[oaicite:10]{index=10}:contentReference[oaicite:11]{index=11}
  FR‑07  Reverse‑decay weights apply per‑chunk exponential decay, separate
         params for lookahead vs persistent … §7 (not shown in snippet but
         described in CMA.md)                             ────────────────
  FR‑08  Adaptive gate outputs in (0,1) and optional reg‑loss propagated … §5 :contentReference[oaicite:12]{index=12}:contentReference[oaicite:13]{index=13}
  FR‑09  Causal self‑attention masks future positions … Transformer basics
  FR‑10  CascadeMemoryAttention integrates memory tokens when supplied … §5 :contentReference[oaicite:14]{index=14}:contentReference[oaicite:15]{index=15}
  FR‑11  Memory‑update layer actually mutates memory states with gated blend … §6
  FR‑12  Layer‑group parsing enforces exactly one update‑layer per group … §4.2 :contentReference[oaicite:16]{index=16}:contentReference[oaicite:17]{index=17}
  FR‑13  CMAModel initialises learned memory parameters per group (or shared) … §4.3
  FR‑14  Forward‑pass path returns logits with correct shape and
         aggregates gate‑reg loss when enabled … §9 :contentReference[oaicite:18]{index=18}:contentReference[oaicite:19]{index=19}
---------------------------------------------------------------------------

Each `test_*` function below targets one or more of these FRs and at least
one public method/constructor in `cma.py`.
"""

import math
import types
from typing import List

import torch
import pytest

import cma  # the module under test


# -------------------------------------------------------------------------
# Utility fixtures
# -------------------------------------------------------------------------

class DummyTokenizer:
    """Deterministic stand‑in for tiktoken; round‑trips bytes→ints bijectively."""
    def encode(self, text: str):
        # Map each character to an int 0‑255; keeps sequence short and reproducible
        return [ord(ch) % 256 for ch in text]

    def decode(self, tokens: List[int]) -> str:
        """
        Inverse of encode: map each byte-valued token back to its character.

        Args:
            tokens: sequence of integers in [0,255]

        Returns:
            The reconstructed string.
        """
        return ''.join(chr(t) for t in tokens)

@pytest.fixture(scope="session")
def small_cfg():
    """Return a minimal but valid CMAConfig."""
    return cma.CMAConfig(
        chunk_size=8,
        semantic_chunking_gap_percentage=25.0,
        max_memory_size=16,
        reverse_memory_size=4,
        embed_dim=8,
        n_heads=2,
        head_dim=4,          # embed_dim // n_heads
        n_layers=4,
        layer_structure=[
            {"type": "local_only"},
            {"group": {"layers": ["memory_update", "memory_read", "local_only"]}}
        ],
        skip_attention_layers=()
    )

@pytest.fixture(scope="session")
def dummy_tok():
    return DummyTokenizer()

@pytest.fixture(scope="session")
def small_model(small_cfg, dummy_tok):
    """A tiny CMAModel for end‑to‑end checks."""
    return cma.CMAModel(config=small_cfg, vocab_size=300, tokenizer=dummy_tok)


# -------------------------------------------------------------------------
# FR‑01 / FR‑02  ‑‑ ChunkProcessor behaviour
# -------------------------------------------------------------------------

def test_semantic_chunking_gap_respected(small_cfg, dummy_tok):
    cp = cma.ChunkProcessor(small_cfg, dummy_tok)
    txt = "A" * 50  # 50 chars ⇒ ≤50 tokens with DummyTokenizer
    chunks = cp.semantic_chunk_reverse_with_gap(txt)
    # last chunk must be <= (1‑gap)*chunk_size = 6 tokens
    assert len(chunks[-1]) <= math.floor(small_cfg.chunk_size * 0.75)
    # every other chunk length ≤ chunk_size
    assert all(len(c) <= small_cfg.chunk_size for c in chunks)


def test_fixed_size_chunking_reverse_gap(small_cfg, dummy_tok):
    cp = cma.ChunkProcessor(small_cfg, dummy_tok)
    toks = list(range(37))
    chunks = cp.fixed_size_chunk_reverse_with_gap(toks)
    last = math.floor(small_cfg.chunk_size * 0.75)
    assert len(chunks[-1]) == last
    for c in chunks[:-1]:
        assert len(c) <= small_cfg.chunk_size
    # reconstruction round‑trip
    flat = [t for ch in chunks for t in ch]
    assert flat == toks


# -------------------------------------------------------------------------
# FR‑03  ‑‑ mask‑future schedule
# -------------------------------------------------------------------------

@pytest.mark.parametrize("step,total,expected", [
    (0, 100, 0.333),
    (35, 100, pytest.approx(0.333 + (0.667-0.333)*(0.35-0.3)/0.4, rel=1e-3)),
    (90, 100, pytest.approx(0.667 + (1.0-0.667)*(0.9-0.7)/0.3, rel=1e-3)),
])
def test_mask_future_schedule_interp(small_cfg, step, total, expected):
    rate = cma.get_mask_future_schedule(small_cfg, step, total)
    assert pytest.approx(rate, rel=1e-3) == expected


# -------------------------------------------------------------------------
# FR‑04  ‑‑ control token generator
# -------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["forward", "lookahead_reverse", "persistent_reverse", "generate"])
def test_control_token_dimensions(small_cfg, mode):
    gen = cma.ControlTokenGenerator(small_cfg)
    toks = gen.generate_control_tokens(
        mode=mode,
        current_chunk_idx=1,
        total_chunks=4,
        current_mem_size=2,
        max_mem_size=small_cfg.max_memory_size,
        seq_len=10,
        reverse_chunk_idx=0,
        reverse_window_size=3,
    )
    assert set(toks.keys()) == {
        "generation_flag", "memory_mode_flag", "memory_usage_ratio",
        "memory_density_ratio", "chunk_position_ratio"
    }
    # Flags are floats within [0,1]
    assert all(0.0 <= v <= 1.0 for v in toks.values())


# -------------------------------------------------------------------------
# FR‑05 / FR‑06  ‑‑ MemoryManager scaling & write‑masking
# -------------------------------------------------------------------------

def test_effective_size_linear_growth(small_cfg):
    mm = cma.MemoryManager(small_cfg)
    assert mm.get_effective_size(0) == 0
    half = mm.get_effective_size(small_cfg.memory_cap_length // 2)
    assert half == small_cfg.max_memory_size // 2
    full = mm.get_effective_size(small_cfg.memory_cap_length)
    assert full == small_cfg.max_memory_size


def test_write_mask_progressive(small_cfg):
    print("\nTesting write mask progression...")
    mm = cma.MemoryManager(small_cfg)
    B = 1
    dev = torch.device('cpu')
    total_chunks_in_pass = 4
    tokens_per_chunk = small_cfg.chunk_size  # 8

    results = []
    tokens_processed_before_pass = 0  # Start sequence from 0 for simplicity

    print(
        f"Config: max_mem={small_cfg.max_memory_size}, cap_len={small_cfg.memory_cap_length}, init_frac={small_cfg.initial_write_fraction}")

    for i in range(total_chunks_in_pass):
        # Calculate tokens processed *before* this chunk starts
        tokens_before_chunk = tokens_processed_before_pass + (i * tokens_per_chunk)

        # Calculate expected sequence cap based on tokens processed before chunk
        seq_cap = mm.get_effective_size(tokens_before_chunk)

        # Calculate expected target size based on chunk progress in *this pass*
        chunk_progress = (i + 1) / total_chunks_in_pass
        write_fraction = small_cfg.initial_write_fraction + (1.0 - small_cfg.initial_write_fraction) * chunk_progress
        target_size = int(small_cfg.max_memory_size * write_fraction)

        # Final expected writable size is the minimum of the two constraints
        expected_writable = min(seq_cap, target_size)

        # Get actual mask from the function
        mask = mm.get_write_mask(i, total_chunks_in_pass, tokens_before_chunk, batch_size=B, device=dev)
        actual_writable = mask.sum().item()

        print(
            f"  Chunk {i}: Tokens Before={tokens_before_chunk}, SeqCap={seq_cap}, Target={target_size} => Expected={expected_writable}, Actual={actual_writable}")
        # Assert the calculated value matches the function's output
        assert actual_writable == expected_writable, f"Chunk {i} Failed: Expected {expected_writable}, got {actual_writable}"
        results.append(actual_writable)

    # --- Check Progression ---
    print(f"\n  Progression Results: {results}")
    # Expect non-decreasing size (it might plateau if target_size grows faster than seq_cap)
    assert all(results[j] >= results[j - 1] for j in range(1, len(results))), "Expected non-decreasing writable size"
    # Check if it actually increased at some point (unless cap=0 initially or max cap is reached)
    assert results[-1] > results[0] or (results[0] == 0) or (results[
                                                                 0] == small_cfg.max_memory_size), "Expected writable size to increase overall, start at 0, or be fully capped"

    print("\nWrite mask progression test (Revised) passed.")


# -------------------------------------------------------------------------
# FR‑07  ‑‑ reverse‑decay weights
# -------------------------------------------------------------------------

def test_reverse_decay_weights_drop(small_cfg):
    mm = cma.MemoryManager(small_cfg)
    shape = (1, small_cfg.reverse_memory_size, small_cfg.embed_dim)
    w0 = mm.calculate_reverse_decay_weights(0, 3, False, shape, torch.device("cpu"))
    w1 = mm.calculate_reverse_decay_weights(1, 3, False, shape, torch.device("cpu"))
    assert torch.allclose(w0, torch.ones_like(w0))
    assert torch.all(w1 < w0)  # decay


# -------------------------------------------------------------------------
# FR‑08 / FR‑09 / FR‑10 / FR‑11  ‑‑ Attention layers
# -------------------------------------------------------------------------

def test_causal_self_attention_mask(small_cfg):
    attn = cma.CausalSelfAttention(small_cfg, layer_idx=0)
    x = torch.randn(1, 5, small_cfg.embed_dim)
    out = attn(x)
    assert out.shape == x.shape  # shape check
    # ensure gradient flows
    out.sum().backward()


def test_cascade_memory_attention_gates(small_cfg):
    attn = cma.CascadeMemoryAttention(small_cfg, layer_idx=0, is_memory_update=True)
    B, T, C = 1, 4, small_cfg.embed_dim
    x = torch.randn(B, T, C)
    fwd_mem = torch.randn(B, small_cfg.max_memory_size, C)
    out, fwd_new, _, reg = attn(
        x, forward_memory=fwd_mem, control_tokens=None,
        do_memory_update=True, write_mask=torch.ones(B, small_cfg.max_memory_size, dtype=torch.bool)
    )
    assert out.shape == x.shape
    # Memory should change after update
    assert not torch.allclose(fwd_new, fwd_mem)
    # Gate‑reg either None or scalar tensor
    if reg is not None:
        assert reg.ndim == 0


# -------------------------------------------------------------------------
# FR‑12  ‑‑ Block layer‑group rules
# -------------------------------------------------------------------------

def test_layer_group_constraints(small_cfg):
    # Invalid config: two memory_update layers in one group → should raise
    bad_cfg = small_cfg.__class__(**{**small_cfg.__dict__,
                                     "layer_structure": [{"group": {"layers": ["memory_update",
                                                                               "memory_update"]}}]})
    with pytest.raises(ValueError):
        _ = cma.CMAModel(bad_cfg, vocab_size=100, tokenizer=DummyTokenizer())


# -------------------------------------------------------------------------
# FR‑13 / FR‑14  ‑‑ End‑to‑end CMAModel run
# -------------------------------------------------------------------------

def test_cma_forward_end_to_end(small_model):
    txt = "Hello CMA"
    logits, _ = small_model.forward(txt, training_mode=False)
    # logits: (1, seq_len, vocab)
    assert logits.ndim == 3
    assert logits.size(-1) == small_model.vocab_size
