import math
import torch

from cma import (
    CMAConfig,
    CMAModel,
    ChunkProcessor,
    MemoryManager,
    ControlTokenGenerator,
    CausalSelfAttention,
    CascadeMemoryAttention,
    get_mask_future_schedule,
)

VOCAB = 50257  # whatever – just needs to be > any token id we pass


def main() -> None:
    cfg = CMAConfig()
    model = CMAModel(cfg, VOCAB)
    model.eval()  # we’re not updating weights

    # ---------- 1. simple forward paths ---------------------------------------
    short_out, _ = model("Hello world!")  # string input
    assert short_out.shape == (1, 3, VOCAB)
    model.reset_state()
    print("String input test passed")

    ids = [100] * 42
    list_out, _ = model(ids)  # list[int] input
    assert list_out.shape == (1, 42, VOCAB), f"{list_out.shape}"
    model.reset_state()
    print("List[int] input test passed")

    # ---------- Test chunk flush and re-chunking -----------------------------
    #print("\nTesting chunk flush and re-chunking...")
    # Fill exactly one chunk to trigger the full memory-update cycle
    long_ids = [123] * cfg.chunk_size
    #print(f"Input: List of {len(long_ids)} tokens (exactly chunk_size)")
    _, _ = model(long_ids)  # Process the input
    # Calculate expected sizes after reverse-with-gap chunking
    expected_current_len = math.floor(cfg.chunk_size * (1 - cfg.semantic_chunking_gap_percentage / 100.0))
    expected_current_len = max(1, expected_current_len)  # Ensure at least 1 token
    expected_closed_len = cfg.chunk_size - expected_current_len
    #print(f"Expected state after cycle:")
    #print(f"  - current_chunk_tokens length: {expected_current_len}")
    #print(f"  - closed_chunks count: 1")
    #print(f"  - closed_chunks[0] length: {expected_closed_len}")
    #print(f"  - total_tokens_processed: {cfg.chunk_size}")
    # Assert the state reflects the re-chunking
    assert len(model.current_chunk_tokens) == expected_current_len, \
        f"Current chunk size mismatch. Expected {expected_current_len}, got {len(model.current_chunk_tokens)}"
    assert len(model.closed_chunks) == 1, \
        f"Expected 1 closed chunk, got {len(model.closed_chunks)}"
    assert len(model.closed_chunks[0]) == expected_closed_len, \
        f"Closed chunk size mismatch. Expected {expected_closed_len}, got {len(model.closed_chunks[0])}"
    assert model.total_tokens_processed == cfg.chunk_size, \
        f"Total tokens processed mismatch. Expected {cfg.chunk_size}, got {model.total_tokens_processed}"
    model.reset_state()  # Clean up after test
    print("Chunk flush and re-chunking test passed.")

    # ---------- 2. generation --------------------------------------------------
    gen = model.generate("Once upon a time", max_new_tokens=8)
    assert len(gen) == 8
    model.reset_state()
    print("Generate test passed")

    # ---------- 3. schedule helper --------------------------------------------
    r0 = get_mask_future_schedule(cfg, 0, 100)
    r_last = get_mask_future_schedule(cfg, 100, 100)
    assert math.isclose(r0, cfg.mask_future_rates[0])
    assert math.isclose(r_last, cfg.mask_future_rates[-1])
    print("get_mask_future_schedule test passed")

    # ---------- 4. chunking + memory utils ------------------------------------
    tok = model.tokenizer
    proc = ChunkProcessor(cfg, tok)
    chunks = proc.semantic_chunk_reverse_with_gap("One. Two. Three.")
    assert sum(len(c) for c in chunks) == len(tok.encode("One. Two. Three."))
    print("Semantic chunk processing test passed.")

    mem = MemoryManager(cfg)
    wmask = mem.get_write_mask(0, 1, 500)
    assert wmask.dtype is torch.bool and wmask.shape == (1, cfg.max_memory_size)
    print("get_write_mask test passed")

    # ---------- 5. bare attention layers --------------------------------------
    dummy = torch.randn(2, 16, cfg.embed_dim)
    causal = CausalSelfAttention(cfg, layer_idx=0)
    assert causal(dummy).shape == dummy.shape
    print("CausalSelfAttention test passed.")

    cascade = CascadeMemoryAttention(cfg, layer_idx=1)
    out, *_ = cascade(dummy, forward_memory=None, reverse_memory=None)
    assert out.shape == dummy.shape
    print("CascadeMemoryAttention test passed")

    # ---------- 6. control-token generator ------------------------------------
    ctok = ControlTokenGenerator(cfg).generate_control_tokens(
        mode="forward",
        current_chunk_idx=0,
        total_chunks=1,
        current_mem_size=128,
        max_mem_size=cfg.max_memory_size,
        seq_len=64,
    )
    assert set(ctok) == {
        "generation_flag",
        "memory_mode_flag",
        "memory_usage_ratio",
        "memory_density_ratio",
        "chunk_position_ratio",
    }
    print("generate_control_tokens test passed.")

    print("All CMA sanity tests passed ✔")


if __name__ == "__main__":
    main()