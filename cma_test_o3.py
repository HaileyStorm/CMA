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
    short_out = model("Hello world!")  # string input
    assert short_out.shape == (1, 3, VOCAB)

    ids = [100] * 42
    list_out = model(ids)  # list[int] input
    assert list_out.shape == (1, 42, VOCAB), f"{list_out.shape}"

    # fill a whole chunk to trigger the full memory-update cycle
    long_ids = [123] * cfg.chunk_size
    _ = model(long_ids)
    assert len(model.current_chunk_tokens) == 0, "chunk flush failed"

    # ---------- 2. generation --------------------------------------------------
    gen = model.generate("Once upon a time", max_new_tokens=8)
    assert len(gen) == 8

    # ---------- 3. schedule helper --------------------------------------------
    r0 = get_mask_future_schedule(0, 100, cfg)
    r_last = get_mask_future_schedule(100, 100, cfg)
    assert math.isclose(r0, cfg.mask_future_rates[0])
    assert math.isclose(r_last, cfg.mask_future_rates[-1])

    # ---------- 4. chunking + memory utils ------------------------------------
    tok = model.tokenizer
    proc = ChunkProcessor(cfg, tok)
    chunks = proc.semantic_chunk_reverse_with_gap("One. Two. Three.")
    assert sum(len(c) for c in chunks) == len(tok.encode("One. Two. Three."))

    mem = MemoryManager(cfg)
    wmask = mem.get_write_mask(0, 1, 500)
    assert wmask.dtype is torch.bool and wmask.shape == (1, cfg.max_memory_size)

    # ---------- 5. bare attention layers --------------------------------------
    dummy = torch.randn(2, 16, cfg.embed_dim)
    causal = CausalSelfAttention(cfg, layer_idx=0)
    assert causal(dummy).shape == dummy.shape

    cascade = CascadeMemoryAttention(cfg, layer_idx=1)
    out, *_ = cascade(dummy, forward_memory=None, reverse_memory=None)
    assert out.shape == dummy.shape

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

    print("All CMA sanity tests passed ✔")


if __name__ == "__main__":
    main()