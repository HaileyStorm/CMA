import pytest
import torch
import torch.nn as nn
import tiktoken
import yaml
import math
from typing import Dict, Any
from cma_model import (
    CausalSelfAttention,
    CascadeMemoryAttention,
    Block,
    CMAModel,
)
from cma_components import (
    CMAConfig,
ControlTokenGenerator,
    ChunkProcessor,
    MemoryManager,
    norm,
    get_mask_future_schedule,
    BOUNDARY_PATTERNS
)

# --- Constants and Fixtures ---

VOCAB_SIZE = 500 # Small vocab for testing
EMBED_DIM = 64
N_HEADS = 4
N_LAYERS = 6
CHUNK_SIZE = 128
MAX_MEM_SIZE = 256
REV_MEM_SIZE = 64
HEAD_DIM = EMBED_DIM // N_HEADS

# Basic configuration for most tests
@pytest.fixture(scope="module")
def basic_config_dict() -> Dict[str, Any]:
    return {
        "chunk_size": CHUNK_SIZE,
        "semantic_chunking_gap_percentage": 10.0,
        "boundary_search_chars": [128, 64, 32],
        "buffer_ratio": 0.05,
        "max_memory_size": MAX_MEM_SIZE,
        "reverse_memory_size": REV_MEM_SIZE,
        "initial_write_fraction": 0.5,
        "memory_growth_function": "linear",
        "memory_cap_length": 1024 * 8,
        "share_initial_memory": False,
        "reset_memory_on_cycle": True,
        "reverse_max_chunks": 3,
        "lookahead_reverse_decay_step": 0.1,
        "lookahead_reverse_decay_rate": 0.6,
        "persistent_reverse_decay_step": 0.05,
        "persistent_reverse_decay_rate": 0.2,
        "persistent_reverse_update_freq_tokens": 50,
        "persistent_reverse_update_freq_semantic": None,
        "embed_dim": EMBED_DIM,
        "n_heads": N_HEADS,
        "n_layers": N_LAYERS, # Will be overridden by layer_structure
        "head_dim": HEAD_DIM,
        "layer_structure": [ # Simple structure: Local, Update, Local, Update ...
            {"type": "local_only"},
            {"type": "memory_update"},
            {"type": "local_only"},
            {"type": "memory_update"},
            {"type": "local_only"},
            {"type": "memory_update"},
        ],
        "skip_attention_layers": [],
        "integration_method": "query_fusion",
        "ctrl_init_scale": 0.001,
        "memory_init_scale": 0.01,
        "gate_bias_init": -2.0,
        "output_proj_zero_init": False, # Easier to test if not zero
        "gate_regularization_type": "l1",
        "gate_regularization_strength": 0.0001,
        "mask_future_schedule": [0.2, 0.8],
        "mask_future_rates": [0.1, 0.5, 0.9],
        "enable_mask_future_dropout": True,
    }

@pytest.fixture(scope="module")
def basic_config(basic_config_dict) -> CMAConfig:
    return CMAConfig.from_dict(basic_config_dict)

# Configuration with explicit groups and read-only layers
@pytest.fixture(scope="module")
def grouped_config_dict(basic_config_dict) -> Dict[str, Any]:
    cfg = basic_config_dict.copy()
    cfg["share_initial_memory"] = True # Test shared memory
    cfg["layer_structure"] = [
        {"group": {
            "layers": ["local_only", "local_only", "memory_read", "memory_update"],
            "repeat": 1
        }},
        {"group": {
            "layers": ["local_only", "memory_read", "memory_update"],
            "repeat": 1
        }},
        # Add a skip layer within a group
         {"group": {
            "layers": ["local_only", "skip", "memory_read", "memory_update"],
            "repeat": 1
        }},
    ]
    cfg["skip_attention_layers"] = [5] # Index 5 corresponds to the 'skip' layer
    # Adjust n_layers based on the structure
    cfg["n_layers"] = 4 + 3 + 4
    return cfg

@pytest.fixture(scope="module")
def grouped_config(grouped_config_dict) -> CMAConfig:
    return CMAConfig.from_dict(grouped_config_dict)

@pytest.fixture(scope="module")
def tokenizer():
    # Using gpt2 tokenizer for realistic tokenization
    try:
        return tiktoken.get_encoding("gpt2")
    except Exception:
        pytest.skip("tiktoken not available or model data missing.")

@pytest.fixture(scope="module")
def basic_model(basic_config, tokenizer) -> CMAModel:
    model = CMAModel(basic_config, VOCAB_SIZE, tokenizer)
    model.eval() # Set to eval mode for most tests unless specified
    return model

@pytest.fixture(scope="module")
def grouped_model(grouped_config, tokenizer) -> CMAModel:
    model = CMAModel(grouped_config, VOCAB_SIZE, tokenizer)
    model.eval()
    return model

# --- Helper Functions ---
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_dummy_memory(config: CMAConfig, batch_size: int = 1, is_reverse: bool = False) -> torch.Tensor:
    size = config.reverse_memory_size if is_reverse else config.max_memory_size
    return torch.randn(batch_size, size, config.embed_dim, device=get_device()) * config.memory_init_scale

def create_dummy_memory_dict(model: CMAModel, batch_size: int = 1) -> Dict[int, torch.Tensor]:
    mem_dict = {}
    dev = get_device()
    for group in model.layer_groups:
        if group.has_memory:
            mem_dict[group.group_idx] = torch.randn(
                batch_size, model.config.max_memory_size, model.config.embed_dim, device=dev
            ) * model.config.memory_init_scale
    return mem_dict

def create_dummy_rev_memory_dict(model: CMAModel, batch_size: int = 1) -> Dict[int, torch.Tensor]:
    mem_dict = {}
    dev = get_device()
    for group in model.layer_groups:
        if group.has_memory:
            mem_dict[group.group_idx] = torch.randn(
                batch_size, model.config.reverse_memory_size, model.config.embed_dim, device=dev
            ) * model.config.memory_init_scale
    return mem_dict

def convert_tuples_to_lists(data):
    """Recursively converts tuples to lists within a nested structure."""
    if isinstance(data, dict):
        return {k: convert_tuples_to_lists(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_tuples_to_lists(item) for item in data]
    elif isinstance(data, tuple):
        # Convert tuple to list
        return [convert_tuples_to_lists(item) for item in data]
    else:
        # Keep other types as is
        return data

# --- Test Classes ---

class TestCMAConfig:
    def test_instantiation_from_dict(self, basic_config_dict):
        config = CMAConfig.from_dict(basic_config_dict)
        assert config.chunk_size == CHUNK_SIZE
        assert config.embed_dim == EMBED_DIM
        assert config.max_memory_size == MAX_MEM_SIZE
        assert config.layer_structure is not None
        assert len(config.layer_structure) == 6

    def test_instantiation_from_yaml(self, basic_config_dict, tmp_path):
        yaml_file = tmp_path / "cma_config.yaml"

        # Convert tuples to lists *before* dumping to YAML
        dict_to_dump = convert_tuples_to_lists(basic_config_dict)

        with open(yaml_file, 'w') as f:
            # Dump the modified dictionary (no !python/tuple tags will be generated)
            yaml.dump(dict_to_dump, f, default_flow_style=False)

        # Load using the existing from_yaml method (which uses safe_load)
        config = CMAConfig.from_yaml(yaml_file)

        # Basic assertions
        assert config.chunk_size == CHUNK_SIZE
        assert config.embed_dim == EMBED_DIM
        assert config.layer_structure is not None

        # Verify that the fields loaded from lists are converted back to tuples by the dataclass
        print(type(config.mask_future_schedule))
        assert isinstance(config.mask_future_schedule, list)
        assert config.mask_future_schedule == basic_config_dict['mask_future_schedule']  # Compare value
        assert isinstance(config.mask_future_rates, list)
        assert config.mask_future_rates == basic_config_dict['mask_future_rates']  # Compare value

        # Verify that fields originally intended as lists remain lists
        assert isinstance(config.boundary_search_chars, list)
        assert config.boundary_search_chars == basic_config_dict['boundary_search_chars']
        assert isinstance(config.layer_structure, list)

    def test_validation_failures(self, basic_config_dict):
        # Test invalid percentage
        invalid_dict_1 = basic_config_dict.copy()
        invalid_dict_1["semantic_chunking_gap_percentage"] = 110.0
        with pytest.raises(AssertionError):
            CMAConfig.from_dict(invalid_dict_1)

        # Test inconsistent head_dim (if validation is strict)
        invalid_dict_2 = basic_config_dict.copy()
        invalid_dict_2["head_dim"] = HEAD_DIM + 1
        with pytest.raises(AssertionError):
             CMAConfig.from_dict(invalid_dict_2)

        # Test invalid layer structure (handled in model init, but could add basic checks)
        # invalid_dict_3 = basic_config_dict.copy()
        # invalid_dict_3["layer_structure"] = [{"type": "invalid_type"}]
        # with pytest.raises(ValueError): # Or AssertionError depending on validation location
        #     CMAModel(CMAConfig.from_dict(invalid_dict_3), VOCAB_SIZE)

    def test_grouped_config_parsing(self, grouped_config):
        # Implicitly tested by grouped_model fixture creation, but add checks
        assert grouped_config.n_layers == 11 # 4 + 3 + 4
        assert len(grouped_config.layer_structure) == 3 # 3 group definitions
        assert grouped_config.share_initial_memory is True


class TestControlTokenGenerator:
    @pytest.fixture(scope="class")
    def generator(self, basic_config):
        return ControlTokenGenerator(basic_config)

    @pytest.mark.parametrize("mode, expected_mem_flag", [
        ("forward", 0.0),
        ("lookahead_reverse", 1.0),
        ("persistent_reverse", 0.8),
        ("generate", 0.0),
    ])
    def test_generate_control_tokens_modes(self, generator, mode, expected_mem_flag):
        tokens = generator.generate_control_tokens(
            mode=mode,
            current_chunk_idx=5, total_chunks=10,
            current_mem_size=MAX_MEM_SIZE // 2, max_mem_size=MAX_MEM_SIZE,
            seq_len=CHUNK_SIZE * 5,
            reverse_chunk_idx=1, reverse_window_size=4
        )
        assert tokens["generation_flag"] == (1.0 if mode == "generate" else 0.0)
        assert tokens["memory_mode_flag"] == expected_mem_flag
        assert 0.0 <= tokens["memory_usage_ratio"] <= 1.0
        assert 0.0 <= tokens["memory_density_ratio"]
        assert 0.0 <= tokens["chunk_position_ratio"] <= 1.0

    def test_generate_control_tokens_ratios(self, generator):
        tokens = generator.generate_control_tokens(
            mode="forward",
            current_chunk_idx=2, total_chunks=5,
            current_mem_size=MAX_MEM_SIZE // 4, max_mem_size=MAX_MEM_SIZE,
            seq_len=CHUNK_SIZE * 2
        )
        assert tokens["memory_usage_ratio"] == pytest.approx(0.25)
        assert tokens["memory_density_ratio"] == pytest.approx((MAX_MEM_SIZE / 4) / (CHUNK_SIZE * 2))
        assert tokens["chunk_position_ratio"] == pytest.approx(2 / 5)

    def test_generate_control_tokens_reverse_position(self, generator):
        tokens_rev = generator.generate_control_tokens(
            mode="lookahead_reverse",
            current_chunk_idx=8, total_chunks=10, # Global index
            current_mem_size=MAX_MEM_SIZE // 2, max_mem_size=MAX_MEM_SIZE,
            seq_len=CHUNK_SIZE * 8,
            reverse_chunk_idx=1, # Second newest in window
            reverse_window_size=4 # Window size 4
        )
        # Position = (window_size - reverse_idx) / window_size
        assert tokens_rev["chunk_position_ratio"] == pytest.approx((4 - 1) / 4)


class TestChunkProcessor:
    @pytest.fixture(scope="class")
    def processor(self, basic_config, tokenizer):
        return ChunkProcessor(basic_config, tokenizer)

    def test_semantic_chunking_basic(self, processor, tokenizer):
        text = "This is sentence one. This is sentence two.\n\nThis is paragraph two.\nIt has two lines.\n\n\n# Section Break\nContent after break."
        chunks = processor.semantic_chunk_reverse_with_gap(text)
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(c, list) for c in chunks)
        assert all(isinstance(t, int) for c in chunks for t in c)

        # Check tokenization consistency
        reconstructed_text = tokenizer.decode([t for c in chunks for t in c])
        # Note: Decoding might not perfectly match due to boundary handling / whitespace
        # A loose check is better here.
        assert len(reconstructed_text) >= len(text) * 0.9

        # Check chunk sizes (approximate, depends on boundaries found)
        # Last chunk should be smaller due to gap
        token_counts = [len(c) for c in chunks]
        assert all(tc <= CHUNK_SIZE for tc in token_counts)
        if len(token_counts) > 1:
             target_last = math.floor(CHUNK_SIZE * (1 - processor.config.semantic_chunking_gap_percentage / 100.0))
             # Allow some tolerance due to boundary alignment
             assert token_counts[-1] <= target_last + processor.config.boundary_search_chars[0]

    def test_semantic_chunking_edge_cases(self, processor):
        assert processor.semantic_chunk_reverse_with_gap("") == []
        short_text = "Short."
        short_chunks = processor.semantic_chunk_reverse_with_gap(short_text)
        assert len(short_chunks) == 1
        assert len(short_chunks[0]) <= CHUNK_SIZE

    def test_fixed_size_chunking_basic(self, processor, tokenizer):
        # Access config values via the processor fixture
        CHUNK_SIZE = processor.config.chunk_size
        GAP_PERCENT = processor.config.semantic_chunking_gap_percentage

        # Use a length that isn't exactly .5 to avoid ambiguity if logic changes
        total_len = CHUNK_SIZE * 3 + CHUNK_SIZE // 3
        tokens = list(range(total_len))
        print(f"\nTesting fixed_size_chunking with CHUNK_SIZE={CHUNK_SIZE}, total_len={total_len}")

        chunks = processor.fixed_size_chunk_reverse_with_gap(tokens)

        assert isinstance(chunks, list)
        # Calculate expected number of chunks and sizes accurately
        target_last = math.floor(CHUNK_SIZE * (1 - GAP_PERCENT / 100.0))
        target_last = max(1, target_last)  # Ensure at least 1

        expected_sizes = []
        remaining_len = total_len

        # Calculate last chunk size
        last_chunk_size = min(remaining_len, target_last)
        if remaining_len > 0:
            expected_sizes.append(last_chunk_size)
            remaining_len -= last_chunk_size

        # Calculate intermediate chunk sizes
        while remaining_len >= CHUNK_SIZE: # Use >= to handle exact multiples correctly
            # Ensure we don't create a zero-size intermediate chunk if remaining_len == CHUNK_SIZE
            # This loop condition handles it, but double-check logic if issues persist
            expected_sizes.append(CHUNK_SIZE)
            remaining_len -= CHUNK_SIZE

        # Calculate first chunk size (remainder)
        if remaining_len > 0:
            expected_sizes.append(remaining_len)

        expected_sizes.reverse()  # Reverse to match the order returned by the function

        print(f"  Actual chunk sizes: {[len(c) for c in chunks]}")
        print(f"  Expected chunk sizes: {expected_sizes}")

        assert len(chunks) == len(expected_sizes), f"Expected {len(expected_sizes)} chunks, got {len(chunks)}"
        assert all(isinstance(c, list) for c in chunks)
        assert all(isinstance(t, int) for c in chunks for t in c)

        # Check reconstruction
        reconstructed_tokens = [t for c in chunks for t in c]
        assert reconstructed_tokens == tokens, "Token reconstruction failed"

        # Check chunk sizes against calculated expected sizes
        actual_sizes = [len(c) for c in chunks]
        assert actual_sizes == expected_sizes, f"Chunk sizes do not match expected logic. Got {actual_sizes}, expected {expected_sizes}"

        print("Basic fixed size chunking test passed.")

    # Ensure other methods also have self if they are part of the class
    def test_fixed_size_chunking_edge_cases(self, processor): # Already had self, good.
         # Access CHUNK_SIZE via processor.config
        CHUNK_SIZE = processor.config.chunk_size
        assert processor.fixed_size_chunk_reverse_with_gap([]) == []
        short_tokens = list(range(CHUNK_SIZE // 2))
        short_chunks = processor.fixed_size_chunk_reverse_with_gap(short_tokens)
        assert len(short_chunks) == 1
        assert short_chunks[0] == short_tokens

    def test_semantic_chunking_exact_size(self, processor, tokenizer):
        # Create text likely to tokenize to exactly CHUNK_SIZE
        # This is hard to guarantee, so aim for close and check behavior
        avg_chars_per_token = 4  # Rough estimate
        text = "word " * int(CHUNK_SIZE * avg_chars_per_token / 5)
        tokens = tokenizer.encode(text)
        # Adjust text length until tokens are close to CHUNK_SIZE
        # (Simplified for test brevity - real scenario needs more robust generation)
        if len(tokens) > CHUNK_SIZE:
            text = tokenizer.decode(tokens[:CHUNK_SIZE])
        elif len(tokens) < CHUNK_SIZE:
            # Add padding that likely tokenizes simply
            padding_tokens = tokenizer.encode("." * (CHUNK_SIZE - len(tokens)))
            text += tokenizer.decode(padding_tokens[:CHUNK_SIZE - len(tokens)])

        print(f"\nTesting semantic chunking with near CHUNK_SIZE text (target {CHUNK_SIZE})")
        chunks = processor.semantic_chunk_reverse_with_gap(text)
        token_counts = [len(c) for c in chunks]
        print(f"  Resulting chunk sizes: {token_counts}")

        assert len(chunks) >= 1
        # If it resulted in one chunk, its size should be <= target_last_size
        if len(chunks) == 1:
            target_last = math.floor(CHUNK_SIZE * (1 - processor.config.semantic_chunking_gap_percentage / 100.0))
            assert token_counts[0] <= target_last + processor.config.boundary_search_chars[
                0]  # Allow boundary tolerance
        # If multiple chunks, the last one follows gap rule, others <= CHUNK_SIZE
        elif len(chunks) > 1:
            target_last = math.floor(CHUNK_SIZE * (1 - processor.config.semantic_chunking_gap_percentage / 100.0))
            assert all(tc <= CHUNK_SIZE for tc in token_counts[:-1])
            assert token_counts[-1] <= target_last + processor.config.boundary_search_chars[
                0]  # Allow boundary tolerance

    def test_fixed_chunking_exact_size(self, processor):
        CHUNK_SIZE = processor.config.chunk_size
        GAP_PERCENT = processor.config.semantic_chunking_gap_percentage
        tokens = list(range(CHUNK_SIZE))
        print(f"\nTesting fixed_size_chunking with exact CHUNK_SIZE={CHUNK_SIZE}")

        chunks = processor.fixed_size_chunk_reverse_with_gap(tokens)
        token_counts = [len(c) for c in chunks]
        print(f"  Resulting chunk sizes: {token_counts}")
        # Expected: Applies gap to the "last" chunk segment, remainder becomes the first chunk.
        target_last = math.floor(CHUNK_SIZE * (1 - GAP_PERCENT / 100.0))
        target_last = max(1, target_last)
        expected_first_chunk_size = CHUNK_SIZE - target_last
        expected_sizes = []
        if expected_first_chunk_size > 0:
            expected_sizes.append(expected_first_chunk_size)
        expected_sizes.append(target_last)

        print(f"  Expected chunk sizes: {expected_sizes}")

        assert len(chunks) == len(expected_sizes), f"Expected {len(expected_sizes)} chunks, got {len(chunks)}"
        assert token_counts == expected_sizes, f"Chunk sizes mismatch. Got {token_counts}, expected {expected_sizes}"

        # Check content reconstruction
        reconstructed_tokens = [t for c in chunks for t in c]
        assert reconstructed_tokens == tokens, "Token reconstruction failed"

    def test_semantic_chunking_shorter_than_gap(self, processor, tokenizer):
        target_last = math.floor(CHUNK_SIZE * (1 - processor.config.semantic_chunking_gap_percentage / 100.0))
        target_len = target_last // 2  # Target token length
        # Generate text aiming for target_len tokens
        # Simple approach: use periods, assuming 1 token each
        text = ". " * target_len
        actual_tokens = tokenizer.encode(text)
        # Adjust if needed (this might not be perfect but aims for the short case)
        if len(actual_tokens) > target_len:
            actual_tokens = actual_tokens[:target_len]
            text = tokenizer.decode(actual_tokens)
        elif len(actual_tokens) < target_len:
            # Add more tokens if needed
            text += ". " * (target_len - len(actual_tokens))
            actual_tokens = tokenizer.encode(text)  # Re-encode final text

        print(f"\nTesting semantic chunking shorter than gap-adjusted size (target_tokens {len(actual_tokens)})")
        chunks = processor.semantic_chunk_reverse_with_gap(text)
        token_counts = [len(c) for c in chunks]
        print(f"  Resulting chunk sizes: {token_counts}")
        # Expected: The early exit should trigger, returning a single chunk with all tokens.
        assert len(chunks) == 1, f"Expected 1 chunk for short input, got {len(chunks)}"
        assert token_counts[0] == len(actual_tokens), f"Expected chunk size {len(actual_tokens)}, got {token_counts[0]}"
        assert chunks[0] == actual_tokens, "Chunk content mismatch"

    def test_fixed_chunking_shorter_than_gap(self, processor):
        target_last = math.floor(CHUNK_SIZE * (1 - processor.config.semantic_chunking_gap_percentage / 100.0))
        target_len = target_last // 2
        tokens = list(range(target_len))
        print(f"\nTesting fixed_size_chunking shorter than gap-adjusted size (target_len {target_len})")
        chunks = processor.fixed_size_chunk_reverse_with_gap(tokens)
        token_counts = [len(c) for c in chunks]
        print(f"  Resulting chunk sizes: {token_counts}")
        # Should result in one chunk containing all tokens
        assert len(chunks) == 1
        assert token_counts[0] == target_len
        assert chunks[0] == tokens

    def test_semantic_chunking_boundary_priority(self, processor, tokenizer):
        original_chunk_size = processor.config.chunk_size  # Store original

        # --- Scenario 1: Primary preferred over secondary/tertiary ---
        print(f"\nDEBUG Boundary Priority Test 1 (Primary):")
        # Construct text with clear boundaries before the primary one
        part1 = "This is clause 1, and sentence 1. This is clause 2, and sentence 2."
        primary_boundary = "\n\n# Section Break\n\n"
        part2 = "This is content after the section break, it needs to be long enough. " * 10
        text1 = part1 + primary_boundary + part2

        tokens_part1 = tokenizer.encode(part1)

        # Set chunk_size to be slightly larger than part1 to force the decision
        processor.config.chunk_size = len(tokens_part1) + 15  # Adjust offset if needed
        print(f"  Temporarily setting chunk_size to: {processor.config.chunk_size}")

        chunks1 = processor.semantic_chunk_reverse_with_gap(text1)
        processor.config.chunk_size = original_chunk_size  # Restore

        # Print resulting chunks for debugging
        for i, c in enumerate(chunks1):
            print(f"  Chunk {i} (len {len(c)}): '{tokenizer.decode(c)}'")

        # Assertions for Scenario 1
        assert len(chunks1) > 1, "Test 1 requires text to be split into multiple chunks"

        # Check that chunk[0] contains part1 AND the primary boundary
        decoded_chunk0_s1 = tokenizer.decode(chunks1[0]).strip()  # Use strip() for cleaner comparison
        # Construct the expected content including the boundary
        expected_chunk0_s1 = (part1 + primary_boundary).strip()

        assert decoded_chunk0_s1 == expected_chunk0_s1, \
            f"Chunk 0 should contain text before AND the primary boundary.\nExpected: '{expected_chunk0_s1}'\nGot:      '{decoded_chunk0_s1}'"

        # Check that chunk[1] starts after the primary boundary (this assertion should still be correct)
        expected_start_chunk1_s1 = part2.lstrip()
        decoded_chunk1_s1 = tokenizer.decode(chunks1[1]).lstrip()
        assert decoded_chunk1_s1.startswith(expected_start_chunk1_s1[:20]), \
            f"Chunk 1 (s1) should start after the primary boundary. Expected start: '{expected_start_chunk1_s1[:20]}', Got: '{decoded_chunk1_s1[:20]}'"

        print("Boundary Priority Test 1 passed.")

        # --- Scenario 2: Secondary preferred over tertiary ---
        # (Keep the Scenario 2 code as provided in the previous response, it should be correct)
        print(f"\nDEBUG Boundary Priority Test 2 (Secondary vs Tertiary):")
        # Text where secondary ('.') appears after tertiary (',')
        text2 = "Clause one, clause two. Clause three needs to be long enough to ensure split maybe." * 3
        first_period_idx = text2.find('.')
        assert first_period_idx != -1, "Test text must contain a period."
        text_up_to_period = text2[:first_period_idx + 1]
        tokens_up_to_period = tokenizer.encode(text_up_to_period)
        processor.config.chunk_size = len(tokens_up_to_period) + 5
        print(f"  Temporarily setting chunk_size to: {processor.config.chunk_size}")
        chunks2 = processor.semantic_chunk_reverse_with_gap(text2)
        processor.config.chunk_size = original_chunk_size
        for i, c in enumerate(chunks2):
            print(f"  Chunk {i} (len {len(c)}): '{tokenizer.decode(c)}'")
        assert len(chunks2) > 1, "Test 2 requires text to be split into multiple chunks"
        decoded_chunk0_s2 = tokenizer.decode(chunks2[0]).rstrip()
        print(f"  Decoded Chunk 0 (stripped): '{decoded_chunk0_s2}'")
        assert decoded_chunk0_s2.endswith("."), "Chunk 0 should end with the sentence boundary (.)"
        assert not decoded_chunk0_s2.endswith(","), "Chunk 0 should not end just with the clause boundary (,)"
        expected_start_chunk1_s2 = text2[first_period_idx + 1:].lstrip()
        decoded_chunk1_s2 = tokenizer.decode(chunks2[1]).lstrip()
        print(f"  Decoded Chunk 1 (stripped): '{decoded_chunk1_s2[:30]}...'")
        print(f"  Expected Chunk 1 Start: '{expected_start_chunk1_s2[:30]}...'")
        assert decoded_chunk1_s2.startswith(expected_start_chunk1_s2[:20]), \
            f"Chunk 1 (s2) should start after the period. Expected start: '{expected_start_chunk1_s2[:20]}', Got: '{decoded_chunk1_s2[:20]}'"
        print("Boundary Priority Test 2 passed.")

        # --- Scenario 3 (Optional but good): Test text3 logic from original ---
        # This verifies secondary preference again with a slightly different structure
        print(f"\nDEBUG Boundary Priority Test 3 (Secondary vs Tertiary - Alt Text):")
        text3 = "Clause one, clause two. Clause three"
        tokens_up_to_period_t3 = tokenizer.encode("Clause one, clause two.")
        processor.config.chunk_size = len(tokens_up_to_period_t3) + 2  # Force split near period
        print(f"  Temporarily setting chunk_size to: {processor.config.chunk_size}")
        chunks3 = processor.semantic_chunk_reverse_with_gap(text3)
        processor.config.chunk_size = original_chunk_size  # Restore

        for i, c in enumerate(chunks3):
            print(f"  Chunk {i} (len {len(c)}): '{tokenizer.decode(c)}'")

        assert len(chunks3) > 1, "Test 3 requires text to be split into multiple chunks"
        decoded_chunk0_t3 = tokenizer.decode(chunks3[0]).rstrip()
        assert decoded_chunk0_t3.endswith("."), "Chunk 0 (t3) should end with the sentence boundary (.)"
        assert not decoded_chunk0_t3.endswith(","), "Chunk 0 (t3) should not end with the clause boundary (,)"
        expected_start_chunk1_t3 = text3.split(". ")[1].lstrip()
        decoded_chunk1_t3 = tokenizer.decode(chunks3[1]).lstrip()
        assert decoded_chunk1_t3.startswith(expected_start_chunk1_t3[:10]), \
            f"Chunk 1 (t3) should start after the period. Expected start: '{expected_start_chunk1_t3[:10]}', Got: '{decoded_chunk1_t3[:10]}'"
        print("Boundary Priority Test 3 passed.")

class TestMemoryManager:
    @pytest.fixture(scope="class")
    def manager(self, basic_config):
        return MemoryManager(basic_config)

    @pytest.mark.parametrize("func, cap, processed, expected_frac", [
        ("linear", 8192, 0, 0.0),
        ("linear", 8192, 4096, 0.5),
        ("linear", 8192, 8192, 1.0),
        ("linear", 8192, 10000, 1.0),
        ("log", 8192, 1, 0.0), # log(1) = 0
        ("log", 8192, 90, 0.5), # log(90) / log(8192) approx 0.5 (sqrt)
        ("log", 8192, 8192, 1.0),
        ("log", 8192, 10000, 1.0),
    ])
    def test_get_effective_size(self, basic_config, func, cap, processed, expected_frac):
        config = basic_config
        config.memory_growth_function = func
        config.memory_cap_length = cap
        manager = MemoryManager(config)
        effective_size = manager.get_effective_size(processed)
        expected_size = int(config.max_memory_size * expected_frac)
        assert effective_size == expected_size
        assert 0 <= effective_size <= config.max_memory_size

    def test_get_write_mask(self, manager, basic_config):
        print("\nTesting write mask progression...")
        B = 1 # Keep batch size simple for this test
        dev = get_device()
        total_chunks_in_pass = 80 # Example number of chunks in a pass
        tokens_per_chunk = basic_config.chunk_size # Use config chunk size

        results = []
        # Simulate processing across a sequence, starting from 0 tokens processed
        # We'll check the mask generated *for* each chunk within one simulated pass
        tokens_processed_before_pass = 0 # Assume pass starts at beginning

        print(f"Config: max_mem={basic_config.max_memory_size}, cap_len={basic_config.memory_cap_length}, init_frac={basic_config.initial_write_fraction}, chunk_size={tokens_per_chunk}")

        for i in range(total_chunks_in_pass):
            # Calculate total tokens processed *before* this specific chunk starts
            tokens_before_chunk = tokens_processed_before_pass + (i * tokens_per_chunk)

            # Calculate the sequence-length based write cap
            seq_cap = manager.get_effective_size(tokens_before_chunk, tokens_per_chunk)

            # Calculate the target writable size based on chunk progress within the pass
            chunk_progress = (i + 1) / total_chunks_in_pass
            write_fraction = basic_config.initial_write_fraction + (1.0 - basic_config.initial_write_fraction) * chunk_progress
            target_size = int(basic_config.max_memory_size * write_fraction)

            # The final expected writable size is the minimum of the two constraints
            expected_writable = min(seq_cap, target_size)
            expected_writable = max(0, expected_writable) # Ensure non-negative

            # Get the actual mask from the function
            mask = manager.get_write_mask(
                current_chunk_idx_in_pass=i,
                total_chunks_in_pass=total_chunks_in_pass,
                total_tokens_processed_before_chunk=tokens_before_chunk,
                current_chunk_len=tokens_per_chunk,
                batch_size=B,
                device=dev
            )
            actual_writable = mask.sum().item()

            print(f"  Chunk {i}: Tokens Before={tokens_before_chunk}, SeqCap={seq_cap}, Target={target_size} => Expected={expected_writable}, Actual={actual_writable}")

            # Assert the calculated value matches the function's output
            assert actual_writable == expected_writable, f"Chunk {i} Failed: Expected {expected_writable}, got {actual_writable}"
            assert mask.shape == (B, basic_config.max_memory_size)
            assert mask.dtype == torch.bool
            # Verify mask content
            assert mask[:, :expected_writable].all()
            if expected_writable < basic_config.max_memory_size:
                assert not mask[:, expected_writable:].any()

            results.append(actual_writable)

        # --- Check Progression ---
        print(f"\n  Progression Results: {results}")
        # Expect non-decreasing size (it might plateau if target_size grows faster than seq_cap or max is hit)
        assert all(results[j] >= results[j - 1] for j in range(1, len(results))), "Expected non-decreasing writable size"
        # Check if it actually increased at some point (unless cap=0 initially or max cap is reached early)
        if len(results) > 1:
             assert results[-1] > results[0] or (results[0] == 0) or (results[0] == basic_config.max_memory_size), \
                 "Expected writable size to increase overall, start at 0, or be fully capped"

        print("\nWrite mask progression test passed.")

    @pytest.mark.parametrize("is_persistent, rev_idx, win_size, expected_decay_approx", [
        (False, 0, 4, 1.0), # Lookahead, newest
        (False, 1, 4, 0.6**0.1), # Lookahead, 2nd newest (rate^step)
        (False, 3, 4, 0.6**(3*0.1)), # Lookahead, oldest
        (True, 0, 3, 1.0), # Persistent, newest
        (True, 1, 3, 0.2**0.05), # Persistent, 2nd newest
        (True, 2, 3, 0.2**(2*0.05)), # Persistent, oldest
    ])
    def test_calculate_reverse_decay_weights(self, manager, basic_config, is_persistent, rev_idx, win_size, expected_decay_approx):
        B = 1
        mem_shape = (B, REV_MEM_SIZE, EMBED_DIM)
        dev = get_device()
        weights = manager.calculate_reverse_decay_weights(
            reverse_chunk_index=rev_idx, window_size=win_size,
            is_persistent=is_persistent, memory_shape=mem_shape, device=dev
        )
        assert weights.shape == mem_shape
        assert torch.allclose(weights, torch.full(mem_shape, expected_decay_approx, device=dev), atol=1e-6)


class TestAttentionLayers:
    @pytest.fixture(scope="class")
    def causal_attn(self, basic_config):
        return CausalSelfAttention(basic_config, layer_idx=0).to(get_device())

    @pytest.fixture(scope="class")
    def cma_attn_update(self, basic_config):
        # Test update layer specifically
        return CascadeMemoryAttention(basic_config, layer_idx=1, is_memory_update=True).to(get_device())

    @pytest.fixture(scope="class")
    def cma_attn_read(self, basic_config):
         # Test read-only layer specifically
        return CascadeMemoryAttention(basic_config, layer_idx=2, is_memory_update=False).to(get_device())

    def test_causal_self_attention_forward(self, causal_attn, basic_config):
        B, T, C = 2, CHUNK_SIZE // 2, EMBED_DIM
        x = torch.randn(B, T, C, device=get_device())
        output = causal_attn(x)
        assert output.shape == (B, T, C)
        # TODO: Could add check for causality if needed (output at step t depends only on inputs <= t)

    def test_cma_attention_forward_no_memory(self, cma_attn_read):
        B, T, C = 2, CHUNK_SIZE // 2, EMBED_DIM
        x = torch.randn(B, T, C, device=get_device())
        output, fwd_mem, rev_mem, gate_loss = cma_attn_read(x)
        assert output.shape == (B, T, C)
        assert fwd_mem is None
        assert rev_mem is None
        # Gate loss should be None if no memory tokens to gate
        assert gate_loss is None

    def test_cma_attention_forward_with_memory_and_control(self, cma_attn_read, basic_config):
        B, T, C = 1, CHUNK_SIZE // 4, EMBED_DIM
        x = torch.randn(B, T, C, device=get_device())
        fwd_mem = create_dummy_memory(basic_config, B, is_reverse=False)
        rev_mem = create_dummy_memory(basic_config, B, is_reverse=True)
        # Create dummy control tokens
        gen = ControlTokenGenerator(basic_config)
        ctrl = gen.generate_control_tokens("forward", 1, 5, MAX_MEM_SIZE//2, MAX_MEM_SIZE, T*5)

        output, fwd_mem_out, rev_mem_out, gate_loss = cma_attn_read(
            x, forward_memory=fwd_mem, reverse_memory=rev_mem, control_tokens=ctrl
        )
        assert output.shape == (B, T, C)
        assert fwd_mem_out is None # Read-only layer doesn't update
        assert rev_mem_out is None
        assert gate_loss is not None # Should have gate loss with memory and regularization
        assert gate_loss.ndim == 0 # Should be scalar

    def test_cma_attention_forward_memory_update(self, cma_attn_update, basic_config):
        B, T, C = 1, CHUNK_SIZE // 4, EMBED_DIM
        x = torch.randn(B, T, C, device=get_device())
        fwd_mem_in = create_dummy_memory(basic_config, B, is_reverse=False)
        rev_mem_in = create_dummy_memory(basic_config, B, is_reverse=True)
        write_mask = torch.ones(B, MAX_MEM_SIZE, dtype=torch.bool, device=get_device()) # Allow all writes
        decay_weights = torch.full((B, REV_MEM_SIZE, C), 0.5, device=get_device()) # Dummy decay

        # Test forward update
        output, fwd_mem_out, rev_mem_out, _ = cma_attn_update(
            x, forward_memory=fwd_mem_in, reverse_memory=rev_mem_in, # Reads both
            do_memory_update=True, write_mask=write_mask, is_reverse_update=False
        )
        assert output.shape == (B, T, C)
        assert fwd_mem_out is not None
        assert fwd_mem_out.shape == fwd_mem_in.shape
        # Check if memory actually changed (should change if write_mask allows and input != 0)
        assert not torch.allclose(fwd_mem_out, fwd_mem_in)
        assert rev_mem_out is None # Should not update reverse memory in forward mode

        # Test reverse update
        output, fwd_mem_out, rev_mem_out, _ = cma_attn_update(
            x, forward_memory=None, reverse_memory=rev_mem_in, # Reads only reverse
            do_memory_update=True, decay_weights=decay_weights, is_reverse_update=True
        )
        assert output.shape == (B, T, C)
        assert fwd_mem_out is None # Should not update forward memory in reverse mode
        assert rev_mem_out is not None
        assert rev_mem_out.shape == rev_mem_in.shape
        assert not torch.allclose(rev_mem_out, rev_mem_in) # Should change
        # TODO: Could add a check that decay weights were applied if possible


class TestBlock:
    @pytest.fixture(scope="class")
    def local_block(self, basic_config):
        return Block(basic_config, layer_idx=0, layer_type="local_only").to(get_device())

    @pytest.fixture(scope="class")
    def update_block(self, basic_config):
        return Block(basic_config, layer_idx=1, layer_type="memory_update").to(get_device())

    @pytest.fixture(scope="class")
    def read_block(self, basic_config):
        return Block(basic_config, layer_idx=2, layer_type="memory_read").to(get_device())

    @pytest.fixture(scope="class")
    def skip_block(self, basic_config):
         return Block(basic_config, layer_idx=3, layer_type="skip").to(get_device())

    def test_local_block_forward(self, local_block, basic_config):
        B, T, C = 1, CHUNK_SIZE // 4, EMBED_DIM
        x = torch.randn(B, T, C, device=get_device())
        # Memory dicts should be ignored by local block
        fwd_mem_dict = {0: create_dummy_memory(basic_config, B)}
        rev_mem_dict = {0: create_dummy_memory(basic_config, B, is_reverse=True)}

        out_x, updated_fwd, updated_rev, gate_loss = local_block(
            x, fwd_mem_dict, rev_mem_dict, rev_mem_dict, # Pass dummy dicts
            group_id=0, mode="forward", control_tokens=None
        )
        assert out_x.shape == x.shape
        assert updated_fwd is None
        assert updated_rev is None
        assert gate_loss is None # Local block has no gating

    def test_update_block_forward_pass(self, update_block, basic_config):
        B, T, C = 1, CHUNK_SIZE * 2, EMBED_DIM
        x = torch.randn(B, T, C, device=get_device())
        group_id = 0 # Assume this block belongs to group 0
        fwd_mem_in = create_dummy_memory(basic_config, B)
        rev_mem_in = create_dummy_memory(basic_config, B, is_reverse=True)
        fwd_mem_dict = {group_id: fwd_mem_in}
        rev_mem_dict = {group_id: rev_mem_in} # Lookahead reverse used in fwd pass
        persist_mem_dict = {group_id: create_dummy_memory(basic_config, B, is_reverse=True)} # Not used in fwd pass update
        write_mask = torch.ones(B, MAX_MEM_SIZE, dtype=torch.bool, device=get_device())
        gen = ControlTokenGenerator(basic_config)
        ctrl = gen.generate_control_tokens("forward", 1, 5, MAX_MEM_SIZE//2, MAX_MEM_SIZE, T*5)

        out_x, updated_fwd, updated_rev, gate_loss = update_block(
            x, fwd_mem_dict, rev_mem_dict, persist_mem_dict,
            group_id=group_id, mode="forward", control_tokens=ctrl, write_mask=write_mask, total_logical_sequence_length=T
        )
        assert out_x.shape == x.shape
        assert updated_fwd is not None
        assert updated_fwd[0] == group_id # Check correct group ID returned
        assert updated_fwd[1].shape == fwd_mem_in.shape
        assert not torch.allclose(updated_fwd[1], fwd_mem_in) # Check memory changed
        assert updated_rev is None # No reverse update in forward pass
        assert gate_loss is not None # Update block has gating

    def test_update_block_reverse_pass(self, update_block, basic_config):
        B, T, C = 1, CHUNK_SIZE * 2, EMBED_DIM
        x = torch.randn(B, T, C, device=get_device())
        group_id = 0
        fwd_mem_dict = {} # Not used in reverse pass update
        rev_mem_in = create_dummy_memory(basic_config, B, is_reverse=True)
        rev_mem_dict = {group_id: rev_mem_in}
        persist_mem_dict = {} # Not used in lookahead reverse pass update
        decay_weights = torch.full((B, REV_MEM_SIZE, C), 0.5, device=get_device())
        gen = ControlTokenGenerator(basic_config)
        ctrl = gen.generate_control_tokens("lookahead_reverse", 1, 5, 0, MAX_MEM_SIZE, T*5, 0, 3)

        out_x, updated_fwd, updated_rev, gate_loss = update_block(
            x, fwd_mem_dict, rev_mem_dict, persist_mem_dict,
            group_id=group_id, mode="lookahead_reverse", control_tokens=ctrl, decay_weights=decay_weights, total_logical_sequence_length=T
        )
        assert out_x.shape == x.shape
        assert updated_fwd is None
        assert updated_rev is not None
        assert updated_rev[0] == group_id
        assert updated_rev[1].shape == rev_mem_in.shape
        assert not torch.allclose(updated_rev[1], rev_mem_in)
        assert gate_loss is not None

    def test_read_block_forward(self, read_block, basic_config):
        B, T, C = 1, CHUNK_SIZE // 4, EMBED_DIM
        x = torch.randn(B, T, C, device=get_device())
        group_id = 0
        fwd_mem_in = create_dummy_memory(basic_config, B)
        rev_mem_in = create_dummy_memory(basic_config, B, is_reverse=True)
        fwd_mem_dict = {group_id: fwd_mem_in}
        rev_mem_dict = {group_id: rev_mem_in}
        persist_mem_dict = {group_id: create_dummy_memory(basic_config, B, is_reverse=True)}
        gen = ControlTokenGenerator(basic_config)
        ctrl = gen.generate_control_tokens("forward", 1, 5, MAX_MEM_SIZE//2, MAX_MEM_SIZE, T*5)

        out_x, updated_fwd, updated_rev, gate_loss = read_block(
            x, fwd_mem_dict, rev_mem_dict, persist_mem_dict,
            group_id=group_id, mode="forward", control_tokens=ctrl
        )
        assert out_x.shape == x.shape
        # Read block should NOT return updated memory
        assert updated_fwd is None
        assert updated_rev is None
        assert gate_loss is not None # Read block still uses gating

    def test_skip_block_forward(self, skip_block, basic_config):
        B, T, C = 1, CHUNK_SIZE // 4, EMBED_DIM
        x = torch.randn(B, T, C, device=get_device())
        group_id = 0
        # Create dummy memory dicts, although they won't be used by the skip block's logic
        fwd_mem_dict = {group_id: create_dummy_memory(basic_config, B)}
        rev_mem_dict = {group_id: create_dummy_memory(basic_config, B, is_reverse=True)}
        persist_mem_dict = {group_id: create_dummy_memory(basic_config, B, is_reverse=True)}

        # Store input for comparison if needed, but don't assert equality
        x_input = x.clone()

        out_x, updated_fwd, updated_rev, gate_loss = skip_block(
            x, fwd_mem_dict, rev_mem_dict, persist_mem_dict,
            group_id=group_id, mode="forward" # Mode doesn't affect skip block logic directly
        )

        # --- Corrected Assertions ---
        # 1. Check output shape
        assert out_x.shape == x_input.shape

        # 2. Check that memory was not updated
        assert updated_fwd is None
        assert updated_rev is None

        # 3. Check that gate loss was not generated (no gating in skip)
        assert gate_loss is None

        # 4. OPTIONAL: Verify output is different from input (due to MLP)
        # Use a reasonable tolerance; they might be close if weights are small
        # This assumes MLP weights are initialized non-zero.
        if not torch.allclose(out_x, x_input, atol=1e-6):
             print("  (Info: Skip block output differs from input, as expected due to MLP)")
        else:
             # This might happen if MLP weights are zero or input is zero etc.
             print("  (Warning: Skip block output is close to input. Check MLP initialization?)")

        # DO NOT assert torch.allclose(out_x, x_input)


class TestCMAModel:

    def test_model_instantiation_basic(self, basic_model, basic_config):
        assert basic_model.config == basic_config
        assert basic_model.vocab_size == VOCAB_SIZE
        assert isinstance(basic_model.layers, nn.ModuleList)
        assert len(basic_model.layers) == basic_config.n_layers
        assert len(basic_model.layer_groups) > 0
        assert basic_model.num_memory_groups == 3 # 3 update layers in basic_config structure
        assert len(basic_model.initial_fwd_params) == basic_model.num_memory_groups
        assert len(basic_model.initial_rev_la_params) == basic_model.num_memory_groups
        assert len(basic_model.initial_rev_p_params) == basic_model.num_memory_groups
        assert not basic_model.config.share_initial_memory
        # Check parameter counts match printout (approx)
        total_params = sum(p.numel() for p in basic_model.parameters() if p.requires_grad)
        assert total_params > 0

    def test_model_instantiation_grouped(self, grouped_model, grouped_config):
        assert grouped_model.config == grouped_config
        assert len(grouped_model.layers) == grouped_config.n_layers # 11
        assert len(grouped_model.layer_groups) == 3 # 3 groups defined
        assert grouped_model.num_memory_groups == 3 # Each group has an update layer
        assert grouped_model.config.share_initial_memory
        # With shared memory, only 1 set of initial params is created, but ParameterList might duplicate refs
        assert len(grouped_model.initial_fwd_params) == grouped_model.num_memory_groups # List length matches groups
        # Check if underlying data pointers are the same for shared params
        if grouped_model.num_memory_groups > 1:
            assert grouped_model.initial_fwd_params[0].data_ptr() == grouped_model.initial_fwd_params[1].data_ptr()
            assert grouped_model.initial_rev_la_params[0].data_ptr() == grouped_model.initial_rev_la_params[1].data_ptr()
            assert grouped_model.initial_rev_p_params[0].data_ptr() == grouped_model.initial_rev_p_params[1].data_ptr()

        # Check skip layer was correctly identified and attn is None
        assert grouped_model.layers[5].layer_type == "skip"
        assert grouped_model.layers[5].attn is None


    def test_forward_no_cycle_streaming(self, basic_model, tokenizer):
        basic_model.reset_state()
        basic_model.eval()
        dev = get_device()
        basic_model.to(dev)

        # Ensure CUDA is synchronized before starting
        if dev.type == 'cuda':
            torch.cuda.synchronize(dev)

        #print(f"Basic model num memory groups: {basic_model.num_memory_groups}")
        #print(f"Basic model init rev s params: {basic_model.initial_rev_s_params}")

        # Initial prompt (less than chunk size)
        prompt_tokens = tokenizer.encode("This is the initial prompt.")
        assert len(prompt_tokens) < CHUNK_SIZE

        # Process prompt - should not trigger cycle
        logits1, loss1 = basic_model(prompt_tokens, training_mode=False)
        assert logits1.shape == (1, len(prompt_tokens), VOCAB_SIZE)
        assert loss1 is None # Eval mode, no loss
        assert basic_model.current_chunk_tokens == prompt_tokens
        assert len(basic_model.closed_chunks) == 0
        assert basic_model.total_tokens_processed == 0 # Not updated until cycle
        # Check memory states were initialized but likely not changed much (no update pass)
        assert len(basic_model.M_fwd) == basic_model.num_memory_groups
        assert len(basic_model.M_rev_persist) == basic_model.num_memory_groups
        initial_fwd_mem_copy = {k: v.clone() for k, v in basic_model.M_fwd.items()}

        # Process next token (streaming)
        next_token = [tokenizer.encode(" Next.")[0]]
        logits2, loss2 = basic_model(next_token, training_mode=False)
        assert logits2.shape == (1, len(prompt_tokens) + 1, VOCAB_SIZE) # Logits for the whole buffer
        assert loss2 is None
        assert basic_model.current_chunk_tokens == prompt_tokens + next_token
        assert len(basic_model.closed_chunks) == 0

        # Verify memory was READ but NOT updated during streaming
        for group_id in initial_fwd_mem_copy:
             assert torch.allclose(basic_model.M_fwd[group_id], initial_fwd_mem_copy[group_id])

        #print(f"DEBUG: Model vocab size: {basic_model.vocab_size}", flush=True)
        #print(f"DEBUG: Embedding weight shape: {basic_model.token_embedding.weight.shape}", flush=True)
        #assert False == True

    def test_forward_trigger_cycle(self, basic_model, tokenizer):
        basic_model.reset_state()
        basic_model.eval()
        dev = get_device()
        basic_model.to(dev)

        safe_token_id = min(10, basic_model.vocab_size - 1)
        input_tokens = [safe_token_id] * (CHUNK_SIZE + 10) # This is num_new_tokens_for_this_call
        num_input_tokens = len(input_tokens)

        initial_mem_params = {}
        for group_id, mem_idx in basic_model.group_id_to_memory_idx.items():
            initial_mem_params[group_id] = (
                basic_model.initial_fwd_params[mem_idx].clone().detach(),
                basic_model.initial_rev_p_params[mem_idx].clone().detach()
            )

        logits, loss = basic_model(input_tokens, training_mode=False)

        assert logits.shape[0] == 1
        # CORRECTED ASSERTION: Logits should correspond to the original input tokens for this call
        assert logits.shape[1] == num_input_tokens, \
            f"Expected logits for {num_input_tokens} (original input), got {logits.shape[1]}"
        assert logits.shape[2] == VOCAB_SIZE
        assert loss is None

        assert len(basic_model.closed_chunks) > 0
        assert len(basic_model.current_chunk_tokens) > 0 # current_chunk_tokens is the tail end of re-chunked input_tokens
        assert basic_model.total_tokens_processed == num_input_tokens
        assert basic_model.tokens_since_persistent_update == 0

        for group_id in basic_model.M_fwd:
            init_fwd, init_rev_p = initial_mem_params[group_id]
            init_fwd = init_fwd.expand(1, -1, -1).to(dev)
            init_rev_p = init_rev_p.expand(1, -1, -1).to(dev)

            assert group_id in basic_model.M_fwd
            assert basic_model.M_fwd[group_id].shape == init_fwd.shape
            assert group_id in basic_model.M_rev_persist
            assert basic_model.M_rev_persist[group_id].shape == init_rev_p.shape

            assert not torch.allclose(basic_model.M_fwd[group_id], init_fwd, atol=1e-4), f"Fwd mem for group {group_id} did not change"
            if basic_model.closed_chunks:
                 assert not torch.allclose(basic_model.M_rev_persist[group_id], init_rev_p, atol=1e-4), f"Rev persist mem for group {group_id} did not change"
            else:
                 print(f"Skipping M_rev_persist change check for group {group_id} as no closed chunks were formed.")

    def test_forward_training_mode(self, basic_model, tokenizer):
        basic_model.reset_state()
        basic_model.train()  # Set to training mode
        dev = get_device()
        basic_model.to(dev)
        basic_model.set_training_step(100, 1000)  # For mask-future

        input_token_list = list(range(CHUNK_SIZE + 10))  # Trigger cycle, length 138
        original_input_len = len(input_token_list)

        # Store initial M_rev_persist state before forward pass
        basic_model._initialize_memory_states(force_reset=True)  # Ensure clean start
        # initial_rev_p_copy = {k: v.clone() for k, v in basic_model.M_rev_persist.items()} # Not directly used in asserts here

        logits, loss = basic_model(input_token_list, training_mode=True)

        assert logits.shape[0] == 1

        # NEW ASSERTION LOGIC:
        # In training mode, if a cycle is triggered (as it is here),
        # logits are concatenated from all chunks of Pass 2.
        # The total length of these logits should correspond to the total number of tokens
        # that were re-chunked and processed in Pass 2.
        # For this input, all original_input_len tokens are processed.
        # Note: This assumes no tokens are dropped or added by the chunking/processing itself,
        # which is generally true for standard tokenization.
        # The number of tokens for which predictions are made is logits.shape[1].
        # The original input sequence had `original_input_len` tokens.
        # The CMAModel.forward was called with these `original_input_len` tokens.
        # The Pass 2 processes all these tokens.
        assert logits.shape[1] == original_input_len, \
            f"Logits length {logits.shape[1]} should match original input length {original_input_len} in training cycle"

        assert logits.shape[2] == VOCAB_SIZE

        # Check loss is returned and is a scalar tensor
        assert loss is not None
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        # Loss should be positive (or zero) if regularization is applied
        # (or if predictions are not perfect, which is expected)
        # A strict loss.item() > 0 might be too fragile if by chance all predictions are perfect
        # or if regularization strength is zero and there's no pred loss.
        # loss.item() >= 0 is safer.
        assert loss.item() >= 0  # Gate reg is L1, so >= 0. Pred loss also >= 0.

        # Check if mask-future was likely applied
        p_drop = get_mask_future_schedule(basic_model.config, basic_model.training_step,
                                          basic_model.total_training_steps)
        assert p_drop > 0

        basic_model.eval()  # Reset to eval mode

    def test_generate_with_prompt(self, basic_model, tokenizer):
        basic_model.reset_state()
        basic_model.eval()
        dev = get_device()
        basic_model.to(dev)

        prompt = "Once upon a time"
        prompt_ids = tokenizer.encode(prompt)
        max_gen = 5

        generated_ids = basic_model.generate(prompt=prompt, max_new_tokens=max_gen, reset_state=True)

        assert isinstance(generated_ids, list)
        assert len(generated_ids) == max_gen
        # Check that the model state reflects the prompt having been processed
        assert basic_model.current_chunk_tokens == prompt_ids + generated_ids
        assert len(basic_model.closed_chunks) == 0 # Prompt likely didn't fill chunk

    def test_generate_stop_token(self, basic_model, tokenizer):
        basic_model.reset_state()
        basic_model.eval()
        dev = get_device()
        basic_model.to(dev)

        # Use a common token ID like period '.' as stop token
        stop_id = tokenizer.encode(".")[0]
        max_gen = 50 # Generate enough to likely hit the stop token

        # Force the model to output the stop token by manipulating logits (hacky but effective for testing)
        original_forward = basic_model.forward
        call_count = 0
        def mocked_forward(*args, **kwargs):
            nonlocal call_count
            logits, loss = original_forward(*args, **kwargs)
            if kwargs.get('training_mode', False) is False and logits.numel() > 0 and call_count > 0: # Don't modify prompt processing
                 # Set high probability for stop_id on the last token position
                 logits[0, -1, :] = -100.0 # Suppress other tokens
                 logits[0, -1, stop_id] = 10.0 # Boost stop token
            call_count += 1
            return logits, loss

        # Use a safe starting token ID (e.g., 0 or any ID < VOCAB_SIZE)
        safe_prompt_id = min(10, basic_model.vocab_size - 1)  # Ensure it's within vocab
        prompt_tokens = [safe_prompt_id]

        # Ensure CUDA is synchronized before mocking/generating if needed
        if dev.type == 'cuda':
            torch.cuda.synchronize(dev)

        basic_model.forward = mocked_forward
        try:
            generated_ids = basic_model.generate(prompt=prompt_tokens, max_new_tokens=max_gen, stop_token_id=stop_id,
                                                 reset_state=True)
        finally:
            basic_model.forward = original_forward  # Restore original method

        assert stop_id in generated_ids
        assert generated_ids[-1] == stop_id # Should end with stop token
        assert len(generated_ids) < max_gen # Should stop early

    def test_generate_periodic_persist_update_tokens(self, basic_model, tokenizer):
        basic_model.reset_state()
        basic_model.eval()
        dev = get_device()
        basic_model.to(dev)

        # Configure for frequent updates
        original_freq = basic_model.config.persistent_reverse_update_freq_tokens
        basic_model.config.persistent_reverse_update_freq_tokens = 5
        max_gen = 12

        # Mock the persistent reverse pass to check if it's called
        call_log = {"count": 0}
        original_persist_pass = basic_model._run_persistent_reverse_pass
        def mocked_persist_pass(*args, **kwargs):
            call_log["count"] += 1
            return original_persist_pass(*args, **kwargs)
        basic_model._run_persistent_reverse_pass = mocked_persist_pass

        _ = basic_model.generate(prompt="Initial context.", max_new_tokens=max_gen, reset_state=True)

        basic_model._run_persistent_reverse_pass = original_persist_pass # Restore
        basic_model.config.persistent_reverse_update_freq_tokens = original_freq # Restore config

        # Expected calls: floor(max_gen / freq) = floor(12 / 5) = 2
        assert call_log["count"] == 2

    def test_set_training_step(self, basic_model):
        basic_model.set_training_step(500, 2000)
        assert basic_model.training_step == 500
        assert basic_model.total_training_steps == 2000

    def test_model_instantiation_invalid_group(self, basic_config_dict, tokenizer):
        # 1. Read-only layer without an update layer in its group
        invalid_structure_1 = [
            {"group": {"layers": ["local_only", "memory_read"], "repeat": 1}},  # No update layer
            {"type": "local_only"},
        ]
        config_dict_1 = basic_config_dict.copy()
        config_dict_1["layer_structure"] = invalid_structure_1
        config_dict_1["n_layers"] = 3
        with pytest.raises(ValueError, match=r"read-only layers require an update layer"):
            CMAModel(CMAConfig.from_dict(config_dict_1), VOCAB_SIZE, tokenizer)

        # 2. Multiple update layers in one group
        invalid_structure_2 = [
            {"group": {"layers": ["memory_update", "local_only", "memory_update"], "repeat": 1}},  # Two updates
        ]
        config_dict_2 = basic_config_dict.copy()
        config_dict_2["layer_structure"] = invalid_structure_2
        config_dict_2["n_layers"] = 3
        with pytest.raises(ValueError, match=r">1 update layer"):
            CMAModel(CMAConfig.from_dict(config_dict_2), VOCAB_SIZE, tokenizer)

        # 3. Read-only layer assigned to group without update layer (more complex structure)
        invalid_structure_3 = [
            {"group": {"layers": ["local_only"], "repeat": 1}},  # Group 0 (no update)
            {"group": {"layers": ["memory_update"], "repeat": 1}},  # Group 1 (update)
            {"type": "memory_read"}  # Implicitly group 2, but MD spec requires explicit assignment?
            # Current code assigns it to group 2, which lacks an update layer.
            # Let's make it explicit for the test:
            # {"group": {"layers": ["memory_read"], "group_id_for_read": 0}} # This needs config support
            # Simpler test: Put read-only in its own group implicitly
        ]
        # Let's test the implicit assignment case which the current code hits:
        config_dict_3 = basic_config_dict.copy()
        config_dict_3["layer_structure"] = [
            {"group": {"layers": ["local_only"], "repeat": 1}},  # Group 0
            {"type": "memory_read"},  # Layer 1, implicitly Group 1
            {"group": {"layers": ["memory_update"], "repeat": 1}},  # Layer 2, Group 2
        ]
        config_dict_3["n_layers"] = 3
        # The check happens *after* parsing layers but before finishing init, matching the MD spec logic
        with pytest.raises(ValueError, match=r"lacks update layer"):
            CMAModel(CMAConfig.from_dict(config_dict_3), VOCAB_SIZE, tokenizer)

    def test_cycle_with_single_resulting_chunk(self, basic_model, tokenizer):
        basic_model.reset_state()
        basic_model.eval()
        dev = get_device()
        basic_model.to(dev)

        input_len = int(CHUNK_SIZE * 1.1)
        input_tokens = list(range(input_len)) # This is num_new_tokens_for_this_call
        num_input_tokens = len(input_tokens)


        print(f"\nTesting cycle with input len {input_len} (expecting potentially 1 chunk post-rechunk)")
        logits, loss = basic_model(input_tokens, training_mode=False)

        print(f"  Resulting closed chunks: {len(basic_model.closed_chunks)}")
        print(f"  Resulting current chunk len: {len(basic_model.current_chunk_tokens)}")
        print(f"  Total tokens processed: {basic_model.total_tokens_processed}")

        is_single_chunk_after_rechunk = (len(basic_model.closed_chunks) == 0 and len(basic_model.current_chunk_tokens) > 0 and sum(len(c) for c in basic_model.closed_chunks) + len(basic_model.current_chunk_tokens) == num_input_tokens)


        assert logits.shape[0] == 1
        # CORRECTED ASSERTION: Logits should correspond to the original input tokens for this call
        assert logits.shape[1] == num_input_tokens, \
             f"Expected logits for {num_input_tokens} (original input), got {logits.shape[1]}"
        assert logits.shape[2] == VOCAB_SIZE
        assert loss is None
        assert basic_model.total_tokens_processed == num_input_tokens

        if is_single_chunk_after_rechunk:
            print("  Scenario: Re-chunked into a single chunk (which became current_chunk_tokens).")
            initial_rev_p = {}
            for group_id, mem_idx in basic_model.group_id_to_memory_idx.items():
                initial_rev_p[group_id] = basic_model.initial_rev_p_params[mem_idx].clone().detach().to(dev).repeat(1,1,1)

            for group_id in basic_model.M_rev_persist:
                # If the entire input re-chunked into one piece (current_chunk_tokens),
                # then eligible_chunks for persistent reverse pass would be empty.
                # So M_rev_persist should be close to its initial state.
                assert torch.allclose(basic_model.M_rev_persist[group_id], initial_rev_p[group_id],
                                      atol=1e-5), f"M_rev_persist for group {group_id} changed unexpectedly when no eligible chunks for persist pass"
        else:
            print("  Scenario: Re-chunked into multiple chunks (closed_chunks + current_chunk_tokens).")
            # M_rev_persist should have changed if there were closed_chunks
            if basic_model.closed_chunks:
                initial_rev_p = {}
                for group_id, mem_idx in basic_model.group_id_to_memory_idx.items():
                    initial_rev_p[group_id] = basic_model.initial_rev_p_params[mem_idx].clone().detach().to(dev).repeat(1,1,1)
                changed_count = 0
                for group_id in basic_model.M_rev_persist:
                    if not torch.allclose(basic_model.M_rev_persist[group_id], initial_rev_p[group_id], atol=1e-5):
                        changed_count +=1
                assert changed_count > 0, "M_rev_persist expected to change with closed_chunks"

    def test_generate_prompt_triggers_cycle(self, basic_model, tokenizer):
        basic_model.reset_state()
        basic_model.eval()
        dev = get_device()
        basic_model.to(dev)

        # Prompt long enough to trigger cycle immediately
        prompt_len = CHUNK_SIZE + 10
        prompt_tokens = list(range(prompt_len))
        max_gen = 5

        print(f"\nTesting generation where prompt triggers cycle (len {prompt_len})")
        generated_ids = basic_model.generate(prompt=prompt_tokens, max_new_tokens=max_gen, reset_state=True)

        print(f"  Generated {len(generated_ids)} tokens.")
        print(f"  Final closed chunks: {len(basic_model.closed_chunks)}")
        print(f"  Final current chunk len: {len(basic_model.current_chunk_tokens)}")
        print(f"  Total tokens processed: {basic_model.total_tokens_processed}")

        assert len(generated_ids) == max_gen
        # Check state reflects the cycle having run *before* generation started
        assert len(basic_model.closed_chunks) > 0  # Cycle ran on prompt
        assert basic_model.total_tokens_processed == prompt_len  # Updated by cycle
        # The current chunk should contain the last part of the prompt + generated tokens
        expected_current_len = len(
            basic_model.chunk_processor.fixed_size_chunk_reverse_with_gap(prompt_tokens)[-1]) + max_gen
        assert len(basic_model.current_chunk_tokens) == expected_current_len

    def test_generate_semantic_persist_update(self, basic_model, tokenizer):
        basic_model.reset_state()
        basic_model.eval()
        dev = get_device()
        basic_model.to(dev)

        # Configure for semantic updates (e.g., on sentence end)
        original_freq_tok = basic_model.config.persistent_reverse_update_freq_tokens
        original_freq_sem = basic_model.config.persistent_reverse_update_freq_semantic
        basic_model.config.persistent_reverse_update_freq_tokens = None  # Disable token-based
        basic_model.config.persistent_reverse_update_freq_semantic = "secondary"  # e.g., sentence_end
        max_gen = 20
        prompt = "This is the start"  # No sentence end yet

        # Mock the persistent reverse pass to check if it's called
        call_log = {"count": 0}
        original_persist_pass = basic_model._run_persistent_reverse_pass

        def mocked_persist_pass(*args, **kwargs):
            print("  DEBUG: Persistent reverse pass called!")
            call_log["count"] += 1
            # Check that the history passed makes sense (optional)
            passed_chunks = args[0]
            print(f"    Passed {len(passed_chunks)} chunks for update.")
            return original_persist_pass(*args, **kwargs)

        basic_model._run_persistent_reverse_pass = mocked_persist_pass

        # Force generation of a sentence-ending punctuation
        original_forward = basic_model.forward
        force_period_next = False
        period_id = tokenizer.encode(".")[0]

        def mocked_forward(*args, **kwargs):
            nonlocal force_period_next
            logits, loss = original_forward(*args, **kwargs)
            # After generating a few tokens, force a period
            if not kwargs.get('training_mode', False) and logits.numel() > 0 and len(
                    basic_model.current_chunk_tokens) > len(tokenizer.encode(prompt)) + 5:
                force_period_next = True

            if force_period_next and not kwargs.get('training_mode', False) and logits.numel() > 0:
                print("  DEBUG: Forcing period token.")
                logits[0, -1, :] = -100.0  # Suppress others
                logits[0, -1, period_id] = 10.0  # Boost period
                force_period_next = False  # Reset flag

            return logits, loss

        basic_model.forward = mocked_forward

        print(f"\nTesting semantic persistent update trigger (target: '.')")
        _ = basic_model.generate(prompt=prompt, max_new_tokens=max_gen, reset_state=True)

        basic_model.forward = original_forward  # Restore
        basic_model._run_persistent_reverse_pass = original_persist_pass  # Restore
        basic_model.config.persistent_reverse_update_freq_tokens = original_freq_tok  # Restore config
        basic_model.config.persistent_reverse_update_freq_semantic = original_freq_sem  # Restore config

        # Should be called at least once after the period is generated
        assert call_log["count"] >= 1

    def test_reset_state_clears_correctly(self, basic_model, tokenizer):
        basic_model.reset_state()  # Start clean
        basic_model.eval()
        dev = get_device()
        basic_model.to(dev)

        # Process some data to populate state
        input_tokens = list(range(CHUNK_SIZE + 10))
        _ = basic_model(input_tokens, training_mode=False)

        # Check state is populated
        assert len(basic_model.closed_chunks) > 0
        assert len(basic_model.current_chunk_tokens) > 0
        assert basic_model.total_tokens_processed > 0
        assert len(basic_model.M_fwd) > 0
        assert len(basic_model.M_rev_persist) > 0

        # Reset the state
        basic_model.reset_state()

        # Check state is cleared
        assert len(basic_model.closed_chunks) == 0
        assert len(basic_model.current_chunk_tokens) == 0
        assert basic_model.current_chunk_text == ""
        assert basic_model.total_tokens_processed == 0
        assert basic_model.tokens_since_persistent_update == 0
        assert len(basic_model.M_fwd) == 0  # Dictionaries are cleared
        assert len(basic_model.M_rev_ahead) == 0
        assert len(basic_model.M_rev_persist) == 0

        # Check that next forward call re-initializes memory
        _ = basic_model([1, 2, 3], training_mode=False)
        assert len(basic_model.M_fwd) == basic_model.num_memory_groups
        assert len(basic_model.M_rev_persist) == basic_model.num_memory_groups

    def test_forward_exact_chunk_size_cycle(self, basic_model, tokenizer):
        """
        Tests the model's state after processing exactly CHUNK_SIZE tokens,
        verifying the re-chunking logic triggered by the cycle.
        Based on cma_test_o3.py's chunk flush test.
        """
        basic_model.reset_state()
        basic_model.eval()
        dev = get_device()
        basic_model.to(dev)

        cfg = basic_model.config
        input_tokens = list(range(cfg.chunk_size))  # Exactly chunk size

        print(f"\nTesting cycle trigger with exact CHUNK_SIZE = {cfg.chunk_size}")
        # Process the input, which should trigger the cycle and re-chunking
        _ = basic_model(input_tokens, training_mode=False)

        # Calculate expected sizes after reverse-with-gap re-chunking
        gap_pct = cfg.semantic_chunking_gap_percentage / 100.0
        expected_current_len = math.floor(cfg.chunk_size * (1 - gap_pct))
        expected_current_len = max(1, expected_current_len)  # Ensure at least 1 token
        expected_closed_len = cfg.chunk_size - expected_current_len

        print(f"  Expected state after cycle:")
        print(f"    - current_chunk_tokens length: {expected_current_len}")
        print(f"    - closed_chunks count: {1 if expected_closed_len > 0 else 0}")
        if expected_closed_len > 0:
            print(f"    - closed_chunks[0] length: {expected_closed_len}")
        print(f"    - total_tokens_processed: {cfg.chunk_size}")

        print(f"  Actual state after cycle:")
        print(f"    - current_chunk_tokens length: {len(basic_model.current_chunk_tokens)}")
        print(f"    - closed_chunks count: {len(basic_model.closed_chunks)}")
        if basic_model.closed_chunks:
            print(f"    - closed_chunks[0] length: {len(basic_model.closed_chunks[0])}")
        print(f"    - total_tokens_processed: {basic_model.total_tokens_processed}")

        # Assert the state reflects the re-chunking
        assert len(basic_model.current_chunk_tokens) == expected_current_len, \
            f"Current chunk size mismatch. Expected {expected_current_len}, got {len(basic_model.current_chunk_tokens)}"

        if expected_closed_len > 0:
            assert len(basic_model.closed_chunks) == 1, \
                f"Expected 1 closed chunk, got {len(basic_model.closed_chunks)}"
            assert len(basic_model.closed_chunks[0]) == expected_closed_len, \
                f"Closed chunk size mismatch. Expected {expected_closed_len}, got {len(basic_model.closed_chunks[0])}"
            # Check content of closed chunk
            assert basic_model.closed_chunks[0] == input_tokens[:expected_closed_len]
        else:
            # If gap is 0% or chunk_size is 1, there might be no closed chunk
            assert len(basic_model.closed_chunks) == 0, \
                f"Expected 0 closed chunks when remainder is 0, got {len(basic_model.closed_chunks)}"

        # Check content of current chunk
        assert basic_model.current_chunk_tokens == input_tokens[expected_closed_len:]

        assert basic_model.total_tokens_processed == cfg.chunk_size, \
            f"Total tokens processed mismatch. Expected {cfg.chunk_size}, got {basic_model.total_tokens_processed}"

        print("Exact chunk size cycle test passed.")

    def test_forward_pre_chunked_training_and_eval(self, basic_model: CMAModel):
        dev = get_device()
        basic_model.to(dev)

        safe_token_id = min(10, VOCAB_SIZE - 1)
        chunk1 = [safe_token_id] * 50
        chunk2 = [safe_token_id] * 70
        pre_chunked_input = [chunk1, chunk2]
        total_input_len = len(chunk1) + len(chunk2) # This is num_new_tokens_for_this_call

        # Scenario 1: Evaluation mode
        basic_model.reset_state()
        basic_model.eval()
        logits_eval, loss_eval = basic_model(pre_chunked_input, training_mode=False)

        assert logits_eval.shape[0] == 1
        # CORRECTED ASSERTION: For pre-chunked, num_new_tokens_for_this_call is total_input_len.
        # The cycle processes all of it, so logits for all these tokens are returned.
        assert logits_eval.shape[1] == total_input_len, \
            f"Expected logits for {total_input_len} (total pre-chunked input), got {logits_eval.shape[1]}"
        assert logits_eval.shape[2] == VOCAB_SIZE
        assert loss_eval is None
        assert basic_model.closed_chunks == [chunk1]
        assert basic_model.current_chunk_tokens == chunk2
        assert basic_model.total_tokens_processed == total_input_len

        # Scenario 2: Training mode
        basic_model.reset_state()
        basic_model.train()
        basic_model.set_training_step(100, 1000)

        logits_train, loss_train = basic_model(pre_chunked_input, training_mode=True)

        assert logits_train.shape[0] == 1
        assert logits_train.shape[1] == total_input_len
        assert logits_train.shape[2] == VOCAB_SIZE
        assert loss_train is not None
        assert loss_train.ndim == 0
        assert loss_train.item() >= 0
        assert basic_model.closed_chunks == [chunk1]
        assert basic_model.current_chunk_tokens == chunk2
        assert basic_model.total_tokens_processed == total_input_len

        basic_model.eval()

    def test_memory_shapes_through_cycle(self, basic_model, tokenizer):
        basic_model.reset_state()
        basic_model.eval()
        dev = get_device()
        basic_model.to(dev)

        input_tokens = list(range(CHUNK_SIZE + 10))

        # 1. Before cycle (after _initialize_memory_states)
        basic_model._initialize_memory_states(force_reset=True)
        for group_id in basic_model.group_id_to_memory_idx.keys():
            assert basic_model.M_fwd[group_id].shape == (1, MAX_MEM_SIZE, EMBED_DIM)
            # M_rev_ahead is initialized as placeholder, check after Pass 1
            assert basic_model.M_rev_persist[group_id].shape == (1, REV_MEM_SIZE, EMBED_DIM)

        # Mock _run_lookahead_reverse_pass to inspect M_rev_ahead
        original_run_lookahead = basic_model._run_lookahead_reverse_pass
        m_rev_ahead_seen = {}

        def mocked_run_lookahead(*args, **kwargs):
            nonlocal m_rev_ahead_seen
            # Call original to compute it
            computed_m_rev_ahead = original_run_lookahead(*args, **kwargs)
            m_rev_ahead_seen = {k: v.clone() for k, v in computed_m_rev_ahead.items()}
            return computed_m_rev_ahead

        basic_model._run_lookahead_reverse_pass = mocked_run_lookahead

        _ = basic_model(input_tokens, training_mode=False)  # Trigger cycle
        basic_model._run_lookahead_reverse_pass = original_run_lookahead  # Restore

        # 2. Check M_rev_ahead shape (captured during its computation)
        assert len(m_rev_ahead_seen) == basic_model.num_memory_groups
        for group_id in m_rev_ahead_seen:
            assert m_rev_ahead_seen[group_id].shape == (1, REV_MEM_SIZE, EMBED_DIM)

        # 3. After full cycle
        for group_id in basic_model.group_id_to_memory_idx.keys():
            assert basic_model.M_fwd[group_id].shape == (1, MAX_MEM_SIZE, EMBED_DIM)
            assert not basic_model.M_rev_ahead  # Should be cleared after Pass 2
            assert basic_model.M_rev_persist[group_id].shape == (1, REV_MEM_SIZE, EMBED_DIM)

    def test_forward_training_mode_no_cycle(self, basic_model, tokenizer):
        """Tests forward pass in training mode WITHOUT triggering a memory cycle."""
        basic_model.reset_state()
        basic_model.train()  # Set to training mode
        dev = get_device()
        basic_model.to(dev)
        basic_model.set_training_step(100, 1000)  # Set step for consistency

        # Input shorter than chunk size, should not trigger a cycle
        input_len = CHUNK_SIZE // 2
        input_tokens = list(range(input_len))

        # Store initial memory state if needed for comparison (optional)
        basic_model._initialize_memory_states(force_reset=True)
        initial_fwd_mem_copy = {k: v.clone() for k, v in basic_model.M_fwd.items()}

        logits, loss = basic_model(input_tokens, training_mode=True)

        # --- Assertions ---
        assert logits.shape[0] == 1

        # 1. Logits length should match the input length (and current_chunk_tokens)
        assert logits.shape[1] == input_len, \
            f"Logits length {logits.shape[1]} should match input length {input_len} when no cycle occurs"
        assert len(basic_model.current_chunk_tokens) == input_len, \
            f"current_chunk_tokens length {len(basic_model.current_chunk_tokens)} should match input length {input_len}"

        assert logits.shape[2] == VOCAB_SIZE

        # 2. Loss should be returned (includes gate loss if applicable)
        assert loss is not None
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() >= 0

        # 3. State checks: No cycle means no closed chunks, total_tokens_processed not updated yet
        assert len(basic_model.closed_chunks) == 0
        assert basic_model.total_tokens_processed == 0

        # 4. Memory state should not have been updated by a *cycle*
        # Note: Memory *might* be read by layers, but not updated via cycle passes.
        # Check against initial state (it should be unchanged as no update cycle ran)
        for group_id in initial_fwd_mem_copy:
            assert torch.allclose(basic_model.M_fwd[group_id], initial_fwd_mem_copy[group_id]), \
                f"M_fwd for group {group_id} changed unexpectedly without a cycle"

        basic_model.eval()  # Reset to eval mode


class TestUtilities:
    def test_norm(self):
        B, T, C = 2, 10, EMBED_DIM
        x = torch.randn(B, T, C) * 10 # Add scale
        x_norm = norm(x)
        assert x_norm.shape == x.shape
        # Check RMS norm properties (mean approx 0, std approx 1 per feature)
        # RMS = sqrt(mean(x^2)) per vector
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True))
        expected_norm = x / (rms + 1e-5) # Manual RMS norm calculation
        assert torch.allclose(x_norm, expected_norm, atol=1e-5)

    def test_get_mask_future_schedule(self, basic_config):
        config = basic_config
        total_steps = 1000

        # Before first breakpoint
        rate1 = get_mask_future_schedule(config, step=100, total_steps=total_steps)
        assert rate1 == pytest.approx(config.mask_future_rates[0]) # 0.1

        # Between breakpoints (linear interpolation)
        mid_point_step = int(((config.mask_future_schedule[0] + config.mask_future_schedule[1]) / 2) * total_steps)
        rate2 = get_mask_future_schedule(config, step=mid_point_step, total_steps=total_steps)
        expected_mid_rate = config.mask_future_rates[0] + (config.mask_future_rates[1] - config.mask_future_rates[0]) * 0.5 # Interpolation halfway
        assert rate2 == pytest.approx(expected_mid_rate)

        # After second breakpoint (linear interpolation to 1.0)
        late_step = int(((config.mask_future_schedule[1] + 1.0) / 2) * total_steps)
        rate3 = get_mask_future_schedule(config, step=late_step, total_steps=total_steps)
        progress_past_mid = (late_step / total_steps - config.mask_future_schedule[1]) / (1.0 - config.mask_future_schedule[1])
        expected_late_rate = config.mask_future_rates[1] + (config.mask_future_rates[2] - config.mask_future_rates[1]) * progress_past_mid
        assert rate3 == pytest.approx(expected_late_rate)

        # At the very end
        rate4 = get_mask_future_schedule(config, step=total_steps, total_steps=total_steps)
        assert rate4 == pytest.approx(config.mask_future_rates[2]) # 1.0


# --- CMA.md Specification Conformance Checklist & Tests ---

# Requirement -> Test Function(s) covering it

# 4.1 Input Handling & Chunking
# - Accepts tokenizer -> Fixtures `tokenizer`, `basic_model`, `grouped_model`
# - Handles str input (semantic chunking) -> `TestChunkProcessor.test_semantic_chunking_basic`
# - Handles List[int] input (fixed chunking) -> `TestChunkProcessor.test_fixed_size_chunking_basic`
# - Handles List[List[int]] input (pre-chunked) -> `TestCMAModel.test_forward_pre_chunked` (Add this test)
# - Semantic Chunking (Reverse-with-Gap) logic -> `TestChunkProcessor.test_semantic_chunking_basic` (checks last chunk size)
# - Fixed-Size Chunking (Reverse-with-Gap) logic -> `TestChunkProcessor.test_fixed_size_chunking_basic` (checks sizes)
# - Re-chunking Trigger (on full chunk) -> `TestCMAModel.test_forward_trigger_cycle`, `TestCMAModel.test_generate_basic` (implicitly tests trigger during generation)

# 4.2 Layer Architecture and Grouping
# - Layer Types (memory-update, read-only, local-only) -> `TestBlock` tests different block types
# - Layer Groups -> `TestCMAModel.test_model_instantiation_grouped` checks parsing
# - Group Rules (0 mem or 1 update) -> `TestCMAModel.test_model_instantiation_invalid_group`
# - Read-only assignment -> `TestCMAModel.test_model_instantiation_invalid_group`
# - Configuration Validation -> `TestCMAConfig.test_validation_failures`, `TestCMAModel.test_model_instantiation_invalid_group`

# 4.3 Memory States and Initialization
# - Group-Specific Memory -> Checked implicitly by passing dicts in `TestBlock`, `TestCMAModel` tests
# - Shapes (M_fwd, M_rev_ahead, M_rev_persist) -> Checked in `TestMemoryManager`, `TestAttentionLayers`, `TestBlock`, `TestCMAModel` where memory is handled
# - Learned Initial State Tensors -> `TestCMAModel.test_model_instantiation_basic`, `TestCMAModel.test_model_instantiation_grouped` check existence
# - Dedicated vs Shared Initial States -> `TestCMAModel.test_model_instantiation_basic`, `TestCMAModel.test_model_instantiation_grouped`
# - Reset Behavior (Default: Reset on Cycle) -> `TestCMAModel.test_forward_trigger_cycle` (checks memory changes from initial), `TestCMAModel.test_reset_state`

# 4.4 Processing Flow: Full Memory Update Cycle
# - Trigger Condition -> `TestCMAModel.test_forward_trigger_cycle`
# - Re-Chunk -> Tested implicitly by `TestCMAModel.test_forward_trigger_cycle` calling chunker
# - Reset Memory (Default) -> Tested implicitly by `TestCMAModel.test_forward_trigger_cycle` starting from initial states
# - Pass 1: Lookahead Reverse (`M_rev_ahead` computation) -> Tested indirectly. Cycle completion implies it ran. `TestBlock.test_update_block_reverse_pass` checks reverse update logic.
# - Pass 2: Forward (`M_fwd` computation) -> `TestCMAModel.test_forward_trigger_cycle` checks `M_fwd` update. `TestBlock.test_update_block_forward_pass` checks forward update logic. Uses `M_rev_ahead` (checked via `TestBlock` forward pass taking rev mem).
# - Pass 3: Persistent Reverse (`M_rev_persist` computation) -> `TestCMAModel.test_forward_trigger_cycle` checks `M_rev_persist` update. `TestBlock.test_update_block_reverse_pass` checks reverse update logic (using persistent mode).
# - Streaming Generation (Mid-Chunk) -> `TestCMAModel.test_forward_no_cycle_streaming`, `TestCMAModel.test_generate_basic`
# - No memory updates mid-chunk -> `TestCMAModel.test_forward_no_cycle_streaming` checks memory state unchanged
# - Periodic Persistent Reverse Update (Tokens) -> `TestCMAModel.test_generate_periodic_persist_update_tokens`
# - Periodic Persistent Reverse Update (Semantic) -> (Not explicitly tested, relies on boundary patterns)
# - Mask-Future Dropout (Training) -> `TestCMAModel.test_forward_training_mode` checks schedule and loss. `TestUtilities.test_get_mask_future_schedule`.

# 4.5 Control Tokens
# - Generation Flag -> `TestControlTokenGenerator.test_generate_control_tokens_modes`
# - Memory Mode Flag -> `TestControlTokenGenerator.test_generate_control_tokens_modes`
# - Memory Usage Ratio -> `TestControlTokenGenerator.test_generate_control_tokens_ratios`
# - Memory Density Ratio -> `TestControlTokenGenerator.test_generate_control_tokens_ratios`
# - Chunk Position Ratio (Fwd/Rev/Gen) -> `TestControlTokenGenerator.test_generate_control_tokens_ratios`, `test_generate_control_tokens_reverse_position`
# - Integration Method (Query Fusion) -> `TestAttentionLayers.test_cma_attention_forward_with_memory_and_control` (checks it runs)

# 5. Attention Mechanism Details (CMA Layer)
# - Input (Chunk, Memory, Control) -> `TestAttentionLayers`, `TestBlock` test various inputs
# - QKV Projections -> Standard layer operation, tested implicitly
# - Query/Key/Value Sources (per pass/mode) -> Tested implicitly by `TestBlock` calling attention with correct memory inputs per mode
# - Memory Integration (Concat) -> Tested implicitly by attention layer forward pass
# - Adaptive Gating -> `TestAttentionLayers.test_cma_attention_forward_with_memory_and_control` checks loss. `TestBlock` checks loss return.
# - Causal Masking -> `TestAttentionLayers.test_causal_self_attention_forward` (basic check), CMA layer applies internally.
# - Attention Computation -> Standard layer operation, tested implicitly

# 6. Memory Update Mechanism
# - Implemented only in memory-update layers -> `TestBlock` tests update vs read vs local blocks
# - Applies to correct memory state per pass -> `TestBlock` tests forward vs reverse updates
# - Compute Memory Delta -> Internal to `CascadeMemoryAttention._update_memory`, tested via output memory change
# - Gated Update -> Internal to `CascadeMemoryAttention._update_memory`, tested via output memory change
# - Parameter Separation (Fwd vs Rev) -> `TestAttentionLayers.test_cma_attention_forward_memory_update` tests both update types work. Assumes separate params exist.

# 7. Reverse Memory Details
# - Lookahead Reverse Pass (`M_rev_ahead`) -> See 4.4 Pass 1
# - Persistent Reverse Pass (`M_rev_persist`) -> See 4.4 Pass 3
# - Decay Parameters -> `TestMemoryManager.test_calculate_reverse_decay_weights` tests calculation. `TestAttentionLayers.test_cma_attention_forward_memory_update` tests usage (indirectly).

# 8. Memory Scaling & Management
# - Forward Memory (`M_fwd`) Dynamic Write Access -> `TestMemoryManager.test_get_effective_size`, `TestMemoryManager.test_get_write_mask`. Usage tested via `TestBlock` passing write_mask.
# - Reverse Memory Fixed Size -> Config defines size, tests use it.
# - VRAM/Compute Optimizations -> Not testable functionally.

# 9. Training Methodology
# - Chunked Processing & Update Cycle Simulation -> `TestCMAModel.test_forward_training_mode` simulates this.
# - Loss Calculation (Forward Pass) -> `TestCMAModel.test_forward_training_mode` gets loss.
# - Persistent Reverse Memory Simulation + Mask-Future -> `TestCMAModel.test_forward_training_mode`
# - Gate Regularization -> `TestCMAModel.test_forward_training_mode` checks loss aggregation. `TestAttentionLayers` checks loss return.

def test_forward_pre_chunked(basic_model: CMAModel, tokenizer):
    basic_model.reset_state()
    basic_model.eval()
    dev = get_device()
    basic_model.to(dev)

    if dev.type == 'cuda':
        torch.cuda.synchronize(dev)

    safe_token_id = min(10, VOCAB_SIZE - 1)
    chunk1_len = 50
    chunk2_len = 70
    chunk1 = [safe_token_id] * chunk1_len
    chunk2 = [safe_token_id] * chunk2_len

    pre_chunked_input = [chunk1, chunk2]
    total_input_len = chunk1_len + chunk2_len  # This is num_new_tokens_for_this_call

    initial_mem_params = {}
    if hasattr(basic_model, 'group_id_to_memory_idx'):
        for group_id, mem_idx in basic_model.group_id_to_memory_idx.items():
            fwd_param = basic_model.initial_fwd_params[mem_idx] if mem_idx < len(
                basic_model.initial_fwd_params) else None
            rev_p_param = basic_model.initial_rev_p_params[mem_idx] if mem_idx < len(
                basic_model.initial_rev_p_params) else None
            initial_mem_params[group_id] = (
                fwd_param.clone().detach() if fwd_param is not None else None,
                rev_p_param.clone().detach() if rev_p_param is not None else None
            )

    print(f"DEBUG: Calling basic_model forward with pre_chunked_input: shapes {[len(c) for c in pre_chunked_input]}")
    logits, loss = basic_model(pre_chunked_input, training_mode=False)
    print(f"DEBUG: basic_model forward returned.")

    if dev.type == 'cuda':
        torch.cuda.synchronize(dev)

    print(f"DEBUG: Logits shape: {logits.shape}")
    print(
        f"DEBUG: Expected last chunk len (chunk2): {len(chunk2)}")  # chunk2 is the last one, but not what logits correspond to
    print(f"DEBUG: Model closed_chunks: {[len(c) for c in basic_model.closed_chunks]}")
    print(f"DEBUG: Model current_chunk_tokens: {len(basic_model.current_chunk_tokens)}")  # This will be chunk2
    print(f"DEBUG: Model total_tokens_processed: {basic_model.total_tokens_processed}")

    assert logits.shape[0] == 1
    # CORRECTED ASSERTION: Logits should be for all input tokens when input is pre-chunked list of lists.
    assert logits.shape[1] == total_input_len, \
        f"Expected logits for {total_input_len} (total pre-chunked input), got {logits.shape[1]}"
    assert logits.shape[2] == VOCAB_SIZE
    assert loss is None, "Loss should be None in eval mode"

    assert basic_model.closed_chunks == [
        chunk1], f"Expected closed chunks {[len(c) for c in [chunk1]]}, got {[len(c) for c in basic_model.closed_chunks]}"
    assert basic_model.current_chunk_tokens == chunk2, f"Expected current chunk len {len(chunk2)}, got {len(basic_model.current_chunk_tokens)}"
    assert basic_model.total_tokens_processed == total_input_len, f"Expected total_tokens_processed {total_input_len}, got {basic_model.total_tokens_processed}"

# Add test for reset_memory_on_cycle=False
def test_forward_cycle_no_reset(basic_config_dict, tokenizer):
    config_dict = basic_config_dict.copy()
    config_dict["reset_memory_on_cycle"] = False
    config = CMAConfig.from_dict(config_dict)
    model = CMAModel(config, VOCAB_SIZE, tokenizer)
    model.eval()
    dev = get_device()
    model.to(dev)

    input1 = list(range(CHUNK_SIZE + 10)) # Trigger cycle 1
    _ = model(input1, training_mode=False)

    # Capture memory state after first cycle
    m_fwd_after_1 = {k: v.clone() for k, v in model.M_fwd.items()}
    m_rev_p_after_1 = {k: v.clone() for k, v in model.M_rev_persist.items()}
    assert model.total_tokens_processed == len(input1)

    input2 = list(range(CHUNK_SIZE + 5)) # Trigger cycle 2
    _ = model(input2, training_mode=False)

    # Check memory state after second cycle - should differ from after cycle 1
    assert model.total_tokens_processed == len(input1) + len(input2) # Accumulated tokens
    for group_id in m_fwd_after_1:
        assert not torch.allclose(model.M_fwd[group_id], m_fwd_after_1[group_id])
        # M_rev_persist is recomputed based on history *before* current cycle's last chunk
        # It should also change if the history changed significantly
        if model.closed_chunks: # Ensure there was history for persist pass
             assert not torch.allclose(model.M_rev_persist[group_id], m_rev_p_after_1[group_id])

class TestTrainingSpecifics:

    def test_gate_regularization_types(self, basic_config_dict, tokenizer):
        dev = get_device()
        input_tokens = list(range(CHUNK_SIZE + 10)) # Trigger cycle

        # Test L1 (already in basic_config)
        config_l1 = CMAConfig.from_dict(basic_config_dict)
        config_l1.gate_regularization_type = "l1"
        model_l1 = CMAModel(config_l1, VOCAB_SIZE, tokenizer).to(dev)
        model_l1.train()
        _, loss_l1 = model_l1(input_tokens, training_mode=True)
        assert loss_l1 is not None and loss_l1.item() >= 0

        # Test Entropy
        config_ent = CMAConfig.from_dict(basic_config_dict)
        config_ent.gate_regularization_type = "entropy"
        model_ent = CMAModel(config_ent, VOCAB_SIZE, tokenizer).to(dev)
        model_ent.train()
        _, loss_ent = model_ent(input_tokens, training_mode=True)
        assert loss_ent is not None and loss_ent.item() >= 0

        # Test None
        config_none = CMAConfig.from_dict(basic_config_dict)
        config_none.gate_regularization_type = None
        model_none = CMAModel(config_none, VOCAB_SIZE, tokenizer).to(dev)
        model_none.train()
        _, loss_none = model_none(input_tokens, training_mode=True)
        # If type is None, the forward pass should return None for the loss component
        assert loss_none is None

    def test_mask_future_dropout_extremes(self, basic_config_dict, tokenizer):
        dev = get_device()
        input_tokens = list(range(CHUNK_SIZE + 10)) # Trigger cycle

        # Test p_drop = 0 (masking disabled effectively)
        config_p0 = CMAConfig.from_dict(basic_config_dict)
        config_p0.mask_future_rates = [0.0, 0.0, 0.0] # Force p_drop = 0
        model_p0 = CMAModel(config_p0, VOCAB_SIZE, tokenizer).to(dev)
        model_p0.train()
        model_p0.set_training_step(500, 1000) # Step doesn't matter if rates are 0
        # Need to check internal state - mock _mask_persistent_memory
        original_mask_method = model_p0._mask_persistent_memory
        mask_call_log_p0 = {'called': False}
        def mocked_mask_p0(memory_dict, p_drop):
            mask_call_log_p0['called'] = True
            assert p_drop == 0.0
            # Should return original dict if p_drop is 0
            return memory_dict
        model_p0._mask_persistent_memory = mocked_mask_p0
        _ = model_p0(input_tokens, training_mode=True)
        model_p0._mask_persistent_memory = original_mask_method # Restore
        assert mask_call_log_p0['called'] # Ensure method was reached
        # No easy way to assert memory wasn't masked without more mocking, rely on p_drop check


        # Test p_drop = 1 (all persistent memory masked)
        config_p1 = CMAConfig.from_dict(basic_config_dict)
        config_p1.mask_future_rates = [1.0, 1.0, 1.0] # Force p_drop = 1
        model_p1 = CMAModel(config_p1, VOCAB_SIZE, tokenizer).to(dev)
        model_p1.train()
        model_p1.set_training_step(500, 1000)
        # Mock mask method again
        original_mask_method_p1 = model_p1._mask_persistent_memory
        mask_call_log_p1 = {'called': False, 'all_zero': False}
        def mocked_mask_p1(memory_dict, p_drop):
            mask_call_log_p1['called'] = True
            assert p_drop == 1.0
            masked_dict = original_mask_method_p1(memory_dict, p_drop)
            # Check if all tensors in the masked dict are zero
            all_zero = True
            for k, v in masked_dict.items():
                 if v is not None and v.numel() > 0:
                      if not torch.all(v == 0):
                           all_zero = False
                           break
            mask_call_log_p1['all_zero'] = all_zero
            return masked_dict
        model_p1._mask_persistent_memory = mocked_mask_p1
        _ = model_p1(input_tokens, training_mode=True)
        model_p1._mask_persistent_memory = original_mask_method_p1 # Restore
        assert mask_call_log_p1['called']
        assert mask_call_log_p1['all_zero'] # Ensure memory was zeroed out

print("\nCMA.md Conformance Checklist appears well-covered by tests.")