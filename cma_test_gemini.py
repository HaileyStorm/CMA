import torch
import tiktoken
import yaml
from pathlib import Path
from typing import List, Dict, Any

from cma import (
    CMAConfig,
    ChunkProcessor,
    MemoryManager,
    ControlTokenGenerator,
    CausalSelfAttention,
    CascadeMemoryAttention,
    Block,
    CMAModel,
    norm,
    get_mask_future_schedule,
    BOUNDARY_PATTERNS # Import for ChunkProcessor test
)

print("--- Starting CMA Model Test Script ---")

# --- 0. Dummy Tokenizer ---
# Use tiktoken for basic encoding/decoding needed by some components
try:
    tokenizer = tiktoken.get_encoding("gpt2")
    VOCAB_SIZE = tokenizer.n_vocab
    print("Tiktoken tokenizer loaded.")
except Exception as e:
    print(f"Could not load tiktoken tokenizer: {e}")
    print("Using a mock tokenizer instead.")
    # Create a simple mock tokenizer if tiktoken fails
    class MockTokenizer:
        n_vocab = 5000
        def encode(self, text: str) -> List[int]:
            return [ord(c) % self.n_vocab for c in text[:100]] # Simple encoding
        def decode(self, tokens: List[int]) -> str:
             # Simple decoding, might not be accurate
             return "".join([chr(t) for t in tokens if t < 128])

    tokenizer = MockTokenizer()
    VOCAB_SIZE = tokenizer.n_vocab


# --- 1. Configuration ---
print("\n--- Testing Configuration Loading from YAML String ---")

# Define the configuration as a YAML formatted multi-line string
yaml_config_string = """
# Chunking
chunk_size: 768 # User specified
semantic_chunking_gap_percentage: 25.0 # Updated default
boundary_search_chars: [256, 64, 32] # Updated default (YAML uses lists)
# boundary_types: null # Omitted to use default from __post_init__
buffer_ratio: 0.1 # Updated default

# Memory
max_memory_size: 3072 # User specified
reverse_memory_size: 320 # User specified
initial_write_fraction: 0.6 # Updated default
memory_growth_function: "linear" # Same as old/new default
memory_cap_length: 49152 # User specified
# share_initial_memory: false # Omitted to use default (False)
# reset_memory_on_cycle: true # Omitted to use default (True)

# Reverse pass
reverse_max_chunks: 4 # Updated default
lookahead_reverse_decay_step: 0.2 # Same as old/new default
lookahead_reverse_decay_rate: 0.5 # Same as old/new default
persistent_reverse_decay_step: 0.05 # Same as old/new default
persistent_reverse_decay_rate: 0.1 # Same as old/new default
persistent_reverse_update_freq_tokens: 128 # Same as old/new default
persistent_reverse_update_freq_semantic: "secondary" # Same as old/new default

# Model architecture
embed_dim: 768 # User specified
n_heads: 6 # User specified
n_layers: 12 # Matches the length of the specified layer_structure
head_dim: 128 # User specified (Note: 768 / 6 = 128, consistent)
layer_structure: # User specified structure
  - type: local_only
  - type: memory_update
  - type: local_only
  - type: local_only
  - type: local_only
  - type: memory_update
  - type: no_attention
  - type: no_attention
  - type: local_only
  - type: local_only
  - type: local_only
  - type: memory_update
skip_attention_layers: [6] # Updated default (YAML uses lists)

# Control tokens
integration_method: "query_fusion" # Same as old/new default
ctrl_init_scale: 0.0001 # Updated default

# Initialization
memory_init_scale: 0.02 # Updated default
gate_bias_init: -1.0 # Same as old/new default
output_proj_zero_init: true # Updated default

# Adaptive gating regularization
gate_regularization_type: null # Updated default (null represents None)
gate_regularization_strength: 0.001 # Updated default

# Future‐masking schedule
mask_future_schedule: [0.3, 0.7] # Updated default (YAML uses lists)
mask_future_rates: [0.3, 0.5, 0.8] # Updated default (YAML uses lists)
# enable_mask_future_dropout: true # Omitted to use default (True)
"""

# Another layer structure to try"
"""
layer_structure:
  - type: local_only
  - type: memory_update
  - type: local_only
  - type: local_only
  - type: local_only
  - type: memory_update
  - type: no_attention
  - type: local_only
  - type: local_only
  - type: local_only
  - type: memory_update
  - type: memory_update
"""

# Parse the YAML string directly into a dictionary
try:
    test_config_dict = yaml.safe_load(yaml_config_string)
    print("YAML string parsed successfully.")

    # Use from_dict to create the config object
    # This effectively tests the loading logic from YAML syntax defined in the string
    config = CMAConfig.from_dict(test_config_dict)
    config.validate() # Check basic assertions
    print(f"CMAConfig created and validated from parsed YAML string.")

except yaml.YAMLError as e:
    print(f"Error parsing YAML string: {e}")
    # Handle error appropriately, maybe raise it or exit
    raise

# --- 2. Chunk Processor ---
print("\n--- Testing Chunk Processor ---")
chunk_processor = ChunkProcessor(config, tokenizer)
sample_text = """Section 1\n\nThis is the first paragraph. It has multiple sentences.
```python
def hello():
    print("Hello")
```

This is the second paragraph.\nAnother sentence."""
sample_tokens = tokenizer.encode(sample_text)

print(f"Testing semantic_chunk_reverse_with_gap on text (length {len(sample_text)})...")
semantic_chunks = chunk_processor.semantic_chunk_reverse_with_gap(sample_text)
print(f"Found {len(semantic_chunks)} semantic chunks. Sizes: {[len(c) for c in semantic_chunks]}")
assert all(len(c) <= config.chunk_size for c in semantic_chunks), "Semantic chunk size exceeded!"
# Basic check: should not be empty if text is not empty
assert bool(semantic_chunks) == bool(sample_text)

print(f"Testing fixed_size_chunk_reverse_with_gap on tokens (length {len(sample_tokens)})...")
fixed_chunks = chunk_processor.fixed_size_chunk_reverse_with_gap(sample_tokens)
print(f"Found {len(fixed_chunks)} fixed chunks. Sizes: {[len(c) for c in fixed_chunks]}")
assert all(len(c) <= config.chunk_size for c in fixed_chunks), "Fixed chunk size exceeded!"
# Basic check: should not be empty if tokens are not empty
assert bool(fixed_chunks) == bool(sample_tokens)


# --- 3. Memory Manager ---
print("\n--- Testing Memory Manager ---")
memory_manager = MemoryManager(config)
seq_len_test = 512
print(f"Testing get_effective_size for seq_len={seq_len_test}...")
eff_size = memory_manager.get_effective_size(seq_len_test)
print(f"Effective memory size: {eff_size}")
assert 0 <= eff_size <= config.max_memory_size

#print("Testing get_write_mask...")
#write_mask = memory_manager.get_write_mask(current_chunk_idx=1, total_chunks=3, seq_len=seq_len_test, batch_size=1)
#print(f"Write mask shape: {write_mask.shape}, Sum: {write_mask.sum()}")
#assert write_mask.shape == (1, config.max_memory_size)

#print("Testing apply_downweighting...")
#dummy_memory = torch.randn(1, config.reverse_memory_size, config.embed_dim)
#downweighted_mem = memory_manager.apply_downweighting(dummy_memory, chunk_indices=[0, 1], is_reverse=True, is_persistent=False)
#print(f"Downweighted memory shape: {downweighted_mem.shape}")
#assert downweighted_mem.shape == dummy_memory.shape
# Check if values changed (simple check)
#assert not torch.equal(downweighted_mem, dummy_memory)


# --- 4. Control Token Generator ---
print("\n--- Testing Control Token Generator ---")
control_gen = ControlTokenGenerator(config)
print("Generating control tokens for 'forward' mode...")
ctrl_fwd = control_gen.generate_control_tokens(
    mode="forward", current_chunk_idx=1, total_chunks=3,
    current_mem_size=eff_size, max_mem_size=config.max_memory_size, seq_len=seq_len_test
)
print(f"Forward control tokens: {ctrl_fwd}")
assert len(ctrl_fwd) == 5

print("Generating control tokens for 'lookahead_reverse' mode...")
ctrl_rev = control_gen.generate_control_tokens(
    mode="lookahead_reverse", current_chunk_idx=0, total_chunks=3, # current_chunk_idx not used directly here
    current_mem_size=eff_size // 2, max_mem_size=config.max_memory_size, seq_len=seq_len_test,
    reverse_chunk_idx=0, reverse_window_size=config.reverse_max_chunks
)
print(f"lookahead reverse control tokens: {ctrl_rev}")
assert len(ctrl_rev) == 5
assert ctrl_rev["memory_mode_flag"] == 1.0


# --- 5. Attention Layers ---
print("\n--- Testing Attention Layers ---")
B, T, C = 1, config.chunk_size, config.embed_dim
M_fwd, M_rev = config.max_memory_size, config.reverse_memory_size
dummy_x = torch.randn(B, T, C)
dummy_fwd_mem = torch.randn(B, M_fwd, C)
dummy_rev_mem = torch.randn(B, M_rev, C)
dummy_ctrl = ctrl_fwd # Use forward controls for basic test

# Test CausalSelfAttention
print("Testing CausalSelfAttention...")
causal_attn = CausalSelfAttention(config, layer_idx=0)
causal_out = causal_attn(dummy_x)
print(f"CausalSelfAttention output shape: {causal_out.shape}")
assert causal_out.shape == (B, T, C)

# Test CascadeMemoryAttention (Read Mode)
print("Testing CascadeMemoryAttention (Read Mode)...")
cma_read = CascadeMemoryAttention(config, layer_idx=1, is_memory_update=False)
cma_read_out, _, _, gate_loss = cma_read(
    dummy_x,
    forward_memory=dummy_fwd_mem,
    reverse_memory=dummy_rev_mem,
    control_tokens=dummy_ctrl
)
print(f"CMA Read output shape: {cma_read_out.shape}")
assert cma_read_out.shape == (B, T, C)
print(f"Gate regularization loss (read): {gate_loss}")
assert gate_loss is None or isinstance(gate_loss, torch.Tensor)

# Test CascadeMemoryAttention (Update Mode)
print("Testing CascadeMemoryAttention (Update Mode)...")
cma_update = CascadeMemoryAttention(config, layer_idx=2, is_memory_update=True)
dummy_write_mask = memory_manager.get_write_mask(1, 3, T, B)
dummy_decay = torch.ones_like(dummy_rev_mem) # Simple decay for testing

# Test forward update
cma_upd_out, upd_fwd, upd_rev, gate_loss = cma_update(
    dummy_x,
    forward_memory=dummy_fwd_mem,
    reverse_memory=dummy_rev_mem, # Provide both for dimension checks
    control_tokens=dummy_ctrl,
    do_memory_update=True,
    write_mask=dummy_write_mask,
    is_reverse_update=False # Test forward update
)
print(f"CMA Update output shape: {cma_upd_out.shape}")
print(f"Updated forward memory shape: {upd_fwd.shape}")
print(f"Updated reverse memory (should be None): {upd_rev}")
print(f"Gate regularization loss (update): {gate_loss}")
assert cma_upd_out.shape == (B, T, C)
assert upd_fwd.shape == (B, M_fwd, C)
assert upd_rev is None # Should not update reverse in this call
assert gate_loss is None or isinstance(gate_loss, torch.Tensor)
assert not torch.equal(upd_fwd, dummy_fwd_mem) # Check memory changed


# Test reverse update
cma_upd_out_rev, upd_fwd_rev, upd_rev_rev, _ = cma_update(
    dummy_x,
    forward_memory=dummy_fwd_mem, # Provide both for dimension checks
    reverse_memory=dummy_rev_mem,
    control_tokens=ctrl_rev, # Use reverse controls
    do_memory_update=True,
    write_mask=None, # No specific write mask for reverse in this test setup
    decay_weights=dummy_decay,
    is_reverse_update=True # Test reverse update
)
print(f"Updated forward memory (should be None): {upd_fwd_rev}")
print(f"Updated reverse memory shape: {upd_rev_rev.shape}")
assert upd_fwd_rev is None # Should not update forward in this call
assert upd_rev_rev.shape == (B, M_rev, C)
assert not torch.equal(upd_rev_rev, dummy_rev_mem) # Check memory changed


# --- 6. Block ---
#print("\n--- Testing Block ---")
# Test a local_only block
#block_local = Block(config, layer_idx=0, layer_type="local_only")
#block_out, _, _, _ = block_local(dummy_x)
#print(f"Local Block output shape: {block_out.shape}")
#assert block_out.shape == (B, T, C)

# Test a memory_update block
#block_update = Block(config, layer_idx=1, layer_type="memory_update")
#block_out_upd, fwd_mem_upd, rev_mem_upd, loss_upd = block_update(
#    dummy_x,
#    forward_memory=dummy_fwd_mem,
#    reverse_memory=dummy_rev_mem,
#   control_tokens=dummy_ctrl,
#    do_memory_update=True,
#    write_mask=dummy_write_mask,
#    decay_weights=dummy_decay,
#    is_reverse_update=False # Test forward update within block
#)
#print(f"Update Block output shape: {block_out_upd.shape}")
#print(f"Update Block forward memory shape: {fwd_mem_upd.shape}")
#print(f"Update Block reverse memory (None): {rev_mem_upd}")
#print(f"Update Block gate loss: {loss_upd}")
#assert block_out_upd.shape == (B, T, C)
#assert fwd_mem_upd.shape == (B, M_fwd, C)
#assert rev_mem_upd is None


# --- 7. Utility Functions ---
print("\n--- Testing Utility Functions ---")
print("Testing norm...")
norm_out = norm(dummy_x)
print(f"Norm output shape: {norm_out.shape}, Mean: {norm_out.mean():.4f}, Std: {norm_out.std():.4f}")
assert norm_out.shape == dummy_x.shape

print("Testing get_mask_future_schedule...")
rate1 = get_mask_future_schedule(config=config, step=100, total_steps=1000)
rate2 = get_mask_future_schedule(config=config, step=500, total_steps=1000)
rate3 = get_mask_future_schedule(config=config, step=900, total_steps=1000)
print(f"Mask future rates at steps 100, 500, 900: {rate1:.3f}, {rate2:.3f}, {rate3:.3f}")
assert 0.0 <= rate1 <= 1.0 and 0.0 <= rate2 <= 1.0 and 0.0 <= rate3 <= 1.0


# --- 8. Full Model ---
print("\n--- Testing Full CMAModel ---")
model = CMAModel(config, vocab_size=VOCAB_SIZE, tokenizer=tokenizer)
model.eval() # Set to eval mode for basic tests (disables dropout if any)

# Test forward pass with string input (will likely trigger chunking logic internally)
print("Testing model.forward() with string input...")
test_input_string = "This is a test sequence that might be longer than the chunk size."
with torch.no_grad():
    output_dict_str, _ = model(test_input_string, training_mode=False)
logits_str = output_dict_str
mem_states_str = {'forward': model.M_fwd, 'reverse_persistent': model.M_rev_persist}
print(f"Logits shape (string input): {logits_str.shape}")
#print(f"Forward memory state shape: {mem_states_str['forward'].shape}")
#print(f"Reverse persistent memory state shape: {mem_states_str['reverse_persistent'].shape}")
assert len(logits_str.shape) == 3 and logits_str.shape[0] == 1 and logits_str.shape[2] == VOCAB_SIZE
#assert mem_states_str['forward'].shape == (1, M_fwd, C)
#assert mem_states_str['reverse_persistent'].shape == (1, M_rev, C)
model.reset_state()

# Test forward pass with token list input
print("\nTesting model.forward() with token list input...")
test_input_tokens = tokenizer.encode("Another short test.")
with torch.no_grad():
     output_dict_tok, _ = model(test_input_tokens, training_mode=False)
logits_tok = output_dict_tok
print(f"Logits shape (token input): {logits_tok.shape}")
assert len(logits_tok.shape) == 3 and logits_tok.shape[0] == 1 and logits_tok.shape[2] == VOCAB_SIZE
# Check if memory states were updated (might not change much with short input)
print("Memory states likely updated internally.")
model.reset_state()

# Test forward pass in training mode (simulated)
print("\nTesting model.forward() in training_mode=True...")
model.train()
model.set_training_step(100, 1000) # For mask_future test
test_input_tokens_train = tokenizer.encode("Training data chunk.")
# Normally you'd compute loss here, but we just check execution
with torch.no_grad(): # Still use no_grad for testing, just checking if it runs
     output_dict_train, _ = model(test_input_tokens_train, training_mode=True)
logits_train = output_dict_train
print(f"Logits shape (training mode): {logits_train.shape}")
assert len(logits_train.shape) == 3
model.reset_state()

model.eval() # Reset to eval

# Test _trigger_memory_update_cycle (indirectly via long input)
print("\nTesting internal memory update cycle (via long input)...")
long_input_tokens = tokenizer.encode(sample_text * 3) # Make it long enough to trigger chunking
print(f"Processing long input ({len(long_input_tokens)} tokens)...")
with torch.no_grad():
     output_long, _ = model(long_input_tokens, training_mode=False)
print("Long input processed, internal cycles likely triggered.")
# Check if internal state looks reasonable
assert model.M_fwd is not None and model.M_rev_persist is not None
#print(f"Final forward memory shape: {model.M_fwd.shape}")
#print(f"Final reverse persistent memory shape: {model.M_rev_persist.shape}")
model.reset_state()

# Test generation (briefly)
print("\nTesting model.generate()...")
context = "Once upon a time"
max_new = 5
with torch.no_grad():
    generated_tokens = model.generate(context, max_new_tokens=max_new, temperature=0.7)
print(f"Generated {len(generated_tokens)} tokens: {generated_tokens}")
print(f"Decoded: {tokenizer.decode(generated_tokens)}")
assert len(generated_tokens) == max_new


print("\n--- CMA Model Test Script Completed Successfully ---")
