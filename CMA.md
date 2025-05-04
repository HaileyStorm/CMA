**Project Description: Cascade Memory Attention (CMA)**

**1. Name:**
Cascade Memory Attention (CMA)

**2. Motivation & Goals**

Standard Transformer attention mechanisms suffer from quadratic scaling in compute and memory complexity with respect to sequence length, severely limiting their practical context window. While recent architectures have introduced techniques like chunking, recurrence, or low-rank approximations, CMA aims to create a synergistic approach combining their strengths.

**Goals:**

* **Effectiveness:** Achieve performance comparable to or exceeding standard attention models on tasks requiring long-range reasoning and coherence.
* **Infinite Sequence Length:** Handle arbitrarily long sequences with fixed VRAM usage during inference, independent of total sequence length.
* **Efficient Compute:** Achieve near-linear compute scaling for long sequences. Minimize redundant computation between related inference steps (e.g., conversational turns).
* **Compatibility:** Integrate smoothly into existing Transformer model structures and training frameworks.
* **Flexible Training:** Support stable training with techniques like curriculum learning for chunk size and memory capacity.

**3. Core Concepts**

CMA processes sequences in chunks, maintaining compressed representations of past information in dedicated memory states associated with specific layer groups. Key components include:

*   **Chunked Processing:** Input sequences are segmented into manageable chunks (see 4.1).
*   **Layer Groups & Types:** Layers are organized into groups. Each group contains either zero memory layers, or exactly one **CMA memory-update layer**, optionally multiple **CMA read-only layers**, and optionally multiple **local-only layers**. (see 4.2).
*   **Group-Specific Memory:** Each layer group containing a memory-update layer manages its own independent memory states.
*   **Forward Memory (`M_fwd`):** A primary memory state accumulating context information as the model processes chunks sequentially forward during the Forward Pass. Persists across chunks within an update cycle.
*   **Lookahead Reverse Memory (`M_rev_std`):** A memory state computed via a backward pass over recent chunks (including the current chunk) *before* the Forward Pass. Its purpose is to provide immediate preceding context to the Forward Pass. It is discarded after the Forward Pass completes.
*   **Persistent Reverse Memory (`M_rev_persist`):** A backward memory state computed over recent chunks *excluding* the current chunk, calculated *after* the Forward Pass. It is maintained between full update cycles (e.g., during streaming generation) to provide continuity and access to recent context not yet consolidated into forward memory.
*   **Adaptive Gating:** A mechanism allowing the model to dynamically control the influence of memory components per token (see 5).
*   **Control Tokens:** Scalar values fused to queries indicating operational mode and progress (see 4.5).

**4. Model Architecture & Workflow**

**4.1. Input Handling & Chunking**

*   The model accepts a `tokenizer` during initialization.
*   The `forward` method handles multiple input types:
    *   `str`: Raw text input. Triggers **semantic chunking (reverse-with-gap)**.
    *   `List[int]`: Flat list of token IDs. Triggers **fixed-size chunking (reverse-with-gap)**.
    *   `List[List[int]]`: Pre-chunked token IDs. Used directly (primarily for specific training/evaluation scenarios).
*   **Semantic Chunking (Reverse-with-Gap - for `str` input):** This is the primary method for defining processing boundaries. It is performed once per input sequence (e.g., document, conversational turn) and subsequently whenever the current chunk fills during generation.
    1.  Identify the total sequence length. Let chunk indices run `0, ..., N`, where `N` is the last/current chunk.
    2.  Start defining boundaries from the end (`k=N` down to `k=1`). For each chunk `k`:
        a.  Determine the known end position `end_k` (start of chunk `k+1`, or sequence end for `k=N`).
        b.  Determine the `target_size` for this chunk (`chunk_size`, adjusted by `gap_percentage` if `k=N`). For optional dynamic chunk sizing, `chunk_size` itself may vary based on total sequence length (see 12).
        c.  Calculate an initial estimated start position: `est_start_k = max(0, end_k - target_size)`.
        d.  Optionally, bias the estimate slightly later: `search_start_point = max(0, est_start_k + buffer_chars)` where `buffer_chars` might be a fraction of `boundary_search_chars[0]`. This provides a margin if the estimate is slightly too early, producing an overly large chunk.
        e.  **Search backward** from `search_start_point` within the configured primary `boundary_search_chars` distance for preferred semantic boundaries (e.g., section break).
        f.  If not found, search backward from `search_start_point` within the secondary character distance for secondary boundaries (e.g., sentence end).
        g.  If not found, search backward from `search_start_point` within the tertiary character distance for tertiary boundaries (e.g., clause end).
        h.  If a boundary is found at `boundary_pos <= search_start_point`, use `boundary_pos` as the actual start of chunk `k`.
        i.  If no boundary is found in any search, use the original `est_start_k` (or perhaps `search_start_point` if biased) as the start of chunk `k`.
        j.  Tokenize the chunk (and save it). If the chunk is too large, re-do the step 2 with a later estimated start position.
    3.  Repeat step 2 for each chunk until the start of chunk 1 is defined. Chunk 0 runs from the beginning of the sequence to the start of chunk 1.
    4.  **Re-chunking Trigger:** This entire reverse-semantic chunking process is performed *once per input sequence* (e.g., document, conversational turn, initial inference context) and subsequently *only* when the "current chunk" (`N`) becomes full (reaches `chunk_size` tokens) through generation. This event also triggers the full memory update cycle (see 4.4).
    * **Determinism & Reproducibility:** Ensure heuristics, buffer values, and tokenizer settings are frozen.

* **Fixed-Size Chunking (Reverse-with-Gap - for `List[int]` input):** This method defines boundaries using fixed lengths, mirroring the reverse-with-gap structure. It is performed once per input sequence and subsequently whenever the current chunk fills.
    1.  Identify the total sequence length in tokens (`total_len`).
    2.  Calculate the target size for the last/current chunk: `last_chunk_target_size = floor(chunk_size * (1 - semantic_chunking_gap_percentage / 100.0))`. Ensure this is at least 1. For optional dynamic chunk sizing, `chunk_size` itself may vary (see 12).
    3.  Define the start index of the last/current chunk: `last_chunk_start = max(0, total_len - last_chunk_target_size)`. The last chunk is `tokens[last_chunk_start:]`.
    4.  Initialize `current_boundary = last_chunk_start`.
    5.  Iteratively define preceding chunk boundaries working backward:
        * While `current_boundary > 0`:
            * Calculate the start index of the next chunk going backward: `prev_chunk_start = max(0, current_boundary - chunk_size)`.
            * The chunk is `tokens[prev_chunk_start:current_boundary]`.
            * Set `current_boundary = prev_chunk_start`.
    6.  This process defines a list of chunks. The first chunk (`tokens[0:current_boundary]` from the last step) may be smaller than `chunk_size`. The last chunk has the target size determined by the gap percentage. All intermediate chunks have size `chunk_size`.
    7.  **Re-chunking Trigger:** Like semantic chunking, this fixed-size reverse chunking process is performed *once per input sequence* and subsequently *only* when the "current chunk" (the last chunk segment) becomes full (reaches `chunk_size` tokens) through generation. This event triggers the full memory update cycle (see 4.4).

**4.2. Layer Architecture and Grouping**

*   The CMA architecture consists of three distinct layer types:
    1.  **CMA memory-update layers**: These layers read from and update the shared memory state of their assigned group. They implement the memory update mechanism (see 6).
    2.  **CMA read-only layers**: These layers read from the shared memory state of their assigned group but *never* update it.
    3.  **Local-only layers**: These layers operate only on the tokens within the current chunk and do *not* interact with any memory state.
*   Layers are organized into one or more **layer groups**.
*   Each layer group **must** contain either:
    *   Exactly zero CMA memory-update layers and zero CMA read-only layers (i.e., only local-only layers).
    *   Or, exactly one CMA memory-update layer. This group may optionally contain zero or more CMA read-only layers and zero or more local-only layers.
*   **Group Assignment:**
    *   CMA read-only layers *must* be explicitly assigned to a group that contains a CMA memory-update layer.
    *   Local-only layers can optionally be assigned to any group for organizational clarity but do not affect memory logic.
    *   If no read-only layers exist and no explicit groups are defined, each CMA memory-update layer implicitly forms its own single-layer group.
*   **Configuration Validation:** Implementations should include assertions to validate the layer group configuration (e.g., raise an error if a read-only layer is defined without a valid group assignment).

**4.3. Memory States and Initialization**

*   **Group-Specific Memory:** Each layer group containing a CMA memory-update layer manages its own independent set of memory state tensors: `M_fwd`, `M_rev_std`, `M_rev_persist`.
*   **Shapes:**
    *   Forward Memory (`M_fwd`): `(B, max_memory_size, D)`. Write access is dynamically scaled (see 8), read access is always to the full size.
    *   Lookahead Reverse Memory (`M_rev_std`): `(B, reverse_memory_size, D)`. Fixed size.
    *   Persistent Reverse Memory (`M_rev_persist`): `(B, reverse_memory_size, D)`. Fixed size.
*   **Initialization:**
    *   Each memory state tensor (`M_fwd`, `M_rev_std`, `M_rev_persist`) for each group is initialized from a corresponding **learned initial state tensor** of the same shape (`initial_M_fwd`, `initial_M_rev_std`, `initial_M_rev_persist`).
    *   These initial state tensors are parameters of the model, updated during training.
    *   **Default:** Each group (identified by its CMA memory-update layer) has its *own dedicated set* of learned initial state parameters.
    *   **Option:** A configuration flag can allow sharing these initial state parameters globally across all groups.
*   **Reset Behavior:** By default, at the beginning of each full memory update cycle (see 4.4), all active memory states (`M_fwd`, `M_rev_std`, `M_rev_persist`) for all groups are **reset** to their respective learned initial state values. An optional configuration can disable this reset, allowing memory states to persist across update cycles (useful for continuous processing scenarios, only resetting on explicit command or model load).

**4.4. Processing Flow: The Full Memory Update Cycle**

A full memory update cycle is triggered **once per input sequence** (e.g., document, conversational turn, initial inference context) and subsequently whenever the **current chunk fills** during generation. This cycle involves re-chunking the entire current sequence and executing three distinct passes in strict order for each layer group containing a memory-update layer.

1.  **Preparation:**
    *   **Re-Chunk:** Perform reverse-semantic-with-gap (or fixed-size) chunking (as described in 4.1) over the *entire* current sequence to define new, consistent chunk boundaries. Let the last chunk be chunk `N`.
    *   **(Default) Reset Memory:** Reset `M_fwd`, `M_rev_std`, and `M_rev_persist` for all groups to their learned initial states (unless persistence across cycles is enabled).
    *   **Update Control Tokens:** Initialize control tokens for the Lookahead Reverse Pass (see 4.5).

2.  **Pass 1: Lookahead Reverse Pass (`M_rev_std` Computation)**
    *   **Purpose:** Compute `M_rev_std` to provide lookahead context (from the current chunk backward) for the subsequent Forward Pass.
    *   **Direction:** Processes chunks backward, starting from the end of the sequence (chunk `N`).
    *   **Scope:** Includes the current chunk `N` and proceeds backward (`N, N-1, ..., N-k+1`) up to `reverse_max_chunks` or the start of the sequence. Uses the newly defined boundaries.
    *   **Processing:** For each chunk in the reverse scope:
        *   Pass the chunk through all layers of the model sequentially.
        *   **Attention:** CMA layers (update and read-only) attend to the current chunk's tokens, the *current* state of `M_rev_std` for their group, and the control tokens.
        *   **Memory Update:** Only the **CMA memory-update layer** in each group updates its group's `M_rev_std` state based on the processing of this chunk, using the Lookahead Reverse decay parameters (see 7) and the memory update mechanism (see 6).
    *   **Result:** The final `M_rev_std` state for each group is stored temporarily.

3.  **Pass 2: Forward Pass (`M_fwd` Computation)**
    *   **Purpose:** Process the sequence chronologically and update the main forward memory state `M_fwd`.
    *   **Direction:** Processes chunks sequentially forward, starting from the first chunk (chunk 0) up to and including the last chunk (`N`). Uses the newly defined boundaries.
    *   **Processing:** For each chunk from 0 to `N`:
        *   Update control tokens for the Forward Pass.
        *   Pass the chunk through all layers of the model sequentially.
        *   **Attention:** CMA layers (update and read-only) attend to the current chunk's tokens, the *current* state of `M_fwd` for their group, the *final* `M_rev_std` computed in Pass 1 (read-only), and the control tokens.
        *   **Memory Update:** Only the **CMA memory-update layer** in each group updates its group's `M_fwd` state based on the processing of this chunk, using the memory update mechanism (see 6). The updated `M_fwd` is carried over to the next chunk in this pass.
    *   **Result:** The final `M_fwd` state for each group (after processing chunk `N`) is stored for subsequent generation steps. The `M_rev_std` computed in Pass 1 is now discarded.

4.  **Pass 3: Persistent Reverse Pass (`M_rev_persist` Computation)**
    *   **Purpose:** Compute `M_rev_persist` to provide recent context (excluding the current chunk) for subsequent generation steps between full update cycles.
    *   **Direction:** Processes chunks backward, starting from the chunk *before* the last one (chunk `N-1`).
    *   **Scope:** Excludes the current chunk `N`. Proceeds backward (`N-1, N-2, ..., N-k`) up to `reverse_max_chunks` or the start of the sequence. Uses the newly defined boundaries.
    *   **Processing:** For each chunk in this reverse scope:
        *   Update control tokens for the Persistent Reverse Pass.
        *   Pass the chunk through all layers of the model sequentially.
        *   **Attention:** CMA layers (update and read-only) attend to the current chunk's tokens, the *current* state of `M_rev_persist` for their group, and the control tokens.
        *   **Memory Update:** Only the **CMA memory-update layer** in each group updates its group's `M_rev_persist` state based on the processing of this chunk, using the persistent reverse decay parameters (see 7) and the memory update mechanism (see 6).
    *   **Result:** The final `M_rev_persist` state for each group is stored for subsequent generation steps.

* **During Streaming Generation (Mid-Chunk):** This occurs when the current chunk is *not yet full* and a full update cycle is not triggered.
    1.  **Check if Chunk Fills:** If adding the next generated token *would* fill the current chunk (reach `chunk_size`), trigger the **Full Memory Update Cycle** (Steps 1-4 above) *first* before generating the token.
    2.  **Generate Token (if chunk not full):**
        *   **Input to Attention:** Let `X_partial` be the sequence of token embeddings generated so far in the *current, partially filled* chunk. The query `q` is derived from the embedding corresponding to the position for the *next* token. Context for `K`, `V` consists of `X_partial` concatenated with the latest `M_fwd` and `M_rev_persist` from *all* relevant groups.
        *   **Layer Processing:** Pass the query and context through all layers.
            *   Local-only layers attend only within `X_partial`.
            *   CMA read-only and memory-update layers attend to `X_partial`, `M_fwd`, and `M_rev_persist` (from their respective groups).
        *   **Causal Masking:** Applied within `X_partial`. Full attention to `M_fwd` and `M_rev_persist`.
        *   Fuse control tokens (reflecting mid-chunk generation mode, ratio flags continue to update) to the query projection (see 4.5).
        *   Generate the next token. Append it to `X_partial`.
        *   **No memory updates occur during mid-chunk generation.**
    3.  **Periodic Persistent Reverse Memory Update (Optional):** Can be triggered independently by token count (`persistent_reverse_update_freq['tokens']`) or semantic signal since the *last* persistent update. This re-runs *only* Pass 3 (Persistent Reverse Pass) over chunks `N-1, N-2, ...` (using the *existing* boundaries, excluding the current partial chunk `N`) to update `M_rev_persist` for all groups. Does *not* trigger Lookahead Reverse or Forward passes and does not reset memory.

* **Training/Inference Parity (Persistent Reverse Memory):**
  - During training, explicitly simulate inference conditions by applying random *mask-future dropout* to the persistent reverse memory state (`M_rev_persist`) provided during the Forward Pass (Pass 2). When simulating the processing of chunk `i` in the forward pass, instead of always providing the `M_rev_persist` computed from ground truth future chunks (`i-1, i-2, …`), occasionally mask or omit parts of this memory with probability `p_mask_future`. This mitigates exposure bias. *Note: This applies to the M_rev_persist used as context during generation/forward passes, not during its own computation in Pass 3.*

* **Recommended dropout schedule:**
  - Start with moderate dropout (`p_mask_future ≈ 0.3`) early in training, and gradually increase (`p_mask_future ≈ 0.7–0.9`) toward the end of training to encourage robust generalization.

**4.5. Control Tokens**

Before processing each chunk (or generating a token), several scalar float values representing the current mode of operation and progress are fused to the query projection in CMA layers:

*   **Generation Flag:** Indicates whether the model is currently processing memory during a full update cycle (0.0), or streaming/generating mid-chunk (1.0).
*   **Memory Mode Flag:** Indicates the current pass within the update cycle or generation mode (e.g., `0.0` for Forward Pass, `1.0` for Lookahead Reverse Pass, `0.8` for Persistent Reverse Pass/Update, potentially another value like `0.5` for mid-chunk generation).
*   **Memory Usage Ratio (`current_mem / max_mem`):** Ratio of the currently *writable* effective forward memory size to the maximum possible forward memory size (`max_memory_size`).
*   **Memory Density Ratio (`current_mem / seq_len`):** Ratio of currently writable forward memory size to the total sequence length processed so far (in tokens).
*   **Chunk Position Ratio:**
    *   Forward Pass: `current_chunk_idx / total_chunks_in_sequence`.
    *   Reverse Passes (Std & Persist): `(N - reverse_chunk_idx) / N`, where N is the number of chunks in the reverse window for that pass, providing a decaying signal (1.0 for most recent chunk down to near 0.0).
    *   Mid-Chunk Generation: Can reflect the relative position within the current partial chunk, e.g., `tokens_in_partial_chunk / chunk_size`.

These signals allow the model (specifically the CMA layers) to adapt its behavior based on the operational mode and memory context.

* **Integration Method:**
  - Control scalar values are formed into a vector \(c\). This vector is projected and added to the query projection:
  \[
  Q_{\text{fused}} = W_Q \cdot X + W_{\text{ctrl}} \cdot c
  \]
  - Where:
    - \(X\) represents the chunk token embeddings (or the next token query embedding during generation).
    - \(c\) is the vector of control scalars (broadcasted per-token).
    - \(W_{\text{ctrl}}\) is a dedicated learned projection initialized conservatively near zero.
* **Stability and Initialization:**
  - Use small-scale initialization (e.g., variance ≈ \(1 \times 10^{-4}\)) for \(W_{\text{ctrl}}\), and consider additional normalization if needed.

**5. Attention Mechanism Details (CMA Layer)**

A `CascadeMemoryAttention` layer integrates chunk and memory information. This applies to both **memory-update** and **read-only** CMA layer types.

*   **Input:** `chunk_input (B, current_chunk_len, D)` (full chunk during update cycles, partial during streaming), memory states (`M_fwd`, `M_rev_std`, `M_rev_persist` as relevant for the current pass/mode, sourced from the layer's group), `control_tokens (vector)`.
*   **QKV Projections:** Standard linear projections.
    *   **Query Source:** Derived from `chunk_input` (full chunk) during update cycle passes. Derived from the *next token position's embedding* relative to `chunk_input` (partial chunk) during streaming generation. Control tokens are fused into the Query projection (see 4.5).
    *   **Key/Value Source:** Derived from `chunk_input` concatenated with the relevant memory states for the current pass/mode.
        *   *Std Reverse Pass:* `chunk_input` + `M_rev_std` (current state).
        *   *Forward Pass:* `chunk_input` + `M_fwd` (current state) + `M_rev_std` (final state from Pass 1).
        *   *Persist Reverse Pass:* `chunk_input` + `M_rev_persist` (current state).
        *   *Streaming Generation:* `chunk_input` (partial) + `M_fwd` (final from Pass 2) + `M_rev_persist` (final from Pass 3 or last periodic update).
*   **Memory Integration:** Memory tokens are prepended (or appended) to the `chunk_input` along the sequence length dimension before Key/Value projection.
*   **Adaptive Gating (Output-level):**
    *   Retain per-head sigmoid gating applied specifically after the attention output calculation for the memory component:
        \[
        g = \sigma(W_{\text{gate}} \cdot q + b_{\text{gate}})
        \]
    *   Apply the gate to scale the output contributions from attending to memory tokens:
        \[
        Y_{\text{mem\_gated}} = g \cdot \text{Attention}(q, K_{\text{mem}}, V_{\text{mem}})
        \]
    *   Initialize gates conservatively (e.g., bias ≈ −1.0). Apply mild gate regularization (e.g., small L1 or entropy penalty).
*   **Causal Masking:** Standard causal masking is applied *within* the `chunk_input` tokens (full or partial). Full attention is allowed from chunk tokens (or the next token query) *to* all relevant memory tokens and control tokens for the current pass/mode.
*   **Attention Computation:** Standard scaled dot-product attention with softmax.

**6. Memory Update Mechanism**

This mechanism is implemented *only* within **CMA memory-update layers** and applies to the specific memory state being updated in the current pass (`M_fwd`, `M_rev_std`, or `M_rev_persist`) for that layer's group.

1.  **Compute Memory Delta (`delta`):** Use an attention mechanism where the *query* is derived from the current memory state being updated (`M_old`), and the *keys/values* are derived from the current chunk's token representations (`X`) after passing through the preceding layers.
    ```
    delta = Attention(Q=f_q(M_old), K=f_k(X), V=f_v(X))
    ```
    Where `f_q, f_k, f_v` are learned transformations (e.g., linear layers).
2.  **Gated Update:** Combine the old memory state and the delta using a learned gate:
    ```
    gate = sigmoid(f_gate(M_old, delta))
    M_new = gate * M_old + (1 - gate) * delta
    ```
    Where `f_gate` is a learned transformation.
3.  **Parameter Separation:**
    *   The forward memory update (`M_fwd`) uses one set of parameters (`f_q, f_k, f_v, f_gate`).
    *   Both reverse memory updates (`M_rev_std`, `M_rev_persist`) share a *separate* set of parameters (`f_q_rev, f_k_rev, f_v_rev, f_gate_rev`).

**7. Reverse Memory Details**

Two distinct reverse passes compute `M_rev_std` and `M_rev_persist` respectively, using the same underlying chunk boundaries defined by the most recent re-chunking event. They share configuration parameters like `reverse_memory_size` and `reverse_max_chunks` but have separate decay parameters and update weights.

*   **Lookahead Reverse Pass (`M_rev_std`)**
    *   **Purpose:** Provide immediate preceding context (including the current chunk) to the Forward Pass.
    *   **Timing:** Runs *once* per full update cycle, *before* the Forward Pass (Pass 1).
    *   **Scope:** Includes the current chunk (`N`) and processes backward (`N, N-1, ...`) up to `reverse_max_chunks`.
    *   **Attention Context:** Attends to current chunk tokens + current `M_rev_std`.
    *   **Update:** Updates `M_rev_std` using the shared reverse memory update parameters and specific `Lookahead_reverse_decay_step`, `Lookahead_reverse_decay_rate` for downweighting during the update calculation.
    *   **Persistence:** The resulting `M_rev_std` is used only by the subsequent Forward Pass and then discarded.

*   **Persistent Reverse Pass (`M_rev_persist`)**
    *   **Purpose:** Maintain recent context (excluding the current chunk) for use during streaming generation between full update cycles.
    *   **Timing:** Runs *once* per full update cycle, *after* the Forward Pass (Pass 3). Can also be run periodically during streaming generation.
    *   **Scope:** Excludes the current chunk (`N`) and processes backward (`N-1, N-2, ...`) up to `reverse_max_chunks`.
    *   **Attention Context:** Attends to current chunk tokens + current `M_rev_persist`.
    *   **Update:** Updates `M_rev_persist` using the shared reverse memory update parameters and specific `persistent_reverse_decay_step`, `persistent_reverse_decay_rate` for downweighting (potentially configured for less decay).
    *   **Persistence:** The resulting `M_rev_persist` is stored and used during subsequent streaming generation steps until the next full update cycle or periodic update.

**8. Memory Scaling & Management**

*   **Forward Memory (`M_fwd`): Dynamic Write Access**
    *   The model always allocates tensors for `max_memory_size` for `M_fwd` per group.
    *   The *effective* number of forward memory tokens that can be *written to* during updates is controlled dynamically via attention masking, based on the total sequence length (scaling up to 100% write access at a configured length).
    *   Within this sequence-determined write cap, the portion actually writable may also grow progressively as chunks are processed in the forward pass (optional, from `initial_write_fraction` up to the cap). Write updates are masked accordingly.
    *   Read access during attention is always to the full `max_memory_size` (masked appropriately if parts haven't been written yet). This ensures compatibility with `torch.compile`.
*   **Reverse Memory (`M_rev_std`, `M_rev_persist`): Fixed Size**
    *   Reverse memory states are computed based on a fixed window (`reverse_max_chunks`) and have a fixed size (`reverse_memory_size`).
    *   Their usage and updates are controlled through attention masks and gating, but their size does not scale dynamically with the total sequence length in the current design (see 12 for future enhancement).
*   **VRAM/Compute Optimizations (Optional):**
    *   Offloading unused memory parameter blocks (initial states) or inactive parts of state tensors to CPU/RAM.
    *   Partitioning memory parameters (`nn.ParameterList`) per group.
    *   Blocking gradients for unused/masked memory slots during training.

**9. Training Methodology**

*   **Chunked Processing & Update Cycle Simulation:** Training data is processed simulating the inference flow. Sequences are chunked using the chosen method (e.g., reverse-semantic-with-gap). The model processes chunks sequentially. When processing reaches the point where the *current* chunk would be considered "filled" (i.e., reaches `chunk_size`), the **full memory update cycle** (Lookahead Reverse -> Forward -> Persistent Reverse) is simulated using the ground truth sequence data.
    *   Memory states are reset (by default) at the start of simulating each sequence's update cycle(s).
    *   Loss is calculated based on predictions made during the **Forward Pass** (Pass 2).
*   **Curriculum Learning:** Gradually increase `chunk_size` (if dynamic sizing is used), `max_memory_size` (or its scaling cap), and sequence length during training.
*   **Persistent Reverse Memory Simulation:** During training, when simulating the Forward Pass (Pass 2) for chunk `i`, the model should attend to an `M_rev_persist` state computed based on the *actual* preceding chunks (`i-1, i-2,...`) from the training data (simulating Pass 3 having run previously). Apply `mask-future dropout` to this provided `M_rev_persist`.
*   **Weight Initialization / Fine-tuning:**
    *   For scratch training: Initialize standard Transformer components normally; initialize **learned initial memory state tensors** randomly near zero; initialize new gates/projections appropriately.
    *   When upgrading existing models (e.g., Gemma): Map compatible layers (e.g., SWA -> Local-only, GA -> CMA memory-update, creating layer groups), initialize new memory components (**including learned initial states** for each new group), adaptive gating, control token logic, and use a phased unfreezing schedule (train memory layers/states first, then unfreeze adapted local layers, etc. - see 11.6).
*   **Numerical Stability Enhancements:**
  *   Replace standard LayerNorm with RMSNorm across CMA layers.
  *   Utilize conservative parameter initialization (e.g., µ-Param or PyTorch defaults).
  *   Explicitly verify numerical stability early in training.
* **Gate Regularisation During Training (Recommended):**
  * When adaptive gating (§5) is enabled with an L1 or entropy regulariser, each block returns an additional scalar gate regularisation loss (`gate_reg_loss`). To ensure regularisation is effectively applied during training, aggregate this loss term across blocks and add it to the main training loss, typically weighted as configured by `gate_regularization_strength`:

```python
# Example aggregation in the training loop:
total_loss = prediction_loss  # your existing training loss
if gate_reg_loss is not None:
    total_loss += gate_reg_loss  # gate regulariser contribution
```

This practice ensures sparse yet flexible gate activations, improving generalisation and preventing gates from permanently saturating at extremes (fully open or closed).
The validation loss should not include the reguliser.

**10. Benefits**

* **True Long Context:** Fixed VRAM allows theoretically infinite sequence processing.
* **Efficiency:** Near-linear compute scaling at long sequences. Reduced computation via memory reuse and periodic updates.
* **Coherence:** Explicit memory states aim to improve long-range dependency modeling and consistency.
* **Flexibility:** Configurable trade-offs between performance, memory usage, and compute.

**11. Application Example: Upgrading Pre-trained Gemma Models**

**11.1 Introduction**

While CMA can be trained from scratch, its compatibility with existing Transformer architectures allows for adapting pre-trained models. This section outlines a strategy for upgrading models from the Gemma family (specifically Gemma 3 variants like 1B, 4B, 12B, 27B, excluding vision components) to incorporate the CMA memory mechanism for extended context capabilities.

**11.2 Gemma Architecture Compatibility**

Gemma 3 models utilize a hybrid attention structure well-suited for conversion to a CMA-based architecture:

* **Layer Ratio:** They typically alternate between 5 Sliding Window Attention (SWA) layers and 1 Global Attention (GA) layer.
* **Sliding Window Size:** The SWA layers operate with a fixed window size of 1024 tokens.
* **Global Attention:** The GA layers perform unmasked attention across the entire (pre-training) context window.

This structure allows for targeted replacement and weight reuse.

**11.3 Layer Mapping Strategy**

The core idea is to map Gemma's attention layers to the defined CMA layer types and organize them into groups:

*   **Gemma Sliding Window Attention (SWA) Layers -> Local-Only Attention Layers:** These layers within the modified model will perform full attention *within* the defined `chunk_size`. They do not interact with memory and can belong to any group or no explicit group.
*   **Gemma Global Attention (GA) Layers -> CMA Memory-Update Layers:** These layers will be responsible for reading from and writing to the CMA memory states. Each original GA layer becomes the single **memory-update layer** for its own **layer group**. It will manage its group's dedicated `M_fwd`, `M_rev_std`, and `M_rev_persist` states. Their original weights are discarded.
*   **CMA Read-Only Layers:** Not directly mapped from Gemma layers in this basic strategy. If desired, some SWA or GA layers could potentially be converted to read-only layers, but they would need to be explicitly assigned to a group containing a memory-update layer.

Other layers (MLPs/FFNs, embeddings, normalization layers - initially) remain unchanged and can be considered part of the groups containing the memory-update layers for organizational purposes.

**11.4 Configuration Alignment**

*   `chunk_size: 1024` (Matches Gemma SWA window).
*   Define layer groups, each centered around one converted GA layer (now a CMA memory-update layer).
*   Set other CMA configurations (memory sizes, scaling, decay, initial state sharing options, etc.).

**11.5 Weight Initialization**

*   **Local-Only Layers (from SWA):** Initialize directly with corresponding Gemma SWA weights. Freeze initially.
*   **CMA Memory-Update Layers (from GA):** Discard original GA weights. Initialize randomly or using standard techniques. Train from the start.
*   **Memory States (`M_fwd`, `M_rev_std`, `M_rev_persist`):** Initialize the **learned initial state tensors** for each group (associated with each former GA layer) near zero or randomly. Train from the start.
*   **Other Components:** Retain original weights (MLPs, embeddings, LayerNorms). Freeze initially.

**11.6 Phased Training and Unfreezing Schedule**

A gradual unfreezing process is recommended:

*   **Phase 1: Memory Initialization Training**
    *   **Frozen:** All inherited weights (Local-Only layers from SWA, MLPs, Embeddings, LayerNorms).
    *   **Trained:** Newly initialized **CMA Memory-Update** layers, their associated **learned initial memory states**, adaptive gating components, control token logic.
    *   **Data:** Focus on sequences filling one or two chunks (~2048 tokens).
    *   **Goal:** Stabilize the core CMA memory update cycle (Std Rev -> Fwd -> Persist Rev) and interaction logic within each group.

*   **Phase 2: Local Attention Integration**
    *   **Frozen:** MLPs, Embeddings, LayerNorms (potentially).
    *   **Trained:** **CMA Memory-Update** layers & states + **Unfreeze** **Local-Only** layers.
    *   **Data:** Sequences filling multiple chunks (~8192 tokens).
    *   **Goal:** Adapt pre-trained local patterns to interact effectively with the group-specific CMA memory across multiple update cycles.

*   **Phase 3: Full Model Fine-tuning (Optional but Recommended)**
    *   **Frozen:** Embeddings (optional).
    *   **Trained:** **CMA Memory-Update** layers & states, **Local-Only** layers + **Unfreeze** LayerNorms and potentially MLPs / LM head.
    *   **Data:** Target long sequences.
    *   **Goal:** Fine-tune the entire architecture jointly.

**11.7 Expected Outcome**

Following this process should result in a Gemma model enhanced with the CMA memory mechanism, capable of processing significantly longer context lengths than the original architecture while retaining much of its pre-trained knowledge and performance characteristics on shorter sequences.

**12. Future Enhancements**

While the proposed CMA architecture offers significant advancements in handling long sequences, several potential enhancements could further extend its capabilities and robustness:

* **Dynamic Retrieval of Chunk Hidden States:**
    * **Concept:** Augment the attention mechanism to allow the model to dynamically retrieve detailed information from specific past chunks during generation, beyond the compressed representations in `M_fwd` and `M_rev_persist`.
    * **Implementation:** Instead of retrieving raw tokens, the model could retrieve the *hidden states* generated when a specific past chunk was initially processed. These states would need to be stored (potentially offloaded to CPU RAM or disk for very long sequences) and accessed via an efficient, non-learnable indexing mechanism (e.g., based on chunk ID or semantic similarity). Within CMA layers, dedicated gates could learn to issue "retrieval queries" based on the current token's context. The retrieved hidden states (potentially representing only relevant parts of the chunk, selected via an attention mechanism over the chunk's states) would be temporarily brought into VRAM and integrated into the attention calculation for the requesting layer(s).
    * **Management:** Retrieved hidden states could be held in VRAM as long as subsequent tokens or layers continue requesting the same data, minimizing redundant retrievals, and disposed of otherwise.
    * **Benefit:** Provides targeted, on-demand access to fine-grained historical details that might be smoothed over in the main memory states, potentially improving factual recall or faithfulness to specific earlier parts of the context.

* **Persistent Cross-Session Memory:**
    * **Concept:** Introduce a true long-term memory mechanism that persists across sessions and user interactions, allowing the model to accumulate knowledge and maintain context over extended periods. Unlike standard RAG which often retrieves raw text, this memory would operate primarily in the latent space, with the model learning *what* to store, update, and retrieve.
    * **Implementation:** This could be realized as a multi-component system:
        1.  **External Latent Store:** A large-scale (potentially user-specific) vector database or key-value store holding learned latent representations of past information, memories, or summarized interactions, stored outside VRAM.
        2.  **Index/Retrieval Network:** A learnable component (potentially integrated within the LLM or as a separate module) that interprets the current context and queries the external store for relevant latent vectors.
        3.  **VRAM Integration:** Retrieved latent vectors are loaded into a dedicated context space (e.g., a separate section within CMA memory or a unique input) for use during generation, likely integrated via attention within CMA layers.
    * **Training:** This component would likely be introduced later in the training process (e.g., during instruction tuning), requiring specialized datasets simulating long-term interactions and information recall over time. The model would need to learn policies for writing summaries or key information (perhaps derived from `M_fwd` states) to the persistent store.
    * **Benefit:** Enables genuine personalization, continuous learning within a user's context, and the ability to handle tasks requiring knowledge accumulated over days, weeks, or longer.

*   **Scaling Reverse Memory Write Access:**
    *   **Concept:** Apply a progressive write access scaling mechanism, similar to that used for forward memory (`M_fwd`), to the reverse memory states (`M_rev_std` and `M_rev_persist`).
    *   **Implementation:** While the full `reverse_memory_size` would still be allocated, the portion of this memory that is *writable* during memory updates in the reverse passes would scale based on the total sequence length processed (or the length of the context relevant to the reverse pass), potentially mirroring the schedule defined for `M_fwd`. Read access would remain to the full effective size determined by masking.
    *   **Benefit:** Creates symmetry in memory handling, potentially improving learning dynamics by preventing aggressive overwriting of fixed-size reverse memory early in long sequences. Aligns the growth of writable context representation capacity across memory types.

*   **Dynamic Chunk Size (for Scratch-Trained Models):**
    *   **Concept:** Allow the `chunk_size` used for processing to vary dynamically based on the total sequence length, rather than being fixed.
    *   **Implementation:** Define `min_chunk_size`, `max_chunk_size`, and `chunk_size_scaling_length` parameters. The actual `chunk_size` used for a given sequence would interpolate between the min and max values, reaching the maximum at the specified scaling length. This would primarily apply to models trained from scratch, as models fine-tuned from architectures with fixed window sizes (like Gemma) benefit from matching that original size.
    *   **Benefit:** Allows the model to use smaller chunks for shorter sequences (potentially reducing padding or computational overhead) and larger chunks for very long sequences (reducing the number of update cycles needed). Requires careful tuning during training.

* **Spectral Harmonic Memory Transform**
    * **Concept:**  
Introduce frequency-domain summaries of chunk representations to replace or complement raw embeddings. After processing each chunk's token embeddings \( X \in \mathbb{R}^{L \times D} \), perform a fast Fourier transform (FFT) along the sequence dimension, retaining only the first \(K\) (lowest-frequency) coefficients per embedding dimension. Store these compact, spectral summaries \( \widehat{X} \in \mathbb{C}^{K \times D} \) as persistent memory states. At attention-read time, query vectors are projected into the same spectral domain, and cross-spectral correlations are computed efficiently. An inverse FFT then maps the attention results back into the token space.

    * **Benefits:**  
      - Captures long-range thematic information efficiently through low-frequency components, improving coherence.
      - Reduces computational complexity to \( O(KD) \) per token for memory reads, significantly lowering inference costs compared to standard attention.
      - Dramatically reduces VRAM footprint for large-context storage.

    * **Potential Pitfalls and Precautions:**  
     - **Information Loss from Frequency Truncation:**  
        Aggressively limiting to low frequencies (small \(K\)) might discard important fine-grained details.  
        **Mitigation:**  
        - Empirically select an optimal \(K\) balancing compression and fidelity.  
        - Optionally maintain a sparse subset of high-frequency coefficients for critical tokens (e.g., section headings, named entities).

     - **Numerical Stability in FFT Operations:**  
       Frequency-domain operations, especially inverse FFTs, can amplify numerical instability if inputs are not carefully normalized.  
        **Mitigation:**  
        - Apply careful layer normalization (e.g., RMSNorm) prior to FFT transforms.  
        - Regularly monitor variance distributions post-IFFT during training and adjust normalizations if needed.

     - **Complex Number Handling:**  
       The complex-valued tensors increase tensor-management complexity and potentially GPU memory demands.  
        **Mitigation:**   
        - Represent complex numbers explicitly as two separate real tensors (real and imaginary), carefully verifying CUDA compatibility and efficiency.
        - Alternatively, only use real-valued cosine transforms (e.g., discrete cosine transform—DCT) if complexity overhead is prohibitive.

     - **Projection Initialization and Stability:**  
        Poor initialization of projection matrices into frequency space can impede learning and convergence.  
        **Mitigation:**  
        - Initialize spectral-domain projection matrices conservatively, near zero, with small variance.
        - Employ gradient clipping or projection regularization to stabilize early training phases.

* **Self-Organizing Memory Tile Grid**
    * **Concept:**  
Maintain memory as a fixed-size 2D "tile map," inspired by Self-Organizing Maps (SOM). Each tile contains a prototype embedding \(T_{h,w} \in \mathbb{R}^{D}\). New chunk summaries are mapped onto the grid by identifying the closest-matching tile via Euclidean distance and updating it (and optionally neighbors) using a moving average. At memory-read time, perform a small convolution across the 2D tile grid to aggregate local and regional context efficiently.

    * **Benefits:**  
      - Maintains constant VRAM usage irrespective of sequence length, ideal for streaming or indefinitely-long contexts.
      - Naturally clusters semantically similar contexts into spatially coherent regions on the grid, facilitating efficient localized context recall.
      - Efficient context aggregation via fast 2D convolutional operations.

    * **Potential Pitfalls and Precautions:**  
      - **Tile Prototype Degradation (Blurring):**  
        Continuous updates risk eroding distinct semantic representations, causing tiles to converge towards overly generalized states.  
        **Mitigation:**  
        - Carefully select a conservative update rate (\(\beta\approx0.9\)–\(0.99\)), ensuring prototypes evolve slowly and retain semantic specificity.
        - Periodically reset or reinitialize unused or underused tiles to preserve grid diversity.

      - **Best-Matching Unit (BMU) Computation Overhead:**  
        Identifying the closest tile involves an \( O(H W D) \) search at every chunk boundary, possibly slowing inference or training.  
        **Mitigation:**  
        - Restrict BMU searches via hierarchical clustering or spatial indexing (e.g., k-d trees or approximate nearest-neighbor methods) to accelerate lookup.
        - Consider performing BMU updates only periodically rather than every chunk if overhead is prohibitive.

      - **Fixed-Size Limitations and Capacity Issues:**  
        A small tile grid may limit representational capacity and lead to overwriting distinct contexts when the model encounters highly diverse content.  
        **Mitigation:**  
        - Dynamically grow or shrink the tile grid size during training based on utilization metrics or semantic diversity measures, balancing representational richness against VRAM constraints.
        - Regularize grid usage via sparsity constraints or encourage uniform distribution of chunk assignments to prevent catastrophic overwrites.

* **Fusion Of Spectral Harmonic Memory and Memory Tiles: Spectral Tile Grid (Recommended)**
    * **Concept:**  
      Combine the Spectral Harmonic Transform and the Self-Organizing Memory Tile Grid by storing each tile prototype directly in the frequency domain (i.e., each tile contains \( K \) low-frequency spectral coefficients instead of raw embeddings). When assigning new chunks, map their frequency-domain summary into the grid via spectral-domain distance metrics (e.g., Euclidean in frequency-space). Convolutions for memory reads are then performed directly in frequency space before inverse-transforming results back into token embeddings.

    * **Why Fuse:**  
      This fusion elegantly unifies spatial locality (tiles) and thematic coherence (spectral components), offering context-aware memory organization with significantly reduced computational complexity and fixed VRAM overhead.

    * **Additional Precautions and Mitigations for Fusion:**  
      - **Complexity and Numerical Stability:**  
        Combining spectral transforms with tile-grid updates requires meticulous numerical management.  
        **Mitigation:**  
        - Use frequency-space RMSNorm or other stable normalization techniques explicitly tailored for spectral data.  
        - Monitor spectral-domain variance closely and apply early gradient clipping if signs of numerical instability appear.

      - **Computational Overhead for Spectral Tile Updates:**  
        Performing spectral transforms at each chunk boundary adds overhead compared to simpler embedding updates.  
        **Mitigation:**  
        - Carefully tune chunk sizes (\(L\)) and frequency truncation (\(K\)) parameters, balancing computational cost and information fidelity.
        - Implement efficient incremental FFT updates (sliding FFT windows) to significantly reduce computational load.

      - **Learning Dynamics and Initialization Issues:**  
        Frequency-domain and tile-grid parameters interact nontrivially; poor initialization could impair early training stages.  
        **Mitigation:**  
        - Start training with conservative initialization and introduce curriculum learning strategies, initially emphasizing short sequences and gradually increasing complexity.
        - Employ spectral-domain dropout or noise injection to encourage robust tile-grid convergence and prevent premature overfitting.

These enhancements represent potential avenues for further research and development, building upon the CMA foundation to create even more powerful and contextually aware language models.

**13. Conclusion**

Cascade Memory Attention (CMA) offers a promising architecture for overcoming the context length limitations of standard Transformers. By combining chunked processing, multiple intelligently managed **group-specific memory states**, a structured **three-pass update cycle**, adaptive gating, and optional fine-tuning strategies, CMA aims to deliver efficient, high-quality language modeling for arbitrarily long sequences, enabling advancements in areas like long-document understanding, extended conversations, and complex code generation.
