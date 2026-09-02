import torch
import torch.nn as nn
import torch.nn.functional as F

#RMSNorm (Root Mean Square Layer Normalization) is a faster, simplified variant of standard LayerNorm
"""
1. Calculate x²
2. Calculate the mean
3. Take reciprocal square root
4. Scale x using that value
5. Multiply by a learnable weight
"""

"""
input_ids ──▶ embed_tokens ──▶ layer 0 ──▶ layer 1 ──▶ … 
──▶ layer 23 ──▶ norm ──▶ lm_head ──▶ logits
"""
class MyQwen(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = QwenModel(config)
        # No lm_head parameter: tie_word_embeddings=True means we reuse embed_tokens.      

    def forward(self, input_ids, return_hidden=False):
        out = self.model(input_ids, return_hidden=return_hidden)
        h, hidden_states = out if return_hidden else (out, None)
        # lm_head: projects 896 → 151936 to get a score for every vocabulary token. 
        # For Qwen2.5-0.5B the config has tie_word_embeddings: true,
        #  which means there is no separate lm_head matrix — the model reuses 
        # the embedding table transposed. So the logits are simply h @ embed_tokens.weight.T. 
        # Creating a separate nn.Linear for this would double the biggest matrix in the model 
        # and there would be no weights in the checkpoint to load into it.
        logits = h @ self.model.embed_tokens.weight.T     # [batch, seq, vocab]

        if return_hidden:
            return logits, hidden_states
        return logits


class QwenModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size
        )
        # Create one DecoderLayer for every Transformer block in Qwen.
        # Each layer has the same architecture, but each layer has
        # its own independent pretrained weights
        self.layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config)
        self.rotary_emb = RotaryEmbedding(config)

    def forward(self, input_ids, return_hidden=False):
        batch, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        # Generate positional cosine and sine values once.
        # All decoder layers use the same position embeddings.
        
        # Convert token IDs into their learned hidden representations.
        h = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(
            h,
            position_ids
        )
        hidden_states = [h]                     # only used for debugging parity
        # Run the hidden states through every Transformer decoder layer.
        for layer in self.layers:
            h = layer(h, cos, sin)
            hidden_states.append(h)
        # Apply the final RMSNorm after the last decoder layer.
        h = self.norm(h)

        if return_hidden:
            return h, hidden_states
        return h

# this is basically the Transformer decoder block, which consists of a 
# self-attention mechanism followed by a feed-forward neural network (MLP). 
# It uses RMSNorm for normalization before the attention and MLP operations. 
# The forward method processes the input tensor through these blocks, applying 
# residual connections to maintain information flow.
class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Self-attention block.
        self.self_attn = AttentionProjections(config)
        # Feed-forward / SwiGLU MLP.
        self.mlp = MLP(config)
        # Normalize the hidden states before self-attention.
        # Qwen uses RMSNorm here.
        self.input_layernorm = RMSNorm(config)
        # Normalize the attention output before sending it through the MLP.
        self.post_attention_layernorm = RMSNorm(config)

    def forward(self, x, cos, sin):
        # Attention sub-block with residual
        residual = x
        # Normalize before self-attention.
        hidden_states  = self.input_layernorm(x)
        hidden_states  = self.self_attn(hidden_states, cos, sin)
        # Add the original input back to the attention output.
        hidden_states  = residual + hidden_states 

        # MLP sub-block with residual
        residual = hidden_states 
        # Normalize before the MLP.
        hidden_states = self.post_attention_layernorm(hidden_states)
        # Run the SwiGLU MLP.
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

class RMSNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.hidden_size)) #learnable weight
        self.eps = config.rms_norm_eps

    def forward(self, x):
        # Calculate the mean square of the input tensor along the last dimension
        # Calculate mean(x^2)
        in_dtype = x.dtype
        x = x.float() 
        mean_square = x.pow(2).mean(dim=-1, keepdim=True)
        # Normalize the input tensor using the mean square and epsilon for numerical stability
        x_normed = x / torch.sqrt(mean_square + self.eps)
        # Scale the normalized tensor by the learnable weight parameter
        return self.weight * x_normed.to(in_dtype)   

# The MLP (Multi-Layer Perceptron) class implements a feedforward neural network layer used in transformer architectures. It consists of three linear transformations: gate projection, up projection, and down projection. The forward method applies the SiLU activation function to the gate projection output and multiplies it with the up projection output before passing it through the down projection to produce the final output.
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = config.hidden_size          # 896
        inter = config.intermediate_size     # 4864
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj   = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)
        """
        def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj
        """
    def forward(self, x):
        # Project x through the gate branch.
        gate = self.gate_proj(x)
        # Apply the SiLU activation to the gate branch.
        gate = torch.nn.functional.silu(gate)
        # Project x through the second MLP branch.
        up = self.up_proj(x)
        # Combine the two branches element-by-element.
        hidden = gate * up
        # Project the intermediate representation back to hidden_size.
        return self.down_proj(hidden)
    
#Qwen2.5-0.5B uses Grouped Query Attention (GQA).
class AttentionProjections(nn.Module):
    def __init__(self, config, debug=False):
        super().__init__()
        self.debug = debug

        # The size of each token's hidden representation.
        self.hidden_size = config.hidden_size
        # Number of Query attention heads.
        self.num_heads = config.num_attention_heads
        # Number of Key/Value heads.
        # Qwen uses Grouped Query Attention (GQA), so this is smaller
        # than the number of Query heads. For Qwen2.5-0.5B, this is 2.
        self.num_kv_heads = config.num_key_value_heads
        # Number of features handled by each individual attention head.
        # Example:
        #   hidden_size = 896
        #   num_heads = 14
        # Therefore:
        #   head_dim = 896 / 14 = 64
        self.head_dim = self.hidden_size // self.num_heads
        # Query projection.
        # Transforms each hidden vector into representations used to ask:
        # "Which other tokens should I pay attention to?"
        # Input shape:
        #   [batch, sequence_length, 896]
        # Output shape:
        #   [batch, sequence_length, 14 * 64]
        # = [batch, sequence_length, 896]
        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=True
        )
        # Key projection.
        # Keys represent what information each token contains.
        # Because Qwen uses only 2 Key/Value heads:
        
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,#   2 * 64 = 128 output features
            bias=True
        )
        # Value projection.
        # Values contain the actual information that will be combined
        # after the attention scores are calculated.
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim, # 2 * 64 = 128 output features
            bias=True
        )
        # Output projection applied after all attention heads
        # are combined back into the hidden dimension.
        #
        # For Qwen2.5-0.5B:
        # 896 -> 896
        self.o_proj = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=False
        )
        # self.rotary_emb = RotaryEmbedding(config)
        # Number of Query heads that share each Key/Value head.
        # For Qwen2.5-0.5B:
        # 14 Query heads / 2 KV heads = 7
        self.num_key_value_groups = (
            self.num_heads // self.num_kv_heads
        )
        # Attention scaling factor used after computing Q @ K^T.
        # Qwen uses:
        #   1 / sqrt(head_dim)
        self.scaling = self.head_dim ** -0.5

    def forward(self, x, cos, sin):
        # Project the same input hidden states into three different spaces:
        #   x -> Query q = [batch, sequence, 896]
        #   x -> Key k = [batch, sequence, 128]
        #   x -> Value v = [batch, sequence, 128]
        # The projections use different pretrained weight matrices.
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        batch_size, sequence_length, _ = x.shape
        # Q currently has shape:
        # [batch_size, sequence_length, 896]
        #
        # Split the 896 dimensions into:
        # 14 attention heads × 64 dimensions per head.
        #
        # New shape:
        # [batch_size, sequence_length, 14, 64]
        # Move the attention-head dimension before the sequence dimension.
        # Before:
        # [batch_size, sequence_length, num_heads, head_dim]
        # After:
        # [batch_size, num_heads, sequence_length, head_dim]
        q = q.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)
        k = k.view(
            batch_size,
            sequence_length,
            self.num_kv_heads,
            self.head_dim
        ).transpose(1, 2)   
        v = v.view(
            batch_size,
            sequence_length,
            self.num_kv_heads,
            self.head_dim
        ).transpose(1, 2)
        # Generate cosine and sine values for each token position.
        # cos, sin = self.rotary_emb(x, position_ids)
        # Add a head dimension so cos and sin broadcast across
        # every attention head.
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        # Save Q and K before applying RoPE.
        q_before_rope = q.clone()
        k_before_rope = k.clone()
        # Apply RoPE to Query and Key.
        # Value does not receive positional rotation.
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        q_post_rope = q.clone()
        k_post_rope = k.clone()
        v_post_rope = v.clone()
        # Repeat the Key heads so their count matches the Query heads.
        #
        # Before:
        # K = [batch, 2, sequence, 64]
        #
        # After:
        # K = [batch, 14, sequence, 64]
        k = repeat_kv(
            k,
            self.num_key_value_groups)
        v = repeat_kv(
            v,
            self.num_key_value_groups)
        # Transpose the last two dimensions of K so we can compare
        # every Query vector against every Key vector.
        #
        # Q:
        # [batch_size, num_heads, sequence_length, head_dim]
        #
        # K transpose:
        # [batch_size, num_heads, head_dim, sequence_length]
        #
        # Matrix multiplication produces:
        # [batch_size, num_heads, sequence_length, sequence_length]
        attention_scores = torch.matmul(
            q,
            k.transpose(-2, -1)
        )
        attention_scores_raw = attention_scores.clone()

        # Scale the attention scores by the square root of the head dimension.
        #
        # This prevents the dot-product values from becoming too large
        # before we apply softmax.
        # Scale the raw attention scores using Qwen's exact scaling factor.
        attention_scores = attention_scores * self.scaling
        # Create an additive causal attention mask.
        # Allowed positions receive 0.
        # Future positions receive -inf.
        #
        # Shape:
        # [sequence_length, sequence_length]
        causal_mask = torch.triu(
            torch.full(
                (
                    sequence_length,
                    sequence_length
                ),
                float("-inf"),
                device=x.device,
                dtype=x.dtype
            ),
            diagonal=1
        )

        # Add dimensions so the mask broadcasts across:
        # batch and attention heads.
        #
        # Shape:
        # [1, 1, sequence_length, sequence_length]
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        # Add the mask to the attention scores.
        attention_scores = attention_scores + causal_mask
        # Replace future-token positions with negative infinity.
        #
        # Allowed positions keep their original attention score.
        # Future positions become -inf, which becomes probability 0
        # after softmax.
        # attention_scores = attention_scores.masked_fill(
        #     causal_mask == 0,
        #     float("-inf")
        # )
        # Convert the masked attention scores into attention probabilities.
        #
        # Softmax is applied over the last dimension, which represents
        # all Key positions that each Query position can attend to.
        #
        # Shape remains:
        # [batch_size, num_heads, sequence_length, sequence_length] 
        attention_weights = torch.softmax(
            attention_scores,
            dim=-1,   
            dtype=torch.float32
        ).to(q.dtype)
        # Use the attention probabilities to compute a weighted combination
        # of the Value vectors.
        #
        # attention_weights:
        # [batch_size, num_heads, sequence_length, sequence_length]
        #
        # v:
        # [batch_size, num_heads, sequence_length, head_dim]
        #
        # Result:
        # [batch_size, num_heads, sequence_length, head_dim]
        attention_output = torch.matmul(
            attention_weights,
            v
        )
        # Move the sequence dimension back before the attention-head dimension.
        # Before:
        # [batch_size, num_heads, sequence_length, head_dim]
        # After:
        # [batch_size, sequence_length, num_heads, head_dim]
        attention_output = attention_output.transpose(1, 2)

        # Combine all attention heads back into the original hidden dimension.
        # For Qwen2.5-0.5B:
        # 14 heads × 64 dimensions = 896 hidden dimensions.
        # Final shape:
        # [batch_size, sequence_length, hidden_size]
        attention_output = attention_output.reshape(
            batch_size,
            sequence_length,
            self.num_heads * self.head_dim
        )
        # Apply the output projection to mix information
        # from all attention heads.
        #
        # Input:
        # [batch_size, sequence_length, hidden_size]
        #
        # Output:
        # [batch_size, sequence_length, hidden_size]
        attention_output = self.o_proj(
            attention_output
        )
        if self.debug:
            print("=== AttentionProjections ===")
            print("Q shape:", q.shape)
            print("K shape:", k.shape)
            print("V shape:", v.shape)
            print("Q before RoPE shape:", q_before_rope.shape)
            print("K before RoPE shape:", k_before_rope.shape)
            print("Q after RoPE shape:", q_post_rope.shape)
            print("K after RoPE shape:", k_post_rope.shape)
            print("V after RoPE shape:", v_post_rope.shape)
            print("Attention scores raw shape:", attention_scores_raw.shape)
            print("Attention output shape:", attention_output.shape)
            return q, k, v, q_before_rope, k_before_rope,q_post_rope, k_post_rope,v_post_rope,attention_scores_raw, attention_output
        else: return attention_output
    
def repeat_kv(hidden_states, num_repeats):
    batch_size, num_kv_heads, sequence_length, head_dim = (
        hidden_states.shape
    )

    # If the number of Query heads equals the number of KV heads,
    # no repetition is necessary.
    if num_repeats == 1:
        return hidden_states

    # Add a new dimension representing how many times each KV head
    # should be shared.
    #
    # Before:
    # [batch, num_kv_heads, sequence, head_dim]
    #
    # After:
    # [batch, num_kv_heads, 1, sequence, head_dim]
    hidden_states = hidden_states[:, :, None, :, :]
    # Expand each KV head across the new group dimension.
    #
    # Example:
    # 2 KV heads × 7 repeats
    #
    # Shape:
    # [batch, 2, 7, sequence, head_dim]
    # Conceptually:
    # K0 -> K0, K0, K0, K0, K0, K0, K0
    # K1 -> K1, K1, K1, K1, K1, K1, K1
    #
    # Shape becomes:
    # [batch_size, 2, 7, sequence_length, 64
    hidden_states = hidden_states.expand(
        batch_size,
        num_kv_heads,
        num_repeats,
        sequence_length,
        head_dim
    )
    # Merge:
    #
    # num_kv_heads × num_repeats
    #
    # For Qwen:
    # 2 × 7 = 14
    #
    # Final shape:
    # [batch, 14, sequence, head_dim]
    return hidden_states.reshape(
        batch_size,
        num_kv_heads * num_repeats,
        sequence_length,
        head_dim
    )

# The RotaryEmbedding class implements the Rotary Positional Embedding (RoPE) technique used in transformer models. It generates cosine and sine values based on token positions and applies them to the Query and Key vectors to encode positional information. The forward method computes the rotation angles for each token position and returns the cosine and sine values for use in attention calculations.
class RotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Each attention head in Qwen2.5-0.5B has 64 dimensions.
        head_dim = (
            config.hidden_size //
            config.num_attention_heads
        )
        # RoPE uses half of the head dimensions to create rotation
        # frequencies because the dimensions are processed in pairs.
        # head_dim = 64
        # dimensions used for frequencies = 0, 2, 4, ..., 62
        inv_freq = 1.0 / (
            config.rope_parameters["rope_theta"] **
            (
                torch.arange(
                    0,
                    head_dim,
                    2,
                    dtype=torch.float32
                ) / head_dim
            )
        )
        # Store the frequencies as part of the module.
        # register_buffer means this is not a trainable parameter.
        self.register_buffer(
            "inv_freq",
            inv_freq,
            persistent=False
        )
    def forward(self, x, position_ids):
        # Convert position IDs to the same floating-point type as inv_freq.
        position_ids = position_ids.float()
        # position_ids:
        # [batch, sequence_length]
        #
        # inv_freq:
        # [head_dim / 2]
        #
        # The matrix multiplication creates rotation angles for every
        # token position and every frequency.
        freqs = torch.einsum(
            "bs,d->bsd",
            position_ids,
            self.inv_freq
        )
        # Duplicate the frequencies so they match the full head dimension.
        # Before:
        # [batch, sequence_length, 32]
        # After:
        # [batch, sequence_length, 64]
        emb = torch.cat(
            (freqs, freqs),
            dim=-1
        )
        # Return the cosine and sine values used to rotate Q and K.
        return emb.cos(), emb.sin()

#this function rotates the last dimension of a tensor by splitting it into two halves, negating the second half, and concatenating them back together. It is used in the Rotary Positional Embedding (RoPE) technique to apply positional information to Query and Key vectors in transformer models.    
def rotate_half(x):
    # Split the last dimension into two equal halves.
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]

    # [x1, x2] -> [-x2, x1]
    return torch.cat(
        (-x2, x1),
        dim=-1
    )  

