import torch
import torch.nn as nn
import torch.nn.functional as F

LAYERNORM_EPSILON = 1e-12  # Layer norm from BERT

def truncated_normal(stddev, dtype=torch.float32):
  def init(tensor):
    return torch.nn.init.trunc_normal_(tensor, std=stddev)
  return init

class Attention(nn.Module):
  """Attention layer that is part of each Transformer layer."""
  def __init__(self, hidden_size, hidden_dropout_prob, num_attention_heads, attention_probs_dropout_prob, initializer_fn):
    super().__init__()
    self.hidden_size = hidden_size
    self.hidden_dropout_prob = hidden_dropout_prob
    self.num_attention_heads = num_attention_heads
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.initializer_fn = initializer_fn

    self.self_attention = nn.MultiheadAttention(hidden_size, num_attention_heads, dropout=attention_probs_dropout_prob)
    self.self_attention.qkv_proj.weight.data = self.initializer_fn(self.self_attention.qkv_proj.weight.data)
    self.self_attention.qkv_proj.bias.data.zero_()
    self.self_attention.out_proj.weight.data = self.initializer_fn(self.self_attention.out_proj.weight.data)
    self.self_attention.out_proj.bias.data.zero_()

    self.dropout = nn.Dropout(hidden_dropout_prob)
    self.layer_norm = nn.LayerNorm(hidden_size, eps=LAYERNORM_EPSILON)

  def forward(self, layer_input, input_mask, deterministic):
    attention_mask = input_mask.unsqueeze(1).unsqueeze(2) # (batch_size, 1, 1, seq_length)
    attention_mask = (1.0 - attention_mask) * -10000.0 # mask out the padding tokens
    attention_output, _ = self.self_attention(layer_input, layer_input, layer_input, attn_mask=attention_mask)
    attention_output = self.dropout(attention_output, deterministic=deterministic)
    attention_output = self.layer_norm(attention_output + layer_input)
    return attention_output

class Mlp(nn.Module):
  """MLP layer that is part of each Transformer layer."""
  def __init__(self, hidden_size, hidden_dropout_prob, intermediate_size, initializer_fn):
    super().__init__()
    self.hidden_size = hidden_size
    self.hidden_dropout_prob = hidden_dropout_prob
    self.intermediate_size = intermediate_size
    self.initializer_fn = initializer_fn

    self.intermediate = nn.Linear(hidden_size, intermediate_size)
    self.intermediate.weight.data = self.initializer_fn(self.intermediate.weight.data)
    self.intermediate.bias.data.zero_()

    self.output = nn.Linear(intermediate_size, hidden_size)
    self.output.weight.data = self.initializer_fn(self.output.weight.data)
    self.output.bias.data.zero_()

    self.dropout = nn.Dropout(hidden_dropout_prob)
    self.layer_norm = nn.LayerNorm(hidden_size, eps=LAYERNORM_EPSILON)

  def forward(self, attention_output, deterministic):
    intermediate_output = F.gelu(self.intermediate(attention_output))
    layer_output = self.output(intermediate_output)
    layer_output = self.dropout(layer_output, deterministic=deterministic)
    layer_output = self.layer_norm(layer_output + attention_output)
    return layer_output

class TransformerLayer(nn.Module):
  """A single Transformer layer."""
  def __init__(self, intermediate_size, hidden_size, hidden_dropout_prob, num_attention_heads, attention_probs_dropout_prob, initializer_fn):
    super().__init__()
    self.intermediate_size = intermediate_size
    self.hidden_size = hidden_size
    self.hidden_dropout_prob = hidden_dropout_prob
    self.num_attention_heads = num_attention_heads
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.initializer_fn = initializer_fn

    self.attention = Attention(hidden_size, hidden_dropout_prob, num_attention_heads, attention_probs_dropout_prob, initializer_fn)
    self.mlp = Mlp(hidden_size, hidden_dropout_prob, intermediate_size, initializer_fn)

  def forward(self, layer_input, input_mask, deterministic):
    attention_output = self.attention(layer_input, input_mask, deterministic)
    layer_output = self.mlp(attention_output, deterministic)
    return layer_output

class Embed(nn.Module):
  """Embeds visual tokens."""
  def __init__(self, embedding_size, hidden_dropout_prob, vocab_size, max_position_embeddings, initializer_fn, hidden_size=None):
    super().__init__()
    self.embedding_size = embedding_size
    self.hidden_dropout_prob = hidden_dropout_prob
    self.vocab_size = vocab_size
    self.max_position_embeddings = max_position_embeddings
    self.initializer_fn = initializer_fn
    self.hidden_size = hidden_size

    self.word_embedder = nn.Embedding(vocab_size, embedding_size)
    self.word_embedder.weight.data = self.initializer_fn(self.word_embedder.weight.data)
    self.position_embeddings = nn.Embedding(max_position_embeddings, embedding_size)
    self.position_embeddings.weight.data = self.initializer_fn(self.position_embeddings.weight.data)
    self.layer_norm = nn.LayerNorm(embedding_size, eps=LAYERNORM_EPSILON)
    if self.hidden_size:
      self.hidden_mapping = nn.Linear(embedding_size, hidden_size)
      self.hidden_mapping.weight.data = self.initializer_fn(self.hidden_mapping.weight.data)
      self.hidden_mapping.bias.data.zero_()
    self.dropout = nn.Dropout(hidden_dropout_prob)

  def forward(self, input_ids, deterministic):
    seq_length = input_ids.shape[-1]
    position_ids = torch.arange(seq_length)[None, :]

    word_embeddings = self.word_embedder(input_ids)
    position_embeddings = self.position_embeddings(position_ids)

    input_embeddings = self.layer_norm(word_embeddings + position_embeddings)
    if self.hidden_size:
      input_embeddings = self.hidden_mapping(input_embeddings)
    input_embeddings = self.dropout(input_embeddings, deterministic=deterministic)

    return input_embeddings


class Bias(nn.Module):
  """Adds a learnable bias to the input.

  Attributes:
    dtype: the dtype of the computation (default: float32).
    bias_init: initializer function for the bias.
  """
  def __init__(self, dtype=torch.float32, bias_init=nn.init.zeros_):
    super().__init__()
    self.dtype = dtype
    self.bias_init = bias_init
    self.bias = None

  def forward(self, inputs):
    """Applies a linear transformation to the inputs along the last dimension.

    Args:
      inputs: The nd-array to be transformed.

    Returns:
      The transformed input.
    """
    inputs = inputs.to(self.dtype)

    bias_shape = inputs.shape[-1]
    if self.bias is None:
      self.bias = nn.Parameter(torch.empty(bias_shape, dtype=self.dtype))
      self.bias_init(self.bias.data)
    bias = self.bias.expand_as(inputs)

    return inputs + bias


class MlmLayer(nn.Module):
  """MLM layer for masked token prediction."""
  def __init__(self, hidden_size, initializer_fn):
    super().__init__()
    self.hidden_size = hidden_size
    self.initializer_fn = initializer_fn

    self.mlm_hidden = nn.Linear(hidden_size, hidden_size)
    self.mlm_hidden.weight.data = self.initializer_fn(self.mlm_hidden.weight.data)
    self.mlm_hidden.bias.data.zero_()
    self.layer_norm = nn.LayerNorm(hidden_size, eps=LAYERNORM_EPSILON)
    self.bias = Bias()

  def forward(self, last_layer, embeddings):
    mlm_hidden = F.gelu(self.mlm_hidden(last_layer))
    mlm_hidden = self.layer_norm(mlm_hidden)
    output_weights = embeddings.t()
    logits = self.bias(torch.matmul(mlm_hidden, output_weights))
    return logits

class Transformer(nn.Module):
  """Transformer modified from BERT."""
  def __init__(self, vocab_size, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=256, initializer_range=0.02):
    super().__init__()
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads
    self.intermediate_size = intermediate_size
    self.hidden_dropout_prob = hidden_dropout_prob
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.max_position_embeddings = max_position_embeddings
    self.initializer_fn = truncated_normal(initializer_range)

    self.embed = Embed(embedding_size=hidden_size, hidden_dropout_prob=hidden_dropout_prob, vocab_size=vocab_size, max_position_embeddings=max_position_embeddings, initializer_fn=self.initializer_fn)
    self.transformer_layers = nn.ModuleList([TransformerLayer(intermediate_size=intermediate_size, hidden_size=hidden_size, hidden_dropout_prob=hidden_dropout_prob, num_attention_heads=num_attention_heads, attention_probs_dropout_prob=attention_probs_dropout_prob, initializer_fn=self.initializer_fn) for _ in range(num_hidden_layers)])
    self.mlm_layer = MlmLayer(hidden_size=hidden_size, initializer_fn=self.initializer_fn)

  def forward(self, input_ids, deterministic=True):
    input_ids = input_ids.to(torch.int32)
    input_embeddings = self.embed(input_ids, deterministic=deterministic)

    layer_input = input_embeddings
    for layer in self.transformer_layers:
      layer_output = layer(layer_input, torch.ones_like(input_ids, dtype=torch.int32), deterministic=deterministic)
      layer_input = layer_output

    word_embedding_matrix = self.embed.word_embedder.weight
    logits = self.mlm_layer(layer_output, word_embedding_matrix)

    return logits
