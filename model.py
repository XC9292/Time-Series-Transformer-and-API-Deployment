import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """
    Positional encoding for time series data.
    Can use either sinusoidal encoding.
    """
    
    def __init__(self, d_model: int, max_len: int = 50):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
    
        # Fixed sinusoidal positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor with positional encoding added
        """
       
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].unsqueeze(0)


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism for time series.
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query, key, value: Tensors of shape (batch_size, seq_len, d_model)
            mask: Optional mask tensor
        Returns:
            output: Tensor of shape (batch_size, seq_len, d_model)
            attention_weights: Tensor of shape (batch_size, n_heads, seq_len, seq_len)
        """
        batch_size, seq_len = query.size(0), query.size(1)
        
        # Linear projections and reshape for multi-head attention
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads and put through final linear layer
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        output = self.w_o(context)
        
        return output, attention_weights


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """
    A single transformer block with self-attention and feed-forward layers.
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection and layer norm
        attn_output, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class TimeSeriesTransformer(nn.Module):
    """
    Transformer model for time series forecasting.
    
    This model can be used for:
    1. Next-step prediction (forecasting)
    2. Sequence-to-sequence prediction
    3. Classification tasks
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 192,
        n_heads: int = 3,
        n_layers: int = 4,
        d_ff: int = 384,
        max_seq_len: int = 50,
        output_dim: int = 1,
        dropout: float = 0.0,
        learnable_pos: bool = False
    ):
        super(TimeSeriesTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.output_dim = output_dim
        # Input projection layer
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output layers
        self.output_projection = nn.Linear(d_model, output_dim)

        
        self.dropout = nn.Dropout(dropout)

        self.act = nn.Sigmoid()

        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal mask for autoregressive prediction."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask == 0  # True for positions that should attend, False for masked
    
    def forward(
        self, 
        x: torch.Tensor, 
        use_causal_mask: bool = True
    ) -> torch.Tensor:
        """
        Forward pass of the transformer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            mask: Optional attention mask
            use_causal_mask: Whether to use causal masking for autoregressive prediction
            
        Returns:
            Output tensor. Shape depends on task_type:
            - forecasting: (batch_size, output_dim)
            - seq2seq: (batch_size, seq_len, output_dim)
            - classification: (batch_size, output_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Input projection and positional encoding
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Create causal mask if needed
        if use_causal_mask:
            causal_mask = self.create_causal_mask(seq_len, x.device)
        else:
            causal_mask = None

        # Pass through transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, causal_mask)

        # Output projection
        output = self.output_projection(x[:, -1, :])

        return self.act(output)


