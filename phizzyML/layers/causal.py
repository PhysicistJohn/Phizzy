"""
Causality-respecting neural network layers for physics applications.
These layers ensure that information only flows forward in time.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
import math


class CausalConv1d(nn.Module):
    """
    1D Convolutional layer that respects causality.
    Only uses past information to predict future values.
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, dilation: int = 1, groups: int = 1,
                 bias: bool = True):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        
        # Calculate padding needed for causal convolution
        self.padding = (kernel_size - 1) * dilation
        
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=0, dilation=dilation,
            groups=groups, bias=bias
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with causal padding.
        
        Args:
            x: Input tensor of shape (batch, channels, time)
        
        Returns:
            Causally convolved output
        """
        # Apply causal padding (pad only on the left)
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class CausalAttention(nn.Module):
    """
    Self-attention layer that respects temporal causality.
    Each time step can only attend to previous time steps.
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 8, 
                 dropout: float = 0.1, bias: bool = True):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, 
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with causal masking.
        
        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim)
            key_padding_mask: Optional mask for padded positions
        
        Returns:
            Output tensor with causal attention applied
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # (batch, heads, seq_len, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        
        # Create causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1
        )
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Apply key padding mask if provided
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
        
        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(attn_output)
        
        return output


class CausalLSTM(nn.Module):
    """
    LSTM that explicitly enforces causality and provides physics-aware features.
    Includes energy tracking and conservation options.
    """
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 bias: bool = True, dropout: float = 0.0, 
                 track_energy: bool = False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.track_energy = track_energy
        
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            bias=bias, dropout=dropout, batch_first=True
        )
        
        if track_energy:
            # Additional layer to compute energy from hidden state
            self.energy_layer = nn.Linear(hidden_size, 1)
    
    def forward(self, x: torch.Tensor, 
                initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                return_energy: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through causal LSTM.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            initial_state: Optional (h0, c0) initial states
            return_energy: Whether to return energy trajectory
        
        Returns:
            Output tensor and optionally energy trajectory
        """
        output, (hn, cn) = self.lstm(x, initial_state)
        
        if self.track_energy and return_energy:
            # Compute energy at each time step
            energy = self.energy_layer(output).squeeze(-1)
            return output, energy
        
        return output


class CausalTransformerBlock(nn.Module):
    """
    Transformer block with causal attention for time series modeling.
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 8, 
                 mlp_ratio: float = 4.0, dropout: float = 0.1,
                 activation: nn.Module = nn.GELU()):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = CausalAttention(embed_dim, num_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            activation,
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connections.
        
        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim)
        
        Returns:
            Output tensor with same shape as input
        """
        # Self-attention with residual
        attn_out = self.attn(self.norm1(x))
        x = x + self.dropout1(attn_out)
        
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        
        return x


class RelativisticCausalLayer(nn.Module):
    """
    Causal layer that respects relativistic constraints.
    Ensures information propagation speed doesn't exceed speed of light.
    """
    
    def __init__(self, spatial_dim: int, feature_dim: int,
                 dx: float = 1.0, dt: float = 1.0, c: float = 1.0):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.feature_dim = feature_dim
        self.dx = dx  # Spatial resolution
        self.dt = dt  # Temporal resolution
        self.c = c    # Speed of light
        
        # Maximum spatial influence per time step
        self.max_influence_radius = int(c * dt / dx)
        
        # Learnable weights for causal influence
        self.spatial_conv = nn.Conv2d(
            feature_dim, feature_dim,
            kernel_size=2 * self.max_influence_radius + 1,
            padding=self.max_influence_radius
        )
        
        # Create causal mask based on light cone
        self._create_causal_mask()
    
    def _create_causal_mask(self):
        """Create mask that enforces light cone constraint."""
        kernel_size = 2 * self.max_influence_radius + 1
        mask = torch.zeros(kernel_size, kernel_size)
        center = self.max_influence_radius
        
        for i in range(kernel_size):
            for j in range(kernel_size):
                # Distance from center
                dist = math.sqrt((i - center)**2 + (j - center)**2)
                # Within light cone?
                if dist <= self.max_influence_radius:
                    mask[i, j] = 1.0
        
        self.register_buffer('causal_mask', mask)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply relativistic causal convolution.
        
        Args:
            x: Input tensor of shape (batch, channels, height, width)
        
        Returns:
            Causally filtered output
        """
        # Apply causal mask to convolution weights
        masked_weight = self.spatial_conv.weight * self.causal_mask.unsqueeze(0).unsqueeze(0)
        
        # Perform convolution with masked weights
        output = F.conv2d(
            x, masked_weight, self.spatial_conv.bias,
            padding=self.max_influence_radius
        )
        
        return output