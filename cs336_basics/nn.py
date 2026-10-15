import torch, math
from einops import rearrange, einsum

class Linear(torch.nn.Module):
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        std = math.sqrt(2 / (in_features + out_features))
        torch.nn.init.trunc_normal_(self.weight, mean = 0.0, std = std, a = -3 * std, b = 3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... i, o i -> ... o")

class Embedding(torch.nn.Module):
    def __init__(self, 
                num_embeddings: int, 
                embedding_dim: int, 
                device: torch.device | None = None, 
                dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = torch.nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.weight, mean = 0.0, std = 1.0, a = -3.0, b = 3.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight[x]

class RMSNorm(torch.nn.Module):
    def __init__(self, 
                 d_model: int, 
                 eps: float = 1e-5,
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones((d_model,), device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, sequence_length, d_model) or (sequence_length, d_model)
        in_type = x.dtype
        x = x.to(torch.float32)

        rms_x = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        x = x / rms_x * self.weight

        return x.to(in_type)

class SiLU(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

class SwiGLU(torch.nn.Module):
    def __init__(self, 
                 d_model: int, 
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = int((8 * d_model / 3) / 64 + 0.5)
        self.w1 = torch.nn.Parameter(torch.empty((self.d_ff, d_model), device=device, dtype=dtype))
        self.w2 = torch.nn.Parameter(torch.empty((d_model, self.d_ff), device=device, dtype=dtype))
        self.w3 = torch.nn.Parameter(torch.empty((self.d_ff, d_model), device=device, dtype=dtype))
        self.silu = SiLU()

        std = math.sqrt(2 / (self.d_ff + d_model))
        torch.nn.init.trunc_normal_(self.w1, mean=0.0, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(self.w2, mean=0.0, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(self.w3, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, sequence_length, d_model) or (sequence_length, d_model)
        gate = einsum(self.w1, x, "ff model, ... seq model -> ... seq ff")
        gate = self.silu(gate)

        ret = einsum(self.w3, x, "ff model, ... seq model -> ... seq ff")
        ret = einsum(gate, ret, "... seq ff, ... seq ff -> ... seq ff")
        ret = einsum(self.w2, ret, "model ff, ... seq ff -> ... seq model")
        return ret

class RotaryPositionEmbedding(torch.nn.Module):
    def __init__(self, 
                 theta: float, 
                 d_k: int, 
                 max_seq_len: int, 
                 device: torch.device | None = None) -> None:
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device = device).float() / d_k))
        pos = torch.arange(max_seq_len, device = device).float()
        angles = einsum(pos, inv_freq, "seq, dk -> seq dk")
        self.register_buffer("cos", angles.cos(), persistent = False)
        self.register_buffer("sin", angles.sin(), persistent = False)

    def forward(self, x: torch.Tensor, token_pos: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, sequence_length, d_k) or (sequence_length, d_k)
        # token_pos: (..., sequence_length)

        cos = self.cos[token_pos] # cos : (..., sequence_length, d_k/2)
        sin = self.sin[token_pos] # sin : (..., sequence_length, d_k/2)
        x1, x2 = x[..., ::2], x[..., 1::2]
        x_rotated = rearrange([x1 * cos - x2 * sin, x1 * sin + x2 * cos], "two ... seq dk -> ... seq (dk two)")
        return x_rotated
        
