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
        self.d_ff = int((8 * d_model / 3) / 64 + 0.5) * 64
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
        

def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    max_val = torch.max(x, dim = dim, keepdim = True).values
    x_exp = torch.exp(x - max_val)
    x_exp_sum = torch.sum(x_exp, dim = dim, keepdim = True)
    return x_exp / x_exp_sum

def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """
        q: (batch_size, q_len, d_k)
        k: (batch_size, k_len, d_k)
        v: (batch_size, k_len, d_v)
        mask: (batch_size, q_len, k_len) or None

    Returns:
        torch.Tensor: Output of the scaled dot-product attention with shape (batch_size, q_len, d_v).
    """

    att = einsum(q, k, "... q_len d_k, ... k_len d_k -> ... q_len k_len")
    att = att / math.sqrt(q.shape[-1])

    if mask is not None:
        att = att.masked_fill(~mask, float("-inf"))
    att = softmax(att, dim = -1)
    att = einsum(att, v, "... q_len k_len, ... k_len d_v -> ... q_len d_v")

    return att

class Multihead_self_attention(torch.nn.Module):
    def __init__(self, 
                 d_model: int, 
                 num_heads: int, 
                 with_rope: bool = True,
                 max_seq_len: int | None = None,
                 theta: float | None = None,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None,
                 ) -> None:
        super().__init__() 

        self.d_model = d_model
        self.num_heads = num_heads
        self.dim_per_head = d_model // num_heads
        self.with_rope = with_rope

        self.rope = RotaryPositionEmbedding(theta = theta, d_k = self.dim_per_head, max_seq_len = max_seq_len, device=device) if with_rope else None

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.out_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x : torch.Tensor, pos : torch.Tensor | None = None) -> torch.Tensor:
        # x: (batch_size, seq_len, d_model)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = rearrange(q, "... seq (h d) -> ... h seq d", h = self.num_heads) # q : (... num_head seq_len dim_per_head)
        k = rearrange(k, "... seq (h d) -> ... h seq d", h = self.num_heads) # k : (... num_head seq_len dim_per_head)
        v = rearrange(v, "... seq (h d) -> ... h seq d", h = self.num_heads) # v : (... num_head seq_len dim_per_head)

        if self.with_rope:
            if pos is not None:
                pos = rearrange(pos, "... seq -> ... 1 seq") # add a singleton dimension for the head axis
                q = self.rope(q, token_pos = pos)
                k = self.rope(k, token_pos = pos)
            else:
                raise ValueError("Position tensor 'pos' must be provided when using rotary position embedding.")

        seq_len = x.shape[-2]
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        att = scaled_dot_product_attention(q, k, v, mask) 

        att = rearrange(att, "... h seq d -> ... seq (h d)")
        out = self.out_proj(att)
        return out

class TransformerBlock(torch.nn.Module):
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 max_seq_len: int,
                 theta: float,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None,
                 ) -> None:
        super().__init__()

        self.attn = Multihead_self_attention(
            d_model=d_model,
            num_heads=num_heads,
            with_rope=True,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype
        )

        self.ff = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

        self.rms_norm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.rms_norm2 = RMSNorm(d_model, device=device, dtype=dtype)


    def forward(self, x: torch.Tensor, pos: torch.Tensor | None = None) -> torch.Tensor:
        y = x + self.attn(self.rms_norm1(x), pos=pos)
        y = y + self.ff(self.rms_norm2(y))
        return y