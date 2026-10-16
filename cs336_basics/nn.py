import torch, math
from torch import Tensor
from einops import rearrange, einsum
from jaxtyping import Float, Int, Bool

# 维度命名约定：
#   ...        任意前导 batch 维（可为空）
#   d_in/d_out Linear 输入/输出维
#   d_model    隐层维；d_ff FFN 中间维；d_k 单头维
#   seq        序列长度；h 头数；vocab_size 词表大小


def _as_weight_dict(state_dict):
    """允许直接传一个 Tensor，视为 {"weight": tensor}。"""
    if isinstance(state_dict, torch.Tensor):
        return {"weight": state_dict}
    return state_dict


class Linear(torch.nn.Module):
    weight: Float[Tensor, "d_out d_in"]

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
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        return einsum(x, self.weight, "... i, o i -> ... o")

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        # 接受 {"weight": W} 或直接传 W，W 形状 (out_features, in_features)
        return super().load_state_dict(_as_weight_dict(state_dict), strict=strict, assign=assign)


class Embedding(torch.nn.Module):
    weight: Float[Tensor, "vocab_size d_model"]

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = torch.nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, x: Int[Tensor, "..."]) -> Float[Tensor, "... d_model"]:
        return self.weight[x]

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        # 接受 {"weight": W} 或直接传 W，W 形状 (num_embeddings, embedding_dim)
        return super().load_state_dict(_as_weight_dict(state_dict), strict=strict, assign=assign)


class RMSNorm(torch.nn.Module):
    weight: Float[Tensor, "d_model"]

    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones((d_model,), device=device, dtype=dtype))

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        in_type = x.dtype
        x = x.to(torch.float32)

        rms_x = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        x = x / rms_x * self.weight

        return x.to(in_type)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        # 接受 {"weight": g} 或直接传 g，g 形状 (d_model,)
        return super().load_state_dict(_as_weight_dict(state_dict), strict=strict, assign=assign)


class SiLU(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return x * torch.sigmoid(x)


class SwiGLU(torch.nn.Module):
    # state_dict keys: w1.weight (d_ff, d_model), w2.weight (d_model, d_ff), w3.weight (d_ff, d_model)
    def __init__(self,
                 d_model: int,
                 d_ff: int | None = None,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.d_model = d_model
        # 默认 d_ff ≈ 8/3 * d_model，向上取到 64 的倍数
        self.d_ff = d_ff if d_ff is not None else int(math.ceil(8 * d_model / 3 / 64)) * 64
        self.w1 = Linear(d_model, self.d_ff, device=device, dtype=dtype)
        self.w2 = Linear(self.d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, self.d_ff, device=device, dtype=dtype)
        self.silu = SiLU()

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        return self.w2(self.silu(self.w1(x)) * self.w3(x))


class RotaryPositionEmbedding(torch.nn.Module):
    cos: Float[Tensor, "max_seq_len half_d_k"]
    sin: Float[Tensor, "max_seq_len half_d_k"]

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

        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        pos = torch.arange(max_seq_len, device=device).float()
        angles = einsum(pos, inv_freq, "seq, dk -> seq dk")
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(
        self,
        x: Float[Tensor, "... seq d_k"],
        token_pos: Int[Tensor, "... seq"],  # 前导维需能与 x 广播，如 (b, 1, seq) vs (b, h, seq, d_k)
    ) -> Float[Tensor, "... seq d_k"]:
        cos = self.cos[token_pos]  # (..., seq, d_k/2)
        sin = self.sin[token_pos]  # (..., seq, d_k/2)
        x1, x2 = x[..., ::2], x[..., 1::2]
        x_rotated = rearrange([x1 * cos - x2 * sin, x1 * sin + x2 * cos], "two ... seq dk -> ... seq (dk two)")
        return x_rotated


def softmax(x: Float[Tensor, "..."], dim: int = -1) -> Float[Tensor, "..."]:
    max_val = torch.max(x, dim=dim, keepdim=True).values
    x_exp = torch.exp(x - max_val)
    x_exp_sum = torch.sum(x_exp, dim=dim, keepdim=True)
    return x_exp / x_exp_sum

def cross_entropy(inputs: Float[Tensor, " batch_size vocab_size"], 
                  targets : Int[Tensor, " batch_size"]) -> torch.Tensor:
    logsumexp = inputs.logsumexp(dim=-1, keepdim=True)
    target_logits = torch.gather(inputs, -1, targets.unsqueeze(dim=-1)).squeeze(dim=-1)

    return torch.mean(logsumexp.squeeze(dim=-1) - target_logits)

def scaled_dot_product_attention(
    q: Float[Tensor, "... q_len d_k"],
    k: Float[Tensor, "... k_len d_k"],
    v: Float[Tensor, "... k_len d_v"],
    mask: Bool[Tensor, "... q_len k_len"] | None = None,  # True = 可见
) -> Float[Tensor, "... q_len d_v"]:
    att = einsum(q, k, "... q_len d_k, ... k_len d_k -> ... q_len k_len")
    att = att / math.sqrt(q.shape[-1])

    if mask is not None:
        att = att.masked_fill(~mask, float("-inf"))
    att = softmax(att, dim=-1)
    att = einsum(att, v, "... q_len k_len, ... k_len d_v -> ... q_len d_v")

    return att


class Multihead_self_attention(torch.nn.Module):
    # state_dict keys: q_proj.weight, k_proj.weight, v_proj.weight, output_proj.weight
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

        self.rope = RotaryPositionEmbedding(theta=theta, d_k=self.dim_per_head, max_seq_len=max_seq_len, device=device) if with_rope else None

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(
        self,
        x: Float[Tensor, "... seq d_model"],
        pos: Int[Tensor, "... seq"] | None = None,
    ) -> Float[Tensor, "... seq d_model"]:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = rearrange(q, "... seq (h d) -> ... h seq d", h=self.num_heads)
        k = rearrange(k, "... seq (h d) -> ... h seq d", h=self.num_heads)
        v = rearrange(v, "... seq (h d) -> ... h seq d", h=self.num_heads)

        if self.with_rope:
            if pos is not None:
                pos = rearrange(pos, "... seq -> ... 1 seq")  # 为 head 维插入单例维
                q = self.rope(q, token_pos=pos)
                k = self.rope(k, token_pos=pos)
            else:
                raise ValueError("Position tensor 'pos' must be provided when using rotary position embedding.")

        seq_len = x.shape[-2]
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        att = scaled_dot_product_attention(q, k, v, mask)

        att = rearrange(att, "... h seq d -> ... seq (h d)")
        out = self.output_proj(att)
        return out


class TransformerBlock(torch.nn.Module):
    # state_dict keys: attn.{q,k,v,output}_proj.weight, ln1.weight, ln2.weight, ffn.w{1,2,3}.weight
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

        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(
        self,
        x: Float[Tensor, "... seq d_model"],
        pos: Int[Tensor, "... seq"] | None = None,
    ) -> Float[Tensor, "... seq d_model"]:
        y = x + self.attn(self.ln1(x), pos=pos)
        y = y + self.ffn(self.ln2(y))
        return y


class TransformerLM(torch.nn.Module):
    def __init__(self,
                 vocab_size: int,
                 context_length: int,
                 d_model: int,
                 num_layers: int,
                 num_heads: int,
                 d_ff: int,
                 rope_theta: float,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None,
                 ) -> None:
        super().__init__()

        self.layers = torch.nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=rope_theta,
                device=device,
                dtype=dtype
            )
            for _ in range(num_layers)
        ])

        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)

        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(
        self,
        x: Int[Tensor, "... seq"],
        pos: Int[Tensor, "... seq"] | None = None,
    ) -> Float[Tensor, "... seq vocab_size"]:
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x, pos=pos)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x

