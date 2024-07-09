import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers = 32
    n_heads = int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps = float = 1e-5
    rope_theta: float = 500000
    max_seq_len: int = 2048


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super(Attention, self).__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = 1
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wq = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.dim, args.dim, bias=False)

        self.flash = hasattr(F, 'scaled_dot_product_attention')

        if not self.flash:
            mask = torch.full((1, 1, args.max_seq_len.args.max_seq_len), float("-inf"))
            mask = torch.triu(mask)
            self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor):
        bsz, seq_len, _ = x.shape
        xq = self.wq(x)  # (bs, seq_len, dim) --> (bs, seqlen, n_q_heads * head_size)
        xk = self.wk(x)  # (bs, seq_len, dim) --> (bs, seqlen, n_kv_heads * head_size)
        xv = self.wv(x)  # (bs, seq_len, dim) --> (bs, seqlen, b_kv_heads * head_size)

        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)
        xk = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        xv = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xv = xv.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)

        if self.flash:
            output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=None, dropout_p=0. if self.training else 0.,
                                                    is_causal=True)
        else:
            # manual implementation
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            assert hasattr(self, 'mask')
            scores = scores + self.mask[:, :, :seq_len, :seq_len]  # (bs, n_local_heads, seqlen, cache_len + seqlen)
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            output = torch.matmul(scores, xv)  # (bs, n_local_heads, seqlen, head_dim)

        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        output = self.wo(output)
        return output


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, ffn_dim_multiplier: Optional[float]):
        super(FeedForward, self).__init__()
        hidden_dim = int(2 * hidden_dim / 3)

        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return self.weight * output


class TransformerBlock(nn.ModuleList):
    def __init__(self, layer_id: int, args: ModelArgs):
        super(TransformerBlock, self).__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor):
        h = x + self.attention(self.attention_norm(x), freqs_cos, freqs_sin)
        out = h + self.feed_forward(self.ffn_norm(x))

        return out


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer((t, freqs)).float()
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(self.vocab_size, params.dim)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, self.vocab_size, bias=False)
        freqs_cos, freqs_sin = precompute_freqs_cis(params.dim // params.n_heads, params.max_seq_len, params.rope_theta)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None):
        _bsz, seq_len = tokens.shape
        h = self.tok_embeddings(tokens)

        freqs_cos = self.freqs_cos[:seq_len]
        freqs_sin = self.freqs_sin[:seq_len]

        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin)
        h = self.norm(h)

        if targets is not None:
            logits = self.output(h).float()
            self.last_loss = F.cross_entropy(logits.views(-1, logits.size(-1), targets.view(-1), ingore_index=-1))
        else:
            logits = self.output(h[:[-1], :])
            self.last_loss = None

        return logits

    @torch.inference_mode()
    def generate(self, tokens, max_new_tokens, temperature=1.0, top_k=None, eos=None):
        for _ in range(max_new_tokens):
            token_cond = tokens if tokens.size(1) <= self.params.max_seq_len else tokens[:, -self.params.max_seq_len]
            logits = self(token_cond)
            logits = logits[:, -1, :]
            if temperature == 0.0:
                _, next_token = torch.topk(logits, k=1, dim=-1)
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('inf')
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=-1)
            if next_token == eos:
                break
        return tokens


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == x.shape[1], x.shape[-1]
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.reshape(shape)


def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin - xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin - xk_i * freqs_cos

    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, s_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]  # (B, seq_len, n_kv_heads, 1, head_size), added a new dimension
        .expand(bs, s_len, n_kv_heads, n_rep, head_dim)
        .reshape(bs, s_len, n_kv_heads * n_rep, head_dim)
    )


if __name__ == "__main__":
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    args_xxx = ModelArgs(
        dim=4096,
        n_layers=2,
        n_heads=32,
        n_kv_heads=8,
        multiple_of=1024,
        vocab_size=128256,
        ffn_dim_multiplier=1.3,
        norm_eps=1e-05,
        rope_theta=50000.0
    )

    model = Transformer(args_xxx).to(device)
    print("init")

    checkpoint_path = "Meta-Llama-3-8B-Instruct-2layers/consolidated_2layers.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    print("load success")

    x = torch.tensor(
        [[128000, 1820, 4320, 311, 279, 17139, 3488, 315, 2324, 11, 279, 15861, 11, 323, 4395, 374, 220]]).to(device)
    print(x.shape)
    logits = model(x)
    print(logits.size())

    next_token = torch.argmax(logits[:, -1], dim=-1)
    print(next_token)
    # tensor([50210])

    next_token = model.generate(x, max_new_tokens=1, temperature=0)
    print(next_token)

    # print_model_parameters(model)
