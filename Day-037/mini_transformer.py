"""
Day-037 — B3 build (part 2): a mini Transformer (decoder-only) from scratch in PyTorch.

Day-036 에서 NumPy 로 MLP 의 forward/backward 를 손코딩했다면,
오늘은 autograd 를 다시 켜고 *구조* 를 손코딩한다:
  - Q/K/V 투영, scaled dot-product, causal mask, multi-head, Pre-LN 블록
  - nn.MultiheadAttention / nn.Transformer 는 쓰지 않는다 (nn.Linear/nn.LayerNorm 까지만)

과제(task): 길이 6 의 숫자열(0~4)을 받아 '정렬된 6개'를 이어서 생성한다.
  예) 0 2 1 1 3 2  ->  0 1 1 2 2 3
디코더-온리 LM 으로 [입력 6개 | 정답 6개] = 12 토큰 시퀀스의 '다음 토큰'을 맞히되,
손실은 정답 절반에만 건다.

실행: uv run python mini_transformer.py
      (uv 프로젝트에 torch 가 없으면) uv run --with torch python mini_transformer.py
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
DEVICE = "cpu"

# ---------------------------------------------------------------- 1. 데이터
VOCAB = 5            # 토큰 0~4
N_IN = 6             # 입력 숫자 개수
SEQ = 2 * N_IN       # 전체 시퀀스 길이 12 = [입력 6 | 정렬 6]
BLOCK = SEQ - 1      # 모델이 보는 컨텍스트 길이 11 (마지막 토큰은 정답으로만 씀)


def make_batch(batch_size: int, gen: torch.Generator):
    """(inp, tgt) 를 만든다. tgt 의 앞부분은 -1 로 두어 손실에서 제외한다."""
    x = torch.randint(0, VOCAB, (batch_size, N_IN), generator=gen)
    s, _ = torch.sort(x, dim=1)
    seq = torch.cat([x, s], dim=1)          # (B, 12)
    inp = seq[:, :-1]                        # (B, 11)  현재 토큰
    tgt = seq[:, 1:].clone()                 # (B, 11)  다음 토큰
    tgt[:, : N_IN - 1] = -1                  # 입력 절반을 '예측'하는 건 무의미 → 무시
    return inp, tgt


# ---------------------------------------------------------------- 2. 모델
class CausalSelfAttention(nn.Module):
    """마스킹 self-attention. Q,K,V 를 한 번에 뽑고 헤드로 쪼갠다."""

    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)   # W_Q, W_K, W_V 를 하나로
        self.proj = nn.Linear(d_model, d_model)      # W_O
        # 하삼각 행렬: tril[i][j] = 1 이면 i 가 j 를 볼 수 있다
        self.register_buffer(
            "tril", torch.tril(torch.ones(BLOCK, BLOCK)).view(1, 1, BLOCK, BLOCK)
        )
        self.last_attn: torch.Tensor | None = None   # 시각화용

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)                       # 각 (B,T,C)
        # (B,T,C) -> (B, n_head, T, d_head)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)     # (B,h,T,T)
        if causal:
            att = att.masked_fill(self.tril[:, :, :T, :T] == 0, float("-inf"))
        att = att.softmax(dim=-1)
        self.last_attn = att.detach()

        y = att @ v                                                  # (B,h,T,d_head)
        y = y.transpose(1, 2).contiguous().view(B, T, C)             # 헤드 합치기
        return self.proj(y)


class Block(nn.Module):
    """Pre-LN 트랜스포머 블록: x + Attn(LN(x)) → x + MLP(LN(x))"""

    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), causal=causal)
        x = x + self.mlp(self.ln2(x))
        return x


class MiniTransformer(nn.Module):
    def __init__(self, d_model: int = 64, n_head: int = 2, n_layer: int = 2):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, d_model)
        self.pos = nn.Embedding(BLOCK, d_model)          # 학습형 위치 임베딩
        self.blocks = nn.ModuleList([Block(d_model, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, VOCAB, bias=False)

    def forward(self, idx: torch.Tensor, causal: bool = True) -> torch.Tensor:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok(idx) + self.pos(pos)                # (B,T,C)
        for blk in self.blocks:
            x = blk(x, causal=causal)
        return self.head(self.ln_f(x))                   # (B,T,VOCAB) 로짓

    @torch.no_grad()
    def generate(self, prompt: torch.Tensor, n_new: int, causal: bool = True):
        """greedy 디코딩: 한 토큰씩 뽑아 뒤에 붙인다 (미래는 정말로 모른다)."""
        idx = prompt
        for _ in range(n_new):
            logits = self(idx[:, -BLOCK:], causal=causal)
            nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


# ---------------------------------------------------------------- 3. 학습/평가
def train(model: nn.Module, steps: int = 3000, causal: bool = True, log: bool = True):
    gen = torch.Generator().manual_seed(1234)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    model.train()
    last = float("nan")
    for step in range(1, steps + 1):
        inp, tgt = make_batch(128, gen)
        logits = model(inp, causal=causal)
        loss = F.cross_entropy(
            logits.reshape(-1, VOCAB), tgt.reshape(-1), ignore_index=-1
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last = loss.item()
        if log and (step == 1 or step % 500 == 0):
            print(f"  step {step:5d}:  loss={last:.4e}")
    return last


@torch.no_grad()
def evaluate(model: nn.Module, n: int = 500, causal: bool = True, seed: int = 999):
    """held-out 입력에 대해 6개를 '생성'해 정렬 정답과 완전일치하는 비율."""
    gen = torch.Generator().manual_seed(seed)
    model.eval()
    x = torch.randint(0, VOCAB, (n, N_IN), generator=gen)
    gold, _ = torch.sort(x, dim=1)
    out = model.generate(x, N_IN, causal=causal)[:, N_IN:]
    exact = (out == gold).all(dim=1).float().mean().item()
    per_tok = (out == gold).float().mean().item()
    return exact, per_tok


def show_attention(model: MiniTransformer):
    """마지막 블록 head 0 의 attention 행렬을 한 예제에 대해 출력."""
    gen = torch.Generator().manual_seed(7)
    x = torch.randint(0, VOCAB, (1, N_IN), generator=gen)
    full = model.generate(x, N_IN)                    # (1,12)
    model(full[:, :BLOCK])                            # attention 캐시 갱신
    att = model.blocks[-1].attn.last_attn[0, 0]       # (T,T)
    toks = full[0].tolist()
    print("\n  예제 시퀀스:", " ".join(map(str, toks[:N_IN])),
          "->", " ".join(map(str, toks[N_IN:])))
    print("  마지막 블록 head0 attention (행=질의 위치, 열=키 위치, 0.00 은 마스킹)")
    print("        " + "".join(f"k{j:<5d}" for j in range(BLOCK)))
    for i in range(BLOCK):
        row = "".join(f"{att[i, j].item():<6.2f}" for j in range(BLOCK))
        print(f"  q{i:<2d} | {row}")


# ---------------------------------------------------------------- 4. main
def main() -> None:
    print("[과제] 길이 6 숫자열(0~4) -> 정렬된 6개를 이어서 '생성'")
    inp, tgt = make_batch(2, torch.Generator().manual_seed(42))
    print(f"  예: inp={inp[0].tolist()}  tgt={tgt[0].tolist()}  (-1 은 손실 제외)\n")

    model = MiniTransformer(d_model=64, n_head=2, n_layer=2)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"[모델] MiniTransformer d_model=64, heads=2, layers=2, 파라미터 {n_param:,}개")

    print("[학습] causal mask ON, AdamW lr=3e-3, batch=128")
    train(model, steps=3000, causal=True)
    exact, per_tok = evaluate(model, causal=True)
    print(f"[결과] held-out 완전일치 정확도 = {exact:.3f}  (토큰 단위 {per_tok:.3f})")

    show_attention(model)

    print("\n[대조 실험] 마스크를 끄면? (causal=False 로 학습·생성)")
    bad = MiniTransformer(d_model=64, n_head=2, n_layer=2)
    bad_loss = train(bad, steps=3000, causal=False, log=False)
    bad_exact, bad_tok = evaluate(bad, causal=False)
    print(f"  학습 손실 = {bad_loss:.4e}  -> 학습 목표는 멀쩡히 맞힌다(미래를 봤으니 당연)")
    print(f"  그런데 생성 정확도 = {bad_exact:.3f} (토큰 단위 {bad_tok:.3f}) -> 붕괴")


if __name__ == "__main__":
    main()
