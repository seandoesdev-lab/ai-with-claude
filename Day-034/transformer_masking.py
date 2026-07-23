"""
Day-034 — Transformer를 완성하다: 인과 마스크 · 인코더 vs 디코더 · Post-/Pre-LN
Completing the Transformer: causal mask, encoder vs decoder, Post-/Pre-LN.

학습 없이(랜덤 초기화) '구조와 마스킹'만 관찰한다. 시드 고정으로 재현 가능.
No training — we only observe *structure and masking*. Seeded for reproducibility.

실행 (Day-028 uv 프로젝트에서):
    uv run python transformer_masking.py
한글이 깨지면 먼저:  $env:PYTHONIOENCODING="utf-8"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# 공통: 마스크를 받는 scaled dot-product attention (Day-032/033 함수의 확장판)
# scaled dot-product attention *with an optional additive mask*.
# ---------------------------------------------------------------------------
def scaled_dot_product_attention(Q, K, V, mask=None):
    d = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / (d ** 0.5)      # (..., L, L)
    if mask is not None:
        # mask == True 인 자리를 -inf 로 → softmax 후 0 (그 위치를 '못 봄')
        scores = scores.masked_fill(mask, float("-inf"))
    attn = F.softmax(scores, dim=-1)                    # (..., L, L)
    out = attn @ V                                      # (..., L, d)
    return out, attn


def causal_mask(L):
    """상삼각(대각선 위) = True → '미래' 위치를 가린다. (L, L) bool.
    A True in row i, col j (j > i) means position i may NOT attend to j."""
    return torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)


def banner(title):
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


# ===========================================================================
# 1부 — 인과 마스크(causal mask)는 '미래'를 -inf 로 가린다
#        A causal mask hides the future by setting it to -inf.
# ===========================================================================
banner("1부 — 인과 마스크: 각 위치는 자기 이하(<=)만 본다")

L, d_model = 6, 16
X = torch.randn(L, d_model)
Wq, Wk, Wv = (nn.Linear(d_model, d_model, bias=False) for _ in range(3))
Q, K, V = Wq(X), Wk(X), Wv(X)

m = causal_mask(L)
print("인과 마스크 (True=가림, 상삼각):")
print(m.int())

out, attn = scaled_dot_product_attention(Q, K, V, mask=m)
print("\n주의 행렬 attn (소수 2자리) — 상삼각이 0 이어야 한다:")
print(torch.round(attn * 100) / 100)
print("\n각 행의 0이 아닌 원소 개수 =", (attn > 1e-6).sum(dim=1).tolist(),
      " -> i번째 위치는 정확히 i+1개(자기 포함, 과거만) 본다")
print("각 행의 합 =", torch.round(attn.sum(dim=1) * 100) / 100, " (여전히 확률분포)")


# ===========================================================================
# 2부 — 인코더(양방향) vs 디코더(단방향): 같은 입력, 다른 마스크
#        Encoder (bidirectional) vs decoder (unidirectional) — same X, different mask.
# ===========================================================================
banner("2부 — 인코더 vs 디코더: 마스크 하나로 갈린다 (-> BERT vs GPT)")

# 인코더: 마스크 없음 → 모든 위치가 서로를 본다(양방향)
_, attn_enc = scaled_dot_product_attention(Q, K, V, mask=None)
# 디코더: 인과 마스크 → 과거만 본다(단방향/autoregressive)
_, attn_dec = scaled_dot_product_attention(Q, K, V, mask=causal_mask(L))

print("인코더 attn: 0인 원소 개수 =", (attn_enc <= 1e-6).sum().item(),
      " (하나도 안 가림 -> 양방향, BERT 계열)")
print("디코더 attn: 0인 원소 개수 =", (attn_dec <= 1e-6).sum().item(),
      f" (상삼각 {L*(L-1)//2}개를 가림 -> 단방향, GPT 계열)")
print("\n마지막 토큰(위치 5)이 보는 범위:")
print("  인코더:", (attn_enc[5] > 1e-6).sum().item(), "개 위치(전체)")
print("  디코더:", (attn_dec[5] > 1e-6).sum().item(), "개 위치(0..5)")
print("첫 토큰(위치 0)이 보는 범위:")
print("  디코더:", (attn_dec[0] > 1e-6).sum().item(), "개 위치(자기 자신뿐)")


# ===========================================================================
# 3부 — Post-LN vs Pre-LN: 잔차·정규화의 '순서'만 다르다
#        Post-LN vs Pre-LN differ only in *where* LayerNorm sits.
# ===========================================================================
banner("3부 — Post-LN vs Pre-LN: LayerNorm 위치만 바꾼 두 블록")


class DecoderBlock(nn.Module):
    """인과 마스크를 쓰는 self-attention + FFN. norm_first 로 Post/Pre-LN 전환."""
    def __init__(self, d_model=16, n_heads=4, d_ff=64, norm_first=False):
        super().__init__()
        self.h, self.dh = n_heads, d_model // n_heads
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)
        )
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.norm_first = norm_first

    def _mha(self, x):
        L = x.size(0)
        def split(t):                     # (L, d_model) → (h, L, dh)
            return t.view(L, self.h, self.dh).transpose(0, 1)
        Q, K, Vv = split(self.Wq(x)), split(self.Wk(x)), split(self.Wv(x))
        out, _ = scaled_dot_product_attention(Q, K, Vv, mask=causal_mask(L))
        out = out.transpose(0, 1).reshape(L, -1)
        return self.Wo(out)

    def forward(self, x):
        if self.norm_first:               # Pre-LN: x + Sublayer(LN(x))
            x = x + self._mha(self.ln1(x))
            x = x + self.ffn(self.ln2(x))
        else:                             # Post-LN: LN(x + Sublayer(x))  (원 논문)
            x = self.ln1(x + self._mha(x))
            x = self.ln2(x + self.ffn(x))
        return x


emb = torch.randn(L, d_model)
post = DecoderBlock(d_model, norm_first=False)
pre = DecoderBlock(d_model, norm_first=True)
print("입력 (L, d_model) =", tuple(emb.shape))
print("Post-LN 출력 shape =", tuple(post(emb).shape), " (모양 보존)")
print("Pre-LN  출력 shape =", tuple(pre(emb).shape),  " (모양 보존)")
print("두 블록은 '연산 순서'만 다르다 — Post: LN(x+f(x)),  Pre: x+f(LN(x))")


# ===========================================================================
# 4부 — 디코더를 여러 개 쌓아 GPT 스타일 백본 만들기 (모양 보존 확인)
#        Stack decoder blocks → a GPT-style backbone (shape preserved).
# ===========================================================================
banner("4부 — 디코더 블록 N개 스택 = GPT 스타일 백본")

N = 6
stack = nn.Sequential(*[DecoderBlock(d_model, norm_first=True) for _ in range(N)])
y = stack(emb)
print(f"{N}개 디코더 블록 통과: {tuple(emb.shape)} -> {tuple(y.shape)}  (여전히 보존)")
print("-> 입출력 모양이 같아 N번 쌓을 수 있다. 인과 마스크 덕에 각 위치는")
print("   자기 이하만 보므로, 다음 토큰 예측(autoregressive LM)에 안전하다.")

print("\n완료. (학습이 없어 값 자체는 랜덤 — 오늘의 초점은 '마스크와 구조'다.)")
