"""
Day-033 — Self-Attention 과 Transformer 인코더 블록 (from scratch, 학습 없음)

어제(Day-032)의 scaled dot-product attention 을 '같은 시퀀스'에 적용하면
Self-Attention 이 된다. 여기에 멀티헤드·위치 인코딩·잔차·층정규화·FFN 을
얹으면 Transformer 인코더 블록이 완성된다.

- 학습(training) 없음: 랜덤 초기화로 '모양(shape)·정보 흐름'을 확인하는 데 집중.
- 시드 고정(torch.manual_seed(0))으로 재현 가능.
- 실행:  uv run python self_attention_intro.py
  (한글이 깨지면 먼저:  $env:PYTHONIOENCODING="utf-8")
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
torch.set_printoptions(precision=3, sci_mode=False)


# ──────────────────────────────────────────────────────────────────────
# 0. 어제의 한 줄 — scaled dot-product attention (Day-032 그대로)
# ──────────────────────────────────────────────────────────────────────
def scaled_dot_product_attention(Q, K, V, mask=None):
    d = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d)   # (..., Lq, Lk)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn = F.softmax(scores, dim=-1)                  # 행 합 = 1
    out = attn @ V                                    # (..., Lq, d)
    return out, attn


# ──────────────────────────────────────────────────────────────────────
# 1. Self-Attention — Q·K·V 를 '같은 시퀀스'에서 뽑는다
# ──────────────────────────────────────────────────────────────────────
def part1_self_attention():
    print("=" * 68)
    print("1부 — Self-Attention: 한 문장이 자기 자신을 참조한다")
    print("=" * 68)

    tokens = ["The", "animal", "didn't", "cross", "because", "it", "was", "tired"]
    L, d_model = len(tokens), 16

    # 각 토큰의 임베딩(랜덤). 실제로는 학습되지만 여기선 흐름만 본다.
    X = torch.randn(L, d_model)

    # Q·K·V 는 '같은 X'를 서로 다른 선형변환으로 투영해 얻는다 (그래서 self-)
    Wq, Wk, Wv = (nn.Linear(d_model, d_model, bias=False) for _ in range(3))
    Q, K, V = Wq(X), Wk(X), Wv(X)

    out, attn = scaled_dot_product_attention(Q, K, V)
    print(f"  입력 X shape = {tuple(X.shape)}  →  출력 shape = {tuple(out.shape)}  (같다!)")
    print(f"  주의 행렬 attn shape = {tuple(attn.shape)}  = (L, L): 각 토큰이 모든 토큰을 본다")
    print(f"  attn 각 행의 합 = {[round(x, 3) for x in attn.sum(dim=-1).tolist()]}  (모두 1.0 = 확률분포)")

    # 'it'(5번째, 0-index) 이 어떤 단어에 주의를 두는지 (랜덤 초기화라 해석은 무의미하지만
    # '자기 자신을 포함한 모든 위치'를 본다는 구조를 확인)
    it_idx = tokens.index("it")
    print(f"\n  '{tokens[it_idx]}' 이(가) 각 단어에 둔 주의:")
    for tok, w in zip(tokens, attn[it_idx].tolist()):
        bar = "█" * int(round(w * 40))
        print(f"    {tok:<9} {w:5.3f} {bar}")
    print("  → 학습이 되면 'it'→'animal' 같은 상호참조(coreference)에 주의가 쏠린다.")


# ──────────────────────────────────────────────────────────────────────
# 2. Multi-Head Attention — 여러 관점을 동시에
# ──────────────────────────────────────────────────────────────────────
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 은 n_heads 로 나누어떨어져야 한다"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, X):
        L, d_model = X.shape
        h, dh = self.n_heads, self.d_head
        # (L, d_model) → (h, L, dh): 헤드 축을 앞으로
        def split(t):
            return t.view(L, h, dh).transpose(0, 1)     # (h, L, dh)
        Q, K, V = split(self.Wq(X)), split(self.Wk(X)), split(self.Wv(X))
        out, attn = scaled_dot_product_attention(Q, K, V)   # (h, L, dh), (h, L, L)
        out = out.transpose(0, 1).reshape(L, d_model)       # 헤드들을 다시 이어붙임(concat)
        return self.Wo(out), attn                           # 마지막 선형변환 Wo


def part2_multihead():
    print("\n" + "=" * 68)
    print("2부 — Multi-Head Attention: 여러 주의를 병렬로")
    print("=" * 68)
    L, d_model, n_heads = 8, 16, 4
    X = torch.randn(L, d_model)
    mha = MultiHeadSelfAttention(d_model, n_heads)
    out, attn = mha(X)
    print(f"  d_model={d_model}, n_heads={n_heads} → 헤드당 차원 d_head={d_model // n_heads}")
    print(f"  입력 {tuple(X.shape)} → 출력 {tuple(out.shape)}  (모양 보존)")
    print(f"  헤드별 주의 행렬 attn shape = {tuple(attn.shape)} = (heads, L, L)")
    print("  → 각 헤드가 '다른 관계'(문법/의미/거리 등)를 따로 학습해 볼 수 있다.")


# ──────────────────────────────────────────────────────────────────────
# 3. Positional Encoding — 순서 정보를 더한다 (사인/코사인)
# ──────────────────────────────────────────────────────────────────────
def sinusoidal_positional_encoding(L, d_model):
    pos = torch.arange(L).unsqueeze(1).float()              # (L, 1)
    i = torch.arange(0, d_model, 2).float()                 # 짝수 인덱스
    div = torch.exp(-math.log(10000.0) * i / d_model)       # 1/10000^(2i/d)
    pe = torch.zeros(L, d_model)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def part3_positional():
    print("\n" + "=" * 68)
    print("3부 — Positional Encoding: self-attention 이 못 보는 '순서'를 주입")
    print("=" * 68)
    L, d_model = 8, 16
    pe = sinusoidal_positional_encoding(L, d_model)
    print(f"  PE shape = {tuple(pe.shape)} = (위치 L, 차원 d_model)")
    print("  왜 필요한가: self-attention 은 '집합(set) 연산'이라 순서를 모른다.")
    print("  → 입력 임베딩에 위치별로 다른 PE 를 '더해' 순서를 알린다.\n")
    # 서로 다른 위치의 PE 는 서로 다르다 → 위치를 구분할 수 있다
    print("  위치 0 vs 위치 1 PE 앞 6개 성분:")
    print(f"    pos0: {[round(x, 3) for x in pe[0, :6].tolist()]}")
    print(f"    pos1: {[round(x, 3) for x in pe[1, :6].tolist()]}")
    # 인접 위치일수록 PE 가 비슷하다(코사인 유사도 큼) — 상대 위치 정보를 담음
    cos01 = F.cosine_similarity(pe[0], pe[1], dim=0).item()
    cos07 = F.cosine_similarity(pe[0], pe[7], dim=0).item()
    print(f"  cos(PE_0, PE_1) = {cos01:.3f}  >  cos(PE_0, PE_7) = {cos07:.3f}")
    print("  → 가까운 위치일수록 인코딩이 비슷하다(부드러운 상대 위치 신호).")


# ──────────────────────────────────────────────────────────────────────
# 4. Transformer 인코더 블록 = MHA + 잔차/LN + FFN + 잔차/LN
# ──────────────────────────────────────────────────────────────────────
class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.mha = MultiHeadSelfAttention(d_model, n_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, X):
        # 1) Self-Attention 서브층 + 잔차 연결 + 층 정규화
        attn_out, _ = self.mha(X)
        X = self.ln1(X + attn_out)          # residual: 원본 X 를 더해 정보/기울기 보존
        # 2) 위치별 Feed-Forward 서브층 + 잔차 + LN
        ff_out = self.ffn(X)
        X = self.ln2(X + ff_out)
        return X


def part4_encoder_block():
    print("\n" + "=" * 68)
    print("4부 — Transformer 인코더 블록 (Self-Attn → +잔차/LN → FFN → +잔차/LN)")
    print("=" * 68)
    L, d_model, n_heads, d_ff = 8, 16, 4, 64
    X = torch.randn(L, d_model)
    pe = sinusoidal_positional_encoding(L, d_model)
    X = X + pe                              # 위치 인코딩을 더해 입력을 완성

    block = TransformerEncoderBlock(d_model, n_heads, d_ff)
    Y = block(X)
    print(f"  입력(임베딩+PE) {tuple(X.shape)} → 인코더 블록 → 출력 {tuple(Y.shape)}  (모양 보존!)")
    print("  모양이 같으므로 블록을 N번 쌓을 수 있다(깊은 Transformer).")

    # 블록을 여러 개 쌓아도 shape 유지 — '순환 없이' 깊이를 얻는다
    stack = nn.Sequential(*[TransformerEncoderBlock(d_model, n_heads, d_ff) for _ in range(6)])
    Z = stack(X)
    print(f"  6개 블록 스택 통과 후 shape = {tuple(Z.shape)}  (여전히 보존)")
    print("  → 이것이 BERT/GPT 의 뼈대다. 순환(recurrence) 없이 모든 위치를 병렬 처리.")


if __name__ == "__main__":
    part1_self_attention()
    part2_multihead()
    part3_positional()
    part4_encoder_block()
    print("\n[끝] Self-Attention → Multi-Head → +Positional → 인코더 블록까지 손으로 확인했다.")
    print("     다음 편(Day-034)에서 이 블록의 세부(왜 잔차·LN 이 필수인지)를 더 깊이 본다.")
