"""
Day-038 — BERT: 마스크를 걷고 '양방향'으로 사전학습하기
(Bidirectional pre-training with Masked Language Modeling)

Day-037 의 미니 Transformer 골격을 재사용하되, 인과 마스크를 '끄고'
BERT 의 사전학습 목적함수(MLM)를 구현해 네 가지를 숫자로 확인한다.

  실험 1. 양방향(BERT) vs 단방향(GPT) — 빈칸 채우기 정확도
  실험 2. 위치별 정확도 — 단방향 모델은 '오른쪽 정보'가 필요한 자리에서만 실패
  실험 3. 80/10/10 마스킹 규칙 — 왜 [MASK] 100% 로 학습하면 안 되는가
  실험 4. 학습된 attention 이 정말 양방향인지 (상삼각이 0 이 아님)

과제(거울 시퀀스 / mirror task):
    무작위 4토큰 x1 x2 x3 x4 를 뽑아  seq = [x1 x2 x3 x4 | x4 x3 x2 x1]
    한 자리를 가리고 그 토큰을 복원한다.
    → 정답은 항상 '거울 짝(mirror partner)' 위치에 있다.
      왼쪽 절반(0~3)의 짝은 '오른쪽'에, 오른쪽 절반(4~7)의 짝은 '왼쪽'에 있다.
      따라서 단방향(causal) 모델은 구조적으로 절반만 풀 수 있다.

실행:  uv run python bert_mlm.py
       (uv 프로젝트에 torch 가 없으면) uv run --with torch python bert_mlm.py
한글이 깨지면 먼저:  $env:PYTHONIOENCODING="utf-8"
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# ── 설정 (config) ──────────────────────────────────────────────────────────
N_WORDS = 8          # 실제 단어 토큰 id: 0..7
MASK_ID = N_WORDS    # [MASK] 토큰 id = 8
VOCAB = N_WORDS + 1  # 9
N_HALF = 4
BLOCK = 2 * N_HALF   # 시퀀스 길이 8

D_MODEL, N_HEAD, N_LAYER = 64, 2, 2
STEPS, BATCH, LR = 3000, 128, 3e-3
IGNORE = -1          # 손실에서 제외할 타깃 값


# ── 데이터 (data) ─────────────────────────────────────────────────────────
def make_mirror(bs):
    """[x1 x2 x3 x4 | x4 x3 x2 x1] 형태의 거울 시퀀스 배치."""
    left = torch.randint(0, N_WORDS, (bs, N_HALF))
    return torch.cat([left, left.flip(1)], dim=1)      # (B, 8)


def corrupt(seq, pos, scheme="mask100"):
    """
    BERT 의 입력 손상(input corruption).
    선택된 위치 pos 만 손상시키고, 타깃은 그 자리의 '원래 토큰'.

    scheme="mask100" : 항상 [MASK] 로 치환                       (나쁜 대안)
    scheme="801010"  : 80% [MASK] / 10% 무작위 / 10% 원본 유지  (BERT 실제 규칙)
    scheme="rand100" : 항상 '틀린 토큰'으로 치환 — 평가 전용(실험 3)
    """
    B = seq.size(0)
    inp = seq.clone()
    tgt = torch.full_like(seq, IGNORE)
    rows = torch.arange(B)

    tgt[rows, pos] = seq[rows, pos]                    # 정답 = 원래 토큰

    if scheme == "mask100":
        inp[rows, pos] = MASK_ID
    elif scheme == "801010":
        r = torch.rand(B)
        to_mask = r < 0.8
        to_rand = (r >= 0.8) & (r < 0.9)               # 나머지 10% 는 원본 유지
        inp[rows[to_mask], pos[to_mask]] = MASK_ID
        inp[rows[to_rand], pos[to_rand]] = torch.randint(
            0, N_WORDS, (int(to_rand.sum()),))
    elif scheme == "rand100":
        # 원본과 '다른' 토큰으로 반드시 바꾼다: (원본 + 1..N-1) mod N
        shift = torch.randint(1, N_WORDS, (B,))
        inp[rows, pos] = (seq[rows, pos] + shift) % N_WORDS
    else:
        raise ValueError(scheme)
    return inp, tgt


# ── 모델 (model) — Day-037 의 블록에 causal 플래그만 추가 ─────────────────
class SelfAttention(nn.Module):
    """causal=False 면 양방향(BERT), True 면 단방향(GPT)."""

    def __init__(self, d_model, n_head, causal):
        super().__init__()
        self.n_head, self.d_head, self.causal = n_head, d_model // n_head, causal
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.register_buffer(
            "tril", torch.tril(torch.ones(BLOCK, BLOCK)).view(1, 1, BLOCK, BLOCK))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if self.causal:                                # ★ 여기만 다르다
            att = att.masked_fill(self.tril[:, :, :T, :T] == 0, float("-inf"))
        att = att.softmax(dim=-1)
        self.last_attn = att.detach()

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, d_model, n_head, causal):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model, n_head, causal)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))                  # Pre-LN 잔차 ①
        x = x + self.mlp(self.ln2(x))                   # Pre-LN 잔차 ②
        return x


class MiniEncoder(nn.Module):
    """causal=False → BERT 류 인코더 / causal=True → GPT 류 디코더."""

    def __init__(self, causal=False, d_model=D_MODEL, n_head=N_HEAD,
                 n_layer=N_LAYER):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, d_model)
        self.pos = nn.Embedding(BLOCK, d_model)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_head, causal) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, VOCAB, bias=False)   # MLM head

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))                  # (B, T, VOCAB)


# ── 학습 / 평가 (train / eval) ────────────────────────────────────────────
def train(causal, scheme, steps=STEPS, log=False):
    model = MiniEncoder(causal=causal)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    for step in range(1, steps + 1):
        seq = make_mirror(BATCH)
        pos = torch.randint(0, BLOCK, (BATCH,))
        inp, tgt = corrupt(seq, pos, scheme)
        loss = F.cross_entropy(
            model(inp).reshape(-1, VOCAB), tgt.reshape(-1), ignore_index=IGNORE)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if log and (step == 1 or step % 500 == 0):
            print(f"    step {step:5d}:  loss={loss.item():.4e}")
    return model, loss.item()


@torch.no_grad()
def accuracy(model, scheme, n=2000):
    """무작위 위치 하나를 가리고 복원 정확도를 잰다."""
    seq = make_mirror(n)
    pos = torch.randint(0, BLOCK, (n,))
    inp, _ = corrupt(seq, pos, scheme)
    pred = model(inp).argmax(-1)
    rows = torch.arange(n)
    return (pred[rows, pos] == seq[rows, pos]).float().mean().item()


@torch.no_grad()
def accuracy_by_position(model, scheme, n=1000):
    """위치 0..7 을 각각 가려 보고 위치별 정확도를 낸다."""
    out = []
    for p in range(BLOCK):
        seq = make_mirror(n)
        pos = torch.full((n,), p)
        inp, _ = corrupt(seq, pos, scheme)
        pred = model(inp).argmax(-1)[:, p]
        out.append((pred == seq[:, p]).float().mean().item())
    return out


@torch.no_grad()
def accuracy_clean(model, n=2000):
    """
    [MASK] 없는 '깨끗한' 입력을 주고 각 자리의 토큰을 그대로 복원하게 한다.
    파인튜닝 상황과 같은 조건 — 실제 문장에는 [MASK] 가 없다.
    """
    seq = make_mirror(n)
    pred = model(seq).argmax(-1)
    return (pred == seq).float().mean().item()


# ── 메인 (main) ───────────────────────────────────────────────────────────
def main():
    print("=" * 76)
    print("[과제] 거울 시퀀스 [x1 x2 x3 x4 | x4 x3 x2 x1] 의 한 자리를 가리고 복원")
    seq0 = make_mirror(1)
    inp0, _ = corrupt(seq0, torch.tensor([2]), "mask100")
    print(f"  원본 seq = {seq0[0].tolist()}")
    print(f"  입력     = {inp0[0].tolist()}   ([MASK]={MASK_ID}, "
          f"정답={seq0[0, 2].item()} — 거울 짝은 위치 {BLOCK - 1 - 2})")
    n_par = sum(p.numel() for p in MiniEncoder().parameters())
    print(f"[모델] MiniEncoder d_model={D_MODEL}, heads={N_HEAD}, "
          f"layers={N_LAYER}, 파라미터 {n_par:,}개")

    # ── 실험 1: 양방향 vs 단방향 ──────────────────────────────────────
    print("\n" + "=" * 76)
    print("[실험 1] 양방향(BERT/MLM) vs 단방향(GPT/causal) — 같은 과제, 같은 크기")
    print("  -- 양방향 인코더 (causal=False) 학습 --")
    bert, loss_b = train(causal=False, scheme="mask100", log=True)
    print("  -- 단방향 디코더 (causal=True) 학습 --")
    gpt, loss_g = train(causal=True, scheme="mask100", log=True)

    print(f"\n  양방향(BERT) : 학습손실={loss_b:.4e}   "
          f"복원 정확도 = {accuracy(bert, 'mask100'):.3f}")
    print(f"  단방향(GPT)  : 학습손실={loss_g:.4e}   "
          f"복원 정확도 = {accuracy(gpt, 'mask100'):.3f}")
    print(f"  (무작위 추측 기준선 = {1 / N_WORDS:.3f}, "
          f"'절반만 푸는' 상한 = {0.5 + 0.5 / N_WORDS:.3f})")

    # ── 실험 2: 위치별 정확도 ────────────────────────────────────────
    print("\n" + "=" * 76)
    print("[실험 2] 위치별 복원 정확도 — 단방향은 '어디서' 실패하는가")
    pb = accuracy_by_position(bert, "mask100")
    pg = accuracy_by_position(gpt, "mask100")
    print("  가린 위치     " + "".join(f"{p:>8d}" for p in range(BLOCK)))
    print("  거울 짝 위치  " + "".join(f"{BLOCK - 1 - p:>8d}" for p in range(BLOCK)))
    print("  짝의 방향     " + "".join(
        f"{'→오른쪽' if BLOCK - 1 - p > p else '←왼쪽':>8s}" for p in range(BLOCK)))
    print("  양방향(BERT)  " + "".join(f"{a:>8.3f}" for a in pb))
    print("  단방향(GPT)   " + "".join(f"{a:>8.3f}" for a in pg))

    # ── 실험 3: 80/10/10 규칙 ────────────────────────────────────────
    print("\n" + "=" * 76)
    print("[실험 3] 왜 80/10/10 인가 — 두 인코더를 세 조건에서 비교")
    bert_801010, _ = train(causal=False, scheme="801010")
    rows = [
        ("[MASK] 로 가림     (사전학습과 동일)", "mask100"),
        ("깨끗한 입력        (파인튜닝과 동일)", None),
        ("틀린 토큰으로 치환 (10% 무작위 규칙)", "rand100"),
    ]
    print(f"  {'조건':<38s}{'A: MASK 100%':>14s}{'B: 80/10/10':>14s}")
    for label, sch in rows:
        a = accuracy_clean(bert) if sch is None else accuracy(bert, sch)
        b = (accuracy_clean(bert_801010) if sch is None
             else accuracy(bert_801010, sch))
        print(f"  {label:<38s}{a:>14.3f}{b:>14.3f}")
    print("  -> 결과: 세 조건 모두 사실상 차이가 없다. 80/10/10 의 이점이 '재현되지 않음'.")
    print("     이 장난감 과제로는 pretrain-finetune 불일치를 잡아낼 수 없기 때문이다:")
    print("       (a) 거울 짝이 100% 신뢰할 수 있는 증거라, 어느 모델이든 '짝 읽기'")
    print("           회로 하나만 배우면 끝난다. 자리에 놓인 토큰을 볼 이유가 없다.")
    print("       (b) 불일치의 실제 피해는 '마스킹되지 않은 자리의 표현'을 하류 과제에")
    print("           쓸 때 생기는데, 이 실험은 손상된 자리만 감독·평가한다.")
    print("     원논문의 ablation 은 GLUE/NER 같은 실제 하류 과제로 측정했다.")
    print("     교훈: 장난감 재현 실험은 '되는 것'만 보여준다. 안 되면 설계를 의심하라.")

    # ── 실험 4: attention 이 정말 양방향인가 ─────────────────────────
    print("\n" + "=" * 76)
    print("[실험 4] 학습된 attention — 가린 자리는 '거울 짝'을 보는가")
    seq = make_mirror(1)
    inp, _ = corrupt(seq, torch.tensor([1]), "mask100")
    _ = bert(inp)
    att = bert.blocks[-1].attn.last_attn[0]              # (h, T, T)
    print(f"  seq={seq[0].tolist()}  입력={inp[0].tolist()}  "
          f"가린 위치=1, 거울 짝=6")
    print("        " + "".join(f"{'k'+str(j):>7s}" for j in range(BLOCK)))
    for h in range(N_HEAD):
        row = att[h, 1]
        print(f"  head{h} q1 " + "".join(f"{v:>7.2f}" for v in row.tolist())
              + f"   <- 최대 k{row.argmax().item()}")
    print("  (상삼각이 0 이 아니다 = 미래를 본다 = 양방향. Day-037 과 정반대)")
    print("=" * 76)


if __name__ == "__main__":
    main()
