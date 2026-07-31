"""
Day-040 — Phase 3 총정리: 같은 과제, 다섯 구조 (One task, five architectures)
(Phase 3 recap: why attention won, measured — and a first look at decoding)

Phase 3 에서 우리가 배운 구조들을 **완전히 같은 과제·같은 예산** 으로 나란히 세운다.
    Day-026 MLP → Day-030 RNN → Day-031 LSTM → Day-033/034/037 Transformer
과제는 '장거리 의존성'이 본질인 것으로 골라, 각 구조의 한계가 **숫자로** 드러나게 한다.

  실험 1. 다섯 구조 대결 — 정확도·파라미터·학습시간을 한 표로
  실험 2. 거리에 따른 정확도 — '순환의 병목'(Day-032)을 눈으로 보기
  실험 3. attention 이 실제로 무엇을 보는가 — 유도 회로 재확인 (Day-039 의 연장)
  실험 4. 디코딩 전략 맛보기 — greedy / 온도 / top-k 가 정확도를 바꾼다 (Phase 4 예고)

과제(연상 회상 / associative recall):
    키 8개(기호 0..7, 순열)와 값 8개(기호 8..15, 중복 허용)를 짝지어 늘어놓고,
    마지막에 키 하나를 다시 제시한다.
        seq = [k1 v1 k2 v2 ... k8 v8 q]      (길이 17)
    모델은 마지막 위치에서 **q 와 짝지어졌던 값** 을 맞혀야 한다.
    답은 문맥에 유일하게 존재하므로 **이론적 상한은 1.000**, 무작위는 1/8 = 0.125.
    질의한 짝의 위치 i 를 바꾸면 '얼마나 멀리 있는 정보를 가져와야 하는가'가 조절된다.

실행:  uv run python phase3_recap.py
       (uv 프로젝트에 torch 가 없으면) uv run --with torch python phase3_recap.py
한글이 깨지면 먼저:  $env:PYTHONIOENCODING="utf-8"
"""

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# ── 설정 (config) ──────────────────────────────────────────────────────────
N_KEY = 8                  # 키 기호 0..7   (매 시퀀스마다 순열 — 각 키는 한 번씩)
N_VAL = 8                  # 값 기호 8..15  (중복 허용)
VOCAB = N_KEY + N_VAL      # 16
N_PAIR = 8
BLOCK = 2 * N_PAIR + 1     # [k v] * 8 + q = 17

D_MODEL, N_HEAD, N_LAYER = 64, 2, 2
STEPS, BATCH, LR = 4000, 128, 3e-3
LR_TF = 3e-4               # Transformer 는 더 작은 lr 이 안정적 (Day-037 과 동일)


# ── 데이터 (data) ─────────────────────────────────────────────────────────
def make_batch(bs, qpos=None):
    """
    seq = [k1 v1 ... k8 v8 q], 정답 = q 와 짝지어진 값.

    qpos=None : 질의할 짝을 0..7 에서 균등하게 뽑는다 (학습·전체 평가)
    qpos=i    : 항상 i 번째 짝을 질의한다 (실험 2 — 거리별 평가)
    """
    keys = torch.argsort(torch.rand(bs, N_PAIR), dim=1)              # 0..7 의 순열
    vals = torch.randint(N_KEY, VOCAB, (bs, N_PAIR))                 # 8..15
    idx = (torch.full((bs,), qpos) if qpos is not None
           else torch.randint(0, N_PAIR, (bs,)))

    seq = torch.empty(bs, BLOCK, dtype=torch.long)
    seq[:, 0:2 * N_PAIR:2] = keys
    seq[:, 1:2 * N_PAIR:2] = vals
    seq[:, -1] = keys.gather(1, idx[:, None]).squeeze(1)             # 질의 = 그 키
    tgt = vals.gather(1, idx[:, None]).squeeze(1)                    # 정답 = 그 값
    return seq, tgt


# ── 구조 1: 위치를 무시하는 MLP (Bag-of-Words) ─────────────────────────────
class BowMLP(nn.Module):
    """Day-007 의 BoW 정신 — 무엇이 나왔는지만 세고 순서를 버린다."""

    def __init__(self, d=D_MODEL):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(VOCAB, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU(),
            nn.Linear(d, VOCAB))

    def forward(self, seq):
        bag = F.one_hot(seq, VOCAB).float().sum(1)                   # (B, 16)
        return self.net(bag)


# ── 구조 2: 위치를 아는 MLP (Day-026) ─────────────────────────────────────
class FlatMLP(nn.Module):
    """한 줄로 펼친 one-hot 을 통째로 먹는 MLP. 순서는 알지만 '비교'를 못 한다."""

    def __init__(self, d=D_MODEL):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(BLOCK * VOCAB, 4 * d), nn.ReLU(),
            nn.Linear(4 * d, 4 * d), nn.ReLU(), nn.Linear(4 * d, VOCAB))

    def forward(self, seq):
        return self.net(F.one_hot(seq, VOCAB).float().flatten(1))


# ── 구조 3·4: RNN / LSTM (Day-030, Day-031) ───────────────────────────────
class Recurrent(nn.Module):
    """마지막 은닉상태 하나로 답한다 — 문맥 전부를 고정 크기 벡터에 눌러 담아야 한다."""

    def __init__(self, cell="rnn", d=D_MODEL, n_layer=1):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, d)
        Cell = nn.LSTM if cell == "lstm" else nn.RNN
        self.rnn = Cell(d, d, num_layers=n_layer, batch_first=True)
        self.head = nn.Linear(d, VOCAB)

    def forward(self, seq):
        out, _ = self.rnn(self.emb(seq))
        return self.head(out[:, -1])                                 # 마지막 시점만


# ── 구조 5: Transformer (Day-037 의 디코더 그대로) ────────────────────────
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.n_head, self.d_head = n_head, d_model // n_head
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
        att = att.masked_fill(self.tril[:, :, :T, :T] == 0, float("-inf"))
        att = att.softmax(dim=-1)
        self.last_attn = att.detach()

        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MiniTransformer(nn.Module):
    def __init__(self, d_model=D_MODEL, n_head=N_HEAD, n_layer=N_LAYER):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, d_model)
        self.pos = nn.Embedding(BLOCK, d_model)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, VOCAB, bias=False)

    def forward(self, seq):
        B, T = seq.shape
        x = self.tok(seq) + self.pos(torch.arange(T, device=seq.device))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))[:, -1]                        # 마지막 위치만


# ── 학습·평가 (train / eval) ──────────────────────────────────────────────
def train(model, steps=STEPS, lr=LR):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    t0 = time.time()
    for _ in range(steps):
        seq, tgt = make_batch(BATCH)
        loss = F.cross_entropy(model(seq), tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return loss.item(), time.time() - t0


@torch.no_grad()
def accuracy(model, seq, tgt):
    return (model(seq).argmax(-1) == tgt).float().mean().item()


def bow_ceiling(n=200_000):
    """
    **순서를 버린 모델(BoW)이 도달할 수 있는 최대 정확도** — 실험 전에 계산해 둔다.

    가방(bag)에는 키가 전부 한 번씩, 질의된 키만 두 번 들어 있다. 그래서 BoW 는
    '무엇을 물었는지'는 알 수 있지만 '어느 값과 짝이었는지'는 알 수 없다 —
    짝짓기는 순서 정보이고, 그걸 버린 것이 BoW 이기 때문이다.
    남는 최선의 전략은 '가방에서 가장 많이 나온 값' 을 답하는 것이다
    (값 하나가 8칸 중 m 칸을 차지하면 그게 정답일 확률이 m/8 이므로).
        => 상한 = E[최대 중복수] / 8
    """
    g = torch.Generator().manual_seed(1234)          # 전역 RNG 를 건드리지 않는다
    vals = torch.randint(N_KEY, VOCAB, (n, N_PAIR), generator=g)
    counts = F.one_hot(vals - N_KEY, N_VAL).sum(1)   # (n, 8) 값별 등장 횟수
    return (counts.max(1).values.float() / N_PAIR).mean().item()


# ── 메인 (main) ───────────────────────────────────────────────────────────
def main():
    print("=" * 78)
    print("[과제] 연상 회상 — [k1 v1 ... k8 v8 q] 를 주고 q 의 짝을 맞힌다")
    s0, t0 = make_batch(1)
    print(f"  seq  = {s0[0].tolist()}")
    print(f"  질의 = {s0[0, -1].item()} (마지막 토큰)   정답 = {t0[0].item()}")
    print(f"  이론 상한 = 1.000 (답은 문맥에 유일)   무작위 = {1 / N_VAL:.3f}")

    seq, tgt = make_batch(4000)                                      # 공통 평가셋

    # ── 실험 1: 다섯 구조 대결 ──────────────────────────────────────────
    print("\n" + "=" * 78)
    print(f"[실험 1] 같은 과제·같은 예산({STEPS} 스텝) — Phase 3 의 구조들을 나란히")
    print(f"  {'구조':<26s}{'파라미터':>10s}{'손실':>8s}{'정확도':>8s}{'학습시간':>9s}")
    zoo = [
        ("BoW MLP (위치 무시)", BowMLP(), LR, "Day-007"),
        ("MLP (펼친 one-hot)", FlatMLP(), LR, "Day-026"),
        ("RNN (tanh)", Recurrent("rnn"), LR, "Day-030"),
        ("LSTM", Recurrent("lstm"), LR, "Day-031"),
        ("Transformer (2층)", MiniTransformer(), LR_TF, "Day-037"),
    ]
    trained = {}
    for label, model, lr, origin in zoo:
        loss, sec = train(model, lr=lr)
        acc = accuracy(model, seq, tgt)
        npar = sum(p.numel() for p in model.parameters())
        print(f"  {label:<26s}{npar:>10,d}{loss:>8.3f}{acc:>8.3f}{sec:>8.1f}s"
              f"   ({origin})")
        trained[label] = model
    print(f"  {'이론 상한':<24s}{'':>10s}{0.0:>8.3f}{1.0:>8.3f}"
          f"   <- 답은 문맥에 유일하다")
    print(f"  {'BoW 상한 (최빈값 전략)':<18s}{'':>10s}{'':>8s}{bow_ceiling():>8.3f}"
          f"   <- 순서를 버리면 여기까지")
    print(f"  {'무작위 추측':<23s}{'':>10s}{math.log(N_VAL):>8.3f}"
          f"{1 / N_VAL:>8.3f}")

    # ── 실험 2: 거리에 따른 정확도 ─────────────────────────────────────
    print("\n" + "=" * 78)
    print("[실험 2] 질의한 짝이 얼마나 앞에 있었나 — '순환의 병목'을 눈으로")
    print("  (짝 0 = 시퀀스 맨 앞 = 가장 멀다,  짝 7 = 질의 바로 앞 = 가장 가깝다)")
    per_dist = {}
    for label in trained:
        accs = []
        for i in range(N_PAIR):
            sq, tg = make_batch(2000, qpos=i)
            accs.append(accuracy(trained[label], sq, tg))
        per_dist[label] = accs
    print("\n  질의한 짝의 위치        " + "".join(f"{i:>7d}" for i in range(N_PAIR)))
    for label, accs in per_dist.items():
        print(f"  {label:<22s}" + "".join(f"{a:>7.3f}" for a in accs))
    print("\n  [읽는 법] 오른쪽(가까운 짝)은 잘하고 왼쪽(먼 짝)에서 무너지면")
    print("            = 고정 크기 은닉상태에 다 담지 못한다는 뜻. 평평하면 거리와 무관.")
    for label, accs in per_dist.items():
        print(f"    {label:<22s} 가까운쪽(5~7) {sum(accs[5:]) / 3:.3f}  "
              f"먼쪽(0~2) {sum(accs[:3]) / 3:.3f}  "
              f"낙폭 {sum(accs[5:]) / 3 - sum(accs[:3]) / 3:+.3f}")

    # ── 실험 3: attention 이 무엇을 보는가 ─────────────────────────────
    print("\n" + "=" * 78)
    print("[실험 3] Transformer 의 마지막 질의는 어디를 보는가 (유도 회로 재확인)")
    tf = trained["Transformer (2층)"]
    sq, tg = make_batch(1)
    with torch.no_grad():
        pred = tf(sq).argmax(-1).item()
    q = sq[0, -1].item()
    kpos = (sq[0, 0:2 * N_PAIR:2] == q).nonzero().item() * 2         # 키가 놓인 위치
    print(f"  seq = {sq[0].tolist()}")
    print(f"  질의 {q} 는 위치 {kpos} 에 있었고, 그 답은 위치 {kpos + 1} 의 "
          f"{sq[0, kpos + 1].item()}  (정답 {tg[0].item()}, 예측 {pred})")
    print("\n      " + "".join(f"{'k' + str(t):>6s}" for t in range(BLOCK)))
    for li, blk in enumerate(tf.blocks):
        att = blk.attn.last_attn[0]
        for h in range(att.size(0)):
            row = att[h, -1]
            mark = "  <== 답이 놓인 칸을 본다" if row.argmax().item() == kpos + 1 else ""
            print(f"  L{li}h{h} " + "".join(f"{v:>6.2f}" for v in row.tolist())
                  + mark)

    # ── 실험 4: 디코딩 전략 맛보기 ─────────────────────────────────────
    print("\n" + "=" * 78)
    print("[실험 4] 같은 모델, 다른 디코딩 — Phase 4 의 첫 질문")
    with torch.no_grad():
        logits = tf(seq)                                             # (4000, 16)
    p_true = logits.softmax(-1).gather(1, tgt[:, None]).mean().item()
    print(f"  학습된 Transformer 가 정답에 준 평균 확률 = {p_true:.3f}")
    print(f"\n  {'디코딩 규칙':<26s}{'정확도':>8s}   설명")
    print(f"  {'greedy (argmax)':<26s}{accuracy(tf, seq, tgt):>8.3f}"
          f"   가장 확률 높은 토큰 하나")
    for T in (0.5, 1.0, 2.0):
        smp = torch.multinomial((logits / T).softmax(-1), 1).squeeze(1)
        acc = (smp == tgt).float().mean().item()
        note = "분포를 날카롭게" if T < 1 else "원분포" if T == 1 else "분포를 평평하게"
        print(f"  {f'sampling  T={T}':<26s}{acc:>8.3f}   {note}")
    for k in (2, 4, 8):
        top = logits.topk(k, dim=-1)
        pick = torch.multinomial(top.values.softmax(-1), 1)
        smp = top.indices.gather(1, pick).squeeze(1)
        acc = (smp == tgt).float().mean().item()
        print(f"  {f'top-k  k={k} (T=1)':<26s}{acc:>8.3f}"
              f"   상위 {k}개만 남기고 뽑기")
    print("\n  => 가중치는 하나인데 '고르는 규칙'만으로 정확도가 흔들린다.")
    print("     정답이 하나인 과제에서는 greedy 가 최선이지만, 글을 쓰는 과제에서는")
    print("     greedy 가 반복에 빠진다 — 이 절충이 Phase 4 '디코딩' 편의 주제다.")
    print("=" * 78)


if __name__ == "__main__":
    main()
