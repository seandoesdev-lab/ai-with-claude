"""
Day-042 — 디코딩 전략: 확률 분포에서 문장을 꺼내는 법
(Decoding strategies: greedy / beam / temperature / top-k / top-p, measured)

[[Day-040]] 에서 우리는 모델이 매 스텝 **다음 토큰의 확률 분포** 를 내놓는 것을 봤고,
[[Day-041]] 에서 텍스트를 토큰으로 바꾸는 법을 배웠다. 남은 질문 하나가 오늘의 주제다.

    "그 분포에서 어떻게 문장을 꺼내는가?"

디코딩은 **모델 밖의 문제** 다 — 가중치를 하나도 바꾸지 않고 규칙만 갈아 끼운다.
그런데 같은 모델·같은 프롬프트가 규칙에 따라 전혀 다른 글을 쓴다. 그것을 측정한다.

  실험 1. 분포를 직접 들여다보기 — 엔트로피와 온도(temperature)가 하는 일
  실험 2. greedy 와 beam search — '확률을 높이면 글이 좋아진다'는 착각 (beam curse)
  실험 3. 표집 계열 — 순수 표집 / 온도 / top-k / top-p 를 같은 자로 재기
  실험 4. 사람의 문장은 '가장 확률 높은 문장'이 아니다 (nucleus sampling 의 핵심 논거)
  실험 5. 반복을 규칙으로 끊기 — no-repeat-ngram 과 repetition penalty

모델: char-level 디코더-only Transformer ([[Day-037]] 의 구조를 축소).
      토크나이저 대신 '글자 하나 = 토큰 하나' 로 두어, 디코딩만 놓고 보게 한다.
코퍼스: **이 트랙이 지금까지 쓴 Day 노트들** (Day-001 … 직전 편). 약 46만 자.
      작은 코퍼스로는 모델이 통째로 암기해 분포가 뾰족해지고, 그러면 온도·top-p 가
      전부 greedy 로 붕괴해 실험 자체가 성립하지 않는다. (직접 확인해 보라 — §4.6)

실행:  uv run python decoding_strategies.py
       (uv 프로젝트에 torch 가 없으면) uv run --with torch python decoding_strategies.py
한글이 깨지면 먼저:  $env:PYTHONIOENCODING="utf-8"
"""

import glob
import math
import os
import re
import time
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# ── 설정 (config) ──────────────────────────────────────────────────────────
BLOCK = 64                       # 문맥 창 (context window), 글자 단위
D_MODEL, N_HEAD, N_LAYER = 128, 4, 3
DROPOUT = 0.1
STEPS, BATCH, LR = 2000, 32, 1e-3
EVAL_EVERY = 250                 # 이 간격으로 val 을 재고 '가장 좋았던 시점'을 보관
MIN_COUNT = 5                    # 이보다 드문 글자는 □ 로 합친다 (어휘 절약)
N_NEW = 200                      # 생성 길이 (글자)
PROMPT = "밀집 검색은 "

# 실험 4 에서 쓸 '사람이 쓴 보류 문장'. 코퍼스에 없어야 한다 (아래에서 검증).
HELD_OUT = "밀집 검색은 어휘 불일치의 벽을 넘기 위해 의미를 좌표로 옮긴다."

# 코퍼스를 못 찾았을 때의 대체용 (트랙 밖에서 이 파일만 돌릴 때)
FALLBACK = ("정보 검색은 질문에 답이 될 문서를 찾아 순서대로 늘어놓는 일이다.\n"
            "언어모델은 다음 토큰의 확률 분포를 내놓는 함수일 뿐이다.\n"
            "디코딩 전략은 그 분포에서 실제 문장을 꺼내는 규칙을 정한다.\n") * 400


# ── 코퍼스 (corpus) ────────────────────────────────────────────────────────
def load_corpus():
    """이 트랙의 Day 노트들을 읽어 하나의 텍스트로 잇는다 (자기 폴더는 제외)."""
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    paths = [p for p in sorted(glob.glob(os.path.join(root, "Day-0*", "Day-0*.md")))
             if os.path.dirname(p) != here]
    if not paths:
        print("    (Day 노트를 못 찾아 대체 코퍼스를 쓴다)")
        return FALLBACK, 0
    out = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            s = f.read()
        s = re.sub(r"^---\n.*?\n---\n", "", s, flags=re.S)   # YAML frontmatter 제거
        s = re.sub(r"```.*?```", "", s, flags=re.S)          # 코드 블록 제거
        s = re.sub(r"[ \t]+", " ", s)                        # 공백 정리
        s = re.sub(r"\n{2,}", "\n", s)
        out.append(s)
    return "".join(out), len(paths)


CORPUS, N_NOTE = load_corpus()
assert HELD_OUT not in CORPUS, "보류 문장이 코퍼스에 있으면 실험 4 가 무의미하다"

# ── 토크나이저: 글자 하나 = 토큰 하나 ────────────────────────────────────
cnt = Counter(CORPUS)
UNK = "□"
chars = sorted({c for c, n in cnt.items() if n >= MIN_COUNT} | set(HELD_OUT) | {UNK})
VOCAB = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi.get(c, stoi[UNK]) for c in s]
decode = lambda ids: "".join(itos[int(i)] for i in ids)

# 학습/검증 분할: 2000자 블록 중 10개마다 1개를 검증으로 뺀다 (한쪽 언어에 쏠리지 않게)
ids_all = torch.tensor(encode(CORPUS), dtype=torch.long)
CHUNK = 2000
blocks = [ids_all[i:i + CHUNK] for i in range(0, len(ids_all) - CHUNK, CHUNK)]
train_data = torch.cat([b for i, b in enumerate(blocks) if i % 10 != 7])
val_data = torch.cat([b for i, b in enumerate(blocks) if i % 10 == 7])


def get_batch(split, bs=BATCH):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - BLOCK - 1, (bs,))
    x = torch.stack([d[i:i + BLOCK] for i in ix])
    y = torch.stack([d[i + 1:i + BLOCK + 1] for i in ix])
    return x, y


# ── 모델 (Day-037 의 미니 디코더를 축소) ─────────────────────────────────
class CausalAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL)
        self.proj = nn.Linear(D_MODEL, D_MODEL)
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        shape = lambda t: t.view(B, T, N_HEAD, C // N_HEAD).transpose(1, 2)
        # is_causal=True 가 인과 마스크 역할 (Day-034) — 미래를 보지 못한다
        y = F.scaled_dot_product_attention(shape(q), shape(k), shape(v),
                                           dropout_p=DROPOUT if self.training else 0.0,
                                           is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(y))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D_MODEL), nn.LayerNorm(D_MODEL)
        self.attn = CausalAttention()
        self.ff = nn.Sequential(
            nn.Linear(D_MODEL, 4 * D_MODEL), nn.GELU(),
            nn.Linear(4 * D_MODEL, D_MODEL), nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))          # Pre-LN (Day-034)
        return x + self.ff(self.ln2(x))


class CharLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, D_MODEL)
        self.pos = nn.Embedding(BLOCK, D_MODEL)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB, bias=False)
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, idx):
        x = self.drop(self.tok(idx) + self.pos(torch.arange(idx.size(1))))
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln_f(x))


@torch.no_grad()
def eval_loss(model, split, iters=20):
    was = model.training
    model.eval()
    tot = 0.0
    for _ in range(iters):
        x, y = get_batch(split)
        tot += F.cross_entropy(model(x).view(-1, VOCAB), y.view(-1)).item()
    model.train(was)
    return tot / iters


def train():
    model = CharLM()
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=STEPS,
                                                pct_start=0.1)
    best, best_state, best_step = float("inf"), None, 0
    t0 = time.time()
    for step in range(1, STEPS + 1):
        x, y = get_batch("train")
        loss = F.cross_entropy(model(x).view(-1, VOCAB), y.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % EVAL_EVERY == 0 or step == STEPS:
            tr, va = eval_loss(model, "train"), eval_loss(model, "val")
            flag = ""
            if va < best:                      # 조기종료용 보관 — 암기하기 전 시점을 쓴다
                best, best_step = va, step
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                flag = "  <= best"
            print(f"    step {step:>5}  train {tr:.3f}  val {va:.3f}"
                  f"  ppl {math.exp(va):>6.1f}{flag}")
    model.load_state_dict(best_state)
    model.eval()
    print(f"    파라미터 {n_par/1e3:.0f}K · 어휘(글자) {VOCAB} · 학습 {time.time()-t0:.0f}s")
    print(f"    → val 이 가장 낮았던 step {best_step} 의 가중치를 쓴다 (val {best:.3f})")
    return model


# ── 디코딩 공통 유틸 ───────────────────────────────────────────────────────
@torch.no_grad()
def next_logits(model, ids):
    """ids: (1, T) → 마지막 위치의 로짓 (V,)"""
    return model(ids[:, -BLOCK:])[0, -1, :]


def entropy_bits(p):
    p = p[p > 0]
    return float(-(p * p.log2()).sum())


@torch.no_grad()
def seq_logp(model, ids, start_at=1):
    """
    lps = [log p(ids[t] | 직전 최대 BLOCK 글자)]  for t = start_at .. T-1.

    채점 문맥을 **생성할 때 모델이 실제로 본 문맥과 똑같이** 맞춘다(직전 BLOCK 글자).
    한 글자씩 다시 forward 하면 200번이 되므로, 창(window)을 batch 로 쌓아 2번만 돈다.
    """
    T = ids.size(1)
    start_at = max(1, start_at)
    lps = []
    short = list(range(start_at, min(BLOCK, T)))           # 앞부분: 문맥이 BLOCK 보다 짧다
    if short:
        lp = F.log_softmax(model(ids[:, :short[-1] + 1]), -1)[0]
        lps += [float(lp[t - 1, ids[0, t]]) for t in short]
    long = list(range(max(BLOCK, start_at), T))            # 나머지: 문맥이 정확히 BLOCK
    if long:
        win = torch.stack([ids[0, t - BLOCK:t] for t in long])       # (N, BLOCK)
        lp = F.log_softmax(model(win), -1)[:, -1, :]                 # (N, V)
        lps += [float(lp[j, ids[0, t]]) for j, t in enumerate(long)]
    return lps


def avg_logp(model, prompt, text):
    """모델이 보는 text 의 평균 log 확률 (자연로그, 글자당). 높을수록 '확률이 높은 글'."""
    ids = torch.tensor([encode(prompt + text)])
    lps = seq_logp(model, ids, start_at=len(encode(prompt)))
    return sum(lps) / len(lps)


# ── 품질 지표 ──────────────────────────────────────────────────────────────
def rep_n(text, n=4):
    """반복률: 전체 n-gram 중 '중복해서 등장한' n-gram 의 비율. 낮을수록 좋다."""
    grams = [text[i:i + n] for i in range(len(text) - n + 1)]
    if not grams:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)


def distinct_n(text, n=2):
    """다양성: 서로 다른 n-gram 의 비율. 높을수록 다양하다."""
    grams = [text[i:i + n] for i in range(len(text) - n + 1)]
    return len(set(grams)) / max(1, len(grams))


def loop_len(text, n=8):
    """가장 길게 이어진 '같은 n-gram 반복' 구간의 길이 (완전한 무한 루프 탐지)."""
    best = 0
    for period in range(1, 41):
        run = 0
        for i in range(len(text) - period):
            run = run + 1 if text[i] == text[i + period] else 0
            best = max(best, run)
    return best


def report(name, text, model, extra=""):
    lp = avg_logp(model, PROMPT, text)
    print(f"  {name:<26} logp/자 {lp:+.3f}   rep-4 {rep_n(text,4):.3f}"
          f"   distinct-2 {distinct_n(text,2):.3f}   최장반복 {loop_len(text):>3}{extra}")
    return lp


def show(text, width=86):
    print(f"      | {text.replace(chr(10), '⏎')[:width]}")


# ── 디코더들 ───────────────────────────────────────────────────────────────
@torch.no_grad()
def greedy(model, prompt, n_new=N_NEW, no_repeat=0, rep_penalty=1.0):
    ids = torch.tensor([encode(prompt)])
    for _ in range(n_new):
        logits = next_logits(model, ids).clone()
        if rep_penalty != 1.0:                            # 이미 쓴 글자에 벌점
            for tid in set(ids[0].tolist()):
                logits[tid] = (logits[tid] / rep_penalty if logits[tid] > 0
                               else logits[tid] * rep_penalty)
        if no_repeat > 0 and ids.size(1) >= no_repeat - 1:
            prefix = tuple(ids[0, -(no_repeat - 1):].tolist())
            seq = ids[0].tolist()
            for i in range(len(seq) - no_repeat + 1):      # 같은 n-gram 재사용 금지
                if tuple(seq[i:i + no_repeat - 1]) == prefix:
                    logits[seq[i + no_repeat - 1]] = -float("inf")
        ids = torch.cat([ids, logits.argmax().view(1, 1)], 1)
    return decode(ids[0, len(encode(prompt)):])


@torch.no_grad()
def beam_search(model, prompt, beams, n_new=N_NEW):
    seqs = torch.tensor([encode(prompt)])
    scores = torch.zeros(1)
    for _ in range(n_new):
        logp = F.log_softmax(model(seqs[:, -BLOCK:])[:, -1, :], -1)   # (B, V)
        cand = (scores.unsqueeze(1) + logp).view(-1)
        top, idx = cand.topk(min(beams, cand.numel()))
        seqs = torch.cat([seqs[idx // VOCAB], (idx % VOCAB).unsqueeze(1)], 1)
        scores = top
    best = int(scores.argmax())                  # 길이가 모두 같아 길이정규화는 무관
    # 두 번째 값 = beam 이 스스로 최대화한 목적함수(누적 logp)를 글자당으로 환산한 것
    return decode(seqs[best, len(encode(prompt)):]), float(scores[best]) / n_new


@torch.no_grad()
def sample(model, prompt, n_new=N_NEW, temp=1.0, top_k=0, top_p=0.0, seed=0):
    """온도 · top-k · top-p 를 한 함수에. 후보 집합 크기와 잘라낸 꼬리 질량도 기록한다."""
    torch.manual_seed(seed)
    ids = torch.tensor([encode(prompt)])
    sizes, cut_mass = [], []
    for _ in range(n_new):
        logits = next_logits(model, ids) / max(temp, 1e-6)
        probs = F.softmax(logits, -1)
        keep = torch.argsort(probs, descending=True)
        if top_k > 0:
            keep = keep[:top_k]
        if top_p > 0.0:
            cum = torch.cumsum(probs[keep], 0)
            keep = keep[:int((cum < top_p).sum()) + 1]     # 경계를 넘는 토큰까지 포함
        sizes.append(len(keep))
        cut_mass.append(max(0.0, 1.0 - float(probs[keep].sum())))
        p = probs[keep] / probs[keep].sum()
        ids = torch.cat([ids, keep[int(torch.multinomial(p, 1))].view(1, 1)], 1)
    stats = (sum(sizes) / len(sizes), min(sizes), max(sizes),
             sum(cut_mass) / len(cut_mass))
    return decode(ids[0, len(encode(prompt)):]), stats


# ══════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 78)
    print("Day-042 — 디코딩 전략: 같은 모델, 다른 규칙")
    print("=" * 78)

    print("\n[0] char-level 미니 LM 학습 (디코딩만 놓고 보기 위한 도구)")
    print(f"    코퍼스: Day 노트 {N_NOTE}편 · {len(CORPUS):,}자"
          f" · 학습 {len(train_data):,} / 검증 {len(val_data):,} 글자")
    model = train()

    # ── 실험 1 ────────────────────────────────────────────────────────────
    print("\n" + "─" * 78)
    print("[1] 분포를 직접 들여다보기 — 엔트로피와 온도")
    print("─" * 78)
    logits = next_logits(model, torch.tensor([encode(PROMPT)]))
    probs = F.softmax(logits, -1)
    order = torch.argsort(probs, descending=True)
    print(f"  프롬프트 {PROMPT!r} 다음 글자의 상위 10개")
    print("    " + "  ".join(f"{itos[int(i)]!r}:{float(probs[i]):.3f}" for i in order[:10]))
    cum = torch.cumsum(probs[order], 0)
    print(f"    어휘 {VOCAB}개 · 엔트로피 {entropy_bits(probs):.2f} bits"
          f" (균등분포라면 {math.log2(VOCAB):.2f})"
          f" · 누적 90%까지 {int((cum<0.9).sum())+1}개"
          f" · 99%까지 {int((cum<0.99).sum())+1}개")

    print("\n  같은 로짓, 온도만 바꿨을 때 (T→0 은 greedy, T→∞ 는 균등분포)")
    print("     온도 T   엔트로피(bits)   최상위 확률   90% 후보수   99% 후보수")
    for T in (0.2, 0.5, 0.7, 1.0, 1.3, 1.7, 2.5):
        p = F.softmax(logits / T, -1)
        s = p[torch.argsort(p, descending=True)]
        c = torch.cumsum(s, 0)
        print(f"    {T:>6.1f}   {entropy_bits(p):>12.2f}   {float(s[0]):>10.3f}"
              f"   {int((c<0.9).sum())+1:>10}   {int((c<0.99).sum())+1:>10}")

    # ── 실험 2 ────────────────────────────────────────────────────────────
    print("\n" + "─" * 78)
    print("[2] greedy 와 beam search — 확률을 높이면 글이 좋아지는가")
    print("─" * 78)
    g = greedy(model, PROMPT)
    report("greedy (= beam 1)", g, model)
    show(g)
    for beams in (2, 4, 8, 16):
        b, beam_score = beam_search(model, PROMPT, beams)
        report(f"beam {beams}", b, model, extra=f"  빔목적함수 {beam_score:+.3f}")
        show(b)
    print("\n  '빔목적함수' = beam 이 스스로 최대화한 누적 logp / 글자. logp/자 와 같은 값이면")
    print("  채점 문맥이 생성 문맥과 정확히 일치한다는 뜻이다 (검산).")
    print("  beam 은 greedy 보다 확률을 크게 올린다. 다만 빔 폭에 대해 '단조'는 아니다 —")
    print("  문맥을 직전 BLOCK 글자로 자른 탓에 목적함수가 스텝마다 일관되지 않고,")
    print("  beam search 자체도 스텝별 가지치기라 폭이 넓다고 항상 더 좋은 해를 주지 않는다.")
    print("  정작 중요한 건 이것이다: 확률이 올라가도 rep-4 · 최장반복은 나아지지 않는다.")
    print("  greedy·beam 전부 §3 의 표집 계열보다 훨씬 심하게 반복한다. 확률 ≠ 글의 품질.")

    # ── 실험 3 ────────────────────────────────────────────────────────────
    print("\n" + "─" * 78)
    print("[3] 표집 계열 — 순수 표집 / 온도 / top-k / top-p")
    print("─" * 78)
    configs = [
        ("순수 표집 (T=1.0)", dict(temp=1.0)),
        ("온도 T=0.7", dict(temp=0.7)),
        ("온도 T=1.3", dict(temp=1.3)),
        ("top-k=5", dict(top_k=5)),
        ("top-k=40", dict(top_k=40)),
        ("top-p=0.9", dict(top_p=0.9)),
        ("top-p=0.95", dict(top_p=0.95)),
        ("top-p=0.9 + T=0.8", dict(top_p=0.9, temp=0.8)),
    ]
    print("  각 설정마다 씨앗(seed) 4개의 평균. '후보수' = 매 스텝 살아남은 후보의 개수.")
    print(f"  {'설정':<19}{'logp/자':>9}{'rep-4':>8}{'dist-2':>8}{'최장반복':>9}"
          f"{'후보수':>8}{'(최소~최대)':>13}{'잘린꼬리':>10}")
    samples = {}
    for name, kw in configs:
        acc = {k: [] for k in "lp rep dis loop avg mn mx cut".split()}
        for seed in range(4):
            txt, (avg_sz, mn, mx, cut) = sample(model, PROMPT, seed=seed, **kw)
            acc["lp"].append(avg_logp(model, PROMPT, txt))
            acc["rep"].append(rep_n(txt, 4)); acc["dis"].append(distinct_n(txt, 2))
            acc["loop"].append(loop_len(txt)); acc["avg"].append(avg_sz)
            acc["mn"].append(mn); acc["mx"].append(mx); acc["cut"].append(cut)
            if seed == 0:
                samples[name] = txt
        m = lambda k: sum(acc[k]) / len(acc[k])
        span = "(%d~%d)" % (min(acc["mn"]), max(acc["mx"]))
        print(f"  {name:<19}{m('lp'):>+9.3f}{m('rep'):>8.3f}{m('dis'):>8.3f}"
              f"{m('loop'):>9.1f}{m('avg'):>8.1f}{span:>13}{m('cut'):>10.4f}")
    print("\n  씨앗 0 의 생성 예시")
    for name in ("순수 표집 (T=1.0)", "온도 T=0.7", "top-k=5", "top-p=0.9",
                 "top-p=0.9 + T=0.8"):
        print(f"    {name}")
        show(samples[name])

    # ── 실험 4 ────────────────────────────────────────────────────────────
    print("\n" + "─" * 78)
    print("[4] 사람의 문장은 '가장 확률 높은 문장'이 아니다")
    print("─" * 78)
    assert HELD_OUT[:len(PROMPT)] == PROMPT, "같은 프롬프트로 시작해야 비교가 공정하다"
    ids = torch.tensor([encode(HELD_OUT)])
    ranks, lps, top1 = [], [], 0
    for t in range(len(encode(PROMPT)), ids.size(1)):
        p = F.softmax(next_logits(model, ids[:, :t]), -1)
        gold = int(ids[0, t])
        r = int((torch.argsort(p, descending=True) == gold).nonzero()[0, 0]) + 1
        ranks.append(r); lps.append(math.log(float(p[gold]) + 1e-12))
        top1 += (r == 1)
    n = len(ranks)
    print(f"  보류 문장(코퍼스에 없음을 assert 로 확인): {HELD_OUT!r}")
    print(f"  길이 {n}자 · 평균 logp/자 {sum(lps)/n:+.3f}"
          f" · greedy 가 그대로 맞힌 글자 {top1}/{n} ({top1/n:.0%})")
    print(f"  순위 중앙값 {sorted(ranks)[n//2]} · 최악 순위 {max(ranks)}"
          f" · 순위 6위 밖(top-5 로는 못 뽑는 글자) {sum(r>5 for r in ranks)}자"
          f" · 20위 밖 {sum(r>20 for r in ranks)}자")
    print(f"  같은 프롬프트의 greedy 생성 logp/자 {avg_logp(model, PROMPT, g):+.3f}")
    print("  → greedy 쪽이 '확률이 더 높다'. 그런데 사람 문장이 더 나은 글이다.")
    print("     확률 최대화 = 좋은 글, 이라는 전제가 여기서 깨진다.")

    # ── 실험 5 ────────────────────────────────────────────────────────────
    print("\n" + "─" * 78)
    print("[5] 반복을 규칙으로 끊기 — no-repeat-ngram · repetition penalty")
    print("─" * 78)
    report("greedy (기준)", g, model)
    show(g)
    for nr in (3, 4, 6):
        report(f"greedy + no-repeat-{nr}gram", greedy(model, PROMPT, no_repeat=nr), model)
    for rp in (1.05, 1.2, 1.5):
        report(f"greedy + rep-penalty {rp}", greedy(model, PROMPT, rep_penalty=rp), model)
    print("\n  예시 — greedy + no-repeat-4gram")
    show(greedy(model, PROMPT, no_repeat=4))
    print("  예시 — greedy + rep-penalty 1.2")
    show(greedy(model, PROMPT, rep_penalty=1.2))
    print("\n  반복이 줄어들 때 logp/자 는 얼마를 대가로 내는가 — 위 표에서 직접 비교하라.")
    print("  (greedy 가 완전한 루프에 빠진 경우라면 대가가 없거나 오히려 이득일 수도 있다.)")

    print("\n" + "=" * 78)
    print("결론: 가중치는 하나도 바꾸지 않았다. 바꾼 것은 '분포에서 꺼내는 규칙'뿐이다.")
    print("=" * 78)


if __name__ == "__main__":
    main()
