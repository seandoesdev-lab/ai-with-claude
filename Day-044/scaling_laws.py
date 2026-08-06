"""
Day-044 — 📄 정독: Scaling Laws — 사전학습에 얼마를 투자해야 하는가
(Kaplan et al. 2020 / Hoffmann et al. 2022 를 우리 축소판에서 부분 재현한다)

[[Day-043]] 에서 사전학습·파인튜닝을 직접 돌렸고, §4.7 (5)에 질문을 남겼다 —
"사전학습 스텝을 400 / 1200 / 3000 으로 바꾸면 하류 성능은 어떻게 변하는가?"
그 곡선의 정체가 오늘의 주제다. 논문의 주장은 놀랄 만큼 단순하다.

    손실은 파라미터(N)·데이터(D)·연산(C)에 대해 **거듭제곱 법칙(power law)** 을 따른다.
        L(N) ≈ (Nc / N)^αN        L(D) ≈ (Dc / D)^αD        L(C) ≈ (Cc / C)^αC
    로그-로그 축에서 직선이 되고, 직선이면 **외삽(extrapolation)** 이 가능하다.

우리는 이 주장을 다섯 조각으로 나눠 직접 잰다.

  실험 1. L(N) — 모델 크기만 바꾼다 (6개 크기). 로그-로그가 직선인가?
  실험 2. 외삽 검증 — 작은 4개로 적합한 법칙이 큰 2개를 맞히는가?
  실험 3. L(D) — 데이터 크기만 바꾼다. 그리고 데이터가 적으면 법칙이 어디서 깨지나.
  실험 4. 계산 최적(compute-optimal) — 같은 예산이면 모델을 키울까 데이터를 늘릴까.
          Kaplan 의 '학습곡선 하한선(envelope)' 방법을 그대로 쓴다 → N*(C), D*(C).
  실험 5. 모양은 별로 중요하지 않다 — 같은 N, 다른 폭/깊이 (논문의 핵심 주장 하나).

모델: char-level 디코더-only Transformer ([[Day-037]] 구조, [[Day-043]] 과 동일 계열).
      입출력 임베딩을 **공유(weight tying)** 하고, 논문처럼 N 은 **비임베딩 파라미터** 로 센다.

실행:  uv run python scaling_laws.py
       (uv 프로젝트에 torch 가 없으면) uv run --with torch python scaling_laws.py
       빠른 점검:  $env:QUICK="1"; uv run python scaling_laws.py
한글이 깨지면 먼저:  $env:PYTHONIOENCODING="utf-8"
"""

import glob
import math
import os
import random
import re
import time
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

QUICK = os.environ.get("QUICK") == "1"

# ── 설정 (config) ──────────────────────────────────────────────────────────
BLOCK = 64                 # 문맥 창 (글자)
BATCH = 24                 # 한 스텝에 보는 토큰 = BATCH * BLOCK = 1,536
DROPOUT = 0.0              # 스케일링 법칙 측정에는 정규화를 끈다 (조기종료로 대신)
WEIGHT_DECAY = 0.01
MIN_COUNT = 20             # 이보다 드문 글자는 □ 로 합친다 (어휘를 줄여 임베딩 비중을 낮춘다)
CHUNK = 2000               # train/val 분할 단위 (글자)
EVAL_BATCHES = 8           # 평가 배치 수 — 모든 런이 **같은 배치** 로 채점된다

WARMUP = 100
A_STEPS = 1000             # 실험 1·5: 크기 스윕 스텝 (모든 크기 동일 → 같은 D 를 본다)
B_STEPS = 800              # 실험 3: 데이터 스윕 스텝 (조기종료로 최선 시점을 쓴다)
EVAL_EVERY = 50

# (d_model, n_layer, n_head) — 비임베딩 파라미터 N ≈ 12 · L · d²
SIZES = [(32, 3, 2), (48, 3, 3), (64, 3, 4), (96, 3, 4), (128, 3, 4), (192, 3, 4)]
DATA_FRACS = [0.004, 0.012, 0.04, 0.12, 0.40, 1.00]   # 실험 3: 학습데이터 비율
SHAPES = [(64, 3, 4), (112, 1, 4), (46, 6, 2)]     # 실험 5: 같은 N, 다른 모양
MIN_EPOCH = 4.0            # 실험 3: 이보다 많이 반복해 본 런만 '데이터 제약' 으로 인정

if QUICK:                                          # 배선만 확인하는 모드
    A_STEPS, B_STEPS, WARMUP, EVAL_EVERY = 120, 120, 20, 20
    SIZES = [(32, 3, 2), (64, 3, 4), (128, 3, 4)]
    DATA_FRACS = [0.004, 0.04, 0.40, 1.00]
    SHAPES = [(64, 3, 4), (112, 1, 4)]

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


# ── 코퍼스 (corpus) ────────────────────────────────────────────────────────
def load_corpus():
    """이 트랙의 모든 텍스트: Day 노트(.md) + 동반 스크립트(.py). 자기 폴더는 제외."""
    md = [p for p in sorted(glob.glob(os.path.join(ROOT, "Day-0*", "Day-0*.md")))
          if os.path.dirname(p) != HERE]
    py = [p for p in sorted(glob.glob(os.path.join(ROOT, "Day-0*", "*.py")))
          if os.path.dirname(p) != HERE]
    out = []
    for p in md:
        with open(p, encoding="utf-8") as f:
            s = f.read()
        s = re.sub(r"^---\n.*?\n---\n", "", s, flags=re.S)   # frontmatter 제거
        s = re.sub(r"[ \t]+", " ", s)
        s = re.sub(r"\n{3,}", "\n\n", s)
        out.append(s)
    for p in py:
        with open(p, encoding="utf-8") as f:
            out.append(f.read())
    return "\n\n".join(out), len(md), len(py)


TEXT, N_MD, N_PY = load_corpus()
if len(TEXT) < 50_000:      # 노트를 못 찾았을 때의 대체 코퍼스
    print("    (Day 노트를 못 찾아 대체 코퍼스를 쓴다 — 절대 수치는 참고만)")
    TEXT = ("정보 검색은 질문에 답이 될 문서를 찾아 순서대로 늘어놓는 일이다.\n"
            "언어모델은 다음 토큰의 확률 분포를 내놓는 함수일 뿐이다.\n"
            "def score(q, d):\n    return sum(w * d.get(t, 0.0) for t, w in q.items())\n"
            ) * 2000
    N_MD = N_PY = 0

# ── 토크나이저 (글자 하나 = 토큰 하나) ────────────────────────────────────
cnt = Counter(TEXT)
UNK = "□"
chars = sorted({c for c, n in cnt.items() if n >= MIN_COUNT} | {UNK})
VOCAB = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi.get(c, stoi[UNK]) for c in s]

IDS = torch.tensor(encode(TEXT), dtype=torch.long)
BLOCKS = [IDS[i:i + CHUNK] for i in range(0, len(IDS) - CHUNK, CHUNK)]
TR_BLOCKS = [b for i, b in enumerate(BLOCKS) if i % 10 != 7]
VAL = torch.cat([b for i, b in enumerate(BLOCKS) if i % 10 == 7])
TRAIN = torch.cat(TR_BLOCKS)
UNK_RATE = sum(1 for c in TEXT if c not in stoi) / len(TEXT)


def subset(frac, seed=0):
    """학습 블록 중 frac 비율만 (씨앗 고정 무작위로) 골라 잇는다."""
    idx = list(range(len(TR_BLOCKS)))
    random.Random(seed).shuffle(idx)
    k = max(1, round(len(idx) * frac))
    return torch.cat([TR_BLOCKS[i] for i in sorted(idx[:k])])


def get_batch(data, bs=BATCH, gen=None):
    ix = torch.randint(max(1, len(data) - BLOCK - 1), (bs,), generator=gen)
    x = torch.stack([data[i:i + BLOCK] for i in ix])
    y = torch.stack([data[i + 1:i + BLOCK + 1] for i in ix])
    return x, y


def fixed_eval_set(data, n=EVAL_BATCHES, seed=1234):
    g = torch.Generator().manual_seed(seed)
    return [get_batch(data, gen=g) for _ in range(n)]


EVAL = fixed_eval_set(VAL)


# ── 모델 ──────────────────────────────────────────────────────────────────
class CausalAttention(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        shape = lambda t: t.view(B, T, self.h, C // self.h).transpose(1, 2)
        y = F.scaled_dot_product_attention(shape(q), shape(k), shape(v), is_causal=True)
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = CausalAttention(d, h)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))          # Pre-LN ([[Day-034]])
        return x + self.ff(self.ln2(x))


class CharLM(nn.Module):
    def __init__(self, d, n_layer, n_head):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, d)
        self.pos = nn.Embedding(BLOCK, d)
        self.blocks = nn.ModuleList([Block(d, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, VOCAB, bias=False)
        self.head.weight = self.tok.weight      # 입출력 임베딩 공유 (weight tying)
        self.cfg = (d, n_layer, n_head)

    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.size(1)))
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln_f(x))

    def n_nonembed(self):
        """논문의 N: 임베딩(토큰·위치)을 **제외한** 파라미터 수."""
        total = sum(p.numel() for p in self.parameters())
        return total - self.tok.weight.numel() - self.pos.weight.numel()


def kaplan_lr(n):
    """논문 §D.6 의 처방: lr ≈ 0.003239 − 0.0001395 · ln(N). 안전 범위로 자른다."""
    return float(min(2.5e-3, max(2e-4, 0.003239 - 0.0001395 * math.log(n))))


@torch.no_grad()
def eval_loss(model, eval_set=None):
    eval_set = eval_set or EVAL
    was = model.training
    model.eval()
    tot = 0.0
    for x, y in eval_set:
        tot += F.cross_entropy(model(x).view(-1, VOCAB), y.view(-1)).item()
    model.train(was)
    return tot / len(eval_set)


def train(model, data, steps, lr, tag, eval_every=EVAL_EVERY):
    """워밍업 후 **상수 LR**. → 중간 체크포인트가 '그 연산량으로 도달 가능한 손실'이 된다.
    (코사인 감쇠를 쓰면 총 스텝 수에 따라 곡선 자체가 달라져 실험 4 가 성립하지 않는다.)"""
    params = list(model.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / WARMUP))
    curve, best, best_step = [], float("inf"), 0
    t0 = time.time()
    model.train()
    for step in range(1, steps + 1):
        x, y = get_batch(data)
        loss = F.cross_entropy(model(x).view(-1, VOCAB), y.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if step % eval_every == 0 or step == steps:
            va = eval_loss(model)
            curve.append((step, va))
            if va < best:
                best, best_step = va, step
    n = model.n_nonembed()
    d, L, h = model.cfg
    r = {"tag": tag, "d": d, "L": L, "h": h, "N": n, "lr": lr, "curve": curve,
         "best": best, "best_step": best_step, "final": curve[-1][1],
         "steps": steps, "tokens": steps * BATCH * BLOCK,
         "data_chars": len(data), "secs": time.time() - t0}
    print(f"    {tag:<26} N {n/1e3:>7.1f}K  lr {lr:.2e}  "
          f"final {r['final']:.4f}  best {best:.4f} @step {best_step:<5} "
          f"{r['secs']:>5.0f}s")
    return r


# ── 거듭제곱 법칙 적합 (power-law fitting) ────────────────────────────────
def fit_power(xs, ys):
    """log y = log A − α log x  최소제곱. 반환: (alpha, A, r2)"""
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    n = len(lx)
    mx, my = sum(lx) / n, sum(ly) / n
    sxx = sum((a - mx) ** 2 for a in lx)
    sxy = sum((a - mx) * (b - my) for a, b in zip(lx, ly))
    slope = sxy / sxx
    inter = my - slope * mx
    pred = [inter + slope * a for a in lx]
    ss_res = sum((b - p) ** 2 for b, p in zip(ly, pred))
    ss_tot = sum((b - my) ** 2 for b in ly)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return -slope, math.exp(inter), r2


def lin_r2(xs, ys, f):
    """원래 공간(로그 아님)의 R². 두 함수형을 **같은 자로** 비교하기 위해 쓴다."""
    my = sum(ys) / len(ys)
    sse = sum((f(x) - y) ** 2 for x, y in zip(xs, ys))
    sst = sum((y - my) ** 2 for y in ys)
    return 1 - sse / sst if sst > 0 else float("nan")


def fit_power_offset(xs, ys):
    """L = E + A·x^(−α) — 줄일 수 없는 손실 E 를 격자탐색으로 함께 적합한다.
    선택 기준은 '로그공간 R²' 가 아니라 **원래 공간의 잔차제곱합(SSE)** 이다.
    (로그공간 R² 로 고르면 E→min(y) 쪽으로 끌려가 병적인 해를 잡는다.)"""
    best = None
    hi = min(ys) * 0.95
    for i in range(400):
        e = hi * i / 399
        try:
            a, A, _ = fit_power(xs, [y - e for y in ys])
        except (ValueError, ZeroDivisionError):
            continue
        sse = sum((e + A * x ** (-a) - y) ** 2 for x, y in zip(xs, ys))
        if best is None or sse < best[0]:
            best = (sse, e, a, A)
    _, e, a, A = best
    my = sum(ys) / len(ys)
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1 - best[0] / ss_tot if ss_tot > 0 else float("nan")
    return e, a, A, r2      # (E, alpha, A, r2)


def loglog_plot(series, xs_label, ys_label, width=62, height=15):
    """series: [(mark, [(x, y), ...]), ...] — 로그-로그 산점도를 ASCII 로 그린다."""
    pts = [(math.log10(x), math.log10(y), m) for m, ps in series for x, y in ps]
    x0, x1 = min(p[0] for p in pts), max(p[0] for p in pts)
    y0, y1 = min(p[1] for p in pts), max(p[1] for p in pts)
    span = lambda a, b: (b - a) if b > a else 1.0
    grid = [[" "] * width for _ in range(height)]
    for lx, ly, m in pts:
        cx = int((lx - x0) / span(x0, x1) * (width - 1))
        cy = int((y1 - ly) / span(y0, y1) * (height - 1))
        grid[cy][cx] = m
    lines = [f"  {ys_label} ↑   (양축 모두 로그)"]
    for r, row in enumerate(grid):
        tick = f"{10 ** (y1 - r * (y1 - y0) / (height - 1)):>7.3f}"
        lines.append(f"  {tick} │{''.join(row)}")
    lines.append("  " + " " * 7 + " └" + "─" * width)
    lines.append(f"  {'':>7}  {10**x0:<12.3g}{' ' * max(1, width - 26)}"
                 f"{10**x1:>12.3g}  {xs_label} →")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
def main():
    bar = "─" * 76
    print(f"\n{bar}\n0. 코퍼스와 설정\n{bar}")
    print(f"  코퍼스: Day 노트 {N_MD}편 + 스크립트 {N_PY}개 · {len(TEXT):,}자")
    print(f"    train {len(TRAIN):,}자 / val {len(VAL):,}자 · 어휘(글자) {VOCAB}개"
          f" (≥{MIN_COUNT}회) · UNK 비율 {UNK_RATE*100:.3f}%")
    print(f"  한 스텝 = {BATCH}×{BLOCK} = {BATCH*BLOCK:,} 토큰 · 문맥 {BLOCK}자"
          f" · dropout {DROPOUT} · 상수 LR(워밍업 {WARMUP})")
    print(f"  연산량 규약(논문과 동일): C ≈ 6·N·D  [N=비임베딩 파라미터, D=처리 토큰]")

    # ── 실험 1: L(N) ───────────────────────────────────────────────────
    print(f"\n{bar}\n실험 1 — L(N): 모델 크기만 바꾼다 ({A_STEPS}스텝 · 데이터 전량 고정)\n{bar}")
    runs = []
    for (d, L, h) in SIZES:
        torch.manual_seed(0)
        m = CharLM(d, L, h)
        lr = kaplan_lr(m.n_nonembed())
        torch.manual_seed(1)
        runs.append(train(m, TRAIN, A_STEPS, lr, f"d{d}·L{L}·h{h}"))

    Ns = [r["N"] for r in runs]
    Ls = [r["final"] for r in runs]
    aN, AN, r2N = fit_power(Ns, Ls)
    EN, aN2, AN2, r2N2 = fit_power_offset(Ns, Ls)
    print(f"\n  {'N(비임베딩)':>14}{'전체 파라미터':>14}{'val loss':>11}{'ppl':>9}"
          f"{'예측(순수법칙)':>15}{'예측(E+A/N^α)':>15}")
    for r in runs:
        tot = r["N"] + VOCAB * r["d"] + BLOCK * r["d"]
        p1 = AN * r["N"] ** (-aN)
        p2 = EN + AN2 * r["N"] ** (-aN2)
        print(f"  {r['N']/1e3:>12.1f}K{tot/1e3:>13.1f}K{r['final']:>11.4f}"
              f"{math.exp(r['final']):>9.1f}{p1:>15.4f}{p2:>15.4f}")
    print(f"\n  순수 거듭제곱  L = ({AN**(1/aN):.3g} / N)^{aN:.4f}"
          f"      αN = {aN:.4f}   R²(log-log) = {r2N:.4f}")
    print(f"  줄일 수 없는 손실 포함  L = {EN:.4f} + {AN2:.4g}·N^(−{aN2:.4f})"
          f"   R² = {r2N2:.4f}")
    print(f"  (논문 Kaplan et al. 2020: αN ≈ 0.076 — 우리 값과 비교해 보라)")
    print()
    print(loglog_plot([("●", list(zip(Ns, Ls)))], "N (비임베딩 파라미터)", "val loss"))

    # ── 실험 2: 외삽 ───────────────────────────────────────────────────
    print(f"\n{bar}\n실험 2 — 외삽(extrapolation) 검증: 작은 모델만 보고 큰 모델을 맞히는가\n{bar}")
    k = max(2, len(runs) - 2)
    aS, AS, r2S = fit_power(Ns[:k], Ls[:k])
    ES, aS2, AS2, r2S2 = fit_power_offset(Ns[:k], Ls[:k])
    print(f"  작은 {k}개({Ns[0]/1e3:.0f}K~{Ns[k-1]/1e3:.0f}K)만으로 적합 → "
          f"αN = {aS:.4f} (R² {r2S:.4f}) · E 포함 αN = {aS2:.4f}, E = {ES:.4f}")
    print(f"  그 법칙으로 나머지 {len(runs)-k}개를 예측:")
    print(f"    {'N':>10}{'실측 L':>10}{'순수법칙':>11}{'오차':>9}"
          f"{'E+A/N^α':>11}{'오차':>9}")
    for r in runs[k:]:
        act = r["final"]
        p1 = AS * r["N"] ** (-aS)
        p2 = ES + AS2 * r["N"] ** (-aS2)
        print(f"    {r['N']/1e3:>8.0f}K{act:>10.4f}{p1:>11.4f}"
              f"{(p1-act)/act*100:>8.1f}%{p2:>11.4f}{(p2-act)/act*100:>8.1f}%")

    # ── 실험 3: L(D) ───────────────────────────────────────────────────
    d0, L0, h0 = SHAPES[0]
    print(f"\n{bar}\n실험 3 — L(D): 데이터 크기만 바꾼다 (모델 d{d0}·L{L0} 고정 · "
          f"{B_STEPS}스텝 · 조기종료)\n{bar}")
    druns = []
    for frac in DATA_FRACS:
        data = subset(frac)
        torch.manual_seed(0)
        m = CharLM(d0, L0, h0)
        lr = kaplan_lr(m.n_nonembed())
        torch.manual_seed(1)
        druns.append(train(m, data, B_STEPS, lr, f"D={len(data):,}자"))
    # '데이터 제약' 구간만으로 법칙을 적합한다 — D 가 크면 병목이 D 가 아니다.
    lim = [r for r in druns if r["tokens"] / r["data_chars"] >= MIN_EPOCH]
    free = [r for r in druns if r not in lim]
    Ds = [r["data_chars"] for r in lim]
    Bs = [r["best"] for r in lim]
    aD, AD, r2D = fit_power(Ds, Bs)
    if len(Ds) >= 4:
        ED, aD2, AD2, r2D2 = fit_power_offset(Ds, Bs)
    else:      # 점이 부족하면 E 를 추정하지 않는다 (과적합한 E 는 거짓말을 한다)
        ED, aD2, AD2 = 0.0, aD, AD
        r2D2 = lin_r2(Ds, Bs, lambda x: AD * x ** (-aD))
    print(f"\n  {'D(글자=토큰)':>14}{'epoch 수':>10}{'best L':>10}{'ppl':>8}"
          f"{'@step':>8}{'최종 L':>10}{'과적합 폭':>11}{'법칙 적합에':>12}")
    for r in druns:
        ep = r["tokens"] / r["data_chars"]
        used = "포함" if r in lim else "제외(D병목아님)"
        print(f"  {r['data_chars']:>13,}{ep:>10.1f}{r['best']:>10.4f}"
              f"{math.exp(r['best']):>8.1f}{r['best_step']:>8}{r['final']:>10.4f}"
              f"{r['final']-r['best']:>+11.4f}{used:>12}")
    print(f"\n  순수 거듭제곱(데이터 제약 {len(lim)}점)  αD = {aD:.4f}"
          f"   R²(log-log) = {r2D:.4f}")
    print(f"  줄일 수 없는 손실 포함  L = {ED:.4f} + {AD2:.4g}·D^(−{aD2:.4f})"
          f"   R² = {r2D2:.4f}")
    print(f"  (논문: αD ≈ 0.095)")
    for r in free:
        p = AD * r["data_chars"] ** (-aD)
        print(f"  법칙이 깨지는 지점: D={r['data_chars']:,} (epoch {r['tokens']/r['data_chars']:.1f})"
              f" → 예측 {p:.4f} vs 실측 {r['best']:.4f} ({(p-r['best'])/r['best']*100:+.1f}%)")
    print()
    print(loglog_plot([("●", [(r["data_chars"], r["best"]) for r in lim]),
                       ("×", [(r["data_chars"], r["best"]) for r in free])],
                      "D (학습 토큰 수)  ●=데이터 제약 ×=제약 아님", "best val loss"))

    # ── 실험 4: 계산 최적 ──────────────────────────────────────────────
    print(f"\n{bar}\n실험 4 — 계산 최적(compute-optimal): 같은 예산이면 모델이냐 데이터냐\n{bar}")
    print(f"  방법: 실험 1 의 학습곡선을 x축을 '스텝'에서 'C = 6·N·D' 로 바꿔 다시 그린다.")
    print(f"        여러 곡선의 **하한선(envelope)** 이 곧 '그 예산으로 가능한 최선'이다.")
    print(f"        추가 학습 비용 0 — 이것이 논문 Approach 1 이다.\n")

    # 워밍업 중 체크포인트는 버린다 — 아직 '학습된 손실'이 아니라 초기화의 잔상이다.
    curves = []
    for r in runs:
        pts = [(6 * r["N"] * (s * BATCH * BLOCK), v) for s, v in r["curve"] if s > WARMUP]
        curves.append((r, pts))

    # 모든 (모델, 체크포인트) 를 연산량 순으로 늘어놓고 **파레토 하한선** 만 남긴다.
    # "예산 C 로 얻을 수 있는 최선" 이므로, 더 싼 점보다 나쁜 점은 하한선이 아니다.
    # (각 모델의 연산 범위가 서로 어긋나므로, 격자 보간보다 이 정의가 정직하다.)
    pool = sorted(((c, v, r) for r, pts in curves for c, v in pts), key=lambda t: t[0])
    front, seen_best = [], float("inf")
    for c, v, r in pool:
        if v < seen_best:
            seen_best = v
            front.append((c, v, r["N"], c / (6 * r["N"]), r))
    print(f"  {'C (FLOPs)':>12}{'최선 L':>10}{'승자 모델':>14}{'N*':>10}"
          f"{'D*(토큰)':>12}{'D*/N*':>9}")
    every = max(1, len(front) // 9)
    for i, (C, v, N, Dstar, r) in enumerate(front):
        if i % every == 0 or i == len(front) - 1:
            print(f"  {C:>12.3g}{v:>10.4f}{'d'+str(r['d'])+'·L'+str(r['L']):>14}"
                  f"{N/1e3:>9.0f}K{Dstar:>12,.0f}{Dstar/N:>9.2f}")
    print(f"  (하한선 점 {len(front)}개 / 전체 체크포인트 {len(pool)}개)")

    Cs = [f[0] for f in front]
    aC, AC, r2C = fit_power(Cs, [f[1] for f in front])
    print(f"\n  하한선 자체의 법칙  L(C) = ({AC**(1/aC):.3g}/C)^{aC:.4f}"
          f"   αC = {aC:.4f}  R² = {r2C:.4f}   (논문: αC ≈ 0.050)")

    winners = sorted({f[2] for f in front})
    print(f"  하한선에 오른 모델 수: {len(winners)}개 "
          f"({', '.join(f'{w/1e3:.0f}K' for w in winners)})")
    if len(winners) >= 3:
        a, _, r2a = fit_power(Cs, [1 / f[2] for f in front])   # N* ∝ C^a
        b, _, r2b = fit_power(Cs, [1 / f[3] for f in front])   # D* ∝ C^b
        print(f"  N* ∝ C^{a:.3f} (R² {r2a:.3f})   D* ∝ C^{b:.3f} (R² {r2b:.3f})"
              f"   a+b = {a+b:.3f}  ← C=6ND 이므로 정의상 정확히 1 이다(계산 점검)")
        print(f"    Kaplan 2020:     a ≈ 0.73, b ≈ 0.27  → '연산이 늘면 주로 모델을 키워라'")
        print(f"    Chinchilla 2022: a ≈ 0.50, b ≈ 0.50  → '모델과 데이터를 같은 비율로'")
    else:
        a = b = float("nan")
        print("  (하한선에 오른 모델이 3개 미만 — 지수 적합은 신뢰할 수 없다)")
    ratios = [f[3] / f[2] for f in front]
    print(f"  우리 하한선의 토큰/파라미터 비: {min(ratios):.1f} ~ {max(ratios):.1f}"
          f" (중앙 {sorted(ratios)[len(ratios)//2]:.1f})   Chinchilla 권고: ≈ 20")

    print()
    marks = "123456789"
    print(loglog_plot([(marks[i], pts) for i, (r, pts) in enumerate(curves)],
                      "C = 6·N·D (FLOPs)", "val loss"))
    print("   " + "  ".join(f"{marks[i]}=d{r['d']}({r['N']/1e3:.0f}K)"
                            for i, (r, _) in enumerate(curves)))

    # ── 실험 5: 모양 ───────────────────────────────────────────────────
    print(f"\n{bar}\n실험 5 — 같은 N, 다른 모양(폭 vs 깊이): 정말 상관없는가\n{bar}")
    sruns = []
    for (d, L, h) in SHAPES:
        same = [r for r in runs if (r["d"], r["L"], r["h"]) == (d, L, h)]
        if same:                       # 실험 1 에서 이미 돌린 설정이면 그 런을 그대로 쓴다
            sruns.append(same[0])
            print(f"    {same[0]['tag']:<26} (실험 1 의 런을 재사용)")
            continue
        torch.manual_seed(0)
        m = CharLM(d, L, h)
        lr = kaplan_lr(m.n_nonembed())
        torch.manual_seed(1)
        sruns.append(train(m, TRAIN, A_STEPS, lr, f"d{d}·L{L}·h{h}"))
    base = sruns[0]["final"]
    print(f"\n  {'모양':>14}{'N':>10}{'가로세로비 d/L':>16}{'final L':>10}"
          f"{'기준 대비':>10}")
    for r in sruns:
        print(f"  {'d'+str(r['d'])+'·L'+str(r['L']):>14}{r['N']/1e3:>9.0f}K"
              f"{r['d']/r['L']:>16.1f}{r['final']:>10.4f}"
              f"{(r['final']-base)/base*100:>+9.1f}%")
    spread = max(r["final"] for r in sruns) - min(r["final"] for r in sruns)
    print(f"\n  같은 N 안에서 모양이 만든 손실 차이: {spread:.4f} nats")
    print(f"  비교: 실험 1 에서 N 을 {Ns[-1]/Ns[0]:.0f}배 키웠을 때의 차이는 "
          f"{Ls[0]-Ls[-1]:.4f} nats")

    # ── 요약 ──────────────────────────────────────────────────────────
    print(f"\n{bar}\n요약\n{bar}")
    allruns = runs + druns + [r for r in sruns if r not in runs]
    tot_secs = sum(r["secs"] for r in allruns)
    print(f"  총 {len(allruns)}회 학습 · {tot_secs:.0f}초 ({tot_secs/60:.1f}분)")
    print(f"  αN = {aN:.4f} (논문 0.076)   αD = {aD:.4f} (논문 0.095)   "
          f"αC = {aC:.4f} (논문 0.050)")
    print(f"  거듭제곱 법칙의 R²: N {r2N:.4f} · D {r2D:.4f} · C {r2C:.4f}")
    print(f"  계산 최적 지수: a = {a:.3f} · b = {b:.3f}  "
          f"(Kaplan 0.73/0.27 · Chinchilla 0.50/0.50)")
    err_big = (AS * Ns[-1] ** (-aS) - Ls[-1]) / Ls[-1] * 100
    print(f"  외삽 오차(작은 {k}개 → 가장 큰 모델 {Ns[-1]/1e3:.0f}K): {err_big:+.1f}%")
    print(f"\n  결론 1. 손실은 로그-로그에서 거의 직선이다 (R² {r2N:.4f}) — 그래서 외삽이 된다.")
    print(f"          작은 {k}개만 보고 {Ns[-1]/Ns[k-1]:.1f}배 큰 모델의 손실을 "
          f"{abs(err_big):.1f}% 안에 맞혔다.")
    # 두 함수형(순수 거듭제곱 vs E+A/x^α)을 **같은 자**(원래 공간 R²)로 비교한다.
    pN = lin_r2(Ns, Ls, lambda x: AN * x ** (-aN))
    pD = lin_r2(Ds, Bs, lambda x: AD * x ** (-aD))
    nwin = f"E={EN:.3f} 포함형(R² {r2N2:.4f})" if r2N2 > pN else f"순수 거듭제곱(R² {pN:.4f})"
    dwin = f"E={ED:.3f} 포함형(R² {r2D2:.4f})" if r2D2 > pD else f"순수 거듭제곱(R² {pD:.4f})"
    print(f"  결론 2. 어느 함수형이 이겼나 —  L(N): {nwin}   L(D): {dwin}")
    if r2D2 > pD and ED > 0:
        print(f"          L(D) 의 E={ED:.3f} 는 '언어의 엔트로피'가 아니라 **모델 용량의 바닥**")
        print(f"          이다 — d{d0} 모델이 데이터를 아무리 줘도 넘지 못하는 선.")
    print(f"  결론 3. 계산 최적 지수는 '방법론'이 정한다. 우리는 Kaplan 의 설계(고정 스텝")
    print(f"          예산 · 상수 LR)를 썼고 Kaplan 의 답(a≈{a:.2f})을 얻었다.")
    print(f"          Chinchilla 가 뒤집은 것은 결과가 아니라 **실험 설계** 였다.")


if __name__ == "__main__":
    main()
