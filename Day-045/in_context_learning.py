"""
Day-045 — 프롬프팅과 문맥 내 학습: 가중치를 건드리지 않고 모델을 부리는 법
(In-Context Learning — 왜 되는가, 언제 안 되는가, 무엇이 흔드는가)

[[Day-043]] 은 가중치를 **바꿔서** 과제를 옮겼고, [[Day-044]] 는 그 가중치를 갖는 데
얼마가 드는지를 읽었다. 오늘은 정반대다 — 가중치는 그대로 두고 **입력만 바꾼다.**

핵심 질문 셋을 실험으로 나눈다.

  Part A. 우리 트랙 코퍼스로 학습한 모델은 few-shot 이 되는가?  (답: 안 된다.
          그런데 '전혀 못한다'가 아니라 **어디까지 되고 어디서 끊기는지** 를 잰다.)
    실험 1. 반복열 유도 테스트 — 문맥을 복사할 줄은 아는가 (induction).
            학습 중 체크포인트마다 재서 **언제 생기는지** 도 본다 (Olsson et al. 2022).
    실험 2. 과제 few-shot 프롬프트 — 데모를 보고 규칙을 따르는가. (아니오)

  Part B. few-shot 이 **되는** 모델을 만든다.
          매 시퀀스마다 새로 뽑은 무작위 대응(bijection)을 데모로 보여 준다.
          외울 수 있는 정답이 없으므로, 손실을 낮추는 유일한 길은 **데모를 읽는 것** 이다.
    실험 3. 샷 수 k 곡선 — 그리고 이론적 '부기(bookkeeping)' 곡선 (k+1)/8 과 비교.
    실험 4. 형식 민감도 — 구분자·화살표·순서만 바꾸면 어떻게 되는가.
    실험 5. 라벨 손상 — Min et al. (2022) 과 우리가 왜 다른 결과를 얻는가.
    실험 6. 데모 위치 — 최근성 편향(recency bias)이 있는가.
    실험 7. 기계 장치 — 유도 회로(previous-token head → induction head)를 절제해 본다.

실행:  uv run --with torch python in_context_learning.py
       빠른 점검:  $env:QUICK="1"; uv run --with torch python in_context_learning.py
한글이 깨지면 먼저:  $env:PYTHONIOENCODING="utf-8"
"""

import glob
import math
import os
import random
import time
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

QUICK = os.environ.get("QUICK") == "1"
DEV = "cpu"
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

torch.manual_seed(0)
random.seed(0)


# ══════════════════════════════════════════════════════════════════════════
#  모델 — [[Day-037]] 의 디코더-only Transformer.
#  다만 오늘은 **어텐션 가중치를 꺼내 보고 헤드를 절제** 해야 하므로
#  F.scaled_dot_product_attention 대신 손으로 편다.
# ══════════════════════════════════════════════════════════════════════════
class CausalSelfAttention(nn.Module):
    def __init__(self, d, h, block):
        super().__init__()
        self.h, self.dh = h, d // h
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.register_buffer(
            "mask", torch.tril(torch.ones(block, block)).view(1, 1, block, block)
        )
        self.dead = ()  # 실험 7: 여기에 넣은 헤드는 출력이 0 이 된다

    def forward(self, x, keep_attn=False):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.h, self.dh).transpose(1, 2)
        k = k.view(B, T, self.h, self.dh).transpose(1, 2)
        v = v.view(B, T, self.h, self.dh).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = torch.softmax(att, dim=-1)
        y = att @ v
        if self.dead:
            y = y.clone()
            for hi in self.dead:
                y[:, hi] = 0.0
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y), (att if keep_attn else None)


class Block(nn.Module):
    def __init__(self, d, h, block):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = CausalSelfAttention(d, h, block)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, keep_attn=False):
        a, att = self.attn(self.ln1(x), keep_attn)
        x = x + a
        return x + self.mlp(self.ln2(x)), att


class GPT(nn.Module):
    def __init__(self, vocab, d, n_layer, n_head, block):
        super().__init__()
        self.block_size = block
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(block, d)
        self.blocks = nn.ModuleList([Block(d, n_head, block) for _ in range(n_layer)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight  # 입출력 임베딩 공유 (Day-044 §1.4)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, keep_attn=False):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        atts = []
        for b in self.blocks:
            x, a = b(x, keep_attn)
            if keep_attn:
                atts.append(a)
        return self.head(self.lnf(x)), atts

    def set_dead(self, spec):
        """spec: {layer_idx: (head_idx, ...)} — 지정한 헤드의 출력을 0 으로 만든다."""
        for i, b in enumerate(self.blocks):
            b.attn.dead = tuple(spec.get(i, ()))


def n_params(model, non_embedding=True):
    n = sum(p.numel() for p in model.parameters())
    if non_embedding:
        n -= model.tok.weight.numel() + model.pos.weight.numel()
    return n


# ══════════════════════════════════════════════════════════════════════════
#  Part A — 우리 트랙 코퍼스로 학습한 char-level 모델
# ══════════════════════════════════════════════════════════════════════════
A_BLOCK = 128
A_BATCH = 16
A_STEPS = 200 if QUICK else 800
A_EVERY = 50 if QUICK else 100
A_D, A_L, A_H = 192, 3, 4
MIN_COUNT = 20


def load_corpus():
    paths = sorted(glob.glob(os.path.join(ROOT, "Day-*", "Day-*.md")))
    paths += sorted(glob.glob(os.path.join(ROOT, "Day-*", "*.py")))
    paths += sorted(glob.glob(os.path.join(ROOT, "*.md")))
    parts, n_md, n_py = [], 0, 0
    for p in paths:
        try:
            parts.append(open(p, encoding="utf-8").read())
        except Exception:
            continue
        if p.endswith(".py"):
            n_py += 1
        else:
            n_md += 1
    text = "\n\n".join(parts)
    return text, n_md, n_py


def build_char_data():
    text, n_md, n_py = load_corpus()
    cnt = Counter(text)
    chars = sorted(c for c, n in cnt.items() if n >= MIN_COUNT)
    stoi = {c: i + 1 for i, c in enumerate(chars)}  # 0 = □ (희귀 글자)
    itos = {i + 1: c for i, c in enumerate(chars)}
    itos[0] = "□"
    ids = torch.tensor([stoi.get(c, 0) for c in text], dtype=torch.long)
    unk = float((ids == 0).float().mean())
    n = len(ids)
    cut = (int(n * 0.9) // 2000) * 2000
    return dict(
        train=ids[:cut], val=ids[cut:], stoi=stoi, itos=itos,
        vocab=len(chars) + 1, chars=len(text), unk=unk, n_md=n_md, n_py=n_py,
    )


def get_batch(data, block, batch, gen):
    ix = torch.randint(len(data) - block - 1, (batch,), generator=gen)
    x = torch.stack([data[i : i + block] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block] for i in ix])
    return x, y


@torch.no_grad()
def induction_probe(model, val, seg=48, n=32, seed=7):
    """반복열 유도 테스트.

    무작위 조각 S 를 이어 붙여 [S, S] 를 만들고, 두 번째 사본에서의 손실이
    첫 번째 사본보다 얼마나 낮은지를 잰다. 낮아진다면 모델이 **문맥을 되읽어
    복사** 하고 있다는 뜻이다 (induction).

    대조군: [S1, S2] (서로 다른 조각). '뒤쪽 위치라서 쉬운 것'이 아님을 보인다.
    """
    g = torch.Generator().manual_seed(seed)
    ix = torch.randint(len(val) - 2 * seg - 2, (n,), generator=g)
    jx = torch.randint(len(val) - 2 * seg - 2, (n,), generator=g)
    rep = torch.stack([torch.cat([val[i : i + seg], val[i : i + seg]]) for i in ix])
    ctl = torch.stack(
        [torch.cat([val[i : i + seg], val[j : j + seg]]) for i, j in zip(ix, jx)]
    )
    out = {}
    model.eval()
    for name, x in (("rep", rep), ("ctl", ctl)):
        logits, _ = model(x[:, :-1])
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1), reduction="none"
        ).view(x.size(0), -1)
        first = loss[:, 8 : seg - 1].mean().item()   # 첫 사본 (앞 8자는 워밍업)
        second = loss[:, seg + 7 :].mean().item()    # 두 번째 사본
        out[name] = (first, second, first - second)
    model.train()
    return out


def train_char_model(d):
    model = GPT(d["vocab"], A_D, A_L, A_H, A_BLOCK).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=0.01)
    gen = torch.Generator().manual_seed(1234)
    trace, t0 = [], time.time()
    for step in range(1, A_STEPS + 1):
        lr = 1.2e-3 * min(1.0, step / 100)
        for pg in opt.param_groups:
            pg["lr"] = lr
        x, y = get_batch(d["train"], A_BLOCK, A_BATCH, gen)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % A_EVERY == 0 or step == A_STEPS:
            pr = induction_probe(model, d["val"])
            trace.append((step, loss.item(), pr["rep"], pr["ctl"]))
    return model, trace, time.time() - t0


@torch.no_grad()
def fewshot_char_probe(model, d, shots):
    """트랙 모델에 '규칙을 따르라'는 데모를 준다.

    프롬프트:  a=1;b=2;c=3;a=      정답은 '1'.
    데모를 읽을 줄 안다면 P('1') 이 압도적이어야 한다.
    """
    pairs = [("a", "1"), ("b", "2"), ("c", "3"), ("d", "4")]
    q, ans = pairs[0]
    demo = "".join(f"{x}={y};" for x, y in pairs[:shots])
    prompt = demo + f"{q}="
    ids = torch.tensor(
        [[d["stoi"].get(c, 0) for c in prompt]], dtype=torch.long
    )[:, -A_BLOCK:]
    logits, _ = model(ids)
    p = torch.softmax(logits[0, -1], dim=-1)
    tgt = d["stoi"].get(ans, 0)
    top = torch.topk(p, 5)
    return (
        prompt,
        p[tgt].item(),
        int((p > p[tgt]).sum().item()) + 1,
        [(d["itos"][int(i)], float(v)) for v, i in zip(top.values, top.indices)],
    )


@torch.no_grad()
def greedy(model, d, prompt, n=60):
    ids = torch.tensor([[d["stoi"].get(c, 0) for c in prompt]], dtype=torch.long)
    for _ in range(n):
        logits, _ = model(ids[:, -A_BLOCK:])
        nxt = logits[0, -1].argmax().view(1, 1)
        ids = torch.cat([ids, nxt], dim=1)
    return "".join(d["itos"][int(i)] for i in ids[0])


# ══════════════════════════════════════════════════════════════════════════
#  Part B — few-shot 이 '되는' 모델을 만든다 (합성 대응 과제)
#
#  어휘:  X0..X7 = 0..7 (입력 심볼)   Y0..Y7 = 8..15 (출력 심볼)   SEP = 16
#  시퀀스: [x y SEP] × k(데모, x 는 서로 다름)  +  [xq yq SEP] × Q(질의)
#  매 시퀀스마다 대응 π 를 새로 뽑는다 → **외울 수 있는 정답이 없다.**
#
#  학습이 되게 만드는 데 결정적이었던 두 가지 (노트 §4.4 에 기록):
#    · k 를 **시퀀스마다** 다르게 뽑는다 (배치마다가 아니라).
#    · 질의를 넉넉히 둔다 (Q = 8). 복사가 이득이 되는 자리를 늘려야 회로가 잡힌다.
# ══════════════════════════════════════════════════════════════════════════
NSYM = 8
SEP, B_VOCAB = 16, 17
B_BLOCK = 64
B_BATCH = 64
B_STEPS = 400 if QUICK else 3000
B_LR = 1e-3
B_D, B_L, B_H = 64, 2, 4
Q_ROUNDS = 8


def _demo_tokens(x, y, fmt):
    """데모 한 쌍의 토큰. 표준형에서 x 와 y 는 **바로 이웃** 이다 —
    유도 회로(previous-token head → induction head)가 성립하는 조건이다."""
    if fmt == "std":
        return [x, y + NSYM, SEP]
    if fmt == "nosep":            # 구분자 제거
        return [x, y + NSYM]
    if fmt == "double_sep":       # 구분자 두 번
        return [x, y + NSYM, SEP, SEP]
    if fmt == "gap":              # 입력과 라벨 사이에 구분자 하나 → 인접성 파괴
        return [x, SEP, y + NSYM, SEP]
    if fmt == "reversed":         # 라벨을 먼저
        return [y + NSYM, x, SEP]
    raise ValueError(fmt)


def make_train_batch(bs, rng):
    seqs = []
    for _ in range(bs):
        perm = list(range(NSYM))
        rng.shuffle(perm)                       # π: x → perm[x] (시퀀스마다 새로)
        xs = list(range(NSYM))
        rng.shuffle(xs)
        k = rng.randint(1, NSYM)                # 데모 수도 시퀀스마다 새로
        s = []
        for x in xs[:k]:
            s += _demo_tokens(x, perm[x], "std")
        for _ in range(Q_ROUNDS):
            xq = rng.randrange(NSYM)            # 데모에 없던 x 도 나온다
            s += _demo_tokens(xq, perm[xq], "std")
        seqs.append(s)
    T = max(len(s) for s in seqs)
    x = torch.full((bs, T), SEP, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, : len(s)] = torch.tensor(s)
    return x[:, :-1], x[:, 1:]


def make_eval_batch(bs, k, seed, *, fmt="std", corrupt="none", force_slot=None):
    """첫 질의의 정답 위치까지만 만든다 (그 뒤는 볼 필요가 없다).

    corrupt: "none" | "all"(모든 데모 라벨 무작위) | "others"(관련 데모만 정답 유지)
    force_slot: 질의와 일치하는 데모를 몇 번째 자리에 둘지 (0-based)
    반환: tokens, 참정답, 화면에 보여 준 라벨, 일치 데모의 y 토큰 위치(없으면 -1)
    """
    rng = random.Random(seed)
    seqs, true_y, shown_y, ypos = [], [], [], []
    for _ in range(bs):
        perm = list(range(NSYM))
        rng.shuffle(perm)
        xs = list(range(NSYM))
        rng.shuffle(xs)
        demo_x = xs[:k]
        if force_slot is not None and k > 0:
            xq = demo_x[force_slot]
        else:
            xq = rng.randrange(NSYM)
        labels = {}
        for x in demo_x:
            if corrupt == "all":
                labels[x] = rng.randrange(NSYM)
            elif corrupt == "others" and x != xq:
                labels[x] = rng.randrange(NSYM)
            else:
                labels[x] = perm[x]
        s, pos = [], -1
        for x in demo_x:
            toks = _demo_tokens(x, labels[x], fmt)
            if x == xq:
                pos = len(s) + toks.index(labels[x] + NSYM)  # 일치 데모의 y 위치
            s += toks
        s.append(xq)                      # 질의 — 여기서 다음 토큰을 예측한다
        seqs.append(s)
        true_y.append(perm[xq])
        shown_y.append(labels.get(xq, perm[xq]))
        ypos.append(pos)
    T = max(len(s) for s in seqs)
    x = torch.full((bs, T), SEP, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, T - len(s) :] = torch.tensor(s)  # 왼쪽 패딩 → 예측 위치는 항상 마지막
    shift = [T - len(s) for s in seqs]
    ypos = [p + sh if p >= 0 else -1 for p, sh in zip(ypos, shift)]
    return x, torch.tensor(true_y), torch.tensor(shown_y), torch.tensor(ypos)


@torch.no_grad()
def icl_eval(model, k, seed=99, bs=512, **kw):
    x, true_y, shown_y, ypos = make_eval_batch(bs, k, seed, **kw)
    logits, _ = model(x)
    pred = logits[:, -1, NSYM : 2 * NSYM].argmax(-1)  # 출력 심볼 중에서만 고른다
    return (
        (pred == true_y).float().mean().item(),
        (pred == shown_y).float().mean().item(),
    )


def train_icl_model():
    model = GPT(B_VOCAB, B_D, B_L, B_H, B_BLOCK).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=B_LR, weight_decay=0.01)
    rng = random.Random(2024)
    trace, t0 = [], time.time()
    for step in range(1, B_STEPS + 1):
        for pg in opt.param_groups:
            pg["lr"] = B_LR * min(1.0, step / 200)
        x, y = make_train_batch(B_BATCH, rng)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % (100 if QUICK else 250) == 0 or step == B_STEPS:
            acc, _ = icl_eval(model, 7, bs=256)
            trace.append((step, loss.item(), acc))
    return model, trace, time.time() - t0


@torch.no_grad()
def attention_report(model, k=7, bs=256):
    """유도 회로의 두 조각을 잰다.

    (1) previous-token head — 위치 t 가 t-1 을 보는가 (이른 층)
    (2) induction head      — 질의의 ARROW 가 **일치 데모의 y** 를 보는가 (늦은 층)
    """
    x, _, _, ypos = make_eval_batch(bs, k, seed=1231)
    keep = ypos >= 0                       # 질의가 데모에 없던 행은 뺀다
    x, ypos = x[keep], ypos[keep]
    _, atts = model(x, keep_attn=True)
    T, B = x.size(1), x.size(0)
    prev, ind = {}, {}
    for li, a in enumerate(atts):
        for h in range(a.size(1)):
            d1 = a[:, h].diagonal(offset=-1, dim1=-2, dim2=-1)  # t → t-1
            prev[(li, h)] = d1[:, 4:].mean().item()
            m = a[:, h, T - 1, :]
            ind[(li, h)] = m[torch.arange(B), ypos].mean().item()
    return prev, ind


# ══════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 78)
    print("Day-045 — 문맥 내 학습(In-Context Learning) 해부")
    print("=" * 78)

    # ── Part A ──────────────────────────────────────────────────────────
    d = build_char_data()
    print(
        f"\n[코퍼스] 노트 {d['n_md']}편 + 스크립트 {d['n_py']}개 · {d['chars']:,}자 "
        f"| 어휘 {d['vocab']} (>={MIN_COUNT}회) | UNK {d['unk']*100:.3f}%"
    )
    print(f"[모델 A] d{A_D}·L{A_L}·h{A_H}, block {A_BLOCK}, {A_STEPS} steps")

    model_a, trace_a, ta = train_char_model(d)
    print(f"         비임베딩 N = {n_params(model_a):,}   학습 {ta:.0f}s")

    print("\n-- 실험 1. 반복열 유도(induction) 테스트 ---------------------------")
    print("   [S,S] 의 두 번째 사본 손실이 첫 사본보다 낮은가?  (대조군 [S1,S2])")
    print("   step |  train |  반복 1st  반복 2nd    d(유도)  | 대조 1st  대조 2nd     d")
    for step, tl, rep, ctl in trace_a:
        print(
            f"   {step:>4} | {tl:6.3f} |   {rep[0]:6.3f}   {rep[1]:6.3f}   "
            f"{rep[2]:+6.3f}  |  {ctl[0]:6.3f}   {ctl[1]:6.3f}  {ctl[2]:+6.3f}"
        )

    print("\n-- 실험 2. 과제 few-shot 프롬프트 (a=1;b=2;c=3;a= -> '1' 인가) -----")
    print("   shots | 프롬프트              P(정답) | 순위 | 상위 5개 예측")
    for s in (0, 1, 2, 3, 4):
        prompt, p, rank, top = fewshot_char_probe(model_a, d, s)
        tops = "  ".join(f"{c!r}:{v:.2f}" for c, v in top)
        print(f"   {s:>5} | {prompt:<20} {p:7.4f} | {rank:>4} | {tops}")
    print(f"   (무작위 추측 = 1/{d['vocab']} = {1/d['vocab']:.4f})")
    print("\n   greedy 생성:")
    print("   " + repr(greedy(model_a, d, "a=1;b=2;c=3;a=", 60)))

    # ── Part B ──────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("Part B — few-shot 이 '되는' 모델 만들기 (매번 새 대응을 데모로)")
    print("=" * 78)
    model_b, trace_b, tb = train_icl_model()
    print(
        f"[모델 B] d{B_D}·L{B_L}·h{B_H}, 어휘 {B_VOCAB}, "
        f"비임베딩 N = {n_params(model_b):,}, {B_STEPS} steps, {tb:.0f}s"
    )
    print("   step |  train loss | k=7 정확도")
    for step, tl, acc in trace_b:
        print(f"   {step:>4} |    {tl:7.4f} |   {acc*100:6.2f}%")

    print("\n-- 실험 3. 샷 수 곡선 vs 이론적 부기 곡선 (k+1)/8 ------------------")
    print("     k | 정확도    이론 (k+1)/8    차이")
    for k in range(NSYM):
        acc, _ = icl_eval(model_b, k)
        th = (k + 1) / NSYM
        print(f"    {k:>2} | {acc*100:6.2f}%      {th*100:6.2f}%    {(acc-th)*100:+6.2f}%p")

    print("\n-- 실험 4. 형식 민감도 (k=7, 내용은 동일 · 형식만 교란) ------------")
    print("     형식                        정확도     기준 대비")
    base = None
    for fmt, label in (
        ("std", "표준        x y ,"),
        ("nosep", "구분자 제거     x y"),
        ("double_sep", "구분자 두 번    x y , ,"),
        ("gap", "사이에 구분자   x , y ,"),
        ("reversed", "순서 뒤집기     y x ,"),
    ):
        acc, _ = icl_eval(model_b, 7, fmt=fmt)
        if base is None:
            base = acc
        print(f"     {label:<26} {acc*100:6.2f}%    {(acc-base)*100:+7.2f}%p")

    print("\n-- 실험 5. 라벨 손상 — Min et al. (2022) 과 왜 다른가 (k=7) --------")
    print("     조건                          참정답 일치   보여 준 라벨 일치")
    for cor, label in (
        ("none", "원본"),
        ("others", "무관한 데모만 손상"),
        ("all", "모든 데모 라벨 무작위"),
    ):
        acc_t, acc_s = icl_eval(model_b, 7, corrupt=cor)
        print(f"     {label:<28} {acc_t*100:7.2f}%      {acc_s*100:7.2f}%")

    print("\n-- 실험 6. 데모 위치 — 최근성 편향이 있는가 (k=7) -------------------")
    print("     일치 데모 위치 |  정확도")
    for slot in range(7):
        acc, _ = icl_eval(model_b, 7, force_slot=slot)
        tag = " (가장 먼저)" if slot == 0 else (" (가장 마지막)" if slot == 6 else "")
        print(f"     {slot+1:>9}번째{tag:<14} {acc*100:6.2f}%")

    print("\n-- 실험 7. 기계 장치 — 유도 회로를 찾아 절제한다 (k=7) --------------")
    prev, ind = attention_report(model_b)
    print("     헤드별 어텐션 질량")
    print("     (층,헤드) | t->t-1 (previous-token) | 질의->일치 데모의 y (induction)")
    for key in sorted(prev):
        print(
            f"       L{key[0]}H{key[1]}    |        {prev[key]:.3f}"
            f"          |        {ind[key]:.3f}"
        )

    base_acc, _ = icl_eval(model_b, 7)
    print(f"\n     절제 전 정확도: {base_acc*100:.2f}%")
    print("     절제 대상 |  정확도   |   낙폭")
    damage = []
    for li in range(B_L):
        for h in range(B_H):
            model_b.set_dead({li: (h,)})
            acc, _ = icl_eval(model_b, 7)
            damage.append(((li, h), acc))
            model_b.set_dead({})
    for (li, h), acc in sorted(damage, key=lambda t: t[1]):
        print(f"       L{li}H{h}     |  {acc*100:6.2f}%  |  {(acc-base_acc)*100:+7.2f}%p")

    # 회로 전체 절제: 층마다 가장 아픈 헤드 하나씩
    l0 = min((t for t in damage if t[0][0] == 0), key=lambda t: t[1])[0]
    l1 = min((t for t in damage if t[0][0] == 1), key=lambda t: t[1])[0]
    model_b.set_dead({0: (l0[1],), 1: (l1[1],)})
    acc_both, _ = icl_eval(model_b, 7)
    model_b.set_dead({})
    print(
        f"\n     회로 동시 절제 L{l0[0]}H{l0[1]} + L{l1[0]}H{l1[1]}: "
        f"{acc_both*100:.2f}%  ({(acc_both-base_acc)*100:+.2f}%p)   "
        f"무작위 추측 = {100/NSYM:.2f}%"
    )
    print("\n완료.")


if __name__ == "__main__":
    main()
