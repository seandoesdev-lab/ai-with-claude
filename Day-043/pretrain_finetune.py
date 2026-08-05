"""
Day-043 — 사전학습과 파인튜닝: 그 분포는 애초에 어떻게 만들어졌는가
(Pre-training & fine-tuning, measured on two real domains)

[[Day-041]] 에서 입구(토크나이저)를, [[Day-042]] 에서 출구(디코딩)를 봤다.
남은 것은 가운데다 — 모델은 p(다음 토큰 | 문맥) 을 **어떻게 갖게 되었는가.**

답은 두 단계다.
    사전학습(pre-training)   : 거대한 '아무 텍스트'로 다음 토큰 예측만 시킨다.
    파인튜닝(fine-tuning)    : 그 가중치에서 출발해, 내가 원하는 작은 과제로 옮긴다.

오늘은 그 두 단계를 **한 파일 안에서 직접 돌리고 측정** 한다. 두 도메인을 쓴다.

    소스 도메인 (사전학습) : 이 트랙의 Day 노트 산문 — 한국어 마크다운
    타깃 도메인 (하류 과제) : 이 트랙의 동반 .py 파일 — 파이썬 코드

둘은 글자를 공유하지만 분포는 전혀 다르다. 그래서 '전이(transfer)'가 공짜가 아니다.

  실험 1. 두 도메인은 얼마나 다른가 — 그리고 사전학습 모델의 zero-shot 성적
  실험 2. 같은 타깃 스텝 예산: 백지에서 vs 사전학습에서 (학습곡선)
  실험 3. 데이터가 적을수록 이득이 크다 — 타깃 데이터 2% / 10% / 100%
  실험 4. 어떻게 옮길 것인가 — 학습률(LR)과 '무엇을 얼릴 것인가(freezing)'
  실험 5. 파국적 망각 — 새 과제를 배우며 원래 능력을 얼마나 잃는가

모델: char-level 디코더-only Transformer ([[Day-037]] 구조의 축소, [[Day-042]] 와 동일).
      토크나이저를 소스·타깃에 **공유** 시킨다 — 이것이 전이의 전제조건이다([[Day-041]]).

실행:  uv run python pretrain_finetune.py
       (uv 프로젝트에 torch 가 없으면) uv run --with torch python pretrain_finetune.py
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

torch.manual_seed(0)

# ── 설정 (config) ──────────────────────────────────────────────────────────
BLOCK = 64                        # 문맥 창 (context window), 글자 단위
D_MODEL, N_HEAD, N_LAYER = 128, 4, 3
DROPOUT = 0.1
BATCH = 32
WARMUP = 50                       # 워밍업 후 '상수 LR' — 짧은 런이 긴 런의 앞부분과 같아진다
PRE_STEPS = 1200                  # 사전학습 스텝
FT_STEPS = 250                    # 파인튜닝(및 동일 예산 백지학습) 스텝
LONG_STEPS = 1200                 # 대조군: 백지에서 오래 학습 (사전학습과 같은 compute)
PRE_LR, SCRATCH_LR, FT_LR = 1e-3, 1e-3, 3e-4
MIN_COUNT = 5                     # 이보다 드문 글자는 □ 로 합친다
CHUNK = 2000                      # train/val 분할 단위 (글자)
BUDGETS = [0.02, 0.10, 1.00]      # 타깃 학습데이터 예산

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


# ── 코퍼스 (corpus) ────────────────────────────────────────────────────────
def load_source():
    """소스 도메인: Day 노트의 '산문'. YAML·코드블록을 지워 코드와 겹치지 않게 한다."""
    paths = [p for p in sorted(glob.glob(os.path.join(ROOT, "Day-0*", "Day-0*.md")))
             if os.path.dirname(p) != HERE]
    out = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            s = f.read()
        s = re.sub(r"^---\n.*?\n---\n", "", s, flags=re.S)   # frontmatter 제거
        s = re.sub(r"```.*?```", "", s, flags=re.S)          # 코드블록 제거 ← 중요
        s = re.sub(r"[ \t]+", " ", s)
        s = re.sub(r"\n{2,}", "\n", s)
        out.append(s)
    return "".join(out), len(paths)


def load_target():
    """타깃 도메인: 동반 .py 파일들 (자기 폴더 제외). 파이썬 코드."""
    paths = [p for p in sorted(glob.glob(os.path.join(ROOT, "Day-0*", "*.py")))
             if os.path.dirname(p) != HERE]
    out = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            out.append(f.read())
    return "\n\n".join(out), len(paths)


SRC, N_MD = load_source()
TGT, N_PY = load_target()

FALLBACK_SRC = ("정보 검색은 질문에 답이 될 문서를 찾아 순서대로 늘어놓는 일이다.\n"
                "언어모델은 다음 토큰의 확률 분포를 내놓는 함수일 뿐이다.\n") * 900
FALLBACK_TGT = ("def score(query, doc):\n"
                "    return sum(w * doc.get(t, 0.0) for t, w in query.items())\n") * 900
if not SRC:
    print("    (Day 노트를 못 찾아 대체 소스 코퍼스를 쓴다)")
    SRC, N_MD = FALLBACK_SRC, 0
if not TGT:
    print("    (.py 를 못 찾아 대체 타깃 코퍼스를 쓴다)")
    TGT, N_PY = FALLBACK_TGT, 0

# ── 공유 토크나이저 (글자 하나 = 토큰 하나) ───────────────────────────────
# 핵심: 어휘를 **두 도메인 합집합** 으로 만든다. 소스에만 맞춘 어휘로 사전학습하면
#       타깃의 글자가 전부 UNK 이 되어 전이가 시작조차 못 한다. ([[Day-041]])
cnt = Counter(SRC) + Counter(TGT)
UNK = "□"
chars = sorted({c for c, n in cnt.items() if n >= MIN_COUNT} | {UNK})
VOCAB = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi.get(c, stoi[UNK]) for c in s]
decode = lambda ids: "".join(itos[int(i)] for i in ids)


def split_blocks(text, val_slot=7, every=10):
    """CHUNK 글자 블록으로 잘라 every 개마다 1개를 검증으로 뺀다."""
    ids = torch.tensor(encode(text), dtype=torch.long)
    blocks = [ids[i:i + CHUNK] for i in range(0, len(ids) - CHUNK, CHUNK)]
    tr = [b for i, b in enumerate(blocks) if i % every != val_slot]
    va = torch.cat([b for i, b in enumerate(blocks) if i % every == val_slot])
    return tr, va


SRC_TR_BLOCKS, SRC_VAL = split_blocks(SRC)
TGT_TR_BLOCKS, TGT_VAL = split_blocks(TGT)
SRC_TRAIN = torch.cat(SRC_TR_BLOCKS)


def budget_data(frac, seed=0):
    """타깃 학습블록 중 frac 비율만 (씨앗 고정 무작위로) 골라 잇는다."""
    idx = list(range(len(TGT_TR_BLOCKS)))
    random.Random(seed).shuffle(idx)
    k = max(1, round(len(idx) * frac))
    return torch.cat([TGT_TR_BLOCKS[i] for i in sorted(idx[:k])])


def get_batch(data, bs=BATCH, gen=None):
    ix = torch.randint(len(data) - BLOCK - 1, (bs,), generator=gen)
    x = torch.stack([data[i:i + BLOCK] for i in ix])
    y = torch.stack([data[i + 1:i + BLOCK + 1] for i in ix])
    return x, y


def fixed_eval_set(data, n=16, seed=1234):
    """평가 배치를 **고정** 한다 — 모든 실행이 똑같은 자로 재야 비교가 성립한다."""
    g = torch.Generator().manual_seed(seed)
    return [get_batch(data, gen=g) for _ in range(n)]


EVAL_SRC = fixed_eval_set(SRC_VAL)
EVAL_TGT = fixed_eval_set(TGT_VAL)


# ── 모델 (Day-037 의 미니 디코더를 축소; Day-042 와 동일 구조) ────────────
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


def set_trainable(model, mode):
    """무엇을 얼릴 것인가. 'full' | 'last' (마지막 블록+ln_f+head) | 'head' (선형 프로브)"""
    for p in model.parameters():
        p.requires_grad = (mode == "full")
    if mode == "last":
        mods = [model.blocks[-1], model.ln_f, model.head]
    elif mode == "head":
        mods = [model.head]
    else:
        return sum(p.numel() for p in model.parameters())
    for m in mods:
        for p in m.parameters():
            p.requires_grad = True
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def eval_loss(model, eval_set):
    was = model.training
    model.eval()
    tot = 0.0
    for x, y in eval_set:
        tot += F.cross_entropy(model(x).view(-1, VOCAB), y.view(-1)).item()
    model.train(was)
    return tot / len(eval_set)


def train_run(tag, model, data, steps, lr, eval_set, mode="full",
              eval_every=25, quiet=False):
    """워밍업 후 상수 LR. → 짧은 런이 긴 런의 '앞부분'과 정확히 같아진다."""
    n_tr = set_trainable(model, mode)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / WARMUP))
    curve = [(0, eval_loss(model, eval_set))]         # step 0 = zero-shot
    best, best_state, best_step = curve[0][1], None, 0
    t0 = time.time()
    for step in range(1, steps + 1):
        x, y = get_batch(data)
        loss = F.cross_entropy(model(x).view(-1, VOCAB), y.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if step % eval_every == 0 or step == steps:
            va = eval_loss(model, eval_set)
            curve.append((step, va))
            if va < best:
                best, best_step = va, step
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    if not quiet:
        print(f"    {tag:<28} val {best:.3f}  ppl {math.exp(best):>7.1f}"
              f"  @step {best_step:<5} 학습가능 {n_tr/1e3:>6.1f}K  {time.time()-t0:.0f}s")
    return {"tag": tag, "curve": curve, "best": best, "best_step": best_step,
            "n_train": n_tr, "state": {k: v.clone() for k, v in model.state_dict().items()}}


def steps_to(curve, target):
    """val 손실이 target 이하로 처음 내려간 스텝 (없으면 None)."""
    for s, v in curve:
        if v <= target:
            return s
    return None


# ── 디코딩 (Day-042 의 top-p 표집을 재사용) ───────────────────────────────
@torch.no_grad()
def generate(model, prompt, n_new=180, temp=0.8, top_p=0.9, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.tensor([encode(prompt)], dtype=torch.long)
    model.eval()
    for _ in range(n_new):
        logits = model(ids[:, -BLOCK:])[0, -1, :] / max(temp, 1e-6)
        p = F.softmax(logits, -1)
        sp, si = torch.sort(p, descending=True)
        keep = (torch.cumsum(sp, 0) - sp) < top_p
        sp = torch.where(keep, sp, torch.zeros_like(sp))
        nxt = si[torch.multinomial(sp / sp.sum(), 1, generator=g)]
        ids = torch.cat([ids, nxt.view(1, 1)], dim=1)
    return decode(ids[0])


def show(text, width=88):
    one = text.replace("\n", "⏎")
    return one[:width] + ("…" if len(one) > width else "")


# ══════════════════════════════════════════════════════════════════════════
def main():
    ppl = lambda x: math.exp(x)
    bar = "─" * 74

    # ── 실험 1: 두 도메인 ──────────────────────────────────────────────
    print(f"\n{bar}\n실험 1 — 두 도메인은 얼마나 다른가\n{bar}")
    print(f"  소스(사전학습): Day 노트 산문 {N_MD}편 · {len(SRC):,}자 "
          f"(train {len(SRC_TRAIN):,} / val {len(SRC_VAL):,})")
    print(f"  타깃(하류과제): .py 파일 {N_PY}개 · {len(TGT):,}자 "
          f"(train {sum(len(b) for b in TGT_TR_BLOCKS):,} / val {len(TGT_VAL):,})")
    print(f"  공유 어휘(글자) {VOCAB}개 — 두 도메인 합집합에서 {MIN_COUNT}회 이상 등장")

    cs, ct = Counter(SRC), Counter(TGT)
    only_s = sum(1 for c in chars if cs.get(c, 0) >= MIN_COUNT and ct.get(c, 0) == 0)
    only_t = sum(1 for c in chars if ct.get(c, 0) >= MIN_COUNT and cs.get(c, 0) == 0)
    ns, nt = sum(cs.values()), sum(ct.values())
    tvd = 0.5 * sum(abs(cs.get(c, 0) / ns - ct.get(c, 0) / nt) for c in chars)
    print(f"  글자 종류: 소스에만 {only_s}개 · 타깃에만 {only_t}개 "
          f"· 1-gram 총변동거리(TVD) {tvd:.3f}  (0=동일, 1=완전히 다름)")
    top_s = " ".join(repr(c) for c, _ in cs.most_common(8) if not c.isspace())
    top_t = " ".join(repr(c) for c, _ in ct.most_common(8) if not c.isspace())
    print(f"  최빈 글자  소스: {top_s}\n             타깃: {top_t}")

    # ── 사전학습 ───────────────────────────────────────────────────────
    print(f"\n{bar}\n사전학습 (pre-training) — 소스 도메인, 다음 글자 예측 하나로만\n{bar}")
    torch.manual_seed(0)
    pre = CharLM()
    n_par = sum(p.numel() for p in pre.parameters())
    print(f"  파라미터 {n_par/1e3:.0f}K · 어휘 {VOCAB} · 문맥 {BLOCK}자 · {PRE_STEPS}스텝")
    torch.manual_seed(1)
    rand_tgt = eval_loss(CharLM(), EVAL_TGT)
    r_pre = train_run("pretrain(소스)", pre, SRC_TRAIN, PRE_STEPS, PRE_LR,
                      EVAL_SRC, eval_every=200)
    PRE_STATE = r_pre["state"]
    src_ppl_pre = ppl(r_pre["best"])
    zero_shot = eval_loss(pre, EVAL_TGT)
    print(f"    → 소스 val ppl {src_ppl_pre:.1f} · 이 모델의 **타깃 zero-shot** "
          f"val {zero_shot:.3f} (ppl {ppl(zero_shot):.1f})")
    print(f"    기준선: 무작위 초기화 모델의 타깃 ppl {ppl(rand_tgt):.1f} "
          f"(균등분포라면 어휘 크기 {VOCAB}) — 파이썬을 한 줄도 안 봤는데 "
          f"{ppl(rand_tgt)/ppl(zero_shot):.1f}배 낫다")

    def fresh_pre():
        m = CharLM()
        m.load_state_dict(PRE_STATE)
        return m

    # ── 실험 2·3: 예산별 백지 vs 파인튜닝 ───────────────────────────────
    print(f"\n{bar}\n실험 2·3 — 같은 타깃 스텝 예산({FT_STEPS}스텝): 백지 vs 사전학습\n{bar}")
    rows = []
    for frac in BUDGETS:
        data = budget_data(frac)
        print(f"  [타깃 데이터 {frac*100:>5.0f}% = {len(data):>7,}자]")
        torch.manual_seed(1)
        r_s = train_run("  백지(scratch)", CharLM(), data, FT_STEPS,
                        SCRATCH_LR, EVAL_TGT)
        torch.manual_seed(1)
        r_f = train_run("  파인튜닝(finetune)", fresh_pre(), data, FT_STEPS,
                        FT_LR, EVAL_TGT)
        reach = steps_to(r_f["curve"], r_s["best"])
        rows.append((frac, len(data), r_s, r_f, reach))
        gain = (ppl(r_s["best"]) - ppl(r_f["best"])) / ppl(r_s["best"]) * 100
        msg = f"{reach}스텝" if reach is not None else "도달 못함"
        print(f"    → ppl {ppl(r_s['best']):.1f} → {ppl(r_f['best']):.1f} "
              f"({gain:+.0f}%) · 파인튜닝이 백지의 최종 성능에 닿는 데 {msg}"
              f" (백지는 {r_s['best_step']}스텝)")

    # 대조군: 백지에서 오래 (사전학습과 같은 compute) — 결국 따라잡는가?
    print(f"\n  [대조군] 타깃 100% 데이터로 백지에서 {LONG_STEPS}스텝 "
          f"(= 사전학습에 쓴 compute 와 동일)")
    torch.manual_seed(1)
    r_long = train_run("  백지-장기(scratch-long)", CharLM(), budget_data(1.0),
                       LONG_STEPS, SCRATCH_LR, EVAL_TGT, eval_every=100)
    r_f100 = rows[-1][3]
    print(f"    → 백지-장기 ppl {ppl(r_long['best']):.1f} vs "
          f"파인튜닝 {FT_STEPS}스텝 ppl {ppl(r_f100['best']):.1f}")

    # ── 실험 4: 학습률과 얼리기 ─────────────────────────────────────────
    print(f"\n{bar}\n실험 4 — 어떻게 옮길 것인가: 학습률(LR)과 얼리기(freezing)\n{bar}")
    data10 = budget_data(0.10)
    print(f"  타깃 10% ({len(data10):,}자) 에서 {FT_STEPS}스텝 고정, 조건만 바꾼다")
    lr_rows = []
    for lr in [3e-3, 1e-3, 3e-4, 1e-4]:
        torch.manual_seed(1)
        r = train_run(f"  full FT · lr {lr:g}", fresh_pre(), data10, FT_STEPS,
                      lr, EVAL_TGT)
        lr_rows.append((lr, r))
    fz_rows = []
    for mode, name in [("head", "head 만 (선형 프로브)"), ("last", "마지막 블록+head")]:
        torch.manual_seed(1)
        r = train_run(f"  {name}", fresh_pre(), data10, FT_STEPS, 1e-3,
                      EVAL_TGT, mode=mode)
        fz_rows.append((mode, r))

    # ── 실험 5: 파국적 망각 ─────────────────────────────────────────────
    print(f"\n{bar}\n실험 5 — 파국적 망각 (catastrophic forgetting)\n{bar}")
    print(f"  파인튜닝 전 소스 val ppl: {src_ppl_pre:.1f}")
    for frac, n, _r_s, r_f, _ in rows:
        m = CharLM()
        m.load_state_dict(r_f["state"])
        after = eval_loss(m, EVAL_SRC)
        print(f"    타깃 {frac*100:>5.0f}% 로 파인튜닝 후 → 소스 ppl "
              f"{ppl(after):>8.1f}  (사전학습 대비 {ppl(after)/src_ppl_pre:>6.1f}배)"
              f" · 타깃 ppl {ppl(r_f['best']):.1f}")
    m = CharLM()
    m.load_state_dict(fz_rows[0][1]["state"])       # head-only
    hp = ppl(eval_loss(m, EVAL_SRC))
    print(f"    head 만 학습(선형 프로브) 후     → 소스 ppl "
          f"{hp:>8.1f}  (사전학습 대비 {hp/src_ppl_pre:>6.1f}배)")
    print("    (주의: head 만 학습한 런은 lr 1e-3, full FT 런은 lr 3e-4 다 —"
          " 얼리기와 학습률이 섞인 비교다)")

    # ── 생성 비교 ───────────────────────────────────────────────────────
    print(f"\n{bar}\n생성 비교 — 프롬프트 'def ' · T=0.8 · top-p 0.9 (Day-042)\n{bar}")
    m_pre = CharLM(); m_pre.load_state_dict(PRE_STATE)
    m_ft = CharLM(); m_ft.load_state_dict(r_f100["state"])
    m_sc = CharLM(); m_sc.load_state_dict(r_long["state"])
    for name, m in [("사전학습만 (타깃 0스텝)", m_pre),
                    (f"파인튜닝 {FT_STEPS}스텝", m_ft),
                    (f"백지-장기 {LONG_STEPS}스텝", m_sc)]:
        print(f"  {name}\n    | {show(generate(m, 'def '))}")

    # ── 요약 ────────────────────────────────────────────────────────────
    print(f"\n{bar}\n요약\n{bar}")
    print(f"  {'타깃 데이터':<14}{'백지 ppl':>10}{'파인튜닝 ppl':>14}"
          f"{'개선':>8}{'따라잡기':>10}")
    for frac, n, r_s, r_f, reach in rows:
        g = (ppl(r_s["best"]) - ppl(r_f["best"])) / ppl(r_s["best"]) * 100
        rr = f"{reach}스텝" if reach is not None else "—"
        print(f"  {frac*100:>4.0f}% ({n:>7,}자){ppl(r_s['best']):>10.1f}"
              f"{ppl(r_f['best']):>14.1f}{g:>7.0f}%{rr:>10}")
    print(f"\n  결론: 사전학습은 '가중치를 미리 좋은 곳에 놓아 두는 일'이다.")
    print(f"        타깃 데이터가 적을수록 그 위치의 값어치가 커진다.")

    # 학습곡선 (실험 2 용 상세 출력)
    print(f"\n  [실험 2 학습곡선] 타깃 100% · val ppl")
    s_curve = dict(rows[-1][2]["curve"])
    f_curve = dict(rows[-1][3]["curve"])
    print(f"    {'step':>6}{'백지':>10}{'파인튜닝':>12}")
    for s in sorted(f_curve):
        a = f"{ppl(s_curve[s]):.1f}" if s in s_curve else "—"
        print(f"    {s:>6}{a:>10}{ppl(f_curve[s]):>12.1f}")


if __name__ == "__main__":
    main()
