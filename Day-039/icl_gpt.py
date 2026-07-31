"""
Day-039 — GPT 계열: 단방향을 고집해서 이긴 쪽의 이야기
(The GPT line: in-context learning emerges from scale, not from a new objective)

Day-037 의 디코더-온리 미니 Transformer 를 '목적함수 그대로' 재사용하되,
데이터를 바꿔 GPT-3 의 핵심 현상 — **문맥 내 학습(in-context learning)** —
을 장난감 규모로 재현한다. 가중치 갱신은 사전학습 때만 일어나고,
평가 때는 **프롬프트에 든 예시만으로** 새 규칙을 푼다.

  실험 1. shot 수에 따른 정확도 — 0-shot -> k-shot 곡선 (GPT-3 Fig 1.2 의 축소판)
          + 이론적 상한(Bayes)과 '검색만 하는' 베이스라인을 함께 계산해 대조
  실험 2. 규모(깊이/폭) 스윕 — 어느 크기에서 문맥 내 학습이 '켜지는가'
  실험 3. 학습된 attention — 유도 헤드(induction head)를 눈으로 찾기
  실험 4. 분포 이탈(OOD) — 전단사가 아닌 사상을 주면 무엇이 무너지는가

과제(사상 추론 / mapping-induction task):
    매 시퀀스마다 8개 기호 위의 **무작위 전단사(bijection) pi** 를 새로 뽑는다.
        seq = [x1 pi(x1) x2 pi(x2) ... x8 pi(x8)]      (길이 16)
    모델은 짝수 위치에서 다음 토큰(= 그 기호의 상)을 예측한다.
    pi 가 매번 달라지므로 **가중치에 답을 저장할 수 없다** —
    앞쪽 짝들을 문맥에서 읽어야만 뒤쪽을 맞힐 수 있다.
    즉 정확도의 상승분 전체가 '문맥 내 학습'이다.

실행:  uv run python icl_gpt.py
       (uv 프로젝트에 torch 가 없으면) uv run --with torch python icl_gpt.py
한글이 깨지면 먼저:  $env:PYTHONIOENCODING="utf-8"
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# ── 설정 (config) ──────────────────────────────────────────────────────────
N_SYM = 8                 # 기호 id 0..7 (특수 토큰 없음)
VOCAB = N_SYM
N_PAIR = 8                # 한 시퀀스에 (질의, 답) 짝 8개
BLOCK = 2 * N_PAIR        # 시퀀스 길이 16

D_MODEL, N_HEAD, N_LAYER = 64, 2, 2
STEPS, BATCH, LR = 6000, 128, 3e-4
SWEEP_STEPS = 6000        # 실험 2 는 모든 구성에 '같은 예산'을 준다
IGNORE = -1


# ── 데이터 (data) ─────────────────────────────────────────────────────────
def make_batch(bs, bijective=True):
    """
    매 행마다 새로운 사상 pi 를 뽑아 [x1 pi(x1) x2 pi(x2) ...] 를 만든다.

    bijective=True  : pi 는 전단사(무작위 순열)  -> 학습·기본 평가
    bijective=False : pi 는 임의 함수(충돌 허용) -> 실험 4 의 분포 이탈 조건
    """
    if bijective:
        pi = torch.argsort(torch.rand(bs, N_SYM), dim=1)      # (B, 8) 순열
    else:
        pi = torch.randint(0, N_SYM, (bs, N_SYM))             # 충돌 허용
    xs = torch.randint(0, N_SYM, (bs, N_PAIR))                # 질의는 iid — 중복 가능
    ys = torch.gather(pi, 1, xs)                              # ys[b,i] = pi[b][xs[b,i]]

    seq = torch.empty(bs, BLOCK, dtype=torch.long)
    seq[:, 0::2] = xs                                         # 짝수 위치 = 질의
    seq[:, 1::2] = ys                                         # 홀수 위치 = 답
    return seq, xs, ys


def targets_of(seq):
    """짝수 위치에서만 손실을 건다: 위치 2i 의 로짓이 y_i(위치 2i+1)를 맞혀야 한다."""
    tgt = torch.full_like(seq, IGNORE)
    tgt[:, 0::2] = seq[:, 1::2]
    return tgt


# ── 모델 (model) — Day-037 의 디코더 그대로 (causal 유지) ──────────────────
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


class MiniGPT(nn.Module):
    def __init__(self, d_model=D_MODEL, n_head=N_HEAD, n_layer=N_LAYER):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, d_model)
        self.pos = nn.Embedding(BLOCK, d_model)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, VOCAB, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))


# ── 학습 (train) ──────────────────────────────────────────────────────────
def train(d_model=D_MODEL, n_head=N_HEAD, n_layer=N_LAYER, steps=STEPS, log=False):
    model = MiniGPT(d_model, n_head, n_layer)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    for step in range(1, steps + 1):
        seq, _, _ = make_batch(BATCH)
        loss = F.cross_entropy(
            model(seq).reshape(-1, VOCAB), targets_of(seq).reshape(-1),
            ignore_index=IGNORE)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if log and (step == 1 or step % 250 == 0):
            print(f"    step {step:5d}:  loss={loss.item():.4f}")
    return model, loss.item()


# ── 이론 상한 (analysis) ──────────────────────────────────────────────────
def ceilings(xs):
    """
    짝 i 에서 **정보이론적으로 가능한 최대 정확도** 세 가지를 표본별로 계산한다.
    (d = 지금까지 등장한 서로 다른 질의 기호 수. 전단사이므로 '이미 쓰인 상' d 개는
     다른 기호의 답이 될 수 없다.)

      bayes    : 문맥을 전부 활용 — 검색 + 배제 추론
                 x_i 를 이미 봤으면 확실히 맞힘(1.0),
                 처음 보는 기호면 '아직 쓰이지 않은 상' 중 균등 추측 -> 1/(8-d)
      retrieval: 검색만 하고 배제는 못함 — 봤으면 1.0, 못 봤으면 1/8
      no_comp  : **attention 을 한 번만 쓸 수 있는(=1층) 모델의 구조적 상한.**
                 "x_i 가 앞에 나왔는가"(짝수 위치와의 내용 일치)와
                 "어떤 상들이 이미 쓰였는가"(홀수 위치의 집합)는 1층으로도 알 수 있다.
                 그러나 "x_i 와 짝지어진 상이 무엇인가"는 알 수 없다 —
                 그건 attention 두 번의 합성(composition)이 필요하다.
                 따라서 봤으면 '쓰인 상' d 개 중 균등 -> 1/d,
                        못 봤으면 '안 쓰인 상' (8-d) 개 중 균등 -> 1/(8-d)
    """
    n = xs.size(0)
    bayes, retr, nocomp = [], [], []
    for i in range(N_PAIR):
        if i == 0:
            bayes.append(1.0 / N_SYM)
            retr.append(1.0 / N_SYM)
            nocomp.append(1.0 / N_SYM)
            continue
        prev = xs[:, :i]                                   # (n, i)
        seen = (prev == xs[:, i:i + 1]).any(dim=1)         # x_i 를 본 적 있나
        onehot = torch.zeros(n, N_SYM)
        onehot.scatter_(1, prev, 1.0)
        d = onehot.sum(dim=1)                              # 서로 다른 기호 수
        p_b = torch.where(seen, torch.ones(n), 1.0 / (N_SYM - d))
        p_r = torch.where(seen, torch.ones(n), torch.full((n,), 1.0 / N_SYM))
        p_n = torch.where(seen, 1.0 / d, 1.0 / (N_SYM - d))
        bayes.append(p_b.mean().item())
        retr.append(p_r.mean().item())
        nocomp.append(p_n.mean().item())
    return bayes, retr, nocomp


def loss_floors(xs):
    """
    같은 두 전략의 **교차엔트로피 손실 하한** — 학습 곡선을 읽는 '눈금'이 된다.
    (정확도가 아니라 손실을 보는 이유: 학습 중 우리가 관찰하는 값이 손실이기 때문)

      random : 항상 균등분포 -> ln 8 = 2.079
      nocomp : 1층 상한. 봤으면 '쓰인 상 d 개'에 균등 -> ln d,
               못 봤으면 '안 쓰인 상'에 균등 -> ln(8-d)
      bayes  : 봤으면 정답에 확률 1 -> 0, 못 봤으면 ln(8-d)
    """
    n = xs.size(0)
    fb, fn = [], []
    for i in range(N_PAIR):
        if i == 0:
            fb.append(math.log(N_SYM))
            fn.append(math.log(N_SYM))
            continue
        prev = xs[:, :i]
        seen = (prev == xs[:, i:i + 1]).any(dim=1)
        onehot = torch.zeros(n, N_SYM)
        onehot.scatter_(1, prev, 1.0)
        d = onehot.sum(dim=1)
        rest = torch.log(N_SYM - d)                        # ln(8-d)
        fb.append(torch.where(seen, torch.zeros(n), rest).mean().item())
        fn.append(torch.where(seen, torch.log(d), rest).mean().item())
    return (sum(fb) / N_PAIR, sum(fn) / N_PAIR, math.log(N_SYM))


def ceilings_seen_unseen(xs):
    """
    acc_seen_unseen 과 **같은 방식**(짝 1..7 을 이어 붙여 집계)으로 상한을 낸다.
    반환: (Bayes-본 기호, Bayes-첫 기호, 1층상한-본 기호, 1층상한-첫 기호)
    """
    n = xs.size(0)
    bs, bu, ns, nu = [], [], [], []
    for i in range(1, N_PAIR):
        prev = xs[:, :i]
        seen = (prev == xs[:, i:i + 1]).any(dim=1)
        onehot = torch.zeros(n, N_SYM)
        onehot.scatter_(1, prev, 1.0)
        d = onehot.sum(dim=1)
        bs.append(torch.ones(int(seen.sum())))             # 검색 성공 -> 1.0
        bu.append(1.0 / (N_SYM - d[~seen]))                # 배제 후 균등
        ns.append(1.0 / d[seen])                           # 검색 불가 -> 쓰인 상 중 균등
        nu.append(1.0 / (N_SYM - d[~seen]))
    return tuple(torch.cat(x).mean().item() for x in (bs, bu, ns, nu))


@torch.no_grad()
def acc_by_shot(model, seq, xs, ys):
    """짝 i 의 정확도 = 위치 2i 의 argmax 가 y_i 인 비율. i = 앞서 본 예시(shot) 수."""
    pred = model(seq).argmax(-1)
    return [(pred[:, 2 * i] == ys[:, i]).float().mean().item()
            for i in range(N_PAIR)]


@torch.no_grad()
def acc_seen_unseen(model, seq, xs, ys):
    """'질의 기호를 문맥에서 이미 봤는가'로 쪼갠 정확도 (짝 1..7 만 집계)."""
    pred = model(seq).argmax(-1)
    ok_seen, ok_unseen = [], []
    for i in range(1, N_PAIR):
        seen = (xs[:, :i] == xs[:, i:i + 1]).any(dim=1)
        ok = (pred[:, 2 * i] == ys[:, i]).float()
        ok_seen.append(ok[seen])
        ok_unseen.append(ok[~seen])
    return (torch.cat(ok_seen).mean().item(),
            torch.cat(ok_unseen).mean().item())


# ── 메인 (main) ───────────────────────────────────────────────────────────
def main():
    print("=" * 78)
    print("[과제] 매 시퀀스마다 새 사상 pi 를 뽑아 [x1 pi(x1) x2 pi(x2) ...] 를 만든다")
    seq0, xs0, ys0 = make_batch(1)
    print(f"  seq  = {seq0[0].tolist()}")
    print(f"  질의 = {xs0[0].tolist()}   (짝수 위치)")
    print(f"  답   = {ys0[0].tolist()}   (홀수 위치, pi 는 이 시퀀스에서만 유효)")
    n_par = sum(p.numel() for p in MiniGPT().parameters())
    print(f"[모델] MiniGPT d_model={D_MODEL}, heads={N_HEAD}, layers={N_LAYER}, "
          f"파라미터 {n_par:,}개  (Day-037 과 같은 디코더-온리, 목적함수도 그대로)")

    # ── 실험 1: shot 수 곡선 ──────────────────────────────────────────
    print("\n" + "=" * 78)
    print("[실험 1] shot 수에 따른 정확도 — 프롬프트의 예시가 늘면 무엇이 일어나는가")
    seq, xs, ys = make_batch(4000)                         # 평가셋을 먼저 뽑아 둔다
    f_bayes, f_nocomp, f_rand = loss_floors(xs)
    print(f"  [학습 곡선을 읽는 눈금 — 손실 하한을 미리 계산해 둔다]")
    print(f"    무작위(문맥 무시)     = {f_rand:.3f} = ln 8")
    print(f"    1층 상한 (배제만)     = {f_nocomp:.3f}")
    print(f"    Bayes 최적 (검색+배제)= {f_bayes:.3f}")
    gpt, loss = train(log=True)
    print(f"  최종 학습손실 = {loss:.4f}")

    model_acc = acc_by_shot(gpt, seq, xs, ys)
    bayes, retr, nocomp = ceilings(xs)

    print("\n  shot 수(앞서 본 예시)   " + "".join(f"{i:>7d}" for i in range(N_PAIR)))
    print("  모델 정확도             " + "".join(f"{a:>7.3f}" for a in model_acc))
    print("  Bayes 상한 (검색+배제)  " + "".join(f"{a:>7.3f}" for a in bayes))
    print("  검색만 (배제 못함)      " + "".join(f"{a:>7.3f}" for a in retr))
    print("  1층 상한 (검색 불가)    " + "".join(f"{a:>7.3f}" for a in nocomp))
    print(f"  무작위 추측 기준선      {1 / N_SYM:.3f} (모든 shot 에서 동일)")

    s, u = acc_seen_unseen(gpt, seq, xs, ys)
    cbs, cbu, cns, cnu = ceilings_seen_unseen(xs)
    print(f"\n  [정확도를 두 갈래로 쪼개기 — 짝 1~7 집계]")
    print(f"    질의 기호를 문맥에서 이미 본 경우 : {s:.3f}   "
          f"(Bayes {cbs:.3f} / 1층 상한 {cns:.3f})  <- '검색'이 되는가")
    print(f"    질의 기호를 처음 보는 경우        : {u:.3f}   "
          f"(Bayes {cbu:.3f} / 무작위 {1 / N_SYM:.3f})  <- '배제'가 되는가")

    # ── 실험 2: 규모 스윕 ────────────────────────────────────────────
    print("\n" + "=" * 78)
    print(f"[실험 2] 규모 스윕 — 문맥 내 학습은 어느 크기에서 '켜지는가' "
          f"(모두 {SWEEP_STEPS} 스텝 동일 예산)")
    print(f"  {'구성':<20s}{'파라미터':>10s}{'손실':>8s}"
          f"{'7-shot':>8s}{'평균':>8s}{'본 기호':>9s}{'첫 기호':>9s}")
    sweep = [
        ("layers=1, d=64", dict(n_layer=1, d_model=64)),
        ("layers=2, d=16", dict(n_layer=2, d_model=16)),
        ("layers=2, d=64", dict(n_layer=2, d_model=64)),
        ("layers=3, d=64", dict(n_layer=3, d_model=64)),
    ]
    for label, kw in sweep:
        m, l = train(steps=SWEEP_STEPS, **kw)
        a = acc_by_shot(m, seq, xs, ys)
        ss, uu = acc_seen_unseen(m, seq, xs, ys)
        npar = sum(p.numel() for p in m.parameters())
        print(f"  {label:<20s}{npar:>10,d}{l:>8.3f}"
              f"{a[-1]:>8.3f}{sum(a) / len(a):>8.3f}{ss:>9.3f}{uu:>9.3f}")
    cbs, cbu, cns, cnu = ceilings_seen_unseen(xs)
    print(f"  {'Bayes 최적':<18s}{'':>10s}{f_bayes:>8.3f}{bayes[-1]:>8.3f}"
          f"{sum(bayes) / len(bayes):>8.3f}{cbs:>9.3f}{cbu:>9.3f}"
          f"   <- 2층 이상이면 도달 가능")
    print(f"  {'1층 구조적 상한':<14s}{'':>10s}{f_nocomp:>8.3f}{nocomp[-1]:>8.3f}"
          f"{sum(nocomp) / len(nocomp):>8.3f}{cns:>9.3f}{cnu:>9.3f}"
          f"   <- 1층이 넘을 수 없는 선")

    # ── 실험 3: 유도 헤드 찾기 ───────────────────────────────────────
    print("\n" + "=" * 78)
    print("[실험 3] 학습된 attention — 유도 헤드(induction head)를 눈으로 찾기")
    # 마지막 질의(위치 14)의 기호가 앞에서 딱 한 번 나왔던 예시를 찾는다.
    seqE, xsE, ysE = make_batch(3000)
    hit = None
    for b in range(seqE.size(0)):
        occ = (xsE[b, :7] == xsE[b, 7]).nonzero().flatten()
        if occ.numel() == 1:
            hit = (b, occ.item())
            break
    b, j = hit
    _ = gpt(seqE[b:b + 1])
    print(f"  seq = {seqE[b].tolist()}")
    print(f"  마지막 질의: 위치 14 의 기호 {xsE[b, 7].item()} — "
          f"같은 기호가 짝 {j}(위치 {2 * j})에 있었고 그 답은 "
          f"위치 {2 * j + 1} 의 {ysE[b, j].item()}  (정답 = {ysE[b, 7].item()})")

    print(f"\n  (1) 유도 회로의 앞단 — 정답이 놓인 자리(위치 {2 * j + 1})가")
    print(f"      자기 질의(위치 {2 * j})를 보는가 = '직전 토큰 헤드'")
    for li, blk in enumerate(gpt.blocks):
        att = blk.attn.last_attn[0]
        for h in range(att.size(0)):
            row = att[h, 2 * j + 1]
            mark = " <== 직전 토큰 헤드" if row.argmax().item() == 2 * j else ""
            print(f"      layer{li} head{h} q{2 * j + 1}: "
                  f"최대 k{row.argmax().item():<2d}({row.max():.2f})   "
                  f"k{2 * j}={row[2 * j]:.2f}{mark}")

    print("\n  (2) 유도 회로의 뒷단 — 마지막 질의(위치 14)의 attention 전체 행")
    print("      " + "".join(f"{'k'+str(t):>6s}" for t in range(0, 15)))
    for li, blk in enumerate(gpt.blocks):
        att = blk.attn.last_attn[0]                        # (h, T, T)
        for h in range(att.size(0)):
            row = att[h, 14]
            mark = "  <== 유도 헤드" if row.argmax().item() == 2 * j + 1 else ""
            print(f"  L{li}h{h} " + "".join(f"{v:>6.2f}" for v in row[:15].tolist())
                  + mark)
    print(f"  (찾는 신호: 같은 기호가 나온 자리(k{2 * j}) 의 '바로 다음 칸'"
          f"(k{2 * j + 1}) 을 보는 헤드)")

    # ── 실험 4: 분포 이탈 ───────────────────────────────────────────
    print("\n" + "=" * 78)
    print("[실험 4] 분포 이탈(OOD) — 전단사가 아닌 사상을 주면 무엇이 무너지는가")
    seqO, xsO, ysO = make_batch(4000, bijective=False)
    accO = acc_by_shot(gpt, seqO, xsO, ysO)
    sO, uO = acc_seen_unseen(gpt, seqO, xsO, ysO)
    print("  shot 수                 " + "".join(f"{i:>7d}" for i in range(N_PAIR)))
    print("  전단사 (학습 분포)      " + "".join(f"{a:>7.3f}" for a in model_acc))
    print("  임의 함수 (분포 이탈)   " + "".join(f"{a:>7.3f}" for a in accO))
    print(f"\n  이미 본 기호  : 전단사 {s:.3f} -> 임의 함수 {sO:.3f}   (검색은 사상 종류와 무관)")
    print(f"  처음 보는 기호: 전단사 {u:.3f} -> 임의 함수 {uO:.3f}   "
          f"(배제 추론은 전단사 '가정'에 의존)")
    print("=" * 78)


if __name__ == "__main__":
    main()
