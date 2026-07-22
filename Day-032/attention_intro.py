"""
Day-032 — Attention 메커니즘 (from RNN bottleneck to "look at all positions")
================================================================================
학습 없이(시드 고정) Attention 의 핵심을 숫자로 관측하는 스크립트.

세 파트:
  1부. Scaled Dot-Product Attention 을 손으로 구현 (softmax(QKᵀ/√d)·V)
  2부. "정렬(alignment)" 관측 — 디코더가 어느 인코더 위치에 주의를 두는지 가중치 확인
  3부. 왜 √d 로 나누나 — 스케일링 없으면 softmax 가 포화(saturate)되어 기울기가 죽는다

실행:  uv run python attention_intro.py
(Windows 콘솔 한글 깨짐 시:  $env:PYTHONIOENCODING="utf-8")
필요 패키지: torch  (Day-028 uv 프로젝트에 이미 설치됨)
"""

import torch
import torch.nn.functional as F

torch.manual_seed(0)


# ──────────────────────────────────────────────────────────────────────────
# 1부. Scaled Dot-Product Attention — 손으로 한 줄씩
# ──────────────────────────────────────────────────────────────────────────
def scaled_dot_product_attention(Q, K, V):
    """
    Q: (Lq, d)  질의(query)   — "나는 무엇을 찾고 있나"
    K: (Lk, d)  키(key)       — "각 위치는 무엇에 대한 것인가"
    V: (Lk, d)  값(value)     — "각 위치가 실제로 담은 내용"
    반환: (출력 (Lq, d), 주의가중치 attn (Lq, Lk))
    """
    d = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / (d ** 0.5)   # (Lq, Lk)  질의·키 유사도, √d 스케일
    attn = F.softmax(scores, dim=-1)                # 각 질의마다 키들에 대한 확률분포(합=1)
    out = attn @ V                                  # 값들의 '주의 가중 평균'
    return out, attn


print("=" * 70)
print("1부. Scaled Dot-Product Attention — softmax(QKᵀ/√d)·V")
print("=" * 70)

Lk, d = 4, 8          # 인코더 위치 4개, 차원 8
K = torch.randn(Lk, d)
V = torch.randn(Lk, d)
q = torch.randn(1, d)  # 질의 1개

out, attn = scaled_dot_product_attention(q, K, V)
print(f"  질의 1개 → 키 {Lk}개에 대한 주의 가중치(softmax, 합=1):")
print("   ", [round(a, 3) for a in attn[0].tolist()], " 합 =", round(attn.sum().item(), 4))
print(f"  출력 = 이 가중치로 값 V 를 섞은 벡터, shape={tuple(out.shape)}")
print("  → RNN 처럼 '하나의 벡터로 압축'하지 않고, 관련 위치를 '직접' 골라 끌어온다.\n")


# ──────────────────────────────────────────────────────────────────────────
# 2부. 정렬(alignment) 관측 — '비슷한 것에 주의가 쏠린다'
# ──────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("2부. 정렬(alignment) — 질의와 '닮은' 키에 주의가 집중된다")
print("=" * 70)

# 4개의 서로 구별되는 키/값을 직교에 가깝게 세팅 (원-핫 유사)
d2 = 4
K2 = torch.eye(d2)                      # 각 위치가 하나의 축을 대표 (4개 '단어')
V2 = torch.tensor([[10., 0, 0, 0],      # 위치0의 '내용'
                   [0, 20., 0, 0],      # 위치1
                   [0, 0, 30., 0],      # 위치2
                   [0, 0, 0, 40.]])     # 위치3
words = ["I", "love", "cats", "."]

# 질의가 위치2("cats")를 강하게 가리키도록 설정
q2 = torch.tensor([[0.2, 0.2, 5.0, 0.2]])   # 3번째 축이 큼 → 위치2 와 내적이 큼
out2, attn2 = scaled_dot_product_attention(q2, K2, V2)
print("  질의가 3번째 위치(cats)를 가리킬 때 주의 가중치:")
for w, a in zip(words, attn2[0].tolist()):
    bar = "█" * int(a * 40)
    print(f"    {w:<5} {a:5.3f} {bar}")
print(f"  → 출력은 거의 V[2]='cats'의 내용({V2[2].tolist()})에 수렴:",
      [round(x, 2) for x in out2[0].tolist()])
print("  이것이 seq2seq 번역에서 '이 출력 단어는 저 입력 단어를 본다'는 정렬이다.\n")


# ──────────────────────────────────────────────────────────────────────────
# 3부. 왜 √d 로 나누나 — 스케일링 없으면 softmax 포화
# ──────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("3부. 왜 √d 로 나누나 — 큰 차원에서 점수가 커져 softmax 가 포화된다")
print("=" * 70)

def entropy(p):
    """주의 분포의 엔트로피(bits). 클수록 '고르게 분산', 작을수록 '한 곳에 쏠림'."""
    return float(-(p * (p.clamp_min(1e-12)).log2()).sum())

for dim in [8, 64, 512]:
    Qd = torch.randn(1, dim)
    Kd = torch.randn(20, dim)
    raw = F.softmax(Qd @ Kd.T, dim=-1)              # 스케일링 없음
    scaled = F.softmax(Qd @ Kd.T / dim ** 0.5, dim=-1)  # √d 스케일링
    print(f"  d={dim:4d} | 스케일 X 엔트로피 {entropy(raw):5.2f} bits "
          f"(max p={raw.max():.3f}) | 스케일 O 엔트로피 {entropy(scaled):5.2f} bits "
          f"(max p={scaled.max():.3f})")

print("""
  해석: 차원 d 가 커지면 내적 QKᵀ 의 분산이 d 에 비례해 커진다.
  스케일링을 안 하면 한 점수만 매우 커져 softmax 가 '거의 원-핫'으로 포화(엔트로피↓)되고,
  그 지점의 기울기가 사실상 0 이 되어 학습이 멈춘다. √d 로 나누면 분산이 1 근처로
  돌아와 분포가 부드럽게 유지된다 — 그래서 'Scaled' dot-product attention 이다.
""")

print("끝. 핵심: Attention = softmax(QKᵀ/√d)·V — 모든 위치를 한 번에 보고,")
print("     질의와 닮은 키의 값을 가중 평균으로 끌어온다. 다음 편: Self-Attention & Transformer.")
