"""
Day-035 — 📄 정독: Attention Is All You Need (Vaswani et al., 2017)
논문의 핵심 주장 2가지를 '학습 없이' 부분 재현(partial reproduction)한다.

  1부. Sinusoidal Positional Encoding (논문 3.5절, 식 그대로)
       PE(pos, 2i)   = sin( pos / 10000^(2i/d_model) )
       PE(pos, 2i+1) = cos( pos / 10000^(2i/d_model) )
       - 값이 [-1, 1] 로 유계인가?
       - "상대 위치를 선형변환으로 표현" 주장 확인 (거리 offset 에 따른 유사도 감소)

  2부. 왜 sqrt(d_k) 로 나누는가 (논문 3.2.1절 각주 4)
       "큰 d_k 에서 내적이 커져 softmax 가 기울기가 아주 작은 영역으로 밀린다."
       - 스케일링 전/후 내적의 분산을 실측 (이론값 d_k 와 비교)
       - 스케일링 없으면 softmax 가 near one-hot 으로 포화됨을 실측

  3부. Base 모델 설정 재현 (논문 Table 3, base row)
       d_model=512, h=8, d_ff=2048, N=6 → d_k=d_v=64, 어텐션/FFN 파라미터 수.

실행:  uv run python attention_paper_reproduce.py
의존:  numpy (Day-002 uv 프로젝트, 이미 설치)  —  `uv add numpy` (필요 시)
재현:  np.random.seed(0) 로 고정.
"""

import numpy as np

np.random.seed(0)
np.set_printoptions(precision=3, suppress=True)


# ────────────────────────────────────────────────────────────────
# 1부. Sinusoidal Positional Encoding (논문 식 그대로)
# ────────────────────────────────────────────────────────────────
def positional_encoding(max_len: int, d_model: int) -> np.ndarray:
    """논문 3.5절의 식을 그대로 구현. 반환 shape = (max_len, d_model)."""
    pos = np.arange(max_len)[:, None]              # (max_len, 1)
    i = np.arange(d_model)[None, :]                # (1, d_model)
    # 10000^(2i/d_model) — 각 차원 i 의 "파장(wavelength)"
    div = np.power(10000.0, (2 * (i // 2)) / d_model)
    angle = pos / div                              # (max_len, d_model)
    pe = np.zeros((max_len, d_model))
    pe[:, 0::2] = np.sin(angle[:, 0::2])           # 짝수 차원 = sin
    pe[:, 1::2] = np.cos(angle[:, 1::2])           # 홀수 차원 = cos
    return pe


def part1_positional_encoding():
    print("=" * 60)
    print("1부. Sinusoidal Positional Encoding (논문 3.5절)")
    print("=" * 60)
    d_model = 16
    pe = positional_encoding(max_len=50, d_model=d_model)

    print(f"  PE shape = {pe.shape}  (max_len=50, d_model={d_model})")
    print(f"  값 범위: min={pe.min():.3f}, max={pe.max():.3f}  → [-1, 1] 유계 확인")

    # (a) pos=0 은 sin(0)=0, cos(0)=1 이 번갈아 → [0,1,0,1,...]
    print(f"  pos=0 의 인코딩(앞 8차원) = {pe[0, :8]}")

    # (b) "상대 위치는 선형변환으로 표현 가능"(논문) — 인접 위치의 유사도가
    #     거리(offset)에만 의존하는지 정규화 내적으로 확인.
    print("\n  [상대 위치 성질] pos=10 과 pos=10+k 의 정규화 내적:")
    base = pe[10] / np.linalg.norm(pe[10])
    for k in [0, 1, 2, 5, 10, 20]:
        other = pe[10 + k] / np.linalg.norm(pe[10 + k])
        print(f"    k={k:2d}: cos유사도 = {base @ other:+.3f}")
    print("  → 가까울수록 유사도가 크게 감소 (거리 정보 인코딩). 단 여러 주파수가")
    print("    겹쳐 멀리서는 완벽히 단조롭진 않다 — d_model 이 작을수록 더 두드러짐.")


# ────────────────────────────────────────────────────────────────
# 2부. 왜 sqrt(d_k) 로 나누는가 (논문 3.2.1절 각주)
# ────────────────────────────────────────────────────────────────
def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def part2_scaling():
    print("\n" + "=" * 60)
    print("2부. 왜 1/sqrt(d_k) 스케일링인가 (논문 3.2.1 각주)")
    print("=" * 60)
    print("  주장: q,k 성분이 평균0·분산1 이면 내적 q·k 의 분산은 d_k.")
    print("        d_k 가 크면 점수가 커져 softmax 가 포화(기울기~=0)된다.\n")

    for d_k in [8, 64, 512]:
        # 평균0, 분산1 인 무작위 q, k 를 많이 뽑아 내적 분포를 관찰
        q = np.random.randn(20000, d_k)
        k = np.random.randn(20000, d_k)
        dot = np.sum(q * k, axis=1)                 # 내적 q·k
        scaled = dot / np.sqrt(d_k)                 # 스케일링 후
        print(f"  d_k={d_k:3d}:  내적 분산={dot.var():7.1f} (이론값 {d_k})"
              f"   |   스케일 후 분산={scaled.var():.3f} (~=1)")

    # softmax 포화 실측: 표준편차 ~sqrt(d_k) 크기의 점수에서, 스케일 유무로
    # 최대 확률이 얼마나 one-hot 에 가까워지는지.
    print("\n  [softmax 포화] 점수 벡터 = 표준정규 * sqrt(d_k), d_k=64:")
    d_k = 64
    scores_raw = np.random.randn(64) * np.sqrt(d_k)   # 스케일 안 한 점수 크기
    p_unscaled = softmax(scores_raw)
    p_scaled = softmax(scores_raw / np.sqrt(d_k))
    print(f"    스케일 없음: max prob = {p_unscaled.max():.3f}  (1.0 에 가까움 = 포화)")
    print(f"    스케일 있음: max prob = {p_scaled.max():.3f}  (완만한 분포 = 기울기 살아있음)")
    print("  → 스케일이 없으면 softmax 가 near one-hot 으로 굳어 학습 기울기가 사라진다.")


# ────────────────────────────────────────────────────────────────
# 3부. Base 모델 설정 재현 (논문 Table 3, base row)
# ────────────────────────────────────────────────────────────────
def part3_base_config():
    print("\n" + "=" * 60)
    print("3부. Transformer 'base' 설정 (논문 Table 3)")
    print("=" * 60)
    d_model, h, d_ff, N = 512, 8, 2048, 6
    d_k = d_v = d_model // h
    print(f"  d_model={d_model}, heads h={h}, d_ff={d_ff}, N(층)={N}")
    print(f"  → d_k = d_v = d_model/h = {d_k}  (헤드당 차원)")

    # 멀티헤드 어텐션 한 층의 투영 파라미터: W_Q,W_K,W_V,W_O (bias 무시)
    mha = 4 * d_model * d_model
    # position-wise FFN 한 층: (d_model→d_ff) + (d_ff→d_model)
    ffn = 2 * d_model * d_ff
    print(f"  MHA 투영 파라미터/층 = 4·d_model^2 = {mha:,}")
    print(f"  FFN 파라미터/층      = 2·d_model·d_ff = {ffn:,}")
    print(f"  인코더 한 층 합(대략) = {mha + ffn:,}")
    print(f"  논문 base 총 파라미터 ~= 65M (임베딩·6+6층·LN 등 포함).")


if __name__ == "__main__":
    part1_positional_encoding()
    part2_scaling()
    part3_base_config()
    print("\n[요약] PE는 학습 없이 위치를 유계 신호로 주입하고, 1/sqrt(d_k) 는")
    print("       내적 분산을 1로 되돌려 softmax 포화를 막는다 — 논문의 두 설계 결정.")
