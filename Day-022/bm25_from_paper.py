# bm25_from_paper.py — Day-022 (📄 논문 정독)
# Robertson & Zaragoza (2009), "The Probabilistic Relevance Framework: BM25 and Beyond"
# BM25 공식의 '유도 사슬'을 부분 재현한다 — 표준 라이브러리(math)만 사용.
# 실행: uv run python bm25_from_paper.py
import sys, math
sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔 한글 깨짐 방지


# ──────────────────────────────────────────────────────────────────────
# (1) RSJ 관련성 가중치 (Robertson–Spärck Jones weight)
#     이항 독립 모델(BIM)에서 '적합/비적합' 로그오즈로 유도되는 항별 가중치.
#     일반형(적합 정보 R개 문서 중 r개가 해당 term 포함):
#       w = log[(r+0.5)/(R-r+0.5)] - log[(n-r+0.5)/(N-n-R+r+0.5)]
#     N=전체 문서 수, n=term 을 가진 문서 수(=df).
# ──────────────────────────────────────────────────────────────────────
def rsj_weight(N, n, R=0, r=0):
    num = (r + 0.5) / (R - r + 0.5)
    den = (n - r + 0.5) / (N - n - R + r + 0.5)
    return math.log(num / den)


# 적합 정보가 전혀 없을 때(R=r=0) → 고전적 IDF 형태로 '자동' 환원된다.
def idf_no_relevance(N, n):
    return math.log((N - n + 0.5) / (n + 0.5))


# 우리 엔진(Day-021)이 쓴 '항상 양수' 변형 (Lucene 스타일: +1 로 음수 방지)
def idf_lucene(N, n):
    return math.log(1 + (N - n + 0.5) / (n + 0.5))


# ──────────────────────────────────────────────────────────────────────
# (2) tf 포화 + 문서길이 정규화 (2-Poisson '엘리트니스' 모델의 간단 근사)
#     포화 계수 S = (k1+1)*tf / (k1*B + tf),   B = (1-b) + b*(dl/avgdl)
#       k1: tf 포화 속도 (k1=0 → 이진 BIM, k1→∞ → 선형 tf)
#       b : 길이 정규화 세기 (b=0 정규화 없음, b=1 완전 정규화)
# ──────────────────────────────────────────────────────────────────────
def length_norm_B(b, dl, avgdl):
    return (1 - b) + b * (dl / avgdl)


def saturation(tf, k1, b, dl, avgdl):
    if tf <= 0:
        return 0.0
    B = length_norm_B(b, dl, avgdl)
    return tf * (k1 + 1) / (k1 * B + tf)


# ──────────────────────────────────────────────────────────────────────
# (3) 완성형 BM25 항 점수 = RSJ가중치 × 포화계수
# ──────────────────────────────────────────────────────────────────────
def bm25_term(N, n, tf, dl, avgdl, k1=1.5, b=0.75, lucene=True):
    idf = idf_lucene(N, n) if lucene else idf_no_relevance(N, n)
    return idf * saturation(tf, k1, b, dl, avgdl)


def rule():
    print("=" * 70)


def demo1_rsj_reduces_to_idf():
    print("\n=== (1) RSJ 가중치는 '적합 정보가 없으면' IDF로 환원된다 ===")
    print("  N=12 (Day-021 코퍼스 크기). n=df.  RSJ(R=r=0) == 고전 IDF 임을 확인")
    print(f"  {'n(df)':>5} | {'RSJ(R=0)':>10} {'IDF(classic)':>13} {'IDF(Lucene)':>12}")
    print("  " + "-" * 46)
    for n in (1, 4, 6, 10):
        w = rsj_weight(12, n, R=0, r=0)
        i_c = idf_no_relevance(12, n)
        i_l = idf_lucene(12, n)
        print(f"  {n:>5} | {w:>10.3f} {i_c:>13.3f} {i_l:>12.3f}")
    print("  → 고전 IDF는 매우 흔한 term(n=10)에서 '음수'가 된다(스팸 억제).")
    print("    Lucene 변형은 +1 로 음수를 막아 우리 bm25_engine._idf 와 일치.")


def demo2_relevance_feedback():
    print("\n=== (2) 적합 피드백(relevance feedback)이 가중치를 끌어올린다 ===")
    print("  N=12, n(df)=4 인 term. 적합 문서 R개 중 r개가 이 term 포함 → RSJ:")
    print(f"  {'R':>2} {'r':>2} | {'RSJ weight':>11}  해석")
    print("  " + "-" * 42)
    for R, r, note in [(0, 0, "적합 정보 없음(=IDF)"),
                       (3, 1, "적합 3편 중 1편만 포함"),
                       (3, 3, "적합 3편 모두 포함 → 강한 신호")]:
        w = rsj_weight(12, 4, R=R, r=r)
        print(f"  {R:>2} {r:>2} | {w:>11.3f}  {note}")
    print("  → 적합 문서에 자주 나오는 term일수록 가중치↑ (Rocchio·PRF의 뿌리).")


def demo3_tf_saturation():
    print("\n=== (3) tf 포화 곡선 — 왜 '한 번 더 나온 단어'의 가치는 점점 줄까 ===")
    print("  포화계수 S=(k1+1)tf/(k1+tf), dl=avgdl(B=1). k1이 포화 속도를 정한다.")
    print(f"  {'tf':>3} | {'k1=0.0':>8} {'k1=1.2':>8} {'k1=2.0':>8}   (k1=0 → 이진)")
    print("  " + "-" * 42)
    for tf in (0, 1, 2, 3, 4, 5, 6, 8):
        s0 = saturation(tf, 0.0, 0.0, 1, 1)
        s1 = saturation(tf, 1.2, 0.0, 1, 1)
        s2 = saturation(tf, 2.0, 0.0, 1, 1)
        print(f"  {tf:>3} | {s0:>8.3f} {s1:>8.3f} {s2:>8.3f}")
    print("  → tf가 커질수록 증가폭이 줄어 (k1+1)로 '포화'. k1=0이면 present=1/absent=0.")


def demo4_length_normalization():
    print("\n=== (4) 문서길이 정규화 — 같은 tf라도 '긴 문서'는 감점 ===")
    print("  k1=1.5, b=0.75, tf=3 고정, avgdl=100. dl만 변화 → B와 포화계수 S:")
    print(f"  {'dl':>4} | {'B':>6} {'S(포화계수)':>12}")
    print("  " + "-" * 30)
    for dl in (50, 100, 200, 400):
        B = length_norm_B(0.75, dl, 100)
        s = saturation(3, 1.5, 0.75, dl, 100)
        print(f"  {dl:>4} | {B:>6.3f} {s:>12.3f}")
    print("  → 평균보다 짧으면 가점(B<1), 길면 감점(B>1). b가 그 세기를 조절.")


def demo5_full_score_matches_engine():
    print("\n=== (5) 완성형 BM25 항 점수 = IDF × 포화계수 (Day-021 엔진과 대조) ===")
    print("  N=12, avgdl=15.92. term 'search'(df=4)가 문서 d4(tf=1,dl=11)에 기여한 점수:")
    idf = idf_lucene(12, 4)
    s = saturation(1, 1.5, 0.75, 11, 15.92)
    score = idf * s
    print(f"  IDF(Lucene) = {idf:.4f}")
    print(f"  포화계수 S  = {s:.4f}   (tf=1, dl=11 < avgdl → 소폭 가점)")
    print(f"  항 점수     = IDF x S = {score:.4f}")
    print("  → 이 값이 bm25_engine.py 의 항별 누적과 동일한 공식임을 확인.")


if __name__ == "__main__":
    rule()
    print("📄 논문 정독 — BM25의 확률론적 유도 (Robertson & Zaragoza, 2009)")
    print("   PRP → BIM → RSJ가중치(=IDF) → 2-Poisson 포화 → 길이정규화 → BM25")
    rule()
    demo1_rsj_reduces_to_idf()
    demo2_relevance_feedback()
    demo3_tf_saturation()
    demo4_length_normalization()
    demo5_full_score_matches_engine()
    print()
    rule()
    print("요약: BM25는 임의의 공식이 아니라, 확률적 적합성 가정에서 '유도'되고")
    print("      2-Poisson 포화를 실용적으로 근사한 결과다. k1·b는 그 근사의 손잡이.")
    rule()
