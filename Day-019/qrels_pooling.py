# qrels_pooling.py — Day-019: 테스트 컬렉션 · 관련성 판단(qrels) · pooling · 어노테이터 일치도
# 표준 라이브러리(math)만 사용. 실행: uv run python qrels_pooling.py
# 다루는 것:
#   (1) pooling — 여러 시스템의 상위 k개를 모아 '판단할 후보 풀'을 만든다(전수 판단 회피)
#   (2) qrels 구축 — 풀 안 문서만 사람이 판단, 풀 밖은 '비관련(0)'으로 가정
#   (3) pooling bias — 풀에 안 든 진짜 관련 문서는 놓친다(가정의 대가)
#   (4) Cohen's kappa — 두 어노테이터의 일치도(우연 일치 보정)
import sys, math
sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔 한글 깨짐 방지

# ─────────────────────────────────────────────────────────────
# 0) 전체 컬렉션(현실에선 수백만 개). 여기선 12개로 축소.
collection = [f"d{i}" for i in range(1, 13)]   # d1 .. d12

# 한 질의 Q에 대해 서로 다른 3개 시스템(예: BM25·TF-IDF·불리언)이 낸 순위(상위→하위)
runs = {
    "BM25":   ["d1", "d2", "d3", "d4", "d5"],
    "TF-IDF": ["d3", "d1", "d6", "d2", "d7"],
    "Boolean":["d8", "d3", "d1", "d9", "d2"],
}

# ─────────────────────────────────────────────────────────────
# 1) POOLING — 각 시스템의 상위 depth개만 모아 합집합 → '판단할 풀'
def build_pool(runs, depth):
    pool = set()
    for ranking in runs.values():
        pool.update(ranking[:depth])
    return pool

DEPTH = 3
pool = build_pool(runs, DEPTH)
print("=== (1) Pooling: 전수 판단을 피한다 ===")
print(f"  전체 컬렉션 크기        : {len(collection)}개")
for name, r in runs.items():
    print(f"  {name:8s} 상위{DEPTH} : {r[:DEPTH]}")
print(f"  풀(depth={DEPTH}) = 상위{DEPTH}들의 합집합 : {sorted(pool)}")
print(f"  → 판단할 문서 {len(pool)}개만 사람이 보면 됨 "
      f"(전수 {len(collection)}개 대비 {len(pool)/len(collection)*100:.0f}%)")

# ─────────────────────────────────────────────────────────────
# 2) qrels 구축 — 풀 안 문서만 사람이 판단. 풀 밖은 '비관련(0)'으로 가정.
#    (아래 '진짜 관련'은 설명용 정답 — 현실엔 없다. pooling bias 시연에 쓴다.)
true_relevant = {"d1", "d3", "d6", "d10"}   # d10 은 어떤 시스템도 상위에 못 올림!

qrels = {}
for doc in collection:
    if doc in pool:
        qrels[doc] = 1 if doc in true_relevant else 0   # 판단됨
    # 풀 밖 문서는 qrels 에 아예 없음 → 평가 시 0(비관련)으로 취급
print("\n=== (2) qrels: 풀 안만 판단, 풀 밖은 0 으로 가정 ===")
judged_rel = sorted(d for d, g in qrels.items() if g == 1)
judged_non = sorted(d for d, g in qrels.items() if g == 0)
print(f"  판단=관련(1) : {judged_rel}")
print(f"  판단=비관련(0): {judged_non}")
print(f"  판단 안 함(풀 밖, 0 가정): {sorted(set(collection) - pool)}")

# ─────────────────────────────────────────────────────────────
# 3) POOLING BIAS — 풀에 안 든 진짜 관련 문서(d10)는 영영 놓친다.
missed = true_relevant - pool
print("\n=== (3) Pooling bias: 가정의 대가 ===")
print(f"  진짜 관련(설명용 정답): {sorted(true_relevant)}")
print(f"  풀이 놓친 관련 문서   : {sorted(missed)}  ← 어느 시스템도 상위{DEPTH}에 못 올려 '비관련'으로 오분류")
print("  → depth 를 키우면 bias↓ 판단비용↑. 새 시스템이 풀 밖 관련 문서를 올리면 과소평가될 수 있다.")

# ─────────────────────────────────────────────────────────────
# 4) COHEN'S KAPPA — 두 어노테이터의 일치도(우연 일치 보정)
#    같은 10개 문서를 두 사람이 관련(1)/비관련(0) 판단. 쌍의 리스트.
judgments = [  # (어노테이터 A, 어노테이터 B)
    (1, 1), (1, 1), (1, 1), (1, 1), (1, 0),
    (0, 1), (0, 0), (0, 0), (0, 0), (1, 1),
]
def cohens_kappa(pairs):
    n = len(pairs)
    both1 = sum(1 for a, b in pairs if a == 1 and b == 1)
    both0 = sum(1 for a, b in pairs if a == 0 and b == 0)
    p_o = (both1 + both0) / n                     # 관측 일치율
    a1 = sum(a for a, _ in pairs) / n             # A 가 1 준 비율
    b1 = sum(b for _, b in pairs) / n             # B 가 1 준 비율
    p_e = a1 * b1 + (1 - a1) * (1 - b1)           # 우연히 일치할 확률
    kappa = (p_o - p_e) / (1 - p_e) if p_e < 1 else 1.0
    return p_o, p_e, kappa

p_o, p_e, kappa = cohens_kappa(judgments)
def interpret(k):
    return ("거의 없음(poor)" if k < 0.0 else "약함(slight)" if k < 0.20 else
            "어느정도(fair)" if k < 0.40 else "보통(moderate)" if k < 0.60 else
            "상당(substantial)" if k < 0.80 else "거의 완벽(almost perfect)")
print("\n=== (4) Cohen's kappa: 어노테이터 일치도(우연 보정) ===")
print(f"  관측 일치율 p_o = {p_o:.3f}   우연 일치 p_e = {p_e:.3f}")
print(f"  Cohen's κ = (p_o - p_e)/(1 - p_e) = {kappa:.3f}  → {interpret(kappa)}")
print("  → 단순 일치율(p_o)은 우연 일치를 포함해 과대평가. κ 는 그걸 빼서 '진짜 합의'를 잰다.")

# ─────────────────────────────────────────────────────────────
# 5) (실험) depth 를 바꾸면 풀 크기와 bias 가 어떻게 변하나
print("\n=== (5) depth ↔ 풀 크기 ↔ 놓친 관련 문서 ===")
for depth in (1, 2, 3, 5):
    p = build_pool(runs, depth)
    miss = true_relevant - p
    print(f"  depth={depth}: 풀 {len(p)}개, 놓친 관련 {sorted(miss) if miss else '없음'}")
