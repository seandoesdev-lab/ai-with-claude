# ir_metrics.py — Day-018: IR 평가지표 from scratch (표준 라이브러리만)
# Precision@k · Recall@k · Average Precision(MAP) · MRR · nDCG
# 실행: uv run python ir_metrics.py
import sys, math
sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔 한글 깨짐 방지

# ──────────────────────────────────────────────────────────────────────────
# 0) 관련성 판단(qrels) — 등급(graded) 0~3. 0 은 비관련(목록에 없으면 0).
#    실제로는 사람이 매기거나 클릭 로그로 추정한다(Day-019 예고).
qrels = {
    "Q1": {"d1": 3, "d2": 2, "d4": 1},   # d1 가장 관련, d4 약하게 관련
    "Q2": {"d3": 3, "d5": 1},
}

# 시스템(예: Day-017 BM25)이 반환한 순위 — 상위→하위
rankings = {
    "Q1": ["d2", "d1", "d3", "d4", "d6"],
    "Q2": ["d6", "d3", "d1", "d5", "d2"],
}

# ──────────────────────────────────────────────────────────────────────────
# 1) 기본 헬퍼
def rel(q, doc):            # 관련도 등급 (없으면 0)
    return qrels[q].get(doc, 0)

def is_rel(q, doc):         # 이진 관련성 (grade > 0)
    return rel(q, doc) > 0

def n_relevant(q):          # 그 질의의 총 관련 문서 수 R
    return sum(1 for g in qrels[q].values() if g > 0)

# ──────────────────────────────────────────────────────────────────────────
# 2) 집합 기반 지표 (순서 일부만 반영: top-k 컷오프)
def precision_at_k(q, ranking, k):
    topk = ranking[:k]
    return sum(is_rel(q, d) for d in topk) / k

def recall_at_k(q, ranking, k):
    R = n_relevant(q)
    topk = ranking[:k]
    return (sum(is_rel(q, d) for d in topk) / R) if R else 0.0

# ──────────────────────────────────────────────────────────────────────────
# 3) 순위 기반 지표 (이진 관련성)
def average_precision(q, ranking):
    """관련 문서를 만날 때마다의 precision@i 를 평균 → AP. MAP 은 AP 의 질의 평균."""
    R = n_relevant(q)
    if R == 0:
        return 0.0
    hits, score = 0, 0.0
    for i, d in enumerate(ranking, 1):     # i = 1-based 순위
        if is_rel(q, d):
            hits += 1
            score += hits / i              # 이 위치에서의 precision@i
    return score / R

def reciprocal_rank(q, ranking):
    """첫 관련 문서의 순위 역수. MRR 은 RR 의 질의 평균."""
    for i, d in enumerate(ranking, 1):
        if is_rel(q, d):
            return 1.0 / i
    return 0.0

# ──────────────────────────────────────────────────────────────────────────
# 4) 등급 기반 지표 — nDCG (관련도의 '정도'와 '위치'를 함께 반영)
def dcg_at_k(q, ranking, k):
    # 선형 이득(linear gain): grade / log2(rank+1)
    return sum(rel(q, d) / math.log2(i + 1) for i, d in enumerate(ranking[:k], 1))

def ndcg_at_k(q, ranking, k):
    dcg = dcg_at_k(q, ranking, k)
    ideal_grades = sorted(qrels[q].values(), reverse=True)[:k]   # 이상적 순위
    idcg = sum(g / math.log2(i + 1) for i, g in enumerate(ideal_grades, 1))
    return (dcg / idcg) if idcg else 0.0

# ──────────────────────────────────────────────────────────────────────────
# 5) 질의별 출력 + 컬렉션 전체(평균)
print("=== 질의별 지표 ===")
aps, rrs, ndcgs = [], [], []
for q in ["Q1", "Q2"]:
    r = rankings[q]
    grades = ", ".join(f"{d}={g}" for d, g in qrels[q].items())
    print(f"\n[{q}] ranking={r}  (관련 grade: {grades})")
    print(f"  P@1={precision_at_k(q, r, 1):.3f}  "
          f"P@3={precision_at_k(q, r, 3):.3f}  P@5={precision_at_k(q, r, 5):.3f}")
    print(f"  R@3={recall_at_k(q, r, 3):.3f}  R@5={recall_at_k(q, r, 5):.3f}")
    ap, rr, nd = average_precision(q, r), reciprocal_rank(q, r), ndcg_at_k(q, r, 5)
    print(f"  AP={ap:.4f}   RR={rr:.4f}   nDCG@5={nd:.4f}")
    aps.append(ap); rrs.append(rr); ndcgs.append(nd)

print("\n=== 컬렉션 전체 (질의 평균) ===")
print(f"  MAP    = {sum(aps)/len(aps):.4f}")
print(f"  MRR    = {sum(rrs)/len(rrs):.4f}")
print(f"  nDCG@5 = {sum(ndcgs)/len(ndcgs):.4f}")

# ──────────────────────────────────────────────────────────────────────────
# 6) nDCG 는 '순서'를 본다 — 같은 관련 집합, 다른 순서 → recall 같아도 nDCG 다름
print("\n=== nDCG 가 '순서'를 본다: Q1 의 세 가지 순위 ===")
variants = {
    "시스템 순위": ["d2", "d1", "d3", "d4", "d6"],
    "이상적 순위": ["d1", "d2", "d4", "d3", "d6"],   # 등급 높은 순 (3,2,1)
    "나쁜 순위  ": ["d4", "d3", "d6", "d2", "d1"],   # 약한 관련을 위로
}
for name, r in variants.items():
    print(f"  {name}  {r} → "
          f"R@5={recall_at_k('Q1', r, 5):.3f}  nDCG@5={ndcg_at_k('Q1', r, 5):.4f}")
print("  → 세 순위 모두 R@5=1.000(관련 3개 다 top5)이지만 nDCG 는 다르다.")
print("    recall 은 '찾았나'만, nDCG 는 '얼마나 위에·얼마나 관련도 높게' 두었나를 본다.")
