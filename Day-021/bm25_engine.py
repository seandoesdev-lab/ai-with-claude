# bm25_engine.py — Day-021 (🛠️ B2 빌드): from-scratch BM25 검색엔진 + nDCG/MAP 평가 하니스
# 표준 라이브러리(math, re)만 사용. 실행: uv run python bm25_engine.py
#
# Day-016(역색인)·017(BM25)·018(nDCG)·019(qrels)·020(파이프라인·EDD)의 조각들을
# 재사용 가능한 "모듈"로 굳혀, 여러 질의에 대한 평균 nDCG 로 파라미터를 EDD 튜닝한다.
import sys, math, re

sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔 한글 깨짐 방지


# ──────────────────────────────────────────────────────────────────────────
# 0) 토큰화 (Day-006)
# ──────────────────────────────────────────────────────────────────────────
def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


# ──────────────────────────────────────────────────────────────────────────
# 1) 역색인 (INDEX-TIME, 한 번) — Day-016
#    term -> {doc_id: tf}, 문서 길이, N, avgdl 을 미리 계산해 둔다.
# ──────────────────────────────────────────────────────────────────────────
class InvertedIndex:
    def __init__(self, docs):
        self.docs = docs
        self.postings = {}     # term -> {doc_id: tf}
        self.doc_len = {}      # doc_id -> 토큰 수
        self._build()

    def _build(self):
        for did, text in self.docs.items():
            toks = tokenize(text)
            self.doc_len[did] = len(toks)
            tf_local = {}
            for t in toks:
                tf_local[t] = tf_local.get(t, 0) + 1
            for t, f in tf_local.items():
                self.postings.setdefault(t, {})[did] = f
        self.N = len(self.docs)
        self.avgdl = sum(self.doc_len.values()) / self.N if self.N else 0.0

    def df(self, term):
        return len(self.postings.get(term, {}))


# ──────────────────────────────────────────────────────────────────────────
# 2) BM25 스코어러 (QUERY-TIME, 매번) — Day-017
#    후보(질의어를 포함한 문서)만 점수화한다.
# ──────────────────────────────────────────────────────────────────────────
class BM25:
    def __init__(self, index, k1=1.5, b=0.75):
        self.ix = index
        self.k1 = k1
        self.b = b

    def _idf(self, term):
        df = self.ix.df(term)
        # BM25 IDF (항상 양수 형태) — Day-017
        return math.log(1 + (self.ix.N - df + 0.5) / (df + 0.5))

    def search(self, query, topk=None):
        ix, k1, b = self.ix, self.k1, self.b
        scores = {}
        for t in tokenize(query):
            if t not in ix.postings:
                continue
            idf = self._idf(t)
            for did, f in ix.postings[t].items():
                denom = f + k1 * (1 - b + b * ix.doc_len[did] / ix.avgdl)
                scores[did] = scores.get(did, 0.0) + idf * (f * (k1 + 1)) / denom
        # 점수 내림차순, 동점은 doc_id 오름차순(결정론적)
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:topk] if topk else ranked


# ──────────────────────────────────────────────────────────────────────────
# 3) 평가 하니스 (EVAL) — Day-018
#    등급형 nDCG@k + 이진 MAP. 여러 질의의 '평균'을 낸다.
# ──────────────────────────────────────────────────────────────────────────
def dcg(gains):
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked, qrels, k):
    gains = [qrels.get(did, 0) for did, _ in ranked[:k]]
    ideal = sorted(qrels.values(), reverse=True)[:k]
    idcg = dcg(ideal)
    return dcg(gains) / idcg if idcg > 0 else 0.0


def average_precision(ranked, qrels):
    # 이진 관련성(등급>0 을 관련으로 간주)으로 AP 계산
    rel_total = sum(1 for g in qrels.values() if g > 0)
    if rel_total == 0:
        return 0.0
    hits, precisions = 0, []
    for i, (did, _) in enumerate(ranked, 1):
        if qrels.get(did, 0) > 0:
            hits += 1
            precisions.append(hits / i)
    return sum(precisions) / rel_total


def evaluate(engine, queries, k=5):
    """여러 질의에 대해 평균 nDCG@k, 평균 AP(=MAP) 를 낸다."""
    ndcgs, aps = [], []
    for q in queries:
        ranked = engine.search(q["query"])
        ndcgs.append(ndcg_at_k(ranked, q["qrels"], k))
        aps.append(average_precision(ranked, q["qrels"]))
    return {
        "mean_ndcg": sum(ndcgs) / len(ndcgs),
        "map": sum(aps) / len(aps),
        "per_query_ndcg": ndcgs,
    }


# ──────────────────────────────────────────────────────────────────────────
# 4) EDD 튜닝 — (k1, b) 격자를 훑어 '평균 nDCG@5' 최고를 고른다 — Day-020
# ──────────────────────────────────────────────────────────────────────────
def grid_search(index, queries, k1_grid, b_grid, k=5):
    results = []
    for k1 in k1_grid:
        for b in b_grid:
            eng = BM25(index, k1=k1, b=b)
            m = evaluate(eng, queries, k=k)
            results.append((k1, b, m["mean_ndcg"], m["map"]))
    # 평균 nDCG 내림차순, 동점은 (k1,b) 작은 쪽
    results.sort(key=lambda r: (-r[2], r[0], r[1]))
    return results


# ══════════════════════════════════════════════════════════════════════════
# 실행부
# ══════════════════════════════════════════════════════════════════════════
docs = {
    "d1": "an inverted index maps terms to postings lists for fast full text search",
    "d2": "bm25 ranks documents using term frequency saturation and length normalization",
    "d3": ("our marketing blog briefly mentions search engine ranking and search engine "
           "ranking tips but then wanders into gardening travel cooking finance sports "
           "weather music movies books fashion cars phones laptops cameras coffee tea and "
           "many more padding words that make this document quite long overall"),
    "d4": "a search engine uses bm25 ranking to score and order documents by relevance",
    "d5": "evaluation metrics like ndcg and map measure the quality of a ranking",
    "d6": "the cat sat on the mat near the sunny window all afternoon long",
    "d7": "tf idf weights each term by its frequency and its rarity in the collection",
    "d8": "a search engine first builds an index then ranks results for a user query",
    "d9": "precision and recall trade off against each other when we tune a ranking system",
    "d10": ("neural network training uses gradient descent and backpropagation to minimize "
            "a loss function over many epochs"),
    "d11": "cosine similarity compares two vectors by the angle between them not their length",
    "d12": "a vector space model represents each document as a weighted vector of its terms",
}

# 여러 질의 + qrels (Day-019). 등급: 3=강한관련, 2/1=약한관련, 없음=비관련(0)
queries = [
    {
        "query": "search engine ranking",
        "qrels": {"d4": 3, "d8": 3, "d1": 1, "d2": 1, "d5": 1},  # d3=0 (스팸성 긴 문서)
    },
    {
        "query": "evaluation metrics for ranking quality",
        "qrels": {"d5": 3, "d9": 2},
    },
    {
        "query": "document vector similarity",
        "qrels": {"d11": 3, "d12": 3, "d7": 1},
    },
]

print("=" * 70)
print("🛠️ B2 빌드 — from-scratch BM25 검색엔진 + 평가 하니스 (Day-021)")
print("=" * 70)

# (1) 색인
ix = InvertedIndex(docs)
print("\n=== (1) 색인 (INDEX-TIME, 1회) ===")
print(f"  문서 수 N = {ix.N}, 평균 문서길이 avgdl = {ix.avgdl:.2f} 토큰")
print(f"  고유 단어(term) 수 = {len(ix.postings)}")
print(f"  df('search') = {ix.df('search')},  df('ranking') = {ix.df('ranking')},  "
      f"df('the') = {ix.df('the')}")

# (2) 질의 — 기본 파라미터로 한 질의 시연
eng = BM25(ix, k1=1.5, b=0.75)
q0 = queries[0]
print(f"\n=== (2) 질의 (QUERY-TIME) — '{q0['query']}' (k1=1.5, b=0.75) ===")
for rank, (did, sc) in enumerate(eng.search(q0["query"], topk=5), 1):
    g = q0["qrels"].get(did, 0)
    mark = f"관련{g}" if g > 0 else "  ·  "
    print(f"  {rank}. {did:<4} score={sc:.4f}  [{mark}]  {docs[did][:52]}")

# (3) 평가 — 여러 질의 평균
m = evaluate(eng, queries, k=5)
print("\n=== (3) 평가 (EVAL) — 3개 질의 (k1=1.5, b=0.75) ===")
for q, nd in zip(queries, m["per_query_ndcg"]):
    print(f"  nDCG@5 = {nd:.4f}   ← '{q['query']}'")
print(f"  ─────────────────────────────────────")
print(f"  평균 nDCG@5 = {m['mean_ndcg']:.4f}   MAP = {m['map']:.4f}")

# (4) EDD 격자 탐색
print("\n=== (4) EDD 격자 탐색 — (k1, b) 로 '평균 nDCG@5' 최고 찾기 ===")
res = grid_search(ix, queries, k1_grid=(1.0, 1.5, 2.0), b_grid=(0.0, 0.25, 0.5, 0.75, 1.0))
print(f"  {'k1':>4} {'b':>5} | {'평균 nDCG@5':>12} {'MAP':>8}")
print(f"  {'-'*4} {'-'*5} | {'-'*12} {'-'*8}")
for i, (k1, b, nd, mp) in enumerate(res):
    flag = "  ← 최고" if i == 0 else ""
    print(f"  {k1:>4} {b:>5} | {nd:>12.4f} {mp:>8.4f}{flag}")
best = res[0]
print(f"\n  → 최적 (k1={best[0]}, b={best[1]}): 평균 nDCG@5={best[2]:.4f}. "
      f"단일 질의가 아닌 '여러 질의 평균' 으로 골라 과적합을 피한다.")
print("=" * 70)
