# prf_expansion.py — Day-023: 질의 확장 & 의사 적합 피드백 (Query Expansion & Pseudo-Relevance Feedback)
# Day-021 BM25 엔진을 그대로 재사용하고, Day-022 에서 본 RSJ 적합 가중치로 확장어를 고른다.
# 표준 라이브러리(math, re)만 사용. 실행: uv run python prf_expansion.py
import sys, math, re

sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔 한글 깨짐 방지


# ── 0) 토큰화 (Day-006) ────────────────────────────────────────────────────
def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


# 확장어 후보에서 걸러낼 불용어 (Day-006). 확장이 'the' 같은 기능어로 오염되는 것 방지.
STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "by", "for", "with",
    "its", "it", "is", "are", "as", "at", "be", "we", "our", "then", "not",
    "each", "other", "over", "many", "all", "near", "into", "but", "that",
    "this", "their", "them", "than", "using", "uses", "use", "like", "first",
}


# ── 1) 역색인 (Day-016) ─────────────────────────────────────────────────────
class InvertedIndex:
    def __init__(self, docs):
        self.docs = docs
        self.postings = {}   # term -> {doc_id: tf}
        self.doc_len = {}    # doc_id -> 토큰 수
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


# ── 2) BM25 스코어러 (Day-017) — 가중 질의(weighted query)를 지원하도록 확장 ──
class BM25:
    def __init__(self, index, k1=1.5, b=0.75):
        self.ix, self.k1, self.b = index, k1, b

    def _idf(self, term):
        df = self.ix.df(term)
        return math.log(1 + (self.ix.N - df + 0.5) / (df + 0.5))  # 항상 양수

    def _score_terms(self, term_weights, topk=None):
        """term -> weight 딕셔너리로 점수화. weight=1 이면 일반 BM25 와 동일."""
        ix, k1, b = self.ix, self.k1, self.b
        scores = {}
        for t, w in term_weights.items():
            if t not in ix.postings:
                continue
            idf = self._idf(t)
            for did, f in ix.postings[t].items():
                denom = f + k1 * (1 - b + b * ix.doc_len[did] / ix.avgdl)
                scores[did] = scores.get(did, 0.0) + w * idf * (f * (k1 + 1)) / denom
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:topk] if topk else ranked

    def search(self, query, topk=None):
        # 일반 검색: 질의어 각각 weight 1 (같은 단어 반복은 합산).
        # 질의에서도 불용어는 제거한다(Day-006) — 'to' 같은 기능어가 점수를 끌지 않도록.
        tw = {}
        for t in tokenize(query):
            if t in STOP:
                continue
            tw[t] = tw.get(t, 0.0) + 1.0
        return self._score_terms(tw, topk)


# ── 3) 평가 하니스 (Day-018) ───────────────────────────────────────────────
def dcg(gains):
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked, qrels, k):
    gains = [qrels.get(did, 0) for did, _ in ranked[:k]]
    idcg = dcg(sorted(qrels.values(), reverse=True)[:k])
    return dcg(gains) / idcg if idcg > 0 else 0.0


def average_precision(ranked, qrels):
    rel_total = sum(1 for g in qrels.values() if g > 0)
    if rel_total == 0:
        return 0.0
    hits, precisions = 0, []
    for i, (did, _) in enumerate(ranked, 1):
        if qrels.get(did, 0) > 0:
            hits += 1
            precisions.append(hits / i)
    return sum(precisions) / rel_total


# ── 4) RSJ 적합 가중치 (Day-022) — 확장어 선택 기준 ─────────────────────────
#     w = log[ ((r+0.5)/(R-r+0.5)) / ((n-r+0.5)/(N-n-R+r+0.5)) ]
#     R=피드백 문서 수, r=그중 t 포함 수, n=df(t), N=전체 문서 수
def rsj_weight(N, n, R, r):
    num = (r + 0.5) / (R - r + 0.5)
    den = (n - r + 0.5) / (N - n - R + r + 0.5)
    return math.log(num / den)


# ── 5) 의사 적합 피드백(PRF) — Rocchio 스타일 질의 확장 ─────────────────────
def prf_expand(bm25, query, fb_docs=3, fb_terms=3, beta=0.5, verbose=False):
    """
    1) 초기 검색 → 상위 fb_docs 편을 '의사 적합(pseudo-relevant)' 으로 가정.
    2) 그 문서들의 단어를 후보로, RSJ 가중치(Day-022)로 점수화.
    3) 상위 fb_terms 개를 확장어로 채택(원 질의어·불용어 제외).
    4) 원 질의어(weight 1.0) + 확장어(weight beta) 로 가중 질의를 만든다(Rocchio α:β).
    반환: (가중 질의 딕셔너리, 채택된 확장어 리스트[(term, rsj)])
    """
    ix = bm25.ix
    q_terms = {t for t in tokenize(query) if t not in STOP}

    # 1) 초기 검색 → 피드백 집합 R
    initial = bm25.search(query, topk=fb_docs)
    R_ids = [did for did, _ in initial]
    R = len(R_ids)

    # 2) 후보 확장어 = 피드백 문서에 등장한 단어(원 질의어·불용어 제외)
    cand = {}  # term -> r (피드백 문서 중 포함 수)
    for did in R_ids:
        for t in set(tokenize(ix.docs[did])):
            if t in q_terms or t in STOP:
                continue
            cand[t] = cand.get(t, 0) + 1

    # 3) RSJ 가중치로 후보 점수화 → 상위 fb_terms 채택
    scored = []
    for t, r in cand.items():
        n = ix.df(t)
        w = rsj_weight(ix.N, n, R, r)
        scored.append((t, w, r, n))
    # 가중치 내림차순, 동점은 df 작은(희귀) 순, 그다음 알파벳
    scored.sort(key=lambda x: (-x[1], x[3], x[0]))
    chosen = scored[:fb_terms]

    if verbose:
        print(f"  피드백 문서(top-{fb_docs}) = {R_ids}")
        print(f"  {'확장 후보':<12}{'RSJ':>8}{'r(피드백포함)':>14}{'df':>5}")
        chosen_set = set(chosen)
        for row in scored[:8]:
            t, w, r, n = row
            star = " ★채택" if row in chosen_set else ""
            print(f"  {t:<12}{w:>8.3f}{r:>14}{n:>5}{star}")

    # 4) Rocchio 가중 질의: 원 질의어 1.0, 확장어 beta
    tw = {t: 1.0 for t in q_terms}
    for t, w, r, n in chosen:
        tw[t] = tw.get(t, 0.0) + beta
    return tw, [(t, w) for t, w, r, n in chosen]


# ══════════════════════════════════════════════════════════════════════════
# 실행부 — Day-021 과 동일한 코퍼스 재사용
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
    # Day-023 추가: '학습된 랭킹(learning to rank)' 개념 묶음 — 어휘 다리(bridge) 실험용.
    # d14 는 관련 문서지만 질의어(learning/rank)를 하나도 안 써서 초기 검색엔 '안 보인다'.
    "d13": "learning to rank builds a model that orders retrieved results",
    "d14": "a neural ranker scores query document pairs with a trained model",
    "d15": "gradient boosted trees are a popular learning to rank model",
    "d16": "deep learning improves a ranking model by learning representations",
}

ix = InvertedIndex(docs)
bm25 = BM25(ix, k1=1.5, b=0.75)


def show_ranking(ranked, qrels, k=5):
    for rank, (did, sc) in enumerate(ranked[:k], 1):
        g = qrels.get(did, 0)
        mark = f"관련{g}" if g > 0 else "  ·  "
        print(f"    {rank}. {did:<4} score={sc:.4f}  [{mark}]  {docs[did][:46]}")


print("=" * 70)
print("Day-023 — 질의 확장 & 의사 적합 피드백 (Query Expansion & PRF / Rocchio)")
print("=" * 70)

# ── (1) 어휘 불일치 문제 (vocabulary mismatch) ──────────────────────────────
q1 = "learning to rank"      # 사용자가 쓴 말
# 관련 문서: d13/d15/d16 은 'learning'·'rank' 를 쓰지만, d14('neural ranker')는
# 같은 주제인데도 'ranker'·'model' 이라 써서 질의어를 하나도 안 담고 있다.
qrels1 = {"d13": 3, "d14": 3, "d15": 2, "d16": 2}
print("\n=== (1) 어휘 불일치 — 질의 'learning to rank' ===")
print("  d14('a neural ranker ... a trained model')는 같은 주제지만 질의어(learning/rank)를")
print("  하나도 안 써서 '보이지 않는 관련 문서'다 — 전형적 어휘 불일치(vocabulary mismatch).")
init1 = bm25.search(q1)
print("  [초기 검색 결과]")
show_ranking(init1, qrels1)
print(f"  → d14 는 top-5 에 없다(질의어 미포함). 초기 nDCG@5 = "
      f"{ndcg_at_k(init1, qrels1, 5):.4f}")

# ── (2) PRF 로 확장어 고르기 (RSJ 가중치 = Day-022) ─────────────────────────
print("\n=== (2) 의사 적합 피드백 — 상위 문서로 확장어 고르기 (RSJ 가중치) ===")
tw1, exp1 = prf_expand(bm25, q1, fb_docs=3, fb_terms=2, beta=0.5, verbose=True)
print(f"  채택된 확장어: {[t for t, _ in exp1]}")

# ── (3) 확장 질의로 재검색 (Rocchio 가중) ───────────────────────────────────
print("\n=== (3) 재검색 — 원 질의(1.0) + 확장어(β=0.5) ===")
exp_ranked1 = bm25._score_terms(tw1)
print("  [확장 후 결과]")
show_ranking(exp_ranked1, qrels1)
print(f"  → nDCG@5 = {ndcg_at_k(init1, qrels1, 5):.4f} (전) → "
      f"{ndcg_at_k(exp_ranked1, qrels1, 5):.4f} (후)   "
      f"MAP {average_precision(init1, qrels1):.4f} → "
      f"{average_precision(exp_ranked1, qrels1):.4f}")

# ── (4) 질의 드리프트(query drift) 위험 — 피드백에 비관련 문서가 섞이면 ──────
print("\n=== (4) 함정 — 질의 드리프트(query drift) ===")
q2 = "search engine"
qrels2 = {"d4": 3, "d8": 3, "d1": 1, "d2": 1, "d5": 1}  # d3=스팸(0)
init2 = bm25.search(q2)
print(f"  질의 '{q2}' 초기 상위:")
show_ranking(init2, qrels2)
print("  ↑ 스팸성 긴 문서 d3(비관련)이 상위에 끼어 피드백 집합을 오염시킨다.")
tw2, exp2 = prf_expand(bm25, q2, fb_docs=3, fb_terms=3, beta=0.5, verbose=True)
print(f"  채택된 확장어: {[t for t, _ in exp2]}")
exp_ranked2 = bm25._score_terms(tw2)
print(f"  → nDCG@5 = {ndcg_at_k(init2, qrels2, 5):.4f} (전) → "
      f"{ndcg_at_k(exp_ranked2, qrels2, 5):.4f} (후)")
# 완화책: 피드백 문서를 줄이거나(fb_docs=2), 확장어를 보수적으로
tw2b, exp2b = prf_expand(bm25, q2, fb_docs=2, fb_terms=2, beta=0.3)
exp_ranked2b = bm25._score_terms(tw2b)
print(f"  [완화] fb_docs=2, fb_terms=2, β=0.3 → 확장어 {[t for t, _ in exp2b]}, "
      f"nDCG@5 = {ndcg_at_k(exp_ranked2b, qrels2, 5):.4f}")

print("=" * 70)
print("요약: PRF 는 '상위 문서 = 임시 정답' 으로 가정해 질의를 넓힌다(어휘 불일치 완화).")
print("      하지만 피드백이 오염되면 질의 드리프트로 악화될 수 있다 — β·문서수로 조절.")
print("=" * 70)
