"""Day-024 — 벡터 공간 모델(VSM)과 코사인 유사도 랭킹
Vector Space Model & Cosine Similarity Ranking (from scratch, stdlib only)

- TF-IDF 가중 벡터를 직접 만든다 (log-tf * idf).
- 코사인 유사도로 랭킹하고, '정규화 안 한 내적(dot product)'과 대조한다.
- 같은 코퍼스에서 BM25 랭킹과 비교해 두 고전 모델의 장단을 본다.
표준 라이브러리(math, re)만 사용 — uv run python vsm_cosine.py
(Windows 콘솔에서 한글이 깨지면: $env:PYTHONIOENCODING="utf-8" 후 실행)
"""
import math
import re
from collections import Counter

# ----------------------------------------------------------------------
# 0. 코퍼스 — d11/d12 가 '벡터 공간 모델·코사인' 문서 (Day-023 예고대로)
# ----------------------------------------------------------------------
DOCS = {
    "d1":  "information retrieval finds relevant documents for a user query",
    "d2":  "an inverted index maps each term to a postings list of documents",
    "d3":  "bm25 scores a document using term frequency saturation and length normalization",
    "d4":  "tf idf weighting multiplies term frequency by inverse document frequency",
    "d5":  "precision and recall measure the quality of a ranked retrieval result",
    "d6":  "ndcg is a ranking metric that rewards relevant documents near the top",
    "d7":  "query expansion adds related terms to improve the recall of a search",
    "d8":  "a search engine builds an index then ranks documents for each query",
    "d9":  "word embeddings place words of similar meaning near each other in a space",
    "d10": "cosine similarity measures the angle between two vectors ignoring their length",
    "d11": "the vector space model represents each document and query as a weighted term vector",
    "d12": "documents are ranked by the cosine similarity between the query vector and each document vector",
    # d13: 'vector' 를 잔뜩 반복한 긴 문서 — 내적은 부풀지만 코사인은 정규화한다
    "d13": "vector vector vector data storage cloud server backup archive vector index vector",
}

STOP = {"a", "an", "the", "of", "for", "to", "and", "by", "in", "is",
        "that", "each", "their", "two", "then", "other", "as"}


def tokenize(text):
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP]


# ----------------------------------------------------------------------
# 1. 색인 통계: df, idf
# ----------------------------------------------------------------------
class Index:
    def __init__(self, docs):
        self.docs = docs
        self.N = len(docs)
        self.toks = {d: tokenize(t) for d, t in docs.items()}
        self.df = Counter()
        for toks in self.toks.values():
            for t in set(toks):
                self.df[t] += 1

    def idf(self, t):
        # log10 idf; 코퍼스에 없는 단어는 0
        return math.log10(self.N / self.df[t]) if self.df[t] else 0.0


# ----------------------------------------------------------------------
# 2. TF-IDF 벡터: w = (1+log10 tf) * idf   (log-frequency weighting)
# ----------------------------------------------------------------------
def tfidf_vector(ix, toks):
    tf = Counter(toks)
    vec = {}
    for t, f in tf.items():
        idf = ix.idf(t)
        if idf == 0.0:
            continue
        vec[t] = (1.0 + math.log10(f)) * idf
    return vec


def dot(a, b):
    small, big = (a, b) if len(a) <= len(b) else (b, a)
    return sum(w * big.get(t, 0.0) for t, w in small.items())


def norm(v):
    return math.sqrt(sum(w * w for w in v.values()))


def cosine(a, b):
    na, nb = norm(a), norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return dot(a, b) / (na * nb)


# ----------------------------------------------------------------------
# 3. BM25 (Day-017/021) — 대조군
# ----------------------------------------------------------------------
class BM25:
    def __init__(self, ix, k1=1.5, b=0.75):
        self.ix, self.k1, self.b = ix, k1, b
        self.len = {d: len(t) for d, t in ix.toks.items()}
        self.avgdl = sum(self.len.values()) / ix.N
        self.tf = {d: Counter(t) for d, t in ix.toks.items()}

    def bm25_idf(self, t):
        n = self.ix.df[t]
        if n == 0:
            return 0.0
        return math.log((self.ix.N - n + 0.5) / (n + 0.5) + 1.0)

    def score(self, query, did):
        s = 0.0
        dl = self.len[did]
        for t in set(tokenize(query)):
            f = self.tf[did].get(t, 0)
            if f == 0:
                continue
            denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            s += self.bm25_idf(t) * (f * (self.k1 + 1)) / denom
        return s

    def search(self, query, topk=5):
        scored = [(d, self.score(query, d)) for d in self.ix.docs]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:topk]


# ----------------------------------------------------------------------
# 4. VSM 검색
# ----------------------------------------------------------------------
def vsm_search(ix, query, use_cosine=True, topk=5):
    qv = tfidf_vector(ix, tokenize(query))
    out = []
    for d in ix.docs:
        dv = tfidf_vector(ix, ix.toks[d])
        sc = cosine(qv, dv) if use_cosine else dot(qv, dv)
        out.append((d, sc))
    out.sort(key=lambda x: (-x[1], x[0]))
    return out[:topk]


def show(title, results):
    print(title)
    for rank, (d, sc) in enumerate(results, 1):
        snippet = DOCS[d][:52]
        print(f"    {rank}. {d:<4} score={sc:.4f}  {snippet}")


# ----------------------------------------------------------------------
def main():
    ix = Index(DOCS)
    bm = BM25(ix)
    query = "vector space model cosine"

    print("=" * 70)
    print("Day-024 — 벡터 공간 모델(VSM) & 코사인 유사도 (Vector Space Model & Cosine)")
    print("=" * 70)
    print(f"\n코퍼스 {ix.N}개 문서, 질의 q = {query!r}\n")

    # (1) 질의 벡터 자체 뜯어보기
    print("=== (1) 질의 벡터 q = TF-IDF 가중 (1+log10 tf)*idf ===")
    qv = tfidf_vector(ix, tokenize(query))
    for t in sorted(qv, key=lambda x: -qv[x]):
        print(f"    {t:<8} df={ix.df[t]}  idf={ix.idf(t):.4f}  w={qv[t]:.4f}")
    print(f"    ||q|| = {norm(qv):.4f}")

    # (2) 코사인 랭킹
    print("\n=== (2) 코사인 유사도 랭킹 (cosine) ===")
    show("  [cosine top-5]", vsm_search(ix, query, use_cosine=True))

    # (3) 정규화 안 한 내적 — 길이 편향
    print("\n=== (3) 대조: 정규화 안 한 내적(dot product) ===")
    show("  [raw dot-product top-5]", vsm_search(ix, query, use_cosine=False))
    print("  ↑ 긴 스팸 문서 d13('vector' 반복)이 내적에선 부풀지만,")
    print("    코사인에선 길이로 나눠 순위가 내려간다 (아래 수치 비교).")

    # (4) d13/d11/d12 코사인 vs 내적 직접 비교
    print("\n=== (4) d11 · d12 · d13 — 코사인 vs 내적 ===")
    print(f"    {'doc':<5}{'|d|(길이)':<10}{'dot(q,d)':<12}{'cosine':<10}")
    for d in ["d11", "d12", "d13"]:
        dv = tfidf_vector(ix, ix.toks[d])
        print(f"    {d:<5}{len(ix.toks[d]):<10}{dot(qv, dv):<12.4f}{cosine(qv, dv):<10.4f}")
    print("  → d13 은 dot 이 크지만(길이·반복) 코사인은 낮다 = 길이 정규화의 효과.")

    # (5) BM25 대조
    print("\n=== (5) 대조군 — 같은 질의로 BM25 랭킹 (Day-017/021) ===")
    show("  [BM25 top-5]", bm.search(query))
    print("  → VSM(코사인)과 BM25 는 상위 문서가 대체로 겹치지만 순위가 다르다.")
    print("    BM25 는 tf 포화·길이 정규화를 확률적으로, VSM 은 벡터 각도로 접근한다.")

    print("\n" + "=" * 70)
    print("요약: VSM 은 문서·질의를 TF-IDF 벡터로 놓고 '방향(각도)'의 유사도로 랭킹한다.")
    print("      코사인은 길이로 나눠 긴 문서 편향을 없앤다. 이 '문서=벡터' 관점이")
    print("      Phase 5 의 밀집 임베딩 검색으로 이어진다.")
    print("=" * 70)


if __name__ == "__main__":
    main()
