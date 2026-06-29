# bm25.py — Day-017: 역색인 위에 BM25 랭킹 from scratch (표준 라이브러리만)
# 실행: uv run python bm25.py
import sys, math
from collections import defaultdict, Counter
sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔 한글 깨짐 방지

# 0) Day-016 기반 컬렉션 — 영화 리뷰 6편.
#    'brilliant' 은 doc0 에만 등장(희소→idf 큼), doc5 는 'acting' 만 도배(tf 폭증).
#    이 설계로 "tf 도배가 1등을 보장하지 않는다"(tf 포화)를 숫자로 보여준다.
corpus = [
    "the movie had brilliant acting and a wonderful story",        # doc0  brilliant1 acting1
    "a boring and dull film with terrible acting",                 # doc1  acting1
    "an amazing soundtrack made this film wonderful",              # doc2
    "the plot was confusing but the visuals were truly amazing",   # doc3
    "i loved the wonderful music and the great cast",              # doc4
    "acting acting acting acting acting shown in this film",       # doc5  acting5 (도배)
]


def tokenize(text):
    return text.lower().split()   # Day-006 의 최소 토큰화


# 1) 색인 시간: 역색인 + 통계(df, 문서길이, avgdl) — Day-016 의 산물 재사용
class BM25Index:
    def __init__(self, k1=1.5, b=0.75):
        self.k1, self.b = k1, b

    def build(self, docs):
        self.N = len(docs)
        self.doc_len = [len(tokenize(d)) for d in docs]
        self.avgdl = sum(self.doc_len) / self.N
        postings = defaultdict(dict)                 # term -> {doc_id: tf}
        for doc_id, text in enumerate(docs):
            for term, tf in Counter(tokenize(text)).items():
                postings[term][doc_id] = tf
        self.postings = postings
        self.df = {t: len(p) for t, p in postings.items()}
        return self

    def idf(self, term):                              # 확률적 IDF (1.4)
        n = self.df.get(term, 0)
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5))

    def score(self, query, doc_id):                  # BM25(D, Q) (1.1)
        s = 0.0
        dl = self.doc_len[doc_id]
        for term in tokenize(query):
            if doc_id not in self.postings.get(term, {}):
                continue                              # 그 문서에 없는 단어는 0 기여
            tf = self.postings[term][doc_id]
            denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            s += self.idf(term) * (tf * (self.k1 + 1)) / denom
        return s

    def search(self, query, top_k=None):             # 역색인으로 후보만 채점 (1.5)
        cand = set()
        for term in tokenize(query):
            cand.update(self.postings.get(term, {}).keys())
        ranked = sorted(((self.score(query, d), d) for d in cand), reverse=True)
        return ranked if top_k is None else ranked[:top_k]


def main():
    idx = BM25Index(k1=1.5, b=0.75).build(corpus)
    print(f"N={idx.N}  avgdl={idx.avgdl:.2f}  문서길이={idx.doc_len}")
    print(f"idf(brilliant)={idx.idf('brilliant'):.3f}  "
          f"idf(acting)={idx.idf('acting'):.3f}  idf(the)={idx.idf('the'):.3f}")

    print("\n=== BM25 랭킹 (k1=1.5, b=0.75) ===")
    for q in ["brilliant acting", "wonderful amazing", "the film"]:
        print(f"\n[질의] {q!r}")
        for rank, (sc, d) in enumerate(idx.search(q), 1):
            print(f"  {rank}. doc{d}  score={sc:.4f}  | {corpus[d]}")

    # ── tf 포화 직관: acting 의 tf 가 5인 doc5 vs tf 1인 doc1 ──
    print("\n=== tf 포화(saturation) 확인: 질의 'acting' ===")
    single = BM25Index(k1=1.5, b=0.0).build(corpus)   # b=0 으로 길이효과 제거, tf효과만
    for d in [1, 5]:
        tf = single.postings["acting"][d]
        print(f"  doc{d}: tf(acting)={tf} → score={single.score('acting', d):.4f}")
    print("  → tf가 5배(1→5)라도 점수는 5배가 아니다(포화). k1을 키우면 격차가 커진다.")

    # ── 손잡이 흔들기: k1, b 를 바꾸면 순위가 어떻게 변하나 ──
    print("\n=== 손잡이(k1,b) 민감도: 질의 'acting brilliant' ===")
    for k1, b in [(0.0, 0.75), (1.5, 0.0), (1.5, 0.75), (2.0, 0.75)]:
        rk = BM25Index(k1=k1, b=b).build(corpus).search("acting brilliant", top_k=3)
        order = " > ".join(f"doc{d}({sc:.2f})" for sc, d in rk)
        print(f"  k1={k1}, b={b:<4}: {order}")


if __name__ == "__main__":
    main()
