# tfidf_search.py — Day-015: TF-IDF 를 '검색 점수'로 쓰기 (IR 미리보기)
# Phase 1 에서 만든 TF-IDF 벡터(Day-007/Day-014)를 '분류'가 아니라
# '질의에 맞는 문서 찾기'(정보 검색, IR)에 그대로 재사용한다.
# 핵심: 문서·질의를 같은 TF-IDF 공간의 L2 정규화 벡터로 만들면
#       내적(dot product) = 코사인 유사도(cosine similarity) = '검색 점수'.
import sys, math
from collections import Counter
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔 한글 깨짐 방지

# 0) 작은 문서 컬렉션(corpus) — '영화 리뷰' 6개를 검색 대상 문서로
corpus = [
    "the movie had brilliant acting and a wonderful story",   # doc0
    "a boring and dull film with terrible acting",            # doc1
    "an amazing soundtrack made this film wonderful",         # doc2
    "the plot was confusing but the visuals were amazing",    # doc3
    "i loved the wonderful music and the brilliant cast",     # doc4
    "a dreadful and boring movie i really disliked",          # doc5
]

# 1) TF-IDF 벡터라이저 (Day-007 수식 → Day-014 에서 from scratch 구현한 것 재사용)
class TfidfVectorizer:
    def fit(self, docs):
        self.vocab = sorted({w for d in docs for w in d.split()})
        self.index = {w: i for i, w in enumerate(self.vocab)}
        N = len(docs)
        df = Counter(w for d in docs for w in set(d.split()))   # 문서빈도(df)
        self.idf = np.array([math.log((1 + N) / (1 + df[w])) + 1 for w in self.vocab])
        return self
    def transform(self, docs):
        rows = []
        for d in docs:
            tf = np.zeros(len(self.vocab))
            for w, c in Counter(d.split()).items():
                if w in self.index:               # 어휘 밖 단어(OOV) 무시
                    tf[self.index[w]] = c
            vec = tf * self.idf                    # tf-idf 가중
            norm = np.linalg.norm(vec)
            rows.append(vec / norm if norm > 0 else vec)   # L2 정규화
        return np.vstack(rows)

# 2) 색인(index) 만들기: 컬렉션 전체를 TF-IDF 행렬로 — 한 번만 계산해 둔다
vec = TfidfVectorizer().fit(corpus)
D = vec.transform(corpus)        # shape = (문서수, 어휘수), 각 행은 L2 정규화됨
print(f"어휘 크기 V = {len(vec.vocab)} | 문서 행렬 D = {D.shape} (문서 x 단어)")

# 3) 검색(retrieval): 질의를 같은 공간으로 보내고 코사인 유사도로 랭킹
def search(query, k=3):
    q = vec.transform([query])[0]      # 질의도 '문서처럼' 같은 TF-IDF 공간으로
    scores = D @ q                     # D·q : 모든 문서와의 내적 = 코사인 유사도
    order = np.argsort(scores)[::-1][:k]
    print(f"\n[질의 Query] {query!r}")
    for rank, i in enumerate(order, 1):
        bar = "#" * int(round(scores[i] * 20))
        print(f"  {rank}. score={scores[i]:.3f} {bar:<20} | doc{i}: {corpus[i]}")

# 4) 세 가지 질의로 '검색엔진'처럼 돌려 보기
for query in [
    "wonderful brilliant acting",     # 긍정 키워드 → doc0/doc4 가 위로
    "boring terrible movie",          # 부정 키워드 → doc1/doc5 가 위로
    "amazing music soundtrack",       # doc2/doc4 가 위로
]:
    search(query)

# ──────────────────────────────────────────────────────────────────
# 관찰 포인트
#  - 같은 TF-IDF 벡터인데, Day-014 에선 '분류'(σ(w·x+b)), 여기선 '검색'(질의·문서 코사인).
#    → 텍스트를 벡터로 만든 한 번의 수고가 여러 과제에 재사용된다.
#  - 흔한 단어(the·a·and)는 idf 가 낮아 점수에 거의 기여하지 않는다 → 자연스러운 불용어 효과.
#  - 모든 문서와 내적을 '전부' 계산했다(brute force). 문서가 100만 개면? → Phase 2 의 '역색인'.
#  - 단순 단어빈도(tf)·길이정규화의 한계는 Phase 2 의 BM25 가 보정한다(Day-016~).
# ──────────────────────────────────────────────────────────────────
