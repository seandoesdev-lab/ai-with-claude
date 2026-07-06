# mini_search.py — Day-020: 색인 → 랭킹 → 평가를 하나의 파이프라인으로 잇고, 평가-주도로 개선
# 표준 라이브러리(math, re)만 사용. 실행: uv run python mini_search.py
# 지금까지의 조각을 통합:
#   - 역색인(Inverted Index)        ← Day-016
#   - BM25 랭킹                       ← Day-017
#   - 평가지표(nDCG)                  ← Day-018
#   - 정답지(qrels)                   ← Day-019
# 그리고 '평가-주도 개발(EDD)': 파라미터를 바꾸고 → nDCG 로 재고 → 더 나은 값을 채택.
import sys, math, re
sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔 한글 깨짐 방지

# ─────────────────────────────────────────────────────────────
# 0) 아주 작은 문서 컬렉션 (현실에선 수백만). 검색 주제 + 잡음 문서 섞음.
docs = {
    "d1": "an inverted index maps terms to postings lists for fast search",
    "d2": "bm25 ranks documents using term frequency saturation and length normalization",
    # d3 = 길고 잡다한 블로그: 질의어를 몇 번 흘리지만 대부분 무관한 내용(비관련).
    "d3": ("our marketing blog briefly mentions search engine ranking and search engine "
           "ranking tips but then wanders into gardening travel cooking finance sports "
           "weather music movies books fashion cars phones laptops cameras coffee tea and "
           "many more padding words that make this document quite long overall"),
    "d4": "a search engine uses bm25 ranking to score and order documents",
    "d5": "evaluation metrics like ndcg measure the quality of a ranking",
    "d6": "the cat sat on the mat near the sunny window all afternoon",
    "d7": "tf idf weights each term by its frequency and its rarity in the collection",
    "d8": "a search engine first builds an index then ranks results for a query",
}

def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())

# ─────────────────────────────────────────────────────────────
# 1) 색인 파이프라인 (INDEX-TIME) — 한 번만. 역색인 + 문서 길이 + 평균 길이.
#    Day-016 역색인을 '문서 빈도(df)'와 '문서 내 빈도(tf)'까지 담도록 확장.
def build_index(docs):
    postings = {}       # term -> {doc_id: tf}
    doc_len = {}        # doc_id -> 토큰 수
    for did, text in docs.items():
        toks = tokenize(text)
        doc_len[did] = len(toks)
        tf_local = {}
        for t in toks:
            tf_local[t] = tf_local.get(t, 0) + 1
        for t, f in tf_local.items():
            postings.setdefault(t, {})[did] = f
    N = len(docs)
    avgdl = sum(doc_len.values()) / N
    return {"postings": postings, "doc_len": doc_len, "N": N, "avgdl": avgdl}

# ─────────────────────────────────────────────────────────────
# 2) 질의 파이프라인 (QUERY-TIME) — 질의마다. BM25 로 점수화 후 정렬.  (Day-017)
def bm25_search(index, query, k1=1.5, b=0.75, topk=None):
    postings, doc_len, N, avgdl = (index["postings"], index["doc_len"],
                                   index["N"], index["avgdl"])
    scores = {}
    for t in tokenize(query):
        if t not in postings:
            continue
        df = len(postings[t])
        idf = math.log(1 + (N - df + 0.5) / (df + 0.5))   # BM25 IDF (항상 양수)
        for did, f in postings[t].items():
            denom = f + k1 * (1 - b + b * doc_len[did] / avgdl)
            scores[did] = scores.get(did, 0.0) + idf * (f * (k1 + 1)) / denom
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))  # 점수↓, 동점은 id↑
    return ranked if topk is None else ranked[:topk]

# ─────────────────────────────────────────────────────────────
# 3) 평가 (EVAL) — 등급형 nDCG@k. (Day-018)
def dcg(gains):
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))  # i=0 → /log2(2)=1

def ndcg_at_k(ranked, qrels, k):
    gains = [qrels.get(did, 0) for did, _ in ranked[:k]]
    ideal = sorted(qrels.values(), reverse=True)[:k]
    idcg = dcg(ideal)
    return dcg(gains) / idcg if idcg > 0 else 0.0

# ─────────────────────────────────────────────────────────────
# 실행
index = build_index(docs)
print("=== (1) 색인 파이프라인 (한 번만 수행) ===")
print(f"  문서 수 N = {index['N']}, 평균 문서길이 avgdl = {index['avgdl']:.2f} 토큰")
print(f"  색인된 고유 단어(term) 수 = {len(index['postings'])}")
print(f"  예: 'search' 의 포스팅(문서→tf) = {index['postings']['search']}")
print(f"      'bm25'   의 포스팅(문서→tf) = {index['postings']['bm25']}")

query = "search engine ranking"
# 이 질의에 대한 정답지(qrels) — 등급형(3=매우 관련, 1=약간 관련, 0=비관련)  (Day-019)
qrels = {"d4": 3, "d8": 3, "d1": 1, "d2": 1, "d5": 1}  # d3·d6·d7 = 0(명시 안 함)

print(f"\n=== (2) 질의 파이프라인 — query = '{query}' (BM25 기본 k1=1.5, b=0.75) ===")
ranked = bm25_search(index, query)
for rank, (did, sc) in enumerate(ranked, 1):
    g = qrels.get(did, 0)
    mark = "관련" + str(g) if g > 0 else "  ·  "
    print(f"  {rank}. {did}  score={sc:.4f}  [{mark}]  {docs[did][:60]}")

print("\n=== (3) 평가 — nDCG@k (정답지로 채점) ===")
for k in (3, 5):
    print(f"  nDCG@{k} = {ndcg_at_k(ranked, qrels, k):.4f}")

# ─────────────────────────────────────────────────────────────
# 4) 평가-주도 개발(EDD) — 길이 정규화 강도 b 를 바꿔가며 nDCG@5 로 최적값 탐색
print("\n=== (4) 평가-주도 개발(EDD): b 를 바꾸면 nDCG@5 는? ===")
best = None
for b in (0.0, 0.25, 0.5, 0.75, 1.0):
    r = bm25_search(index, query, b=b)
    score = ndcg_at_k(r, qrels, 5)
    flag = ""
    if best is None or score > best[1]:
        best = (b, score); flag = "  ← 현재 최고"
    print(f"  b={b:<4}  nDCG@5 = {score:.4f}{flag}")
print(f"  → 이 질의에선 b={best[0]} 가 최적(nDCG@5={best[1]:.4f}). "
      f"'추측' 이 아니라 '측정' 으로 고른다.")

print("\n=== (5) 전체 그림 ===")
print("  색인(1회) ─▶ 질의마다 BM25 랭킹 ─▶ qrels 로 nDCG 채점 ─▶ 파라미터 바꿔 재측정(EDD)")
print("  이 루프가 곧 다음 편 🛠️ B2 빌드(BM25 엔진 + nDCG from scratch)의 뼈대다.")
