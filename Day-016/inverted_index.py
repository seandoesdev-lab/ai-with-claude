# inverted_index.py — Day-016: 역색인(inverted index) from scratch + 불린 검색
# 표준 라이브러리만 사용 (설치 불필요). 실행: uv run python inverted_index.py
import sys
from collections import defaultdict, Counter
sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔 한글 깨짐 방지

# 0) Day-015 와 같은 작은 컬렉션(corpus) — 영화 리뷰 6편
corpus = [
    "the movie had brilliant acting and a wonderful story",   # doc0
    "a boring and dull film with terrible acting",            # doc1
    "an amazing soundtrack made this film wonderful",         # doc2
    "the plot was confusing but the visuals were amazing",    # doc3
    "i loved the wonderful music and the brilliant cast",     # doc4
    "a dreadful and boring movie i really disliked",          # doc5
]

def tokenize(text):
    return text.lower().split()   # Day-006 의 최소 토큰화(공백 분리)

# 1) 색인 시간(indexing time): 역색인 구축
#    postings[term] = {doc_id: tf}  →  정렬된 (doc_id, tf) 목록으로 확정
class InvertedIndex:
    def build(self, docs):
        self.N = len(docs)
        self.doc_len = [len(tokenize(d)) for d in docs]       # 문서 길이(뒤의 BM25 준비)
        raw = defaultdict(dict)
        for doc_id, text in enumerate(docs):
            for term, tf in Counter(tokenize(text)).items():
                raw[term][doc_id] = tf
        # 포스팅을 doc_id 오름차순으로 '정렬'해 확정 (교집합 병합을 위해)
        self.postings = {t: sorted(d.items()) for t, d in raw.items()}
        self.df = {t: len(p) for t, p in self.postings.items()}  # 문서빈도
        return self

    def docs_for(self, term):                  # 그 단어가 든 문서 ID 집합(정렬 유지)
        return [doc_id for doc_id, _tf in self.postings.get(term, [])]

# 2) 정렬된 두 리스트의 교집합/합집합 (선형 병합, O(n+m))
def intersect(a, b):
    i = j = 0; out = []
    while i < len(a) and j < len(b):
        if a[i] == b[j]: out.append(a[i]); i += 1; j += 1
        elif a[i] < b[j]: i += 1
        else: j += 1
    return out

def union(a, b):
    i = j = 0; out = []
    while i < len(a) and j < len(b):
        if a[i] == b[j]: out.append(a[i]); i += 1; j += 1
        elif a[i] < b[j]: out.append(a[i]); i += 1
        else: out.append(b[j]); j += 1
    out.extend(a[i:]); out.extend(b[j:])
    return out

# 3) 불린 질의: 질의 단어들의 포스팅을 차례로 AND / OR
def boolean_search(idx, query, mode="AND"):
    terms = tokenize(query)
    lists = [idx.docs_for(t) for t in terms]
    if not lists:
        return []
    if mode == "AND" and any(len(l) == 0 for l in lists):
        # AND 인데 한 단어라도 컬렉션에 없으면 결과 없음
        return []
    combine = intersect if mode == "AND" else union
    result = lists[0]
    for l in lists[1:]:
        result = combine(result, l)
    return result

# ── 실행: 역색인 구축 후 들여다보기 ───────────────────────────
idx = InvertedIndex().build(corpus)
print(f"문서 수 N = {idx.N} | 사전(term) 크기 = {len(idx.postings)}")
print("\n[역색인 일부] term (df) → 포스팅 [doc_id:tf]")
for term in ["brilliant", "acting", "wonderful", "boring", "amazing", "the"]:
    postings = idx.postings.get(term, [])
    pretty = " → ".join(f"[{d}:{tf}]" for d, tf in postings)
    print(f"  {term:<10} (df={idx.df.get(term,0)})  {pretty}")

# ── 불린 검색 데모: 후보가 얼마나 좁혀지는가 ──────────────────
print("\n=== 불린 검색 (후보 추림) ===")
for q, mode in [("brilliant acting", "AND"),
                ("brilliant acting", "OR"),
                ("boring dull", "OR"),
                ("wonderful amazing", "AND")]:
    hits = boolean_search(idx, q, mode)
    print(f"\n[{mode}] {q!r}")
    print(f"  후보 문서: {hits}  (전체 {idx.N}개 중 {len(hits)}개만 채점하면 됨)")
    for i in hits:
        print(f"    - doc{i}: {corpus[i]}")

# ── 속도 직관: brute force 대비 절약량 ───────────────────────
q = "brilliant acting"
cand = boolean_search(idx, q, "AND")
print(f"\n=== 계산량 비교: 질의 {q!r} ===")
print(f"  brute force(Day-015): 문서 {idx.N}개 전부 내적  → {idx.N}회")
print(f"  역색인(오늘):         후보 {len(cand)}개만 채점   → {len(cand)}회")
saved = (1 - len(cand) / idx.N) * 100
print(f"  → 채점 대상 {saved:.0f}% 절약 (문서가 많을수록 격차는 더 벌어진다)")
