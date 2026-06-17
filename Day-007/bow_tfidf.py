# bow_tfidf.py — Bag-of-Words & TF-IDF를 표준 라이브러리만으로 직접 구현
# 실행: uv run python bow_tfidf.py
import sys, re, math
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")   # Windows 콘솔 한글 깨짐 방지

corpus = [
    "the cat sat on the mat",     # D0
    "the dog sat on the log",     # D1
    "cats and dogs are friends",  # D2
    "the cat and the dog play",   # D3
]

def tokenize(text):
    return re.findall(r"[a-z]+", text.lower())

docs = [tokenize(d) for d in corpus]

# 1) 어휘(vocabulary): 등장한 모든 단어를 정렬해 인덱스(차원) 부여
vocab = sorted({t for toks in docs for t in toks})
idx = {t: i for i, t in enumerate(vocab)}
print("어휘 크기(vocab size):", len(vocab))
print("어휘:", vocab)

# 2) Bag-of-Words 벡터: 순서를 버리고 '빈도'만 센다
def bow(tokens):
    v = [0] * len(vocab)
    for term, c in Counter(tokens).items():
        v[idx[term]] = c
    return v

bow_matrix = [bow(toks) for toks in docs]
print("\n[BoW] D0 =", bow_matrix[0])
print("[BoW] D3 =", bow_matrix[3])

# 3) IDF = log10(N / df).  흔한 단어일수록 작아진다.
N = len(corpus)
df = Counter()
for toks in docs:
    for term in set(toks):       # 문서당 1번만 세기 = document frequency
        df[term] += 1
idf = {t: math.log10(N / df[t]) for t in vocab}
print("\n[IDF] (df=등장 문서 수, 흔할수록 idf 작다)")
for t in vocab:
    print(f"  {t:<8} df={df[t]}  idf={idf[t]:.4f}")

# 4) TF-IDF = tf * idf
def tfidf(tokens):
    tf = Counter(tokens)
    return {t: tf[t] * idf[t] for t in tf}

print("\n[TF-IDF] D0 의 단어별 가중치 (클수록 그 문서를 잘 대표)")
for t, w in sorted(tfidf(docs[0]).items(), key=lambda kv: -kv[1]):
    print(f"  {t:<8} tf={Counter(docs[0])[t]}  tfidf={w:.4f}")

# 5) 코사인 유사도로 문서 검색(ranking)  ← Day-004 와 연결
def cosine(a, b):
    common = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in common)
    na = math.sqrt(sum(w * w for w in a.values()))
    nb = math.sqrt(sum(w * w for w in b.values()))
    return dot / (na * nb) if na and nb else 0.0

query = "cat and dog"
q_vec = tfidf(tokenize(query))
doc_vecs = [tfidf(toks) for toks in docs]
scores = [(i, cosine(q_vec, dv)) for i, dv in enumerate(doc_vecs)]
scores.sort(key=lambda x: -x[1])

print(f'\n[검색] 질의(query) = "{query}"')
for i, s in scores:
    print(f"  D{i}  score={s:.4f}  | {corpus[i]}")
