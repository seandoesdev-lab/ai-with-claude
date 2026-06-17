# sklearn_tfidf.py (선택) — 표준 도구(scikit-learn)로 BoW·TF-IDF·검색
# 준비: uv add scikit-learn
# 실행: uv run python sklearn_tfidf.py
import sys
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.stdout.reconfigure(encoding="utf-8")

corpus = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "cats and dogs are friends",
    "the cat and the dog play",
]

# (1) Bag-of-Words 행렬
cv = CountVectorizer()
bow = cv.fit_transform(corpus)
print("어휘:", cv.get_feature_names_out())
print("BoW 행렬(밀집):\n", bow.toarray())

# (2) TF-IDF 행렬 + 코사인 검색
tv = TfidfVectorizer()        # 기본: smooth idf + L2 정규화
X = tv.fit_transform(corpus)
q = tv.transform(["cat and dog"])
sims = cosine_similarity(q, X)[0]   # 질의 vs 각 문서
ranked = sorted(range(len(corpus)), key=lambda i: -sims[i])
print('\n[검색] query = "cat and dog"')
for i in ranked:
    print(f"  D{i}  score={sims[i]:.4f}  | {corpus[i]}")

# scikit-learn 의 IDF 는 스무딩(log((1+N)/(1+df))+1) + L2 정규화를 쓰므로
# 점수의 절댓값은 bow_tfidf.py 와 다르지만, 순위는 같다 — D3 가 1위.
