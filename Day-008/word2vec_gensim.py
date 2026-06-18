# word2vec_gensim.py (선택) — 진짜 '학습된' 밀집 임베딩 맛보기
# 설치:  uv add gensim
import sys
sys.stdout.reconfigure(encoding="utf-8")

from gensim.models import Word2Vec

# 같은 토이 코퍼스 (실전 Word2Vec 은 수십억 단어가 필요하다!)
corpus = [
    "the cat drinks milk".split(),
    "the dog drinks water".split(),
    "the cat eats fish".split(),
    "the dog eats meat".split(),
    "i drive my car".split(),
    "she drives her car".split(),
]

# Skip-gram(sg=1) 으로 학습. vector_size=20 → 단어 하나가 20차원 '밀집(dense)' 벡터.
model = Word2Vec(
    corpus, vector_size=20, window=2, min_count=1,
    sg=1, epochs=300, seed=0, workers=1,   # workers=1 + seed → 재현 가능
)

print("cat 벡터의 모양(shape):", model.wv["cat"].shape, "  ← 20차원 밀집 벡터")
print("cat 벡터 앞 5개 값:", model.wv["cat"][:5].round(3))
print("\ncat 과 가까운 단어 Top3:", model.wv.most_similar("cat", topn=3))
print("car 과 가까운 단어 Top3:", model.wv.most_similar("car", topn=3))
print("\n(작은 코퍼스라 이웃이 다소 흔들립니다 — '경향'만 보세요.")
print(" cat 의 이웃에 dog 가, car 의 이웃에 drive 계열이 오면 성공입니다.)")

# ─────────────────────────────────────────────────────────────────────
# (선택, 인터넷 다운로드 ~66MB) 사전학습된 GloVe 로 그 유명한 유추(analogy)
#   주석을 풀고 실행하면 king - man + woman ≈ queen 을 직접 봅니다.
# ─────────────────────────────────────────────────────────────────────
# import gensim.downloader as api
# glove = api.load("glove-wiki-gigaword-50")   # 처음 1회만 다운로드
# print("\nking - man + woman ≈ ?",
#       glove.most_similar(positive=["king", "woman"], negative=["man"], topn=1))
# # → [('queen', 0.85...)]  근사적으로 queen 이 1위
