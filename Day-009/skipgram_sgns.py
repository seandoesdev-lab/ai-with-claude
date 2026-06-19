# skipgram_sgns.py — Skip-gram + Negative Sampling(SGNS)을 NumPy로 처음부터
# Day-009. "긍정 쌍은 가깝게, 무작위 부정 쌍은 멀게" 당기는 학습을 직접 구현한다.
import sys
import numpy as np
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")   # Windows 콘솔 한글 깨짐 방지
rng = np.random.default_rng(0)             # 재현성(seed). 환경마다 결과가 흔들리지 않게.

# ── 0) 작은 코퍼스 (cat/dog 는 비슷한 문맥, car 는 다른 동네) ──────────────
corpus = [
    "the cat drinks milk",
    "the dog drinks water",
    "the cat eats fish",
    "the dog eats meat",
    "a cat chases a mouse",
    "a dog chases a cat",
    "i drive my car",
    "she drives her car",
    "the car needs fuel",
    "we wash the car",
]

# ── 1) 토큰화 & 어휘(vocabulary) ─────────────────────────────────────────
docs = [s.split() for s in corpus]
counts = Counter(w for d in docs for w in d)
vocab = sorted(counts)
w2i = {w: i for i, w in enumerate(vocab)}
V = len(vocab)

# ── 2) 긍정 쌍 (center, context) 만들기: 창(window) 안의 이웃을 모두 ─────
WINDOW = 2
pos_pairs = []
for d in docs:
    ids = [w2i[w] for w in d]
    for i, center in enumerate(ids):
        lo, hi = max(0, i - WINDOW), min(len(ids), i + WINDOW + 1)
        for j in range(lo, hi):
            if j != i:
                pos_pairs.append((center, ids[j]))   # (가운데, 주변)

# ── 3) 네거티브 샘플링 분포: unigram^0.75 (흔한 단어를 살짝 누른 빈도) ────
freq = np.array([counts[w] for w in vocab], dtype=float)
neg_p = freq ** 0.75
neg_p /= neg_p.sum()

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# ── 4) 임베딩 두 벌: 중심 벡터 W, 문맥 벡터 C (최종엔 W 를 단어 임베딩으로) ─
DIM = 10
W = (rng.random((V, DIM)) - 0.5) / DIM   # center embeddings (사용할 임베딩)
C = (rng.random((V, DIM)) - 0.5) / DIM   # context embeddings (학습용 보조)

K = 5          # 긍정 쌍 1개당 부정 샘플 수
LR = 0.05      # 학습률(learning rate)
EPOCHS = 300

# ── 5) 학습 루프: 각 긍정 쌍 + K개의 부정 쌍을 이진 분류로 학습 ──────────
for epoch in range(EPOCHS):
    rng.shuffle(pos_pairs)
    total_loss = 0.0
    for center, ctx in pos_pairs:
        negs = rng.choice(V, size=K, p=neg_p)        # 부정 샘플 K개
        targets = np.concatenate(([ctx], negs))      # [긍정] + [부정 K개]
        labels = np.zeros(K + 1); labels[0] = 1.0    # 긍정=1, 부정=0

        v = W[center]            # (DIM,)   중심 단어 벡터
        u = C[targets]           # (K+1, DIM) 대상 단어들의 문맥 벡터
        preds = sigmoid(u @ v)   # (K+1,)   각 대상이 "진짜 이웃일" 확률

        # 손실(모니터링): 긍정은 1에 가깝게, 부정은 0에 가깝게
        total_loss += -(np.log(preds[0] + 1e-9) + np.log(1 - preds[1:] + 1e-9).sum())

        # 경사하강(gradient descent): err = preds - labels
        err = preds - labels                 # (K+1,)
        grad_v = err @ u                     # (DIM,)   중심 벡터 기울기
        grad_u = np.outer(err, v)            # (K+1, DIM) 대상 벡터 기울기
        W[center] -= LR * grad_v
        C[targets] -= LR * grad_u

    if (epoch + 1) % 50 == 0:
        print(f"epoch {epoch + 1:3d}  avg loss = {total_loss / len(pos_pairs):.4f}")

# ── 6) 결과: 코사인 유사도로 이웃 확인 (Day-004) ─────────────────────────
def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def most_similar(word, topn=3):
    v = W[w2i[word]]
    sims = [(w, cos(v, W[w2i[w]])) for w in vocab if w != word]
    return sorted(sims, key=lambda x: -x[1])[:topn]

print("\ncat 과 가까운 단어:", most_similar("cat"))
print("dog 과 가까운 단어:", most_similar("dog"))
print("car 과 가까운 단어:", most_similar("car"))
print("\nsim(cat, dog) =", round(cos(W[w2i['cat']], W[w2i['dog']]), 4))
print("sim(cat, car) =", round(cos(W[w2i['cat']], W[w2i['car']]), 4))
