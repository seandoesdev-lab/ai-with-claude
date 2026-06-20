# sentiment_baseline.py — 임베딩 평균 vs TF-IDF 로 감정 분류(sentiment) 베이스라인
# Day-010 (AI with Claude). Windows + uv 환경 기준. 의존성은 numpy + scikit-learn 둘뿐.
#   uv add numpy scikit-learn
#   uv run python sentiment_baseline.py
#
# 핵심 아이디어:
#   (A) TF-IDF + 로지스틱 회귀   — Day-007 의 희소(sparse) 벡터 베이스라인
#   (B) 단어 임베딩 평균 + 로지스틱 회귀 — Day-008/009 의 밀집(dense) 벡터.
#       임베딩은 Day-009 의 Skip-gram+네거티브 샘플링(SGNS)을 NumPy로 직접 학습.
import sys
import numpy as np
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score

sys.stdout.reconfigure(encoding="utf-8")   # Windows 콘솔 한글 깨짐 방지
SEED = 0
rng = np.random.default_rng(SEED)

# ----------------------------------------------------------------------
# 0) 작은 감정 데이터셋 (toy sentiment dataset). 1 = 긍정, 0 = 부정
#    실제 연구는 IMDB(5만 리뷰) 같은 대형 데이터를 쓰지만 여기선 개념 시연용.
# ----------------------------------------------------------------------
pos = [
    "this movie was great and i loved it",
    "an amazing film with a wonderful story",
    "the acting was brilliant and moving",
    "a fantastic and enjoyable film overall",
    "i really liked this great movie",
    "a wonderful and amazing story i loved",
    "brilliant acting and a fantastic story",
    "i loved the wonderful and great film",
    "an enjoyable and brilliant movie overall",
    "the story was amazing and i really liked it",
    "a great film with wonderful acting",
    "i enjoyed this amazing and brilliant movie",
]
neg = [
    "this movie was terrible and i hated it",
    "an awful film with a boring story",
    "the acting was bad and dull",
    "a dreadful and boring film overall",
    "i really disliked this terrible movie",
    "an awful and boring story i hated",
    "bad acting and a dreadful story",
    "i hated the awful and terrible film",
    "a boring and dreadful movie overall",
    "the story was terrible and i really disliked it",
    "a bad film with awful acting",
    "i disliked this terrible and dreadful movie",
]
data = [(t, 1) for t in pos] + [(t, 0) for t in neg]

# 학습/평가 분할 (Day-005: train/test split). 각 클래스의 끝 2개씩을 테스트로.
def split(items):
    return items[:-2], items[-2:]
pos_tr, pos_te = split([(t, 1) for t in pos])
neg_tr, neg_te = split([(t, 0) for t in neg])
train, test = pos_tr + neg_tr, pos_te + neg_te
X_train_txt, y_train = [t for t, _ in train], np.array([y for _, y in train])
X_test_txt,  y_test  = [t for t, _ in test],  np.array([y for _, y in test])


def report(name, y_true, y_pred):
    print(f"\n===== {name} =====")
    print(f"accuracy = {accuracy_score(y_true, y_pred):.3f}")
    print(classification_report(y_true, y_pred,
          target_names=["neg(0)", "pos(1)"], zero_division=0))


# ----------------------------------------------------------------------
# A) 베이스라인 1 — TF-IDF + 로지스틱 회귀 (Day-007 의 희소 벡터)
# ----------------------------------------------------------------------
vec = TfidfVectorizer()
Xtr_tfidf = vec.fit_transform(X_train_txt)
Xte_tfidf = vec.transform(X_test_txt)
clf_tfidf = LogisticRegression(max_iter=1000, random_state=SEED)
clf_tfidf.fit(Xtr_tfidf, y_train)
report("TF-IDF + LogReg", y_test, clf_tfidf.predict(Xte_tfidf))


# ----------------------------------------------------------------------
# B-1) 단어 임베딩 학습 — Day-009 의 Skip-gram + 네거티브 샘플링(SGNS), NumPy로
# ----------------------------------------------------------------------
def train_sgns(sentences, dim=30, window=2, k=5, lr=0.05, epochs=400, seed=0):
    r = np.random.default_rng(seed)
    docs = [s.split() for s in sentences]
    counts = Counter(w for d in docs for w in d)
    vocab = sorted(counts)
    w2i = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)
    pairs = []
    for d in docs:
        ids = [w2i[w] for w in d]
        for i, c in enumerate(ids):
            lo, hi = max(0, i - window), min(len(ids), i + window + 1)
            for j in range(lo, hi):
                if j != i:
                    pairs.append((c, ids[j]))
    neg_p = np.array([counts[w] for w in vocab], dtype=float) ** 0.75
    neg_p /= neg_p.sum()
    W = (r.random((V, dim)) - 0.5) / dim     # center 벡터(최종 임베딩)
    C = (r.random((V, dim)) - 0.5) / dim     # context 벡터
    sig = lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
    for _ in range(epochs):
        r.shuffle(pairs)
        for c, o in pairs:
            negs = r.choice(V, size=k, p=neg_p)
            tgts = np.concatenate(([o], negs))
            labels = np.zeros(k + 1); labels[0] = 1.0
            v, u = W[c], C[tgts]
            err = sig(u @ v) - labels         # 예측 − 정답 (Day-009)
            W[c] -= lr * (err @ u)
            C[tgts] -= lr * np.outer(err, v)
    return W, w2i, dim         # center 행렬을 최종 임베딩으로 사용 (Day-009)


emb, w2i, DIM = train_sgns(X_train_txt, seed=SEED)


# B-2) 문장 벡터 = 단어 임베딩들의 평균 (mean pooling). OOV 단어는 건너뜀.
def sentence_vector(text, emb, w2i, dim):
    vecs = [emb[w2i[w]] for w in text.split() if w in w2i]
    if not vecs:
        return np.zeros(dim)
    m = np.mean(vecs, axis=0)
    return m / (np.linalg.norm(m) + 1e-9)   # L2 정규화: 길이 1로 (Day-004)


Xtr_emb = np.vstack([sentence_vector(t, emb, w2i, DIM) for t in X_train_txt])
Xte_emb = np.vstack([sentence_vector(t, emb, w2i, DIM) for t in X_test_txt])
clf_emb = LogisticRegression(max_iter=1000, random_state=SEED)
clf_emb.fit(Xtr_emb, y_train)
report("MeanEmbedding(SGNS) + LogReg", y_test, clf_emb.predict(Xte_emb))


# ----------------------------------------------------------------------
# C) 일반화 점검 — 학습에 없던 새 문장 (단, 단어는 어휘 안에 있어야 함)
# ----------------------------------------------------------------------
new_sentences = [
    "a wonderful and amazing film i loved",   # 긍정 기대
    "a boring and awful movie i hated",       # 부정 기대
]
print("\n===== 새 문장 예측 (MeanEmbedding) =====")
for s in new_sentences:
    v = sentence_vector(s, emb, w2i, DIM).reshape(1, -1)
    p = clf_emb.predict_proba(v)[0, 1]
    print(f"  P(pos)={p:.3f}  ->  {'긍정' if p >= 0.5 else '부정'}   | {s}")
