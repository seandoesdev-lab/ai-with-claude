# tfidf_logreg.py — 🛠️ B1 빌드 마일스톤
# TF-IDF 벡터화 + 로지스틱 회귀(시그모이드+경사하강)를 NumPy 만으로 from scratch
# (scikit-learn 없이 — 단, 4.5 검증 단계에서만 sklearn 과 대조)
import sys, math
from collections import Counter
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔 한글 깨짐 방지
np.random.seed(0)

# ──────────────────────────────────────────────────────────────────────
# 0) Day-010·Day-013 과 '같은' 작은 감정 데이터셋 (1=긍정, 0=부정)
# ──────────────────────────────────────────────────────────────────────
pos = [
    "this movie was great and i loved it", "an amazing film with a wonderful story",
    "the acting was brilliant and moving", "a fantastic and enjoyable film overall",
    "i really liked this great movie", "a wonderful and amazing story i loved",
    "brilliant acting and a fantastic story", "i loved the wonderful and great film",
    "an enjoyable and brilliant movie overall", "the story was amazing and i really liked it",
    "a great film with wonderful acting", "i enjoyed this amazing and brilliant movie",
]
neg = [
    "this movie was terrible and i hated it", "an awful film with a boring story",
    "the acting was bad and dull", "a dreadful and boring film overall",
    "i really disliked this terrible movie", "an awful and boring story i hated",
    "bad acting and a dreadful story", "i hated the awful and terrible film",
    "a boring and dreadful movie overall", "the story was terrible and i really disliked it",
    "a bad film with awful acting", "i disliked this terrible and dreadful movie",
]

def split(items): return items[:-2], items[-2:]   # 각 클래스 끝 2개를 테스트로 (Day-005)
pos_tr, pos_te = split([(t, 1) for t in pos])
neg_tr, neg_te = split([(t, 0) for t in neg])
train, test = pos_tr + neg_tr, pos_te + neg_te
X_train_txt, y_train = [t for t, _ in train], np.array([y for _, y in train], dtype=float)
X_test_txt,  y_test  = [t for t, _ in test],  np.array([y for _, y in test],  dtype=float)


# ──────────────────────────────────────────────────────────────────────
# 1) TF-IDF 벡터라이저 from scratch  (Day-007 의 수식을 코드로)
#    tf  = 문서 안 단어 빈도
#    idf = log((1+N)/(1+df)) + 1   (smooth idf, sklearn 기본과 동일)
#    tfidf = tf * idf  → 문서벡터를 L2 정규화
# ──────────────────────────────────────────────────────────────────────
class TfidfVectorizer:
    def fit(self, docs):
        self.vocab = sorted({w for d in docs for w in d.split()})
        self.index = {w: i for i, w in enumerate(self.vocab)}
        N = len(docs)
        df = Counter(w for d in docs for w in set(d.split()))   # 문서빈도(document freq)
        self.idf = np.array([math.log((1 + N) / (1 + df[w])) + 1 for w in self.vocab])
        return self

    def transform(self, docs):
        rows = []
        for d in docs:
            tf = np.zeros(len(self.vocab))
            for w, c in Counter(d.split()).items():
                if w in self.index:           # 학습 어휘 밖 단어는 무시(OOV)
                    tf[self.index[w]] = c
            vec = tf * self.idf                # tf-idf 가중
            norm = np.linalg.norm(vec)         # L2 정규화 → 문서 길이 영향 제거
            rows.append(vec / norm if norm > 0 else vec)
        return np.vstack(rows)

    def fit_transform(self, docs):
        return self.fit(docs).transform(docs)


# ──────────────────────────────────────────────────────────────────────
# 2) 로지스틱 회귀 from scratch  (Day-009·Day-010 의 시그모이드+경사하강)
#    z = Xw + b ; ŷ = σ(z)
#    손실 = 이진 교차엔트로피(BCE) + L2 정규화
#    경사: dw = Xᵀ(ŷ−y)/n + λw ,  db = mean(ŷ−y)
# ──────────────────────────────────────────────────────────────────────
def sigmoid(z):
    return np.where(z >= 0, 1 / (1 + np.exp(-z)),
                    np.exp(z) / (1 + np.exp(z)))   # 수치 안정 시그모이드

class LogisticRegression:
    def __init__(self, lr=0.5, epochs=300, l2=0.0):
        self.lr, self.epochs, self.l2 = lr, epochs, l2

    def fit(self, X, y, verbose=False):
        n, d = X.shape
        self.w = np.zeros(d)      # 0 초기화 → 결정론적(seed 무관)
        self.b = 0.0
        for ep in range(self.epochs):
            p = sigmoid(X @ self.w + self.b)
            err = p - y
            dw = X.T @ err / n + self.l2 * self.w
            db = err.mean()
            self.w -= self.lr * dw
            self.b -= self.lr * db
            if verbose and (ep == 0 or (ep + 1) % 100 == 0):
                eps = 1e-12
                loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
                print(f"  epoch {ep+1:>3}  BCE loss = {loss:.4f}")
        return self

    def predict_proba(self, X):
        return sigmoid(X @ self.w + self.b)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


def metrics(name, y_true, y_pred):
    y_true = y_true.astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    acc = (tp + tn) / len(y_true)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    print(f"\n===== {name} =====")
    print(f"  accuracy={acc:.3f}  precision={prec:.3f}  recall={rec:.3f}  f1={f1:.3f}")
    print(f"  (tp={tp} fp={fp} fn={fn} tn={tn})")


# ──────────────────────────────────────────────────────────────────────
# 3) 조립: 텍스트 → TF-IDF → 로지스틱 회귀 → 평가
# ──────────────────────────────────────────────────────────────────────
vec = TfidfVectorizer().fit(X_train_txt)
Xtr = vec.transform(X_train_txt)
Xte = vec.transform(X_test_txt)
print(f"어휘 크기 V = {len(vec.vocab)} | 학습행렬 = {Xtr.shape} (문서 x 단어)")

print("\n[학습] BCE 손실이 줄어드는 과정:")
clf = LogisticRegression(lr=0.5, epochs=300, l2=0.0).fit(Xtr, y_train, verbose=True)
metrics("TF-IDF + LogReg (from scratch, NumPy)", y_test, clf.predict(Xte))

# 학습된 가중치로 '가장 긍정/부정' 단어 보기 (해석 가능성)
order = np.argsort(clf.w)
vocab = np.array(vec.vocab)
print("\n[가중치] 가장 긍정적 단어 top5:", [str(w) for w in vocab[order[::-1][:5]]])
print("[가중치] 가장 부정적 단어 top5:", [str(w) for w in vocab[order[:5]]])

# 새 문장 일반화 테스트
print("\n[일반화] 학습에 없던 새 문장:")
for s in ["a wonderful and amazing film i loved", "a boring and awful movie i hated"]:
    p = float(clf.predict_proba(vec.transform([s]))[0])
    print(f"  P(pos)={p:.3f} -> {'긍정' if p >= 0.5 else '부정'} | {s}")

# ──────────────────────────────────────────────────────────────────────
# 4) (선택) sklearn 과 대조 — 재현성 점검 (Day-012 Pass 3 의 정신)
#    `uv add scikit-learn` 후 주석 해제
# ──────────────────────────────────────────────────────────────────────
# from sklearn.feature_extraction.text import TfidfVectorizer as SkTfidf
# from sklearn.linear_model import LogisticRegression as SkLR
# skv = SkTfidf(); A = skv.fit_transform(X_train_txt); B = skv.transform(X_test_txt)
# sk = SkLR(max_iter=1000).fit(A, y_train.astype(int))
# metrics("sklearn TF-IDF + LogReg (대조)", y_test, sk.predict(B))
