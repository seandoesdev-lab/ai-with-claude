# naive_bayes.py — 나이브 베이즈(Multinomial NB)로 감정 분류 (순수 파이썬, 외부 라이브러리 0)
# Day-013 · Phase 1 (NLP 기초)
# 핵심: P(class | words) ∝ P(class) * ∏ P(word | class)  — 베이즈 정리 + 조건부 독립 가정
#       확률은 로그 공간에서 더하고, 안 본 단어는 라플라스(add-α) 스무딩으로 0 확률을 막는다.
import sys, math
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔 한글 깨짐 방지

# 0) Day-010 과 '같은' 작은 감정 데이터셋 (1=긍정, 0=부정) — 분류기만 바꿔 공정 비교
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

# train/test 분할 (Day-005): 각 클래스 끝 2개씩을 테스트로
def split(items):
    return items[:-2], items[-2:]

pos_tr, pos_te = split([(t, 1) for t in pos])
neg_tr, neg_te = split([(t, 0) for t in neg])
train, test = pos_tr + neg_tr, pos_te + neg_te
X_train, y_train = [t for t, _ in train], [y for _, y in train]
X_test,  y_test  = [t for t, _ in test],  [y for _, y in test]


class MultinomialNB:
    """순수 파이썬 멀티노미얼 나이브 베이즈 + 라플라스(add-alpha) 스무딩."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha  # alpha=0 이면 스무딩 없음(클래스 내 미관측 단어 → log 0 → -inf)

    def fit(self, docs, labels):
        self.classes = sorted(set(labels))
        self.vocab = sorted({w for d in docs for w in d.split()})
        V = len(self.vocab)
        N = len(docs)
        self.logprior = {}
        self.loglik = {c: {} for c in self.classes}   # log P(word | class)
        self.logunseen = {}                            # 어휘엔 있으나 그 클래스에서 못 본 단어용
        for c in self.classes:
            c_docs = [d for d, y in zip(docs, labels) if y == c]
            self.logprior[c] = math.log(len(c_docs) / N)            # log P(class)
            counts = Counter(w for d in c_docs for w in d.split())  # 클래스 내 단어 빈도
            denom = sum(counts.values()) + self.alpha * V           # 정규화 분모(스무딩 포함)
            for w in self.vocab:
                numer = counts[w] + self.alpha
                # 스무딩 없이(alpha=0) 클래스 내 미관측 단어면 확률 0 → log 0 = -inf
                self.loglik[c][w] = math.log(numer / denom) if numer > 0 else float("-inf")
            self.logunseen[c] = math.log(self.alpha / denom) if self.alpha > 0 else float("-inf")
        return self

    def log_scores(self, doc):
        """클래스별 로그 점수: log P(class) + Σ log P(word | class)."""
        scores = {}
        for c in self.classes:
            s = self.logprior[c]
            for w in doc.split():
                if w in self.loglik[c]:
                    s += self.loglik[c][w]
                # 학습 어휘 밖(완전 처음 보는) 단어는 무시 — 모든 클래스에 공평
            scores[c] = s
        return scores

    def predict(self, doc):
        s = self.log_scores(doc)
        return max(s, key=s.get)

    def predict_proba_pos(self, doc):
        """긍정(1) 확률 = softmax(로그 점수)의 1번 성분."""
        s = self.log_scores(doc)
        m = max(s.values())
        exps = {c: math.exp(v - m) for c, v in s.items()}  # 수치 안정 softmax
        z = sum(exps.values())
        return exps.get(1, 0.0) / z


def metrics(name, y_true, y_pred):
    """이진(양성=1) 정확도·정밀도·재현율·F1 — Day-005 의 지표를 손으로."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    acc = (tp + tn) / len(y_true)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    print(f"\n===== {name} =====")
    print(f"  accuracy={acc:.3f}  precision={prec:.3f}  recall={rec:.3f}  f1={f1:.3f}")
    print(f"  (tp={tp} fp={fp} fn={fn} tn={tn})")


if __name__ == "__main__":
    # A) 스무딩 있는 NB (alpha=1.0)
    nb = MultinomialNB(alpha=1.0).fit(X_train, y_train)
    pred = [nb.predict(d) for d in X_test]
    metrics("MultinomialNB (alpha=1.0, 라플라스 스무딩)", y_test, pred)

    # B) 스무딩 없는 NB (alpha=0) — 클래스 내 미관측 단어 하나가 점수를 통째로 -inf 로
    nb0 = MultinomialNB(alpha=0.0).fit(X_train, y_train)
    # 'dull' 은 neg 에만 등장 → pos 클래스에서는 log P('dull'|pos)=log0=-inf
    s = nb0.log_scores("the acting was dull")
    print("\n[스무딩 OFF] log_scores('the acting was dull') =",
          {c: ("-inf" if v == float("-inf") else round(v, 2)) for c, v in s.items()})

    # C) 가장 '긍정/부정' 다운 단어 — 로그우도비 log P(w|pos) - log P(w|neg)
    ratio = {w: nb.loglik[1][w] - nb.loglik[0][w] for w in nb.vocab}
    top_pos = sorted(ratio, key=ratio.get, reverse=True)[:5]
    top_neg = sorted(ratio, key=ratio.get)[:5]
    print("\n[로그우도비] 가장 긍정적 단어:", ", ".join(f"{w}(+{ratio[w]:.2f})" for w in top_pos))
    print("[로그우도비] 가장 부정적 단어:", ", ".join(f"{w}({ratio[w]:.2f})" for w in top_neg))

    # D) 새 문장 일반화 점검 (Day-010 과 같은 두 문장)
    print()
    for s_txt in ["a wonderful and amazing film i loved", "a boring and awful movie i hated"]:
        p = nb.predict_proba_pos(s_txt)
        print(f"  P(pos)={p:.3f} -> {'긍정' if p >= 0.5 else '부정'} | {s_txt}")
