# ngram_lm.py — n-gram 언어모델: 카운트로 다음 단어 예측 + 스무딩 + 퍼플렉서티 + 생성
# Day-011 (AI with Claude). 외부 라이브러리 없이 순수 파이썬으로 동작합니다.
#
# 실행: uv run python ngram_lm.py
import sys
import math
import random
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding="utf-8")   # Windows 콘솔 한글 깨짐 방지
random.seed(0)

# ─────────────────────────────────────────────────────────────────────────────
# 0) 아주 작은 학습 코퍼스 (개념 시연용). 문장 경계를 위해 <s>(시작)·</s>(끝) 토큰 사용.
# ─────────────────────────────────────────────────────────────────────────────
CORPUS = [
    "i love this movie",
    "i love this great film",
    "i really love this film",
    "this movie is great",
    "this film is great and fun",
    "i hate this boring movie",
    "i hate this film",
    "this movie is boring",
    "this film is boring and dull",
    "i really hate this boring film",
]


def tokenize(sentence, n=2):
    """공백 토큰화 + 문장 경계 토큰. 시작 토큰 <s> 를 n-1 개 붙여
    첫 단어도 길이 n-1 의 문맥을 갖도록 한다(특히 trigram 이상에서 중요)."""
    return ["<s>"] * (n - 1) + sentence.split() + ["</s>"]


# ─────────────────────────────────────────────────────────────────────────────
# 1) n-gram 카운트 만들기
#    bigram(n=2): 직전 1단어(context)로 다음 단어를 예측 → P(w_t | w_{t-1})
# ─────────────────────────────────────────────────────────────────────────────
def build_counts(corpus, n=2):
    """원문 코퍼스를 받아 context(앞 n-1개) → {다음단어: 횟수} 카운트와 어휘를 만든다."""
    ctx_counts = defaultdict(Counter)        # context → Counter(다음단어)
    vocab = set()
    for sentence in corpus:
        toks = tokenize(sentence, n=n)
        for w in toks:
            vocab.add(w)
        for i in range(len(toks) - n + 1):
            context = tuple(toks[i:i + n - 1])   # 앞 n-1개
            nxt = toks[i + n - 1]                 # 그 다음 1개
            ctx_counts[context][nxt] += 1
    vocab.discard("<s>")     # <s> 는 예측 대상(다음 단어)이 되지 않음
    return ctx_counts, vocab


# ─────────────────────────────────────────────────────────────────────────────
# 2) add-k(라플라스) 스무딩이 적용된 확률
#    P(w | context) = (count(context, w) + k) / (count(context) + k*V)
#    → 학습에서 한 번도 안 본 조합도 0 이 아니라 작은 확률을 갖는다.
# ─────────────────────────────────────────────────────────────────────────────
def prob(ctx_counts, vocab, context, word, k=1.0):
    V = len(vocab)
    c_ctx = ctx_counts.get(context, Counter())
    numerator = c_ctx[word] + k
    denominator = sum(c_ctx.values()) + k * V
    return numerator / denominator


# ─────────────────────────────────────────────────────────────────────────────
# 3) 퍼플렉서티(perplexity): 모델이 테스트 문장을 얼마나 "덜 놀라며" 예측하나.
#    PP = exp( - (1/N) * Σ log P(w_i | context) ).  낮을수록 좋다.
# ─────────────────────────────────────────────────────────────────────────────
def perplexity(ctx_counts, vocab, sentence, n=2, k=1.0):
    toks = tokenize(sentence, n=n)
    log_sum = 0.0
    count = 0
    for i in range(n - 1, len(toks)):
        context = tuple(toks[i - n + 1:i])
        p = prob(ctx_counts, vocab, context, toks[i], k=k)
        log_sum += math.log(p)
        count += 1
    return math.exp(-log_sum / count)


# ─────────────────────────────────────────────────────────────────────────────
# 4) 생성(generation): 시작 토큰부터 다음 단어를 확률대로 "샘플링" 해 문장을 만든다.
# ─────────────────────────────────────────────────────────────────────────────
def generate(ctx_counts, vocab, n=2, k=1.0, max_len=12):
    sorted_vocab = sorted(vocab)
    context = tuple(["<s>"] * (n - 1))
    out = []
    for _ in range(max_len):
        probs = [prob(ctx_counts, vocab, context, w, k=k) for w in sorted_vocab]
        nxt = random.choices(sorted_vocab, weights=probs, k=1)[0]
        if nxt == "</s>":
            break
        out.append(nxt)
        context = tuple((list(context) + [nxt])[-(n - 1):]) if n > 1 else ()
    return " ".join(out)


# ─────────────────────────────────────────────────────────────────────────────
# 데모
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    N = 2  # bigram
    ctx_counts, vocab = build_counts(CORPUS, n=N)
    print(f"어휘 크기 V = {len(vocab)}  (예: {sorted(vocab)[:6]} ...)")

    print("\n[다음 단어 예측] context='this' 다음에 올 단어 확률 Top-5:")
    ctx = ("this",)
    ranked = sorted(vocab, key=lambda w: prob(ctx_counts, vocab, ctx, w, k=1.0), reverse=True)
    for w in ranked[:5]:
        print(f"  P({w:>7} | this) = {prob(ctx_counts, vocab, ctx, w, k=1.0):.3f}")

    print("\n[퍼플렉서티] (낮을수록 모델이 덜 놀란다)")
    for s in ["i love this movie", "i hate this boring film", "purple monster eats algebra"]:
        pp = perplexity(ctx_counts, vocab, s, n=N, k=1.0)
        print(f"  PP = {pp:7.2f}  | {s}")

    print("\n[문장 생성] bigram 으로 새 문장 5개 샘플링:")
    for _ in range(5):
        print("  -", generate(ctx_counts, vocab, n=N, k=1.0))

    # 보너스: trigram(N=3). 문맥이 길어지면 더 그럴듯해지지만 데이터 희소성이 커진다.
    # 작은 데이터에서 add-1(k=1) 은 V배 스무딩 질량이 카운트를 압도해 거의 균일분포가 된다.
    # → k 를 작게(0.05) 주면 학습 문장을 거의 "외워" 더 유창하게 생성한다(과적합의 다른 얼굴).
    print("\n[trigram 비교] N=3, k=1.0 (과한 스무딩 → 거의 무작위):")
    cc3, vocab3 = build_counts(CORPUS, n=3)
    for _ in range(3):
        print("  -", generate(cc3, vocab3, n=3, k=1.0))
    print("[trigram 비교] N=3, k=0.05 (약한 스무딩 → 학습문장을 거의 암기):")
    for _ in range(3):
        print("  -", generate(cc3, vocab3, n=3, k=0.05))
