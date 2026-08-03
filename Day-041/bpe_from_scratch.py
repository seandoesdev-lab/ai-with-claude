"""
Day-041 — BPE 토크나이저 from scratch (byte-level Byte-Pair Encoding)

의존성 없음(순수 파이썬 표준 라이브러리). 실행:
    uv run python bpe_from_scratch.py
한글이 깨지면 먼저:  $env:PYTHONIOENCODING="utf-8"

다섯 가지 실험
  1) BPE 가 무엇을 '학습'하는가 — merge 규칙이 생기는 순서
  2) 어휘 크기를 키우면 같은 문장이 어떻게 달라지는가 (300 / 500 / 1000 / 3000)
  3) 왜 한국어가 토큰을 더 먹는가 — 언어의 성질인가, 학습 데이터의 성질인가
  4) 학습에 없던 문자는 어떻게 되는가 (byte fallback = OOV 없음)
  5) 'strawberry 의 r 은 몇 개인가' 실패의 기계적 원인
"""

from collections import Counter
import re
import time

# ──────────────────────────────────────────────────────────────────────────
# 0. 학습 코퍼스 — 한국어/영어 각각 비슷한 분량 (이 트랙의 주제로 채웠다)
# ──────────────────────────────────────────────────────────────────────────

CORPUS_KO = """
정보 검색은 사용자의 질의에 가장 적합한 문서를 찾아 순위를 매기는 일이다.
역색인은 단어에서 문서 목록으로 가는 사전이며, 검색 속도의 핵심이다.
BM25 는 단어 빈도의 포화와 문서 길이 정규화를 함께 고려하는 랭킹 함수이다.
질의어가 문서에 글자 그대로 없으면 어휘 기반 검색은 그 문서를 찾지 못한다.
이 문제를 어휘 불일치라고 부르며, 임베딩은 그 해법 가운데 하나이다.
단어 임베딩은 비슷한 문맥에 나타나는 단어를 비슷한 벡터로 만든다.
분포 가설은 단어의 의미가 함께 나타나는 단어들로 결정된다고 말한다.
신경망은 여러 층의 선형 변환과 비선형 활성함수를 쌓아 만든 함수이다.
역전파는 연쇄법칙으로 손실의 기울기를 뒤로 흘려 가중치를 갱신한다.
순환 신경망은 은닉 상태를 다음 시점으로 넘기며 시퀀스를 처리한다.
고정 크기의 은닉 상태는 긴 문장의 정보를 모두 담기에는 좁은 병목이다.
어텐션은 모든 위치를 저장해 두고 필요할 때 직접 참조하는 방식이다.
트랜스포머는 순환을 완전히 버리고 자기 어텐션만으로 시퀀스를 다룬다.
인코더는 양방향으로 문맥을 읽고, 디코더는 왼쪽 문맥만 보며 생성한다.
사전학습은 큰 말뭉치에서 일반적인 표현을 먼저 배우는 단계이다.
파인튜닝은 사전학습된 가중치를 특정 과제에 맞게 다시 조정하는 것이다.
문맥 내 학습은 가중치를 바꾸지 않고 프롬프트의 예시만으로 과제를 푼다.
언어모델은 앞의 토큰들을 보고 다음 토큰의 확률 분포를 내놓는 함수이다.
토큰화는 텍스트를 모델이 다루는 최소 단위로 쪼개는 전처리 과정이다.
한국어는 조사와 어미가 붙는 교착어라서 단어 단위 어휘가 크게 늘어난다.
검색 증강 생성은 검색한 문서를 프롬프트에 넣어 답을 생성하는 구조이다.
평가 지표는 정밀도와 재현율에서 시작해 순위를 반영하는 지표로 확장된다.
벡터 데이터베이스는 임베딩을 저장하고 가장 가까운 이웃을 빠르게 찾는다.
하이브리드 검색은 어휘 기반 점수와 의미 기반 점수를 함께 사용한다.
리랭킹은 먼저 넓게 후보를 뽑고 더 정확한 모델로 상위만 다시 정렬한다.
학습률은 한 번의 갱신에서 가중치를 얼마나 움직일지 정하는 값이다.
과적합은 학습 데이터에만 맞고 새로운 데이터에서는 성능이 떨어지는 현상이다.
정규화는 모델이 지나치게 복잡해지는 것을 막아 일반화를 돕는 기법이다.
경사하강법은 기울기의 반대 방향으로 조금씩 움직여 손실을 줄여 간다.
확률 분포에서 다음 토큰을 뽑는 규칙을 디코딩 전략이라고 부른다.
"""

CORPUS_EN = """
Information retrieval is the task of finding and ranking documents for a query.
An inverted index maps a term to the list of documents that contain the term.
BM25 is a ranking function with term frequency saturation and length normalization.
If the query words do not appear literally in a document, lexical search fails.
This failure is called vocabulary mismatch, and embeddings are one answer to it.
Word embeddings place words that appear in similar contexts near each other.
The distributional hypothesis says meaning is determined by the company a word keeps.
A neural network stacks linear transformations and nonlinear activation functions.
Backpropagation sends the gradient of the loss backward through the chain rule.
A recurrent network carries a hidden state forward across the sequence.
A fixed size hidden state is a narrow bottleneck for a long sentence.
Attention stores every position and reads the ones that matter on demand.
The transformer drops recurrence entirely and uses only self attention.
An encoder reads context in both directions while a decoder looks only left.
Pretraining learns general representations from a very large corpus first.
Finetuning adjusts pretrained weights for one particular downstream task.
In context learning solves a task from prompt examples without weight updates.
A language model maps previous tokens to a distribution over the next token.
Tokenization splits raw text into the smallest units the model can consume.
English words are mostly separated by spaces, which makes segmentation easier.
Retrieval augmented generation puts retrieved documents into the prompt.
Evaluation starts from precision and recall and extends to ranked metrics.
A vector database stores embeddings and finds nearest neighbours quickly.
Hybrid retrieval combines a lexical score with a semantic similarity score.
Reranking retrieves a wide candidate set and reorders the top results.
The learning rate controls how far the weights move at each update step.
Overfitting means the model fits training data but generalizes poorly.
Regularization keeps a model from becoming needlessly complex.
Gradient descent takes small steps in the direction that reduces the loss.
The rule for sampling the next token is called a decoding strategy.
"""

# 평가용(학습에 쓰지 않은) 문장
TEST_KO = "밀집 검색은 질문과 문서를 같은 벡터 공간에 두고 가까운 것을 고른다."
TEST_EN = "Dense retrieval embeds the question and the document into one shared vector space."


# ──────────────────────────────────────────────────────────────────────────
# 1. BPE 학습 — 가장 자주 인접하는 바이트 쌍을 반복해서 합친다
# ──────────────────────────────────────────────────────────────────────────

def pretokenize(text):
    """단어 경계를 보존한 사전 분할. 앞 공백을 단어에 붙인다(GPT-2 관행의 단순화)."""
    return re.findall(r"\s*\S+", text)


def pair_stats(words, freqs):
    """현재 분절 상태에서 인접 쌍의 등장 빈도."""
    stats = Counter()
    for w, syms in words.items():
        f = freqs[w]
        for pair in zip(syms, syms[1:]):
            stats[pair] += f
    return stats


def merge_word(syms, pair):
    """한 단어 안에서 pair 를 전부 하나로 합친다."""
    a, b = pair
    out, i, n = [], 0, len(syms)
    while i < n:
        if i < n - 1 and syms[i] == a and syms[i + 1] == b:
            out.append(a + b)
            i += 2
        else:
            out.append(syms[i])
            i += 1
    return tuple(out)


def train_bpe(corpus, vocab_size, trace=0):
    """byte-level BPE 학습. 반환: (merges, 학습로그). merges 의 '순서'가 곧 규칙이다."""
    freqs = Counter(pretokenize(corpus))
    # 시작점: 모든 단어를 '바이트 하나'씩 쪼갠 상태. 기본 어휘 = 256개 바이트.
    words = {w: tuple(bytes([b]) for b in w.encode("utf-8")) for w in freqs}

    merges, log = [], []
    for step in range(vocab_size - 256):
        stats = pair_stats(words, freqs)
        if not stats:
            break
        pair, count = stats.most_common(1)[0]
        if count < 2:                      # 한 번만 나오는 쌍은 합칠 가치가 없다
            break
        merges.append(pair)
        words = {w: merge_word(s, pair) for w, s in words.items()}
        if step < trace:
            log.append((step + 1, pair[0] + pair[1], count))
    return merges, log


def build_ranks(merges):
    return {pair: i for i, pair in enumerate(merges)}


def encode(text, ranks):
    """학습 때와 같은 순서로 merge 를 적용한다(먼저 배운 규칙이 먼저)."""
    tokens = []
    for w in pretokenize(text):
        syms = [bytes([b]) for b in w.encode("utf-8")]
        while len(syms) > 1:
            best_rank, best_i = None, -1
            for i in range(len(syms) - 1):
                r = ranks.get((syms[i], syms[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_rank is None:
                break
            syms[best_i:best_i + 2] = [syms[best_i] + syms[best_i + 1]]
        tokens.extend(syms)
    return tokens


def show_one(t):
    """토큰 하나를 눈으로 보기 좋게.
    공백은 ·, 글자를 완성하지 못한 조각은 ⟨16진수⟩ 로 드러낸다."""
    prefix = ""
    if t[:1] == b" ":
        prefix, t = "·", t[1:]
    try:
        return prefix + t.decode("utf-8").replace("\n", "\\n")
    except UnicodeDecodeError:
        return prefix + "⟨" + t.hex() + "⟩"     # 글자 경계를 가로지르는 조각


def show(tokens):
    return "|".join(show_one(t) for t in tokens)


def partial_ratio(tokens):
    """글자를 완성하지 못한 토큰의 비율 — 바이트 BPE 가 그 언어를 얼마나 '못 배웠나'."""
    bad = 0
    for t in tokens:
        try:
            t.decode("utf-8")
        except UnicodeDecodeError:
            bad += 1
    return bad / len(tokens)


def header(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ──────────────────────────────────────────────────────────────────────────
# 실험 1 — BPE 는 무엇을 '학습'하는가
# ──────────────────────────────────────────────────────────────────────────

BALANCED = CORPUS_KO + CORPUS_EN

header("실험 1 — merge 규칙이 생기는 순서 (균형 코퍼스, 처음 24개)")
t0 = time.time()
merges_bal, log = train_bpe(BALANCED, vocab_size=3000, trace=24)
train_sec = time.time() - t0
print(f"  학습 시간 {train_sec:.2f}s · merge 규칙 {len(merges_bal)}개 · 어휘 {256 + len(merges_bal)}개\n")
print(f"  {'#':>3}  {'합쳐진 조각':<16} {'빈도':>5}   {'#':>3}  {'합쳐진 조각':<16} {'빈도':>5}")
half = (len(log) + 1) // 2
for i in range(half):
    l = log[i]
    row = f"  {l[0]:>3}  {show([l[1]]):<16} {l[2]:>5}"
    if i + half < len(log):
        r = log[i + half]
        row += f"   {r[0]:>3}  {show([r[1]]):<16} {r[2]:>5}"
    print(row)

print("\n  주목: 초기 merge 는 대부분 '한 글자를 완성하는' 일이다.")
print("  한국어 한 글자는 UTF-8 로 3바이트라, 글자 하나가 되기까지 merge 가 2번 필요하다.")

# ──────────────────────────────────────────────────────────────────────────
# 실험 2 — 어휘 크기를 키우면 분절이 어떻게 변하나
# ──────────────────────────────────────────────────────────────────────────

header("실험 2 — 같은 문장, 어휘 크기만 다르게")

for vs in (300, 500, 1000, 3000):
    merges = train_bpe(BALANCED, vocab_size=vs)[0]
    got = 256 + len(merges)                    # 실제로 도달한 어휘 크기
    ranks = build_ranks(merges)
    ko, en = encode(TEST_KO, ranks), encode(TEST_EN, ranks)
    note = "" if got == vs else f"  (요청 {vs} -> 실제 {got}: merge 고갈)"
    print(f"\n  [어휘 {got}]  KO {len(ko):>3}토큰 · EN {len(en):>3}토큰{note}")
    print(f"    KO  {show(ko)}")
    print(f"    EN  {show(en)}")

print("\n  어휘를 키울수록 조각이 길어진다 = 같은 문장이 적은 토큰이 된다.")
print("  대신 어휘 자체가 커져 임베딩 행렬(V x d)과 출력 소프트맥스가 함께 커진다.")
print("  그리고 어휘는 무한히 키울 수 없다 — 코퍼스에 '두 번 이상 붙어 나오는 쌍'이")
print("  떨어지는 순간 merge 가 멈춘다. 큰 어휘는 큰 데이터를 전제한다.")

# ──────────────────────────────────────────────────────────────────────────
# 실험 3 — 한국어는 왜 토큰을 더 먹는가
# ──────────────────────────────────────────────────────────────────────────

header("실험 3 — 언어의 성질인가, 학습 데이터의 성질인가")

VOCAB = 3000
setups = [
    ("균형      (KO 1 : EN 1)", CORPUS_KO * 1 + CORPUS_EN * 1),
    ("영어 편중  (KO 1 : EN 9)", CORPUS_KO * 1 + CORPUS_EN * 9),
    ("한국어 편중(KO 9 : EN 1)", CORPUS_KO * 9 + CORPUS_EN * 1),
]

print("\n  평가 문장 (학습 코퍼스에 없음)")
print(f"    KO: {TEST_KO}")
print(f"        {len(TEST_KO)}자, UTF-8 {len(TEST_KO.encode())}바이트")
print(f"    EN: {TEST_EN}")
print(f"        {len(TEST_EN)}자, UTF-8 {len(TEST_EN.encode())}바이트")

print(f"\n  {'학습 코퍼스 구성':<26} {'어휘':>5} {'KO토큰':>6} {'EN토큰':>6} {'KO자/톡':>8} {'EN자/톡':>8} {'비용배수':>8} {'KO 미완성':>9}")
print("  " + "-" * 88)
for name, corpus in setups:
    merges = train_bpe(corpus, vocab_size=VOCAB)[0]
    ranks = build_ranks(merges)
    tk, te = encode(TEST_KO, ranks), encode(TEST_EN, ranks)
    nk, ne = len(tk), len(te)
    ck, ce = len(TEST_KO) / nk, len(TEST_EN) / ne
    print(f"  {name:<26} {256 + len(merges):>5} {nk:>6} {ne:>6} {ck:>8.2f} {ce:>8.2f}"
          f" {ce / ck:>7.2f}x {partial_ratio(tk):>8.0%}")

print("\n  '비용배수' = 같은 글자 수를 보낼 때 한국어가 영어의 몇 배 토큰을 쓰는가.")
print("  'KO 미완성' = 한 글자를 완성하지 못한 조각(⟨16진수⟩)의 비율.")
print("  영어 편중 코퍼스에서 배수가 커지고, 한국어 편중에서는 뒤집힌다.")
print("  => 한국어가 본질적으로 비싼 게 아니라, 어휘를 누구 데이터로 만들었느냐의 문제다.")

# ──────────────────────────────────────────────────────────────────────────
# 실험 4 — 미지의 문자는 어떻게 되는가 (byte fallback)
# ──────────────────────────────────────────────────────────────────────────

header("실험 4 — 학습에 없던 문자: OOV 가 없다는 뜻")

ranks_bal = build_ranks(merges_bal)
for s in ["딥러닝", "🍓 딸기", "Ωμέγα", "龘"]:
    toks = encode(s, ranks_bal)
    print(f"  {s:<10} -> {len(toks):>2}토큰  {show(toks)}")
print("\n  어떤 문자열도 바이트로는 표현되므로 <UNK> 가 필요 없다.")
print("  대신 학습에 없던 문자는 '바이트 낱개'로 흩어져 토큰을 많이 먹는다(⟨..⟩ = 미완성 조각).")

# ──────────────────────────────────────────────────────────────────────────
# 실험 5 — strawberry 의 r 은 몇 개인가
# ──────────────────────────────────────────────────────────────────────────

header("실험 5 — 'strawberry 의 r 개수' 실패의 기계적 원인")

CORPUS_BERRY = (CORPUS_EN * 6) + "\nstrawberry blueberry raspberry blackberry " * 40
merges_b = train_bpe(CORPUS_BERRY, vocab_size=1200)[0]
ranks_b = build_ranks(merges_b)
vocab_ids = {tok: 256 + i for i, tok in enumerate(a + b for a, b in merges_b)}


def ids_of(toks):
    return [vocab_ids.get(t, t[0] if len(t) == 1 else -1) for t in toks]


print("\n  모델이 실제로 보는 것 (어휘 1200, berry 계열을 자주 본 코퍼스):")
for w in ["strawberry", " strawberry", "raspberry", "blueberry"]:
    toks = encode(w, ranks_b)
    print(f"    {w!r:<14} -> {len(toks)}토큰  {show(toks):<26} ID {ids_of(toks)}")

toks = encode("strawberry", ranks_b)
print(f"\n  'strawberry' 안의 r 은 3개다. 그런데 토큰 경계는 {show(toks)} 이다.")
print(f"  모델의 입력은 글자열이 아니라 위 조각에 배정된 정수 ID {ids_of(toks)} 이다.")
print("  즉 'r 이 몇 개인가'는 입력에 직접 드러나 있지 않고, 임베딩이 철자를")
print("  얼마나 기억하느냐에 달린 간접 추론 문제가 된다.")

spaced = " ".join("strawberry")
print("\n  한 글자씩 띄우면(= 토큰 경계를 강제로 글자에 맞추면) 문제가 쉬워진다:")
print(f"    {spaced!r}")
print(f"    -> {len(encode(spaced, ranks_b))}토큰  {show(encode(spaced, ranks_b))}")
print("  같은 모델·같은 질문인데 토큰화만 바꿔서 난이도가 달라진다는 점이 핵심이다.")

header("끝 — §4.6 의 실험 제안을 직접 돌려 보세요")
