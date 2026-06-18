# cooccurrence.py — 분포 가설(distributional hypothesis)을 손으로 확인하기
# "비슷한 문맥에 나오는 단어는 비슷한 뜻을 가진다"를 표준 라이브러리만으로 증명한다.
import sys, re, math
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding="utf-8")   # Windows 콘솔 한글 깨짐 방지

corpus = [
    "the cat drinks milk",     # S0
    "the dog drinks water",    # S1
    "the cat eats fish",       # S2
    "the dog eats meat",       # S3
    "i drive my car",          # S4
    "she drives her car",      # S5
]

def tokenize(text):
    return re.findall(r"[a-z]+", text.lower())

docs = [tokenize(s) for s in corpus]
vocab = sorted({w for d in docs for w in d})

# 1) 동시출현(co-occurrence) 세기: 같은 문장에 함께 나온 단어를 센다(문장 전체를 window로).
#    co[w][c] = 단어 w 와 문맥어 c 가 같은 문장에 함께 나온 횟수.
co = defaultdict(Counter)
for d in docs:
    for i, w in enumerate(d):
        for j, c in enumerate(d):
            if i != j:
                co[w][c] += 1

# 각 단어의 '문맥 벡터(context vector)' = 그 단어 주변에 어떤 단어들이 나오는가
def cosine(a, b):
    common = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0

print("어휘 크기(vocab size):", len(vocab))
print("\n[cat 의 문맥 벡터]", dict(co["cat"]))
print("[dog 의 문맥 벡터]", dict(co["dog"]))
print("[car 의 문맥 벡터]", dict(co["car"]))

# 2) 문맥 벡터끼리 코사인 유사도 (Day-004 와 연결)
pairs = [("cat", "dog"), ("cat", "car"), ("dog", "car")]
print("\n[문맥 기반 코사인 유사도]")
for a, b in pairs:
    print(f"  sim({a}, {b}) = {cosine(co[a], co[b]):.4f}")

print("\n→ cat·dog 는 문맥(the/drinks/eats)을 공유 → 유사도 높음.")
print("→ car 는 다른 동네(i/drive/my...) → cat·dog 와 유사도 0.")
print("  이것이 '분포 가설': 의미는 '함께 나타나는 단어들'에 새어 있다.")
