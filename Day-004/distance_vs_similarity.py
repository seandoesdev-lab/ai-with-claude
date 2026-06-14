# distance_vs_similarity.py — 유클리드 거리 vs 코사인 유사도
import sys
import numpy as np

# Windows 콘솔(cp949)에서 한글·특수기호가 깨지거나 에러 나는 것을 막아 줍니다.
sys.stdout.reconfigure(encoding="utf-8")

def euclidean(a, b):
    """두 벡터 사이의 유클리드 거리(L2): 직선 거리. 작을수록 가깝다."""
    return np.sqrt(((a - b) ** 2).sum())

def cosine(a, b):
    """코사인 유사도: 방향(각도)이 얼마나 같은가. -1~1, 클수록 비슷."""
    return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("=" * 56)
print("[1] '거리'는 멀고가까움, '코사인'은 방향(크기 무시)")
print("=" * 56)
a = np.array([1.0, 0.0])
b = np.array([0.0, 1.0])
c = np.array([2.0, 0.0])   # a 와 '같은 방향', 길이만 2배
print(f"a={a}, b={b}, c={c}")
print(f"  d(a,b)={euclidean(a,b):.3f}   cos(a,b)={cosine(a,b):.3f}  (직각->0)")
print(f"  d(a,c)={euclidean(a,c):.3f}   cos(a,c)={cosine(a,c):.3f}  (같은 방향->1)")
print("-> a와 c는 방향이 같아 코사인=1.0, 하지만 길이가 달라 거리는 1.0")

print("\n" + "=" * 56)
print("[2] 검색에서 왜 보통 '코사인'을 쓰나 — 문서 길이 문제")
print("=" * 56)
# 단어 [고양이, 강아지, 자동차] 의 등장 횟수(term frequency) 벡터
query = np.array([1, 1, 0])      # 짧은 질문: 고양이 강아지
short = np.array([1, 1, 0])      # 같은 주제, 짧은 문서
long_ = np.array([10, 10, 1])    # 같은 주제, 긴 문서 (횟수만 큼)
junk  = np.array([0, 0, 5])      # 다른 주제: 자동차
for name, d in [("short(같은주제·짧음)", short),
                ("long (같은주제·긺)", long_),
                ("junk (다른주제)", junk)]:
    print(f"  {name:20s} 거리={euclidean(query,d):6.2f}  코사인={cosine(query,d):.3f}")
print("-> 거리로는 'long'이 멀어 보이지만(길이 탓!), 코사인은 주제가 같음을 알아챔")

print("\n" + "=" * 56)
print("[3] 정규화하면: (유클리드 거리)^2 = 2 * (1 - 코사인유사도)")
print("=" * 56)
def unit(v):
    return v / np.linalg.norm(v)
x = np.array([3.0, 4.0, 0.0])
y = np.array([4.0, 0.0, 3.0])
xu, yu = unit(x), unit(y)
d2  = euclidean(xu, yu) ** 2
rel = 2 * (1 - cosine(xu, yu))
print(f"  단위벡터 거리^2 = {d2:.4f}")
print(f"  2 * (1 - cos)   = {rel:.4f}   <- 같다!")
print("-> 정규화된 임베딩에서는 '거리 정렬'과 '코사인 정렬'이 같은 순위를 준다")

print("\n" + "=" * 56)
print("[4] 같은 데이터, 두 랭킹 (거리 작을수록 가까움 vs 코사인 클수록 비슷)")
print("=" * 56)
docs = np.array([[1.0, 1.0, 0.0],
                 [10.0, 10.0, 1.0],
                 [0.0, 0.0, 5.0],
                 [2.0, 1.0, 0.0]])
q = np.array([1.0, 1.0, 0.0])

dist = np.sqrt(((docs - q) ** 2).sum(axis=1))            # 각 문서까지 거리
qn = q / np.linalg.norm(q)
dn = docs / np.linalg.norm(docs, axis=1, keepdims=True)  # 행별 정규화 (Day 003!)
cos = dn @ qn                                            # 코사인 유사도

print("문서별 거리   :", dist.round(3), " -> 가까운 순:", np.argsort(dist))
print("문서별 코사인 :", cos.round(3), " -> 비슷한 순:", np.argsort(cos)[::-1])
print("문서 1(긴 같은주제)이 거리 랭킹에선 밀리고, 코사인 랭킹에선 상위인지 비교!")
