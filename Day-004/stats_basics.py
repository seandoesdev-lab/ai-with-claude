# stats_basics.py — 중심/퍼짐, 정규분포, 표준화(z-score)
import sys
import numpy as np

# Windows 콘솔(cp949)에서 한글·특수기호가 깨지거나 에러 나는 것을 막아 줍니다.
sys.stdout.reconfigure(encoding="utf-8")

rng = np.random.default_rng(42)   # seed 고정 → 재현 가능

print("=" * 56)
print("[1] 중심과 퍼짐: 평균(mean)·분산(var)·표준편차(std)")
print("=" * 56)
scores = np.array([60, 70, 80, 90, 100], dtype=float)
print("점수            :", scores)
print("평균   mean     :", scores.mean())          # 80.0
print("분산   variance :", scores.var())           # 200.0 (편차 제곱의 평균)
print("표준편차 std    :", round(scores.std(), 4)) # 14.1421...

# '직접' 계산해서 공식 확인하기
dev = scores - scores.mean()                        # 편차(deviation)
print("편차            :", dev)
print("분산(직접 계산) :", (dev ** 2).mean())       # == scores.var()

# 모분산(ddof=0, 기본) vs 표본분산(ddof=1)
print("표본분산 ddof=1 :", round(scores.var(ddof=1), 4))  # N-1 로 나눔 → 더 큼

print("\n" + "=" * 56)
print("[2] 정규분포(normal) 만들고 통계 재기")
print("=" * 56)
# 평균 170, 표준편차 6 인 '키' 데이터 10,000개를 정규분포에서 샘플링
heights = rng.normal(loc=170, scale=6, size=10_000)
print("표본 평균    :", round(heights.mean(), 3))   # ~170
print("표본 표준편차:", round(heights.std(), 3))     # ~6

# 68-95-99.7 규칙을 '비율'로 직접 검증
mu, sd = heights.mean(), heights.std()
theory = [68.3, 95.4, 99.7]
for k in (1, 2, 3):
    inside = np.mean(np.abs(heights - mu) < k * sd) * 100
    print(f"  평균 +/-{k}sigma 안의 비율: {inside:5.1f}%  (이론값 {theory[k-1]}%)")

print("\n" + "=" * 56)
print("[3] 텍스트 히스토그램으로 '종 모양' 직접 보기")
print("=" * 56)
counts, edges = np.histogram(heights, bins=15)
peak = counts.max()
for c, left, right in zip(counts, edges[:-1], edges[1:]):
    bar = "#" * int(round(c / peak * 40))
    print(f"{left:6.1f} ~ {right:6.1f} | {bar} {c}")

print("\n" + "=" * 56)
print("[4] 표준화(standardization, z-score) — 스케일 맞추기")
print("=" * 56)
# 특징 2개: 키(cm, ~170)와 몸무게(kg, ~65). 단위·스케일이 전혀 다름!
feat = np.array([[172, 68],
                 [160, 55],
                 [185, 90],
                 [168, 60]], dtype=float)   # shape (4, 2): 4명 x 2특징
print("원본 (키, 몸무게):\n", feat)
mu = feat.mean(axis=0)          # 열별(특징별) 평균  → (2,)
sd = feat.std(axis=0)           # 열별 표준편차       → (2,)
z = (feat - mu) / sd            # 브로드캐스팅: (4,2) - (2,) / (2,)
print("표준화 후 z (평균0, 표준편차1):\n", z.round(2))
print("표준화 후 열별 평균    :", z.mean(axis=0).round(6))  # ~ [0, 0]
print("표준화 후 열별 표준편차:", z.std(axis=0).round(6))   # ~ [1, 1]
