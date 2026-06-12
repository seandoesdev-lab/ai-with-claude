# first_numpy.py — 첫 NumPy 배열과 "벡터화" 체험
# 실행: uv run python first_numpy.py  (프로젝트에서 `uv add numpy` 후)
import numpy as np

# 1) 파이썬 리스트가 아니라 NumPy 배열(ndarray)을 만든다
a = np.array([1, 2, 3, 4, 5])
print("배열 a       :", a)

# 2) 벡터화(vectorization): 반복문 없이 '한 줄'로 모든 원소에 연산
print("a * 2        :", a * 2)          # [ 2  4  6  8 10]
print("a 의 제곱     :", a ** 2)         # [ 1  4  9 16 25]
print("a 의 평균     :", a.mean())       # 3.0
print("a 의 표준편차 :", a.std().round(3))

# 3) 2차원 배열(행렬)과 모양(shape)
M = np.array([[1, 2, 3],
              [4, 5, 6]])
print("행렬 M 의 shape:", M.shape)       # (2, 3) → 2행 3열
print("M 의 전체 합   :", M.sum())        # 21
print("열별 합(axis=0):", M.sum(axis=0)) # [5 7 9]
