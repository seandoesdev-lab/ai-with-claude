# arrays_and_indexing.py — 배열의 해부 + 인덱싱/슬라이싱/마스킹
# 실행: uv run python arrays_and_indexing.py
import numpy as np

print("=" * 50)
print("[1] 배열의 해부도 (shape / ndim / dtype)")
print("=" * 50)

M = np.array([[1, 2, 3],
              [4, 5, 6]])
print("M =\n", M)
print("shape :", M.shape)   # (2, 3)
print("ndim  :", M.ndim)    # 2
print("dtype :", M.dtype)   # int64 (환경따라 int32일 수 있음)
print("size  :", M.size)    # 6

# dtype 을 지정해 만들기 (딥러닝은 float32 를 즐겨 씀)
f = np.array([1, 2, 3], dtype=np.float32)
print("float32 배열:", f, "| dtype:", f.dtype)

print("\n" + "=" * 50)
print("[2] 배열 만들기")
print("=" * 50)
print("arange(0,10,2) :", np.arange(0, 10, 2))
print("linspace(0,1,5):", np.linspace(0, 1, 5))
print("zeros((2,3))   :\n", np.zeros((2, 3)))
print("eye(3)         :\n", np.eye(3))

rng = np.random.default_rng(42)          # seed 고정 → 재현 가능
print("random (2,3)   :\n", rng.random((2, 3)).round(3))

print("\n" + "=" * 50)
print("[3] 인덱싱 & 슬라이싱 (2차원은 [행, 열])")
print("=" * 50)
G = np.arange(10, 19).reshape(3, 3)
print("G =\n", G)
print("G[0, 2]   (0행 2열):", G[0, 2])     # 12
print("G[1]      (1행 전체):", G[1])        # [13 14 15]
print("G[:, 0]   (0열 전체):", G[:, 0])     # [10 13 16]
print("G[0:2, 1:](부분 사각형):\n", G[0:2, 1:])

print("\n" + "=" * 50)
print("[4] 팬시 인덱싱 & 불리언 마스킹")
print("=" * 50)
a = np.array([10, 20, 30, 40, 50])
print("a              :", a)
print("a[[0, 2, 4]]   :", a[[0, 2, 4]])     # [10 30 50]
mask = a > 25
print("mask (a>25)    :", mask)             # [F F T T T]
print("a[a > 25]      :", a[a > 25])        # [30 40 50]

b = a.copy()
b[b > 25] = 0                               # 조건 위치만 0으로
print("조건부 수정 후 :", b)                 # [10 20 0 0 0]

print("\n" + "=" * 50)
print("[5] 뷰(view) vs 복사(copy) - 가장 흔한 함정")
print("=" * 50)
original = np.array([0, 1, 2, 3, 4])
view = original[1:4]        # 슬라이스 = 뷰 (복사 아님!)
view[0] = 999               # 뷰를 고치면...
print("original :", original)   # [  0 999   2   3   4]  ← 원본도 바뀜!

original2 = np.array([0, 1, 2, 3, 4])
safe = original2[1:4].copy()    # .copy() 로 진짜 복사
safe[0] = 999
print("original2:", original2)  # [0 1 2 3 4]            ← 원본 그대로 (안전)
