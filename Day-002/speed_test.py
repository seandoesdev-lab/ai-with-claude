# speed_test.py — 순수 파이썬 반복 vs NumPy 벡터화 속도 비교
# 실행: python speed_test.py
import time
import numpy as np

N = 5_000_000  # 500만 개

# (A) 순수 파이썬 리스트 + 반복문
py_list = list(range(N))
t0 = time.perf_counter()
py_result = [x * 2 for x in py_list]
t1 = time.perf_counter()
print(f"순수 파이썬 반복 : {t1 - t0:.4f} 초")

# (B) NumPy 배열 + 벡터화
np_arr = np.arange(N)
t2 = time.perf_counter()
np_result = np_arr * 2
t3 = time.perf_counter()
print(f"NumPy 벡터화     : {t3 - t2:.4f} 초")

print(f"NumPy 가 약 {(t1 - t0) / (t3 - t2):.1f} 배 빠름")
