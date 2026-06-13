# broadcasting.py — 브로드캐스팅 규칙 + "임베딩 유사도 검색" 미니 체험
# 실행: uv run python broadcasting.py
import numpy as np

print("=" * 50)
print("[1] 브로드캐스팅 기본")
print("=" * 50)
a = np.array([1, 2, 3])
print("a + 10        :", a + 10)            # 스칼라가 퍼짐 → [11 12 13]

M = np.array([[1, 2, 3],
              [4, 5, 6]])                    # (2,3)
row = np.array([10, 20, 30])                 # (3,)
print("M + row (각 행에 더해짐):\n", M + row) # (1,3)→(2,3)로 늘림

# (3,1) + (1,4) → (3,4) 격자 만들기 (구구단표!)
col = np.arange(1, 4).reshape(3, 1)          # (3,1)
lin = np.arange(1, 5).reshape(1, 4)          # (1,4)
print("구구단 격자 (3,1)x(1,4)=(3,4):\n", col * lin)

print("\n" + "=" * 50)
print("[2] 행별 정규화 (axis + keepdims + 브로드캐스팅)")
print("=" * 50)
# 가짜 '임베딩' 행렬: 문서 3개, 각 4차원 벡터 (실제론 384/768차원)
emb = np.array([[3.0, 4.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0]])        # shape (3, 4)
print("emb shape:", emb.shape)

# 각 행의 길이(L2 norm) 구하기 → (3,1) 로 유지
norms = np.sqrt((emb ** 2).sum(axis=1, keepdims=True))  # (3,1)
print("각 문서 벡터의 길이(norm):\n", norms)

# 브로드캐스팅 한 줄로 "단위벡터"로 정규화 ( (3,4) / (3,1) )
unit = emb / norms
print("정규화된 임베딩 (각 행 길이=1):\n", unit.round(3))
print("정규화 검증(각 행 길이):", np.sqrt((unit ** 2).sum(axis=1)).round(3))

print("\n" + "=" * 50)
print("[3] 코사인 유사도로 '검색' 흉내 내기")
print("=" * 50)
# 질문(query) 벡터 1개
query = np.array([3.0, 4.0, 0.0, 0.0])
query_unit = query / np.sqrt((query ** 2).sum())   # (4,) 정규화

# 정규화된 벡터끼리는 '내적(dot)' = 코사인 유사도
#   unit: (3,4),  query_unit: (4,)  →  결과 (3,)  ← 문서 3개의 점수
scores = unit @ query_unit          # @ 는 행렬곱(matmul)
print("질문 vs 각 문서 유사도:", scores.round(3))

# 점수 높은 순으로 문서 순위 매기기 (검색 결과!)
ranking = np.argsort(scores)[::-1]  # 내림차순 정렬한 '인덱스'
print("검색 순위(문서 번호)   :", ranking)
for rank, doc_id in enumerate(ranking, 1):
    print(f"  {rank}위: 문서 {doc_id}  (유사도 {scores[doc_id]:.3f})")
