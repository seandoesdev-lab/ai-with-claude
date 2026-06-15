# metrics_from_scratch.py — 분류 평가지표를 손으로 만들기
#   혼동행렬(confusion matrix) → 정확도/정밀도/재현율/F1
#   그리고 똑같은 개념이 '검색(IR)'에서 어떻게 쓰이는지까지.
import sys
import numpy as np

# Windows 콘솔(cp949)에서 한글·특수기호가 깨지거나 에러 나는 것을 막아 줍니다.
sys.stdout.reconfigure(encoding="utf-8")

print("=" * 60)
print("[1] 스팸 분류 예제 — 1=스팸(positive), 0=정상(negative)")
print("=" * 60)

# 특징(features)으로부터 모델이 내놓은 '예측'과, 사람이 매긴 '정답(label)'
y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])   # 실제 정답 (스팸 4개)
y_pred = np.array([1, 1, 0, 1, 0, 0, 1, 0, 0, 0])   # 모델 예측

# --- 혼동행렬(confusion matrix)의 네 칸 ---
TP = int(np.sum((y_pred == 1) & (y_true == 1)))   # True  Positive : 스팸을 스팸이라 함 (맞음)
FP = int(np.sum((y_pred == 1) & (y_true == 0)))   # False Positive : 정상을 스팸이라 함 (오경보)
FN = int(np.sum((y_pred == 0) & (y_true == 1)))   # False Negative : 스팸을 놓침
TN = int(np.sum((y_pred == 0) & (y_true == 0)))   # True  Negative : 정상을 정상이라 함

print("              예측=스팸   예측=정상")
print(f"실제=스팸        TP={TP}        FN={FN}      (스팸 {TP + FN}개 중 {TP}개를 잡음)")
print(f"실제=정상        FP={FP}        TN={TN}      (정상 {FP + TN}개 중 {FP}개를 오경보)")


def safe_div(a, b):
    """0으로 나누는 상황(예측을 하나도 안 한 경우 등)을 0으로 처리."""
    return a / b if b else 0.0


accuracy = safe_div(TP + TN, len(y_true))   # 전체 중 맞춘 비율
precision = safe_div(TP, TP + FP)           # '스팸'이라 한 것 중 진짜 스팸 비율
recall = safe_div(TP, TP + FN)              # 진짜 스팸 중 실제로 잡아낸 비율
f1 = safe_div(2 * precision * recall, precision + recall)   # 정밀도·재현율의 조화평균

print("\n--- 평가지표 (직접 계산) ---")
print(f"정확도   accuracy  = (TP+TN)/전체   = {accuracy:.3f}  (전체 중 맞춘 비율)")
print(f"정밀도   precision = TP/(TP+FP)     = {precision:.3f}  (스팸이라 한 것이 진짜일 확률)")
print(f"재현율   recall    = TP/(TP+FN)     = {recall:.3f}  (진짜 스팸을 놓치지 않은 비율)")
print(f"F1 score           = 2PR/(P+R)      = {f1:.3f}  (정밀도·재현율의 균형)")

print("\n[핵심] 정밀도와 재현율은 보통 '트레이드오프(trade-off)' 관계다.")
print("  · 스팸을 '의심되면 다 차단' → recall↑ 이지만 정상메일 오차단↑ → precision↓")
print("  · '확실할 때만 차단'        → precision↑ 이지만 스팸을 놓침↑    → recall↓")
print("  그래서 둘을 함께 보는 F1, 그리고 상황별 우선순위(병 진단은 recall 중시 등)가 중요하다.")


print("\n" + "=" * 60)
print("[2] 같은 개념이 '검색(IR)'에서는 이렇게 — Phase 2 미리보기")
print("=" * 60)
# IR 에서: precision = 가져온 것 중 관련 비율, recall = 관련 있는 것 중 가져온 비율
relevant = {1, 3, 5, 7}            # 실제로 '관련 있는' 문서 id (정답 집합)
retrieved = {1, 2, 3, 8}           # 검색엔진이 실제로 '가져온' 문서 id
hit = relevant & retrieved         # 둘 다에 속함 = 잘 찾은 것(교집합)

p_ir = safe_div(len(hit), len(retrieved))   # 가져온 4개 중 관련 2개
r_ir = safe_div(len(hit), len(relevant))    # 관련 4개 중 2개를 가져옴

print(f"관련 문서(relevant) : {sorted(relevant)}")
print(f"검색 결과(retrieved): {sorted(retrieved)}")
print(f"맞게 찾음(hit)      : {sorted(hit)}")
print(f"  precision = |hit|/|retrieved| = {len(hit)}/{len(retrieved)} = {p_ir:.3f}")
print(f"  recall    = |hit|/|relevant|  = {len(hit)}/{len(relevant)} = {r_ir:.3f}")
print("\n-> '스팸/정상'이든 '검색 결과 관련/무관'이든 똑같은 정밀도·재현율 사고방식이다.")
print("   Phase 2에서 이 개념을 MAP·nDCG·MRR 같은 '순위 평가지표'로 확장한다.")
