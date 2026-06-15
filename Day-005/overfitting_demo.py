# overfitting_demo.py — 과적합(overfitting) vs 일반화(generalization)를 '눈으로' 보기
#
# 핵심 아이디어:
#   다항식 회귀(polynomial regression)의 '차수(degree)'를 올릴수록 모델은 더 복잡해진다.
#   - 훈련 오차(train error)는 차수가 커질수록 계속 줄어든다 (외우면 되니까).
#   - 하지만 '처음 보는' 테스트 오차(test error)는 어느 순간부터 다시 커진다 = 과적합!
#   이 한 장의 표가 "왜 데이터를 나눠야 하는가"를 통째로 설명한다.
import sys
import warnings
import numpy as np

# Windows 콘솔(cp949)에서 한글·특수기호가 깨지거나 에러 나는 것을 막아 줍니다.
sys.stdout.reconfigure(encoding="utf-8")

rng = np.random.default_rng(1)    # seed 고정 → 매번 같은 결과(재현 가능)


# 1) '진짜 규칙(true function)'은 부드러운 곡선. 우리는 이걸 '모른다'고 가정하고,
#    여기에 약간의 잡음(noise)을 더해 현실의 '관측 데이터'를 만든다.
def true_function(x):
    return np.sin(1.2 * x)         # 우리가 학습으로 '되찾고 싶은' 숨은 정답 곡선


N = 60
x_all = np.sort(rng.uniform(0.0, 5.0, size=N))
y_all = true_function(x_all) + rng.normal(0.0, 0.25, size=N)   # 잡음 섞인 관측값

# 2) 훈련/테스트 분할 (train/test split) — 70%로 배우고, 30%로 '처음 보는 듯' 평가
idx = rng.permutation(N)
n_train = int(N * 0.7)
tr_idx, te_idx = idx[:n_train], idx[n_train:]
x_tr, y_tr = x_all[tr_idx], y_all[tr_idx]
x_te, y_te = x_all[te_idx], y_all[te_idx]
print(f"전체 {N}개 → 훈련(train) {len(x_tr)}개 / 테스트(test) {len(x_te)}개\n")


def rmse(pred, true):
    """평균제곱근오차(RMSE): 예측이 정답에서 평균적으로 얼마나 빗나갔나. 낮을수록 좋음."""
    return np.sqrt(np.mean((pred - true) ** 2))


print("=" * 62)
print(" 차수(degree)별  훈련 오차 vs 테스트 오차  (RMSE, 낮을수록 좋음)")
print("=" * 62)

results = []
with warnings.catch_warnings():
    # 차수가 너무 높으면 numpy가 '수치적으로 불안정하다'고 경고하는데,
    # 그 경고 자체가 사실 '과적합 영역'이라는 신호다. 출력만 깔끔히 숨긴다.
    warnings.simplefilter("ignore")
    for deg in [1, 3, 5, 9, 15]:
        coef = np.polyfit(x_tr, y_tr, deg)          # 훈련 데이터에만 맞춤(fit)
        tr_err = rmse(np.polyval(coef, x_tr), y_tr)  # 본 적 있는 데이터의 오차
        te_err = rmse(np.polyval(coef, x_te), y_te)  # 처음 보는 데이터의 오차
        results.append((deg, tr_err, te_err))

best_deg = min(results, key=lambda r: r[2])[0]       # 테스트 오차가 가장 낮은 차수
min_deg = results[0][0]                              # 가장 단순한(작은) 차수

print(f"{'degree':>6} | {'train':>8} | {'test':>8} |  해석")
print("-" * 62)
for deg, tr_err, te_err in results:
    if deg == min_deg:
        note = "과소적합(underfit): 너무 단순해 둘 다 큼"
    elif deg == best_deg:
        note = "★ 일반화 최고: 테스트 오차 최소"
    elif deg > best_deg:
        note = "과적합(overfit): 복잡할수록 test 오차↑"
    else:
        note = "개선 중: train·test 함께 줄어듦"
    print(f"{deg:>6} | {tr_err:8.3f} | {te_err:8.3f} |  {note}")

print(f"\n-> 차수가 커질수록 train 오차는 계속 줄지만, test 오차는 deg={best_deg} 부근에서")
print("   최소가 되고 그 뒤로 다시 커진다. 바로 이 'test 오차의 U자 곡선'이 과적합의 증거다.")
print("   (모델이 잡음까지 통째로 외워버리면, 새 데이터에선 오히려 헷갈린다.)")

# (선택) matplotlib 가 있으면 그림으로도 저장 — 없으면 조용히 건너뜀
try:
    import matplotlib.pyplot as plt

    xs = np.linspace(0, 4, 300)
    plt.figure(figsize=(8, 5))
    plt.scatter(x_tr, y_tr, c="black", label="train data", zorder=3)
    plt.scatter(x_te, y_te, facecolors="none", edgecolors="red", label="test data", zorder=3)
    plt.plot(xs, true_function(xs), "g--", label="true function", linewidth=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for deg, color in [(1, "#999999"), (best_deg, "#5B8FF9"), (15, "#E8684A")]:
            coef = np.polyfit(x_tr, y_tr, deg)
            plt.plot(xs, np.polyval(coef, xs), color=color, label=f"degree {deg}")
    plt.ylim(-2, 2)
    plt.legend()
    plt.title("Underfit (deg1) vs Good vs Overfit (deg15)")
    plt.tight_layout()
    plt.savefig("overfitting.png", dpi=120)
    print("\n(보너스) overfitting.png 저장 완료 — 직선(과소), 적당한 곡선, 요동치는 곡선(과적합)을 비교해 보세요.")
except ImportError:
    print("\n(matplotlib 없음 — 그림은 생략. 보고 싶으면: uv add matplotlib)")
