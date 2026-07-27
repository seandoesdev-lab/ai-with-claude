"""
Day-036 — 🛠️ B3 빌드 (1부): NumPy로 MLP + 역전파를 손코딩하다.
(Build B3, Part 1: A Multi-Layer Perceptron with hand-written backprop, NumPy only.)

- PyTorch autograd 없이, [[Day-026]] 순전파 + [[Day-027]] 역전파(연쇄법칙)를
  '직접' 구현해 2-class 나선형(spiral) 데이터를 분류하도록 학습시킨다.
- 마지막에 '수치 미분(numerical gradient)'과 대조해 우리가 손으로 유도한
  해석적 기울기(analytic gradient)가 맞는지 검증(gradient check)한다.

실행:  uv run python mlp_from_scratch.py
필요:  uv add numpy
재현:  np.random.default_rng(0) 고정 → 매번 같은 결과.
"""

import numpy as np

RNG = np.random.default_rng(0)


# ----------------------------------------------------------------------
# 0. 데이터: 두 갈래 나선형 (선형 분리 불가 → 비선형 MLP가 필요한 이유)
# ----------------------------------------------------------------------
def make_spiral(n_per_class=100, n_classes=2, noise=0.20):
    X = np.zeros((n_per_class * n_classes, 2))
    y = np.zeros(n_per_class * n_classes, dtype=int)
    for c in range(n_classes):
        idx = range(n_per_class * c, n_per_class * (c + 1))
        r = np.linspace(0.0, 1.0, n_per_class)                      # 반지름
        t = np.linspace(c * 4, (c + 1) * 4, n_per_class) + RNG.normal(0, noise, n_per_class)
        X[idx] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[idx] = c
    return X, y


# ----------------------------------------------------------------------
# 1. 활성화 / 손실 (모두 NumPy로 직접)
# ----------------------------------------------------------------------
def relu(z):
    return np.maximum(0.0, z)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)      # 수치 안정화 (overflow 방지)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy(probs, y):
    n = y.shape[0]
    # 정답 클래스의 확률에 -log. 1e-12 로 log(0) 방지.
    return -np.mean(np.log(probs[np.arange(n), y] + 1e-12))


# ----------------------------------------------------------------------
# 2. MLP: 2 → H → 2 (은닉 1층, ReLU, softmax 출력)
#    파라미터를 dict 로 들고, forward 는 중간값(cache)을 함께 반환한다.
# ----------------------------------------------------------------------
def init_params(d_in=2, d_hidden=16, d_out=2):
    # He 초기화 (ReLU에 적합): 분산 2/fan_in
    return {
        "W1": RNG.normal(0, np.sqrt(2.0 / d_in), (d_in, d_hidden)),
        "b1": np.zeros(d_hidden),
        "W2": RNG.normal(0, np.sqrt(2.0 / d_hidden), (d_hidden, d_out)),
        "b2": np.zeros(d_out),
    }


def forward(params, X):
    z1 = X @ params["W1"] + params["b1"]     # (N, H)  선형
    a1 = relu(z1)                            # (N, H)  비선형
    z2 = a1 @ params["W2"] + params["b2"]    # (N, 2)  로짓
    probs = softmax(z2)                      # (N, 2)  확률
    cache = {"X": X, "z1": z1, "a1": a1, "probs": probs}
    return probs, cache


def backward(params, cache, y):
    """연쇄법칙을 손으로 따라가며 dW1,db1,dW2,db2 를 구한다 ([[Day-027]])."""
    X, a1, probs = cache["X"], cache["a1"], cache["probs"]
    n = y.shape[0]

    # (1) softmax + cross-entropy 의 결합 미분: dL/dz2 = (probs - onehot)/n
    dz2 = probs.copy()
    dz2[np.arange(n), y] -= 1.0
    dz2 /= n                                 # (N, 2)

    # (2) 2층 파라미터
    dW2 = a1.T @ dz2                         # (H, 2)
    db2 = dz2.sum(axis=0)                    # (2,)

    # (3) 은닉층으로 역전파: da1 = dz2·W2ᵀ, ReLU 미분(z1>0)에서 게이트
    da1 = dz2 @ params["W2"].T               # (N, H)
    dz1 = da1 * (cache["z1"] > 0)            # ReLU': z>0 이면 1, 아니면 0

    # (4) 1층 파라미터
    dW1 = X.T @ dz1                          # (d_in, H)
    db1 = dz1.sum(axis=0)                    # (H,)

    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


# ----------------------------------------------------------------------
# 3. 학습 루프: 전량(batch) 경사하강법
# ----------------------------------------------------------------------
def train(X, y, d_hidden=16, lr=1.0, epochs=2000):
    params = init_params(d_hidden=d_hidden)
    for ep in range(1, epochs + 1):
        probs, cache = forward(params, X)
        loss = cross_entropy(probs, y)
        grads = backward(params, cache, y)
        for k in params:                     # theta <- theta - lr * grad
            params[k] -= lr * grads[k]
        if ep == 1 or ep % 400 == 0:
            acc = (probs.argmax(1) == y).mean()
            print(f"  epoch {ep:>4}:  loss={loss:.4f}   train_acc={acc:.3f}")
    return params


# ----------------------------------------------------------------------
# 4. 기울기 검증 (gradient check): 해석적 vs 수치적
#    수치 미분: dL/dθ ≈ (L(θ+ε) - L(θ-ε)) / 2ε   (중심차분)
#    둘의 상대오차가 1e-6 수준이면 backprop 이 정확하다는 강력한 증거.
# ----------------------------------------------------------------------
def gradient_check(params, X, y, eps=1e-5, n_samples=5):
    probs, cache = forward(params, X)
    grads = backward(params, cache, y)

    worst = 0.0
    for name in ["W1", "b1", "W2", "b2"]:
        P = params[name]
        flat = P.reshape(-1)
        picks = RNG.choice(flat.size, size=min(n_samples, flat.size), replace=False)
        for idx in picks:
            orig = flat[idx]
            flat[idx] = orig + eps
            lp, _ = forward(params, X); Lp = cross_entropy(lp, y)
            flat[idx] = orig - eps
            lm, _ = forward(params, X); Lm = cross_entropy(lm, y)
            flat[idx] = orig                       # 원복
            num = (Lp - Lm) / (2 * eps)            # 수치 기울기
            ana = grads[name].reshape(-1)[idx]     # 해석 기울기
            rel = abs(num - ana) / max(1e-12, abs(num) + abs(ana))
            worst = max(worst, rel)
    return worst


# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 64)
    print("Day-036 — NumPy MLP + 역전파 from scratch")
    print("=" * 64)

    X, y = make_spiral(n_per_class=100, n_classes=2, noise=0.20)
    print(f"\n[데이터] 나선형 2-class: X={X.shape}, y={y.shape} (선형 분리 불가)")

    print("\n[학습] MLP(2 -> 16 -> 2), ReLU, softmax-CE, full-batch GD, lr=1.0")
    params = train(X, y, d_hidden=16, lr=1.0, epochs=2000)

    probs, _ = forward(params, X)
    acc = (probs.argmax(1) == y).mean()
    print(f"\n[결과] 최종 train accuracy = {acc:.3f}")

    # 손코딩 backprop 이 맞는지 수치미분으로 검증
    worst = gradient_check(params, X, y)
    print(f"\n[검증] gradient check 최대 상대오차 = {worst:.2e}", end="  ")
    print("-> 1e-6 수준이면 backprop 정확 (OK)" if worst < 1e-4 else "-> 재점검 필요 (FAIL)")

    print("\n요점: autograd 없이 순전파.역전파를 손으로 구현해도")
    print("      나선형(비선형)을 100% 가깝게 분류하고, 수치미분과 일치한다.")
