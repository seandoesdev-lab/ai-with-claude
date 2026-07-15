"""
Day-027 — 역전파(Backpropagation): 신경망이 '스스로' 배우는 법
From scratch (NumPy only). Day-026 의 forward pass 에 '역방향 학습'을 붙여
XOR 를 데이터만 보고 풀도록 2층 MLP 를 훈련한다.

핵심: forward(순전파) -> loss(얼마나 틀렸나) -> backward(연쇄법칙으로 기울기) -> 경사하강 갱신.
실행:  uv run python backprop.py   (한글 깨지면 먼저: $env:PYTHONIOENCODING="utf-8")
"""
import numpy as np

# ---------------------------------------------------------------------------
# 0) 활성함수와 그 미분 (backprop 에 미분이 반드시 필요)
# ---------------------------------------------------------------------------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_grad(a):        # a = sigmoid(z) 일 때 dσ/dz = a(1-a)
    return a * (1.0 - a)

# ---------------------------------------------------------------------------
# 1) 데이터 — XOR (Day-026 와 동일). 이번엔 '가중치를 손으로' 넣지 않고 '학습'한다.
# ---------------------------------------------------------------------------
X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])   # (4,2)
y = np.array([[0.], [1.], [1.], [0.]])                   # (4,1) XOR 정답

# ---------------------------------------------------------------------------
# 2) 파라미터 초기화 — 재현 위해 seed 고정, 작은 난수 (대칭 깨기)
# ---------------------------------------------------------------------------
rng = np.random.default_rng(7)
H = 4                              # 은닉 뉴런 수
W1 = rng.normal(0, 1.0, size=(2, H)); b1 = np.zeros((1, H))   # 입력2 -> 은닉H
W2 = rng.normal(0, 1.0, size=(H, 1)); b2 = np.zeros((1, 1))   # 은닉H -> 출력1

lr = 0.5
EPOCHS = 8000

def forward(Xb):
    z1 = Xb @ W1 + b1;  a1 = sigmoid(z1)     # 은닉층
    z2 = a1 @ W2 + b2;  a2 = sigmoid(z2)     # 출력층(확률)
    return z1, a1, z2, a2

def bce(pred, target):                       # 이진 교차엔트로피 (Day-014 와 같은 손실)
    eps = 1e-12
    return float(-np.mean(target*np.log(pred+eps) + (1-target)*np.log(1-pred+eps)))

print("=" * 72)
print("Day-027 — 역전파(Backpropagation): XOR 를 '데이터로' 학습")
print("         Training a 2-layer MLP on XOR with backprop (NumPy only)")
print("=" * 72)

# --- 학습 전 예측 (아직 배우지 않음) ---
_, _, _, a2 = forward(X)
print(f"\n[학습 전] loss={bce(a2,y):.4f}  예측={a2.ravel().round(2)}  (정답 0,1,1,0)")

# ---------------------------------------------------------------------------
# 3) 학습 루프 — forward -> loss -> backward(연쇄법칙) -> 경사하강 갱신
# ---------------------------------------------------------------------------
N = X.shape[0]
history = []
for epoch in range(1, EPOCHS + 1):
    # (a) 순전파
    z1, a1, z2, a2 = forward(X)

    # (b) 손실
    loss = bce(a2, y)

    # (c) 역전파 — 출력에서 입력 쪽으로 기울기를 '되돌려' 흘린다
    #     BCE + sigmoid 결합 미분의 유명한 결과: dL/dz2 = (a2 - y)/N
    dz2 = (a2 - y) / N                 # (4,1)
    dW2 = a1.T @ dz2                   # (H,1)  = 은닉활성 x 출력오차
    db2 = dz2.sum(axis=0, keepdims=True)
    da1 = dz2 @ W2.T                   # (4,H)  오차를 은닉층으로 전파
    dz1 = da1 * sigmoid_grad(a1)       # 연쇄법칙: 활성함수 미분을 곱한다
    dW1 = X.T @ dz1                    # (2,H)
    db1 = dz1.sum(axis=0, keepdims=True)

    # (d) 경사하강 갱신 (Day-014 와 동일한 규칙: 파라미터 -= lr * 기울기)
    W2 -= lr * dW2; b2 -= lr * db2
    W1 -= lr * dW1; b1 -= lr * db1

    if epoch in (1, 100, 500, 1000, 2000, 4000, 8000):
        history.append((epoch, loss))

print("\n[학습 곡선] epoch 별 손실(loss)이 줄어드는가")
print("  epoch |   loss")
for e, l in history:
    print(f"  {e:5d} | {l:.4f}")

# --- 학습 후 예측 ---
_, _, _, a2 = forward(X)
pred_bin = (a2 > 0.5).astype(int).ravel()
print("\n[학습 후] XOR 예측")
print("  x1 x2 | p(=1)  | 예측 | 정답")
for i, (x1, x2) in enumerate(X.astype(int)):
    print(f"   {x1}  {x2} | {a2[i,0]:.3f}  |  {pred_bin[i]}   |  {int(y[i,0])}")
acc = float((pred_bin == y.ravel().astype(int)).mean())
print(f"  정확도(accuracy) = {acc:.2f}  (4/4 = 1.00 이면 XOR 완전 학습)")
print(f"  최종 loss = {bce(a2,y):.4f}  (학습 전 대비 크게 감소)")

print("\n" + "=" * 72)
print("결론: 순전파로 답을 내고 -> 손실로 오차를 재고 -> 연쇄법칙으로 그 오차를")
print("      각 가중치까지 '거꾸로' 흘려 기울기를 구한 뒤 -> 경사하강으로 조금씩")
print("      갱신한다. 이 4단계 반복이 곧 '학습'. Day-026 의 손코딩 가중치를")
print("      이제 데이터가 스스로 찾았다.  다음: PyTorch autograd 가 이 backward 를 자동화.")
print("=" * 72)
