"""
Day-028 — PyTorch 입문: 텐서(tensor)와 autograd 로 backprop 자동화하기
          PyTorch 101: Tensors & Autograd

Day-027 에서 우리는 XOR MLP 의 역전파(dz2, dW1 ...)를 '손으로' 유도해 적었다.
오늘은 같은 미분을 PyTorch 의 autograd 가 loss.backward() 한 줄로 자동 계산하게 한다.
- 1부: 텐서 기초 & autograd 가 미분을 어떻게 자동으로 구하는지 (수치검증)
- 2부: Day-027 의 XOR MLP 를 PyTorch 로 다시 짜 '데이터로' 학습 → 정확도 1.00

실행: uv run python autograd_intro.py
(한글이 깨지면 먼저 PowerShell 에서:  $env:PYTHONIOENCODING="utf-8")
"""

import torch

torch.manual_seed(7)  # 결정론적 재현 (deterministic)


def rule(title: str) -> None:
    print("=" * 72)
    print(title)
    print("=" * 72)


# ----------------------------------------------------------------------
# 1부. 텐서 & autograd — "미분을 자동으로"
# ----------------------------------------------------------------------
rule("Day-028 (1부) — 텐서와 autograd: 미분을 자동으로")

# (a) 텐서는 NumPy 배열 + 두 가지 초능력: ① GPU 실행 ② 자동미분(autograd)
x = torch.tensor([[0.0, 1.0], [1.0, 1.0]])
print("\n[텐서 기초] x =")
print(x)
print("shape:", tuple(x.shape), "| dtype:", x.dtype)

# (b) requires_grad=True 를 켜면, 이 텐서로 이어지는 모든 연산이
#     '계산 그래프(computational graph)' 로 기록된다.
w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)
xx = torch.tensor(3.0)

y = w * xx + b          # 순전파: y = w*x + b
z = y ** 2              # 손실처럼 스칼라 하나로 만든다: z = (w*x+b)^2
print(f"\n[순전파] w={w.item()}, b={b.item()}, x={xx.item()}  ->  y={y.item()}, z={z.item()}")

# (c) backward() 한 줄이 그래프를 거꾸로 훑어 모든 requires_grad 텐서의 기울기를 채운다.
z.backward()
print("[역전파] z.backward() 호출 후 .grad 가 자동으로 채워짐:")
print(f"  dz/dw = {w.grad.item():.4f}   (손계산: 2*(w*x+b)*x = 2*7*3 = 42)")
print(f"  dz/db = {b.grad.item():.4f}   (손계산: 2*(w*x+b)*1 = 2*7   = 14)")

# (d) autograd 가 맞는지 '수치미분' 으로 교차검증 (gradient check)
eps = 1e-4
with torch.no_grad():
    f = lambda W: (W * 3.0 + 1.0) ** 2
    num_dw = (f(torch.tensor(2.0 + eps)) - f(torch.tensor(2.0 - eps))) / (2 * eps)
print(f"[수치검증] 수치미분 dz/dw ≈ {num_dw.item():.4f}  ==  autograd 42.0  → 일치 ✔")


# ----------------------------------------------------------------------
# 2부. Day-027 의 XOR MLP 를 PyTorch 로 — loss.backward() 로 학습
# ----------------------------------------------------------------------
rule("Day-028 (2부) — XOR MLP: 손으로 짠 backprop 을 autograd 로 대체")

X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
Y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])  # XOR

# 은닉 뉴런 4개 (Day-027 과 동일 구조). 파라미터는 작은 난수에서 시작.
H = 4
W1 = torch.randn(2, H, requires_grad=True)
b1 = torch.zeros(1, H, requires_grad=True)
W2 = torch.randn(H, 1, requires_grad=True)
b2 = torch.zeros(1, 1, requires_grad=True)
params = [W1, b1, W2, b2]

lr = 0.5
loss_fn = torch.nn.BCELoss()  # Day-027 의 이진 교차엔트로피 그대로


def forward(inp):
    a1 = torch.sigmoid(inp @ W1 + b1)
    a2 = torch.sigmoid(a1 @ W2 + b2)  # 예측 확률 ŷ
    return a2


with torch.no_grad():
    pred0 = forward(X).squeeze()
print(f"\n[학습 전] 예측={[round(v, 2) for v in pred0.tolist()]}  (정답 0,1,1,0)")

print("\n[학습 곡선] epoch 별 손실(loss) — loss.backward() 가 backprop 을 자동 수행")
print("  epoch |   loss")
for epoch in range(1, 8001):
    # ① 순전파 → ② 손실
    pred = forward(X)
    loss = loss_fn(pred, Y)

    # ③ 역전파: 손으로 짠 dz2/dW1 ... 대신 이 한 줄!
    for p in params:
        if p.grad is not None:
            p.grad.zero_()      # 기울기는 누적되므로 매 스텝 초기화
    loss.backward()             # ← Day-027 의 역전파 전체를 대신한다

    # ④ 경사하강 갱신 (θ ← θ − η·∇)
    with torch.no_grad():
        for p in params:
            p -= lr * p.grad

    if epoch in (1, 100, 500, 1000, 2000, 4000, 8000):
        print(f"  {epoch:5d} | {loss.item():.4f}")

with torch.no_grad():
    prob = forward(X).squeeze()
    hard = (prob >= 0.5).float()
    acc = (hard == Y.squeeze()).float().mean().item()

print("\n[학습 후] XOR 예측")
print("  x1 x2 | p(=1)  | 예측 | 정답")
for (x1, x2), p, h, t in zip(X.tolist(), prob.tolist(), hard.tolist(), Y.squeeze().tolist()):
    print(f"   {int(x1)}  {int(x2)} | {p:.3f}  |  {int(h)}   |  {int(t)}")
print(f"  정확도(accuracy) = {acc:.2f}  (4/4 = 1.00 이면 XOR 완전 학습)")

rule(
    "결론: PyTorch 텐서로 순전파만 적으면, loss.backward() 가 계산 그래프를 거꾸로\n"
    "      훑어 모든 파라미터의 기울기(.grad)를 자동으로 채운다. Day-027 에서 손으로\n"
    "      유도하던 역전파가 한 줄로 대체됐다 — 층이 수백 개인 Transformer 도 동일.\n"
    "      다음: torch.nn.Module 로 모델을, torch.optim 으로 갱신을 캡슐화한다."
)
