"""
Day-029 — torch.nn.Module 과 torch.optim: 모델과 학습을 깔끔하게 캡슐화하기
          Encapsulating Models with nn.Module & Optimizers with torch.optim

Day-028 에서 우리는 W1, b1, W2, b2 를 '손으로' 선언하고, grad 초기화·갱신도
for 문으로 돌렸다. 오늘은 그 XOR MLP 를 PyTorch 표준 구조로 다시 쓴다:
  - nn.Module          : 모델을 하나의 클래스로 (파라미터는 nn.Linear 가 자동 소유)
  - nn.Sequential      : 층을 레고처럼 조립
  - torch.optim.SGD/Adam : 갱신 규칙을 최적화기(optimizer)에 위임
학습 루프의 세 관용구가 optimizer.zero_grad() -> loss.backward() -> optimizer.step()
로 바뀐다. 결과(정확도 1.00)는 Day-028 과 같고, 코드가 훨씬 짧고 재사용 가능해진다.

실행: uv run python nn_module_optim.py
(한글이 깨지면 먼저 PowerShell 에서:  $env:PYTHONIOENCODING="utf-8")
"""

import torch
import torch.nn as nn

torch.manual_seed(7)  # 결정론적 재현 (deterministic)


def rule(title: str) -> None:
    print("=" * 72)
    print(title)
    print("=" * 72)


# XOR 데이터 (Day-026~028 과 동일)
X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
Y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])


# ----------------------------------------------------------------------
# 1부. nn.Module — 모델을 하나의 클래스로
# ----------------------------------------------------------------------
rule("Day-029 (1부) — nn.Module 로 XOR MLP 를 클래스로 캡슐화")


class XORNet(nn.Module):
    """Day-028 의 손선언 (W1,b1,W2,b2) 을 nn.Linear 두 개로 대체.

    nn.Linear(in, out) 은 내부에 weight/bias 텐서를 requires_grad=True 로
    '자동' 으로 만들어 소유한다 -> 우리가 randn 으로 직접 선언할 필요가 사라진다.
    """

    def __init__(self, hidden: int = 8):
        super().__init__()  # nn.Module 초기화 (파라미터 등록 기계장치 켜기)
        self.fc1 = nn.Linear(2, hidden)   # Day-028 의 W1(2xH)+b1 에 해당
        self.fc2 = nn.Linear(hidden, 1)   # Day-028 의 W2(Hx1)+b2 에 해당

    def forward(self, x):
        h = torch.tanh(self.fc1(x))       # 은닉층 (tanh 비선형성)
        return self.fc2(h)                # logit (sigmoid 전 값)


model = XORNet(hidden=8)
print("\n[모델 구조]")
print(model)

# nn.Module 은 자기 안의 모든 파라미터를 .parameters() 로 한 번에 내어 준다.
n_params = sum(p.numel() for p in model.parameters())
print("\n[파라미터] model.parameters() 가 학습 대상 텐서를 자동 수집:")
for name, p in model.named_parameters():
    print(f"  {name:12s} shape={tuple(p.shape)}  requires_grad={p.requires_grad}")
print(f"  총 학습 파라미터 수 = {n_params}  (= 2*8+8 + 8*1+1)")


# ----------------------------------------------------------------------
# 2부. torch.optim — 갱신을 최적화기에 위임 & 표준 학습 루프
# ----------------------------------------------------------------------
rule("Day-029 (2부) — torch.optim 으로 표준 학습 루프")

# BCEWithLogitsLoss = sigmoid + BCELoss 를 하나로 (수치적으로 더 안정적).
# 그래서 forward 는 sigmoid 없이 'logit' 을 그대로 내보낸다.
loss_fn = nn.BCEWithLogitsLoss()

# 최적화기: 파라미터 묶음과 학습률만 넘기면, 갱신 규칙을 대신 수행한다.
# SGD 는 Day-028 의 (θ -= lr*grad) 과 정확히 같은 규칙. Adam 은 적응형(빠름).
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

print("\n[학습 곡선] Day-028 의 세 관용구가 optim 3줄로 바뀐다")
print("  epoch |   loss   (zero_grad -> backward -> step)")
for epoch in range(1, 2001):
    logits = model(X)                 # ① 순전파 (logit)
    loss = loss_fn(logits, Y)         # ② 손실

    optimizer.zero_grad()             # ③ grad 초기화 (Day-028: for p: p.grad.zero_())
    loss.backward()                   #    역전파 자동 (동일)
    optimizer.step()                  # ④ 갱신 위임 (Day-028: with no_grad: p -= lr*grad)

    if epoch in (1, 50, 100, 300, 600, 1000, 2000):
        print(f"  {epoch:5d} | {loss.item():.4f}")

# 평가: 추론 때는 기울기가 필요 없으니 no_grad + logit 을 sigmoid 로 확률화
model.eval()
with torch.no_grad():
    prob = torch.sigmoid(model(X)).squeeze()
    hard = (prob >= 0.5).float()
    acc = (hard == Y.squeeze()).float().mean().item()

print("\n[학습 후] XOR 예측")
print("  x1 x2 | p(=1)  | 예측 | 정답")
for (x1, x2), p, h, t in zip(X.tolist(), prob.tolist(), hard.tolist(), Y.squeeze().tolist()):
    print(f"   {int(x1)}  {int(x2)} | {p:.3f}  |  {int(h)}   |  {int(t)}")
print(f"  정확도(accuracy) = {acc:.2f}  (4/4 = 1.00 이면 XOR 완전 학습)")


# ----------------------------------------------------------------------
# 3부. nn.Sequential — 클래스도 필요 없이 층을 레고처럼
# ----------------------------------------------------------------------
rule("Day-029 (3부) — nn.Sequential: 단순 적층은 클래스 없이 한 줄로")

seq = nn.Sequential(
    nn.Linear(2, 8),
    nn.Tanh(),
    nn.Linear(8, 1),
)
print("\n[nn.Sequential 모델]")
print(seq)
print("동일한 순전파를 클래스 정의 없이 표현. 학습 루프는 2부와 완전히 같다.")

# state_dict: 학습된 가중치를 저장/복원하는 표준 통로 (배포·체크포인트의 기초)
print("\n[state_dict] 저장 가능한 파라미터 목록 (torch.save/load 로 체크포인트):")
for k, v in model.state_dict().items():
    print(f"  {k:12s} {tuple(v.shape)}")

rule(
    "결론: nn.Module 이 파라미터를, torch.optim 이 갱신을 대신 관리한다.\n"
    "      Day-028 의 손선언·수동 갱신이 사라지고 학습 루프는\n"
    "      zero_grad -> backward -> step 세 줄로 표준화됐다. 이 골격이\n"
    "      다음 편(RNN/시퀀스)과 Phase 3 의 미니 Transformer(B3)로 그대로 이어진다."
)
