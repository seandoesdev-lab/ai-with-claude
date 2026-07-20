"""
Day-030 — RNN 입문: 순서가 있는 데이터를 다루는 신경망
(Recurrent Neural Networks: handling sequences)

실행 (Day-028/029 의 uv 프로젝트에서, torch 이미 설치됨):
    uv run python rnn_intro.py
Windows 콘솔에서 한글이 깨지면 먼저:
    $env:PYTHONIOENCODING="utf-8"

구성
  1부  RNN 순환식을 '손으로' 시간축을 따라 펼쳐(unroll) 계산 — 같은 Whh 가
       매 시점 재사용되며 hidden state 가 어떻게 흘러가는지 눈으로 확인.
  2부  nn.RNN 으로 '첫 비트 기억하기' 과제를 학습 — 마지막 시점까지 첫 입력을
       hidden state 에 실어 나르는 '기억'이 실제로 생기는지 정확도로 확인.
       (학습 루프는 Day-029 의 zero_grad -> backward -> step 세 줄 그대로)
  3부  '순서 민감성' 대조 — 같은 비트를 뒤집으면(첫 비트가 바뀌면) 예측도 바뀐다.
"""

import torch
import torch.nn as nn


# =========================================================================
# 1부 — RNN 순환식을 손으로 펼쳐 보기 (unrolling through time)
#   h_t = tanh(Wxh · x_t + Whh · h_{t-1} + b)      (핵심 순환식)
#   * 같은 Wxh, Whh, b 를 '매 시점' 재사용한다 = parameter sharing
# =========================================================================
def part1_unroll():
    print("=" * 72)
    print("Day-030 (1부) — RNN 순환식을 시간축으로 펼쳐 계산")
    print("=" * 72)

    torch.manual_seed(0)
    H, D = 4, 1                      # hidden size, input size
    Wxh = torch.randn(H, D) * 0.5    # 입력 -> hidden
    Whh = torch.randn(H, H) * 0.5    # hidden -> hidden  (여기로 '기억'이 흐른다)
    bh = torch.zeros(H)

    def rnn_step(x_t, h_prev):       # 한 시점 전진
        return torch.tanh(Wxh @ x_t + Whh @ h_prev + bh)

    seq = [torch.tensor([1.0]), torch.tensor([0.0]), torch.tensor([1.0])]
    h = torch.zeros(H)               # h_0 : 초기 기억(빈 상태)
    print("  h_0 =", [round(v, 3) for v in h.tolist()], " (초기 상태, 아직 아무것도 안 봄)")
    for t, x_t in enumerate(seq, 1):
        h = rnn_step(x_t, h)         # h_{t-1} 을 다시 넣는다 = 순환(recurrence)
        print(f"  x_{t}={x_t.item():.0f} -> h_{t} =", [round(v, 3) for v in h.tolist()])
    print("  ↑ 같은 Whh 가 매 시점 재사용됨(파라미터 공유). h 가 과거를 요약해 흐른다.")


# =========================================================================
# 2부 — nn.RNN 으로 '첫 비트 기억하기' 학습
#   입력: 길이 L 의 0/1 시퀀스,  정답 label = '첫 번째 비트'
#   RNN 은 첫 비트를 hidden state 에 담아 마지막 시점까지 실어 날라야 맞힌다.
# =========================================================================
def make_data(n, L):
    X = torch.randint(0, 2, (n, L, 1)).float()   # (N, L, 1): N개 시퀀스, 각 스텝은 스칼라 1개
    y = X[:, 0, 0].clone()                        # 정답 = 첫 시점의 비트
    return X, y


class FirstBitRNN(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()                        # Day-029: nn.Module 첫 줄 규칙
        self.rnn = nn.RNN(input_size=1, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)          # 마지막 hidden -> logit
    def forward(self, x):
        out, h_n = self.rnn(x)                    # out:(N,L,H)  h_n:(1,N,H) 마지막 상태
        last = out[:, -1, :]                      # 마지막 시점의 hidden state (= h_L)
        return self.head(last).squeeze(1)         # (N,) logit


def part2_train():
    print("\n" + "=" * 72)
    print("Day-030 (2부) — nn.RNN 으로 '첫 비트 기억하기' 학습")
    print("=" * 72)

    torch.manual_seed(7)
    L = 6
    Xtr, ytr = make_data(512, L)
    Xte, yte = make_data(256, L)

    model = FirstBitRNN(hidden=8)
    loss_fn = nn.BCEWithLogitsLoss()              # Day-029: sigmoid+BCE 안정판
    opt = torch.optim.Adam(model.parameters(), lr=0.01)

    print(f"  시퀀스 길이 L={L}  (첫 비트를 {L}스텝 뒤까지 기억해야 맞힘)")
    for epoch in range(1, 301):
        logits = model(Xtr)
        loss = loss_fn(logits, ytr)
        opt.zero_grad(); loss.backward(); opt.step()   # ← Day-029 세 줄 관용구 그대로
        if epoch == 1 or epoch % 50 == 0:
            with torch.no_grad():
                pred = (torch.sigmoid(model(Xte)) > 0.5).float()
                acc = (pred == yte).float().mean().item()
            print(f"  epoch {epoch:3d} | loss {loss.item():.4f} | test acc {acc:.3f}")
    print("  ↑ 정확도가 1.0 에 가까워지면, RNN 이 첫 비트를 hidden state 로 '기억'해 낸 것.")
    return model, L


# =========================================================================
# 3부 — 순서 민감성 대조: 첫 비트를 바꾸면 예측이 바뀐다
# =========================================================================
def part3_order(model, L):
    print("\n" + "=" * 72)
    print("Day-030 (3부) — 순서가 의미를 바꾼다 (order matters)")
    print("=" * 72)
    a = torch.tensor([[[1.], [0.], [0.], [1.], [1.], [0.]]])   # 첫 비트 = 1
    b = a.clone(); b[0, 0, 0] = 0.0                            # 같은 뒤쪽, 첫 비트만 0
    with torch.no_grad():
        pa = torch.sigmoid(model(a)).item()
        pb = torch.sigmoid(model(b)).item()
    print(f"  첫비트=1 시퀀스 -> p(첫비트=1) = {pa:.3f}  (예측 {int(pa > 0.5)})")
    print(f"  첫비트=0 시퀀스 -> p(첫비트=1) = {pb:.3f}  (예측 {int(pb > 0.5)})")
    print("  ↑ 나머지가 같아도 '첫' 원소만 다르면 결과가 뒤집힌다 — 순서를 실제로 쓰고 있다.")


if __name__ == "__main__":
    part1_unroll()
    model, L = part2_train()
    part3_order(model, L)
