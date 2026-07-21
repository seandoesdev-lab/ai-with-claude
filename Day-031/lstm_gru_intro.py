"""
Day-031 — LSTM & GRU: 게이트로 장기 기억 지키기 (Gating for long-term memory)

실행 (Day-028 의 uv 프로젝트에서, torch 이미 설치됨):
    uv run python lstm_gru_intro.py
Windows 콘솔에서 한글이 깨지면 먼저:
    $env:PYTHONIOENCODING="utf-8"

구성 (모두 학습 없이 수 초 안에 끝나고, 시드 고정으로 재현 가능)
  1부: LSTM 셀을 손으로 한 스텝 — forget/input/output 게이트와 셀 상태 c_t 를 직접 계산
  2부: '기울기 소실'을 눈으로 — 마지막 출력의 기울기가 '첫 시점' 입력까지
        얼마나 살아남는지를 시퀀스 길이 L 을 늘려가며 측정 (RNN vs LSTM)
        → 오늘 배운 vanishing gradient 를 학습 없이 직접 관측
  3부: 파라미터 수 비교 — 바닐라 RNN < GRU < LSTM (게이트가 늘수록 비용도 늘)
"""

import torch
import torch.nn as nn


# =====================================================================
# 1부 — LSTM 셀을 손으로 한 스텝 (게이트의 정체를 눈으로 확인)
# =====================================================================
def part1_lstm_by_hand():
    print("=" * 62)
    print("1부 — LSTM 셀 한 스텝을 손으로 (게이트 4개 + 셀 상태 c_t)")
    print("=" * 62)
    torch.manual_seed(0)
    H, D = 3, 1  # hidden 크기 3, 입력 차원 1

    def W():
        return torch.randn(H, D) * 0.5

    def U():
        return torch.randn(H, H) * 0.5

    Wf, Uf, bf = W(), U(), torch.zeros(H)  # forget 게이트
    Wi, Ui, bi = W(), U(), torch.zeros(H)  # input 게이트
    Wg, Ug, bg = W(), U(), torch.zeros(H)  # 후보 기억(candidate) g̃
    Wo, Uo, bo = W(), U(), torch.zeros(H)  # output 게이트

    def sig(x):
        return torch.sigmoid(x)

    def lstm_step(x_t, h_prev, c_prev):
        f = sig(Wf @ x_t + Uf @ h_prev + bf)          # 무엇을 '잊을까' (0=지움,1=유지)
        i = sig(Wi @ x_t + Ui @ h_prev + bi)          # 무엇을 '새로 쓸까'
        g = torch.tanh(Wg @ x_t + Ug @ h_prev + bg)   # 새로 쓸 후보 내용
        o = sig(Wo @ x_t + Uo @ h_prev + bo)          # 무엇을 '내보낼까'
        c = f * c_prev + i * g                        # 셀 상태 = 유지 + 새로 쓰기 (덧셈!)
        h = o * torch.tanh(c)                         # 은닉 상태 = 셀을 걸러 출력
        return h, c, f

    seq = [torch.tensor([1.0]), torch.tensor([0.0]), torch.tensor([1.0])]
    h = torch.zeros(H)
    c = torch.zeros(H)
    print("  c_0 =", [round(v, 3) for v in c.tolist()], "(빈 장기기억)")
    for t, x_t in enumerate(seq, 1):
        h, c, f = lstm_step(x_t, h, c)
        print(
            f"  x_{t}={x_t.item():.0f} | forget f_{t}=",
            [round(v, 2) for v in f.tolist()],
            "| c_%d=" % t,
            [round(v, 3) for v in c.tolist()],
        )
    print("  → 핵심: c_t = f*c_(t-1) + i*g̃ 로 '덧셈' 갱신된다.")
    print("    RNN 의 반복 '곱셈'(tanh(Whh·h)) 과 달리, 덧셈 경로가 기울기 고속도로다.\n")


# =====================================================================
# 2부 — 기울기 소실(vanishing gradient)을 학습 없이 직접 관측
#   마지막 시점 출력의 손실을 '첫 시점 입력'까지 역전파했을 때,
#   그 기울기 크기가 시퀀스 길이 L 이 커질수록 어떻게 되는가?
# =====================================================================
def grad_to_first(cell, L, seed=0, h=16, forget_bias=None):
    torch.manual_seed(seed)
    Cell = {"RNN": nn.RNN, "LSTM": nn.LSTM}[cell]
    net = Cell(1, h, batch_first=True)
    if cell == "LSTM" and forget_bias is not None:
        # forget 게이트 편향을 크게 → sigmoid(f)≈1 : '기억을 유지하라'로 열어 둠
        # (실제 학습된 장기기억 LSTM 이 스스로 도달하는 상태를 흉내)
        for nm, p in net.named_parameters():
            if "bias" in nm:
                n = p.shape[0] // 4
                p.data[n : 2 * n].fill_(forget_bias)  # 순서: i, f, g, o
    x = torch.randn(1, L, 1, requires_grad=True)      # 입력 시퀀스(기울기 추적)
    out, _ = net(x)
    out[:, -1, :].sum().backward()                    # 마지막 시점 출력에서 역전파
    return x.grad[0, 0, 0].abs().item()               # '첫 시점' 입력이 받는 기울기 크기


def part2_vanishing():
    print("=" * 62)
    print("2부 — 기울기 소실을 눈으로 : 마지막 출력 → '첫 시점' 기울기 크기")
    print("=" * 62)
    print("   L | 바닐라 RNN | LSTM(forget 열림)  ← 길이가 길수록 어떻게 되나")
    print("  ---+------------+------------------")
    for L in [10, 20, 40, 80, 120]:
        rnn = grad_to_first("RNN", L)
        lstm = grad_to_first("LSTM", L, forget_bias=5.0)  # sigmoid(5)=0.993
        print(f"  {L:3d} | {rnn:.2e}  | {lstm:.2e}")
    print("  → RNN 은 L 이 커질수록 기울기가 e-3 → e-34 로 '소멸'한다(먼 과거 학습 실패).")
    print("    LSTM 은 forget 게이트를 열어 두면(f≈1) 덧셈 경로 덕에 기울기가 살아 남는다.\n")


# =====================================================================
# 3부 — 파라미터 수 비교 : RNN < GRU < LSTM (게이트가 늘면 비용도 는다)
# =====================================================================
def n_params(cell, hidden=16):
    Cell = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[cell]
    net = Cell(1, hidden, batch_first=True)
    return sum(p.numel() for p in net.parameters())


def part3_param_cost():
    print("=" * 62)
    print("3부 — 셀별 파라미터 수 (input=1, hidden=16) : 게이트 비용")
    print("=" * 62)
    for cell, gates in [("RNN", "게이트 없음(1개 변환)"), ("GRU", "게이트 3개분(reset·update·후보)"),
                        ("LSTM", "게이트 4개분(forget·input·output·후보)")]:
        print(f"  {cell:4s} | {n_params(cell):5d} params | {gates}")
    print("  → GRU 는 게이트를 2개로 합쳐 LSTM 보다 가볍고 대개 비슷한 성능을 낸다.\n")


if __name__ == "__main__":
    part1_lstm_by_hand()
    part2_vanishing()
    part3_param_cost()
    print("끝. 게이트가 '곱셈 반복(기울기 소실)'을 '덧셈 경로'로 바꿔 먼 과거를 지킨다.")
