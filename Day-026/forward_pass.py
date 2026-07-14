# -*- coding: utf-8 -*-
"""
Day-026 — 신경망의 첫 원리: 퍼셉트론 → MLP, 그리고 순전파(forward pass)
Neural Networks from First Principles: Perceptron -> MLP & the Forward Pass

핵심 메시지 (What this file proves, numerically):
  (1) 뉴런 하나 = 선형 모델 w·x + b + 활성함수.  (AND 같은 '선형 분리 가능' 문제는 푼다)
  (2) XOR 는 직선 하나로 못 가른다 -> 뉴런 하나로는 절대 못 푼다.
  (3) 비선형 없이 층을 쌓으면 결국 '한 층'으로 붕괴한다:  W2(W1 x) = (W2 W1) x.
  (4) 활성함수(ReLU)를 끼운 2층 MLP 는 XOR 를 정확히 푼다. (순전파를 손으로 따라가며 확인)

의존성: numpy 만 사용.  실행:  uv add numpy;  uv run python forward_pass.py
출력은 결정론적(random 없음) — 아래 노트의 '실제 출력'과 그대로 일치한다.
"""

import numpy as np


# ----------------------------------------------------------------------
# 활성함수 (Activation functions)
# ----------------------------------------------------------------------
def step(z):
    """계단 함수(step): z>=0 이면 1, 아니면 0. 초창기 퍼셉트론의 활성함수."""
    return (z >= 0).astype(int)


def relu(z):
    """ReLU: max(0, z). 오늘날 신경망의 표준 비선형 활성함수."""
    return np.maximum(0.0, z)


# ----------------------------------------------------------------------
# (1) 뉴런 하나 = 선형 모델 + 활성함수  ->  AND 게이트는 풀 수 있다
# ----------------------------------------------------------------------
def neuron(x, w, b, act):
    """한 뉴런의 순전파: 가중합 z = w·x + b 를 구하고 활성함수를 통과."""
    z = float(np.dot(w, x)) + b        # w^T x + b  (Day-004/Day-014 의 선형 점수와 같은 식!)
    return act(np.array(z))


def demo_single_neuron_and():
    print("-" * 72)
    print("(1) 뉴런 하나 = w·x + b + 활성함수  ->  AND 게이트(선형 분리 가능)는 푼다")
    print("-" * 72)
    w = np.array([1.0, 1.0]); b = -1.5   # x1+x2-1.5 >= 0  <=>  둘 다 1일 때만 참
    print(f"  가중치 w={w.tolist()}, 편향 b={b}, 활성함수=step")
    print("  x1 x2 |  z=w·x+b  | step(z) | AND 정답")
    for x1, x2 in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        x = np.array([x1, x2], dtype=float)
        z = float(np.dot(w, x)) + b
        y = int(step(np.array(z)))
        print(f"   {x1}  {x2} |  {z:+.1f}    |    {y}    |   {x1 & x2}")
    print("  -> 직선 x1+x2=1.5 하나로 (1,1)만 반대편에 놓여 완벽히 분리된다.\n")


# ----------------------------------------------------------------------
# (2) XOR 는 직선 하나로 못 가른다  ->  뉴런 하나로는 불가능
# ----------------------------------------------------------------------
def demo_single_neuron_xor_fails():
    print("-" * 72)
    print("(2) XOR: 직선 하나로 못 가른다 -> 뉴런 하나로는 어떤 w,b 로도 실패")
    print("-" * 72)
    xor = {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 0}
    # 'OR'에 가까운 최선의 시도 한 가지를 보여준다: w=[1,1], b=-0.5
    w = np.array([1.0, 1.0]); b = -0.5
    print(f"  최선의 시도 예: w={w.tolist()}, b={b}, 활성함수=step")
    print("  x1 x2 |  z=w·x+b  | 예측 | XOR 정답 | 맞음?")
    correct = 0
    for (x1, x2), t in xor.items():
        x = np.array([x1, x2], dtype=float)
        z = float(np.dot(w, x)) + b
        y = int(step(np.array(z)))
        ok = (y == t)
        correct += ok
        print(f"   {x1}  {x2} |  {z:+.1f}    |  {y}   |    {t}     |  {'O' if ok else 'X'}")
    print(f"  -> 4개 중 {correct}개만 맞음. (1,1)이 늘 어긋난다.")
    print("     이유: XOR 의 정답 1들 (0,1),(1,0) 은 대각선으로 마주보아,")
    print("     어떤 직선(w·x+b=0)으로도 0들과 갈라낼 수 없다 (선형 분리 불가능).\n")


# ----------------------------------------------------------------------
# (3) 비선형 없이 층을 쌓으면 '한 층'으로 붕괴한다
# ----------------------------------------------------------------------
def demo_linear_collapse():
    print("-" * 72)
    print("(3) 활성함수 없이 층을 쌓으면?  W2(W1 x) = (W2·W1) x  ->  결국 선형 1층")
    print("-" * 72)
    # 임의로 고정한 두 개의 '선형 층' (활성함수 없음)
    W1 = np.array([[2.0, -1.0],
                   [0.5,  3.0]])
    W2 = np.array([[1.0,  4.0],
                   [-2.0, 1.0]])
    x = np.array([1.0, 2.0])

    two_layers = W2 @ (W1 @ x)      # 층을 순서대로 통과 (비선형 없음)
    collapsed  = (W2 @ W1) @ x      # 두 행렬을 미리 곱해 '한 층'으로
    print(f"  입력 x = {x.tolist()}")
    print(f"  W2 @ (W1 @ x) = {two_layers.tolist()}   (2개 층을 차례로)")
    print(f"  (W2 @ W1) @ x = {collapsed.tolist()}   (하나로 합친 층)")
    print(f"  두 결과가 같은가? -> {np.allclose(two_layers, collapsed)}")
    print("  -> 비선형이 없으면 층을 100개 쌓아도 표현력은 선형 1층과 똑같다.")
    print("     그래서 '깊게' 쌓는 의미가 없다. 활성함수(비선형)가 딥러닝의 심장인 이유.\n")


# ----------------------------------------------------------------------
# (4) ReLU 를 끼운 2층 MLP 는 XOR 를 정확히 푼다 (순전파 추적)
# ----------------------------------------------------------------------
def demo_mlp_xor():
    print("-" * 72)
    print("(4) 2층 MLP + ReLU 로 XOR 풀기 — 순전파(forward pass)를 손으로 따라가기")
    print("-" * 72)
    # 은닉층(hidden layer): 2개 뉴런, ReLU
    #   h1 = ReLU(x1 + x2 + 0),  h2 = ReLU(x1 + x2 - 1)
    W1 = np.array([[1.0, 1.0],
                   [1.0, 1.0]])
    b1 = np.array([0.0, -1.0])
    # 출력층(output layer): 1개 뉴런, 선형
    #   y = 1*h1 + (-2)*h2 + 0
    W2 = np.array([[1.0, -2.0]])
    b2 = np.array([0.0])

    print("  구조: 입력 2 -> [은닉 2, ReLU] -> [출력 1, 선형]")
    print("  은닉:  h = ReLU(W1·x + b1),   W1=[[1,1],[1,1]], b1=[0,-1]")
    print("  출력:  y = W2·h + b2,          W2=[1,-2],       b2=[0]")
    print()
    print("  x1 x2 |   h(은닉, ReLU 후)   |   y=출력  | XOR 정답")
    xor = {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 0}
    allok = True
    for (x1, x2), t in xor.items():
        x = np.array([x1, x2], dtype=float)
        h = relu(W1 @ x + b1)          # 은닉층 순전파 (비선형!)
        y = float((W2 @ h + b2)[0])    # 출력층 순전파 (선형) — 1원소 벡터에서 스칼라 추출
        allok &= (round(y) == t)
        print(f"   {x1}  {x2} |   [{h[0]:.0f}, {h[1]:.0f}]{' '*13}|    {y:+.0f}    |   {t}")
    print(f"  -> 4개 모두 정답? {allok}")
    print("  핵심: 은닉층 h2=ReLU(x1+x2-1) 가 '둘 다 1일 때'만 켜져,")
    print("        출력에서 -2 로 눌러 (1,1)을 0 으로 되돌린다.")
    print("        비선형(ReLU) 덕분에 '두 직선'을 접어 곡선 경계를 만든 셈.\n")


def main():
    print("=" * 72)
    print("Day-026 — 신경망의 첫 원리: 퍼셉트론 -> MLP, 그리고 순전파(forward pass)")
    print("         From a single neuron to an MLP: why we need nonlinearity")
    print("=" * 72 + "\n")
    demo_single_neuron_and()
    demo_single_neuron_xor_fails()
    demo_linear_collapse()
    demo_mlp_xor()
    print("=" * 72)
    print("결론: 뉴런 하나 = 선형 모델(w·x+b) + 활성함수.  선형 문제(AND)는 풀지만")
    print("      XOR 같은 비선형 문제는 못 푼다.  층을 쌓되 '비선형 활성함수'를 끼워야")
    print("      비로소 표현력이 커진다(선형 붕괴 회피).  이것이 MLP·딥러닝의 출발점.")
    print("      다음: 이 가중치를 '데이터로 학습'하는 법 = 역전파(backpropagation).")
    print("=" * 72)


if __name__ == "__main__":
    main()
