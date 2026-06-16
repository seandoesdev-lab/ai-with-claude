# korean_tokenize.py — 공백 분리가 한국어에서 왜 어휘 폭발을 일으키나
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

sentences = [
    "학교에 간다",
    "학교에서 공부한다",
    "학교는 멀다",
    "학교를 좋아한다",
]

# 공백으로만 토큰화
ws_tokens = [t for s in sentences for t in s.split()]
print("공백 토큰:    ", ws_tokens)
print("어휘 크기:    ", len(set(ws_tokens)))   # '학교'가 매번 다른 토큰!

# '학교' 가 들어간 어절들 — 전부 서로 다른 토큰으로 셈됨
hak = [t for t in ws_tokens if t.startswith("학교")]
print("'학교' 변형:  ", Counter(hak))
print("→ 사람은 같은 '학교'로 보지만, 공백 토크나이저는", len(set(hak)), "개의 다른 토큰으로 봅니다.")
