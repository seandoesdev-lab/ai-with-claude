# tokenizer_basics.py — 공백 분리 → 정규식 토큰화로 진화시키기
import re
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")   # Windows 콘솔 한글 깨짐 방지

text = "I'm loving NLP!! Natural-language processing isn't easy, but it's fun."

# (1) 가장 순진한 방법: 공백으로만 자르기
naive = text.split()
print("① split():       ", naive)

# (2) 소문자화 + 정규식: 단어(영문/숫자/아포스트로피)와 구두점을 분리
#     \w+(?:'\w+)? = 단어, 그리고 don't·it's 의 축약형을 한 토큰으로 유지
pattern = re.compile(r"\w+(?:'\w+)?|[^\w\s]")
regex_tokens = pattern.findall(text.lower())
print("② regex 토큰화:  ", regex_tokens)

# (3) 어휘(vocabulary)와 type/token 수 세기
counts = Counter(regex_tokens)
print("토큰 수(tokens):  ", len(regex_tokens))
print("어휘 크기(types): ", len(counts))
print("최빈 토큰 3개:    ", counts.most_common(3))
