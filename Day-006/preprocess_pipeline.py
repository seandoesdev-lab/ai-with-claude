# preprocess_pipeline.py — 정규화 → 토큰화 → 불용어 제거 파이프라인
import re
import sys
import unicodedata

sys.stdout.reconfigure(encoding="utf-8")

# 아주 작은 영어 불용어 목록 (실무에선 nltk/spaCy 의 목록을 씀)
STOPWORDS = {"the", "a", "an", "is", "are", "of", "to", "and", "in", "on", "it"}


def normalize(text: str) -> str:
    # 유니코드 정규화(NFKC): 전각/이형 문자를 표준형으로 통일
    text = unicodedata.normalize("NFKC", text)
    # 곱슬 따옴표 ’ → 곧은 따옴표 '  (don’t → don't)
    text = text.replace("’", "'")
    return text.lower()


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+(?:'\w+)?|[^\w\s]", text)


def remove_stopwords(tokens: list[str]) -> list[str]:
    # 불용어와 '단독 구두점'을 함께 걸러냄
    return [t for t in tokens if t not in STOPWORDS and not re.fullmatch(r"[^\w\s]", t)]


def preprocess(text: str) -> list[str]:
    return remove_stopwords(tokenize(normalize(text)))


if __name__ == "__main__":
    doc = "The cat is on the mat, and the DOG is in the house!"
    print("원문:        ", doc)
    print("정규화:      ", normalize(doc))
    print("토큰화:      ", tokenize(normalize(doc)))
    print("불용어 제거: ", preprocess(doc))
