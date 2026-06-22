# paper_reader.py — 논문 읽기 입문 실습 도구
# (1) 3-pass 독해 보조: 초록을 구조적으로 훑고(pass 1)
# (2) 핵심 5질문 체크리스트 자동 점검
# (3) 한 장짜리 요약 노트(Markdown) 자동 생성
#
# 외부 라이브러리 없음 — 파이썬 표준 라이브러리만 사용(re, collections).
# 실행:  uv run python paper_reader.py
import re
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔 한글 깨짐 방지

# ──────────────────────────────────────────────────────────────────────────
# 실습용 진짜 초록(abstract) — Bengio et al. (2003),
# "A Neural Probabilistic Language Model" (요지를 옮긴 학습용 텍스트)
# 이 논문은 Day-009(임베딩)·Day-011(n-gram) 과 직결된다: n-gram 의
# '데이터 희소성'을 단어를 벡터로 학습해 푸는 첫걸음.
# ──────────────────────────────────────────────────────────────────────────
ABSTRACT = """\
A goal of statistical language modeling is to learn the joint probability
function of sequences of words in a language. This is intrinsically difficult
because of the curse of dimensionality: a word sequence on which the model
will be tested is likely to be different from all the word sequences seen
during training. Traditional but very successful approaches based on n-grams
obtain generalization by concatenating very short overlapping sequences seen
in the training set. We propose to fight the curse of dimensionality by
learning a distributed representation for words which allows each training
sentence to inform the model about an exponential number of semantically
neighboring sentences. The model learns simultaneously a distributed
representation for each word along with the probability function for word
sequences, expressed in terms of these representations. Generalization is
obtained because a sequence of words that has never been seen before gets
high probability if it is made of words that are similar to words forming an
already seen sentence. We report experiments showing that this approach
significantly improves on state-of-the-art n-gram models, and that the
proposed approach allows to take advantage of longer contexts.
"""

PAPER_META = {
    "title": "A Neural Probabilistic Language Model",
    "authors": "Bengio, Ducharme, Vincent, Jauvin",
    "year": 2003,
    "venue": "JMLR",
}

# 논문에서 자주 쓰이는 '신호어(signal words)' — 문제/한계/기여/검증을 가리키는 단서.
SIGNALS = {
    "problem": ["difficult", "curse", "problem", "challenge", "hard"],
    "limitation_of_prior": ["traditional", "n-grams", "however", "but", "fail"],
    "key_idea": ["propose", "distributed representation", "learn"],
    "validation": ["experiments", "report", "improves", "state-of-the-art", "results"],
}


def sentences(text):
    """아주 단순한 문장 분리기(마침표 기준). 학습용으로 충분."""
    flat = re.sub(r"\s+", " ", text.strip())
    return [s.strip() for s in re.split(r"(?<=[.!?]) ", flat) if s.strip()]


def pass1_skim(text):
    """3-pass 중 1단계(훑기): 길이·문장 수·핵심 키워드를 한눈에."""
    sents = sentences(text)
    words = re.findall(r"[a-zA-Z][a-zA-Z'-]+", text.lower())
    stop = {
        "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is",
        "are", "be", "by", "that", "this", "with", "as", "it", "we", "will",
        "which", "each", "about", "these", "them", "from", "has", "have",
    }
    content = [w for w in words if w not in stop and len(w) > 2]
    top = Counter(content).most_common(8)
    print("── PASS 1: 훑기(skim) ──")
    print(f"  문장 수      : {len(sents)}")
    print(f"  단어 수      : {len(words)}")
    print(f"  첫 문장(주제): {sents[0]}")
    print(f"  끝 문장(결론): {sents[-1]}")
    print(f"  반복 키워드  : {', '.join(f'{w}×{c}' for w, c in top)}")
    return sents


def find_signals(text):
    """핵심 5질문 단서를 신호어로 자동 탐지(완벽하진 않아도 길잡이)."""
    low = text.lower()
    print("\n── 핵심 5질문 단서 자동 탐지 ──")
    labels = {
        "problem": "① 무슨 문제를 푸나?",
        "limitation_of_prior": "② 기존 방법의 한계는?",
        "key_idea": "③ 핵심 아이디어는?",
        "validation": "④ 어떻게 검증했나?",
    }
    for key, label in labels.items():
        hits = [w for w in SIGNALS[key] if w in low]
        mark = "✅" if hits else "⚠️ (직접 찾아보세요)"
        extra = ("단서: " + ", ".join(hits)) if hits else ""
        print(f"  {label} {mark} {extra}")
    print("  ⑤ 한계/남은 문제는?  ⚠️ 보통 초록엔 약함 — 본문 Discussion 에서 확인")


def make_summary_note(meta, answers):
    """한 장짜리 요약 노트(Markdown) 생성 — 정독 회차마다 이 템플릿을 채운다."""
    md = [
        f"# 📄 요약 노트 — {meta['title']} ({meta['year']})",
        f"> {meta['authors']} · {meta['venue']} {meta['year']}",
        "",
        "| 5질문 | 한 줄 답 |",
        "|------|---------|",
        f"| ① 문제 | {answers['problem']} |",
        f"| ② 기존 한계 | {answers['prior']} |",
        f"| ③ 핵심 아이디어 | {answers['idea']} |",
        f"| ④ 검증 | {answers['validation']} |",
        f"| ⑤ 한계 | {answers['limitation']} |",
        "",
        f"**한 줄 요약**: {answers['oneliner']}",
    ]
    return "\n".join(md)


if __name__ == "__main__":
    print(f"제목: {PAPER_META['title']} ({PAPER_META['year']})\n")

    pass1_skim(ABSTRACT)
    find_signals(ABSTRACT)

    # 내가 직접 읽고 채운 5질문(예시 답) — 정독 시 본인 말로 다시 쓴다.
    my_answers = {
        "problem": "단어 시퀀스의 결합확률을 학습하는 통계적 언어모델",
        "prior": "n-gram 은 짧은 부분열만 외워 차원의 저주·데이터 희소성에 약함",
        "idea": "단어를 '분산표현(distributed representation, 임베딩)'으로 학습해 유사 단어로 일반화",
        "validation": "n-gram 대비 perplexity 개선, 더 긴 문맥 활용을 실험으로 입증",
        "limitation": "학습이 무겁고 느림(당시 기준) — 초록엔 명시 약함, 본문서 확인",
        "oneliner": "단어를 벡터로 학습해 n-gram 의 희소성을 부드러운 일반화로 푼 신경망 LM의 출발점",
    }
    note = make_summary_note(PAPER_META, my_answers)
    print("\n── 자동 생성된 한 장짜리 요약 노트(Markdown) ──\n")
    print(note)

    # 파일로도 저장(정독 회차마다 이렇게 노트가 쌓인다)
    out = "summary_bengio2003.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write(note + "\n")
    print(f"\n[저장됨] {out}")
