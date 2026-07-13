#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day-025 — Phase 2 총정리 & 딥러닝으로 가는 다리
(Phase 2 Recap & the Bridge to Deep Learning)

Phase 2 의 모든 어휘 기반(lexical) 랭킹 모델 — 역색인(Day-016), BM25(Day-017/022),
VSM/코사인(Day-024) — 이 공유하는 '한 가지 근본 약점'을 수치로 못 박는다:
  ▶ 어휘 불일치(vocabulary/lexical mismatch):
    같은 뜻이라도 '글자'가 다르면(car↔automobile↔vehicle) 점수는 0 이다.
표준 라이브러리만 사용(math, collections, re). 결정론적 — 실행하면 아래 출력 그대로 재현.
"""
import sys
import math
import re
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔 한글 깨짐 방지

# ---------------------------------------------------------------------------
# 0) 코퍼스 — 전부 '자동차' 이야기인데, 쓰는 단어만 다르다.
#    d2/d4/d5 만 'automobile' 이라는 글자를 쓴다.
# ---------------------------------------------------------------------------
CORPUS = {
    "d0": "the used car market cooled sharply last quarter",          # car (동의어) — 관련
    "d1": "shopping for a new vehicle on a tight budget",             # vehicle (동의어) — 관련
    "d2": "an automobile factory opened downtown creating new jobs",  # automobile — 그러나 '공장/일자리' 이야기
    "d3": "electric cars are cheaper to run than gas cars",           # cars (동의어) — 관련
    "d4": "automobile automobile automobile clearance discount sale", # automobile 반복 — 키워드 스터핑
    "d5": "how to purchase and register your automobile step by step",# automobile — 구매, 진짜 관련
}


def tok(text):
    return re.findall(r"[a-z]+", text.lower())


class BM25:
    """Day-017/021 에서 만든 BM25 를 표준 라이브러리로 압축 재현."""

    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.ids = list(corpus.keys())
        self.docs = {i: tok(t) for i, t in corpus.items()}
        self.N = len(self.docs)
        self.len = {i: len(d) for i, d in self.docs.items()}
        self.avgdl = sum(self.len.values()) / self.N
        self.tf = {i: Counter(d) for i, d in self.docs.items()}
        df = Counter()
        for d in self.docs.values():
            df.update(set(d))
        self.df = df

    def idf(self, t):
        # BM25 의 확률적 idf (음수 방지형)
        return math.log(1 + (self.N - self.df.get(t, 0) + 0.5) / (self.df.get(t, 0) + 0.5))

    def score(self, i, qtoks):
        s = 0.0
        dl = self.len[i]
        for t in qtoks:
            f = self.tf[i].get(t, 0)
            if f == 0:
                continue
            denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            s += self.idf(t) * (f * (self.k1 + 1)) / denom
        return s

    def rank(self, query):
        q = tok(query)
        scored = [(i, self.score(i, q)) for i in self.ids]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return q, scored


def show(bm, query, note=""):
    q, ranked = bm.rank(query)
    print(f"\n[질의 Query] {query!r}   (토큰 {q})  {note}")
    for rank, (i, sc) in enumerate(ranked, 1):
        flag = " ← 관련(동의어)이지만 0점!" if sc == 0.0 else ""
        bar = "#" * int(round(sc * 6))
        print(f"  {rank}. {i}  score={sc:6.3f}  {bar:<18} {CORPUS[i]}{flag}")


def main():
    print("=" * 72)
    print("Day-025 — Phase 2 총정리: 어휘 기반 검색의 '벽'을 수치로 보기")
    print("         (Why every lexical ranker shares one wall: vocabulary mismatch)")
    print("=" * 72)

    bm = BM25(CORPUS)
    print(f"\n코퍼스 {bm.N}개 문서 · 평균 길이 avgdl = {bm.avgdl:.2f} 토큰")
    print("모든 문서가 사실상 '자동차'를 말하지만, 쓰는 단어가 제각각이다:")
    print("  car(d0) · vehicle(d1) · automobile(d2,d4,d5) · cars(d3)")

    # (1) 순수 어휘 불일치: 질의어 'automobile' — 동의어 문서는 전부 0점
    print("\n" + "-" * 72)
    print("(1) 질의 = 'automobile'  ->  car/vehicle/cars 문서는 '완벽히' 0점")
    print("-" * 72)
    show(bm, "automobile")

    # (2) 거울상: 질의어 'car' — 이번엔 automobile 문서가 전부 0점.
    #     게다가 'cars'(복수)조차 'car' 와 다른 글자라 0점(형태소/스테밍의 벽).
    print("\n" + "-" * 72)
    print("(2) 질의 = 'car'  ->  거울상: automobile 문서가 0점. 'cars'(복수)도 0점!")
    print("-" * 72)
    show(bm, "car")
    print("  -> d3('electric cars ...')는 뜻이 같은데도 'cars'!='car' 라 0점.")
    print("     자연어의 형태 변화(단/복수)조차 '글자 매칭'은 못 넘는다 (스테밍의 필요).")

    # (3) 키워드 스터핑에 대한 BM25 의 방어(부분적)
    print("\n" + "-" * 72)
    print("(3) 'automobile' 반복(d4)은 BM25 의 tf 포화가 눌러 준다 (Day-017)")
    print("-" * 72)
    q = tok("automobile")
    for i in ("d4", "d5", "d2"):
        f = bm.tf[i]["automobile"]
        print(f"  {i}: tf(automobile)={f}, |d|={bm.len[i]}, "
              f"score={bm.score(i, q):.3f}")
    print("  -> d4 는 3번 반복해도 tf 포화·길이 정규화로 점수가 무한정 오르지 않는다.")
    print("     하지만 이건 '같은 글자'의 과잉만 막을 뿐, '다른 글자=같은 뜻'은 못 잇는다.")

    print("\n" + "=" * 72)
    print("결론: 역색인·BM25·VSM 은 모두 '질의어와 문서어의 글자 겹침'을 센다.")
    print("      -> 동의어·다른 표현은 원천적으로 0점 (어휘 불일치의 벽).")
    print("      이 벽을 넘으려면 '글자'가 아니라 '학습된 의미 벡터'가 필요하다.")
    print("      -> 신경망·임베딩·트랜스포머(Phase 3) 로 가는 이유.")
    print("=" * 72)


if __name__ == "__main__":
    main()
